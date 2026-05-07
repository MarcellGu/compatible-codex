use crate::common::ResponseEvent;
use crate::common::ResponseStream;
use crate::error::ApiError;
use crate::telemetry::SseTelemetry;
use codex_client::ByteStream;
use codex_client::StreamResponse;
use eventsource_stream::Eventsource;
use futures::StreamExt;
use serde::Deserialize;
use serde::Serialize;
use serde_json::Value;
use std::sync::Arc;
use std::sync::OnceLock;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio::time::Instant;
use tokio::time::timeout;
use tracing::debug;
use tracing::trace;

#[derive(Debug, Serialize, Deserialize)]
pub struct AnthropicStreamEvent {
    pub event: Option<String>,
    pub data: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct AnthropicMessageStart {
    pub r#type: String,
    pub message: AnthropicMessage,
}

#[derive(Debug, Deserialize)]
pub struct AnthropicMessage {
    pub id: Option<String>,
    pub r#type: Option<String>,
    pub role: Option<String>,
    pub content: Option<Vec<AnthropicContentBlock>>,
    pub model: Option<String>,
    pub stop_reason: Option<String>,
    pub stop_sequence: Option<String>,
    pub usage: Option<AnthropicUsage>,
}

#[derive(Debug, Deserialize)]
pub struct AnthropicContentBlock {
    pub r#type: String,
    pub text: Option<String>,
    pub id: Option<String>,
    pub name: Option<String>,
    pub input: Option<String>,
    pub content: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct AnthropicUsage {
    pub input_tokens: i64,
    pub output_tokens: i64,
}

#[derive(Debug, Deserialize)]
pub struct AnthropicContentBlockStart {
    pub r#type: String,
    pub index: usize,
    pub content_block: AnthropicContentBlock,
}

#[derive(Debug, Deserialize)]
pub struct AnthropicContentBlockDelta {
    pub r#type: String,
    pub index: usize,
    pub delta: AnthropicDelta,
}

#[derive(Debug, Deserialize)]
pub struct AnthropicDelta {
    #[serde(rename = "type")]
    pub delta_type: String,
    pub text: Option<String>,
    pub partial_json: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct AnthropicMessageDelta {
    pub r#type: String,
    pub delta: AnthropicMessageStopDelta,
    pub usage: Option<AnthropicUsage>,
}

#[derive(Debug, Deserialize)]
pub struct AnthropicMessageStopDelta {
    pub stop_reason: Option<String>,
    pub stop_sequence: Option<String>,
}

impl From<&AnthropicUsage> for codex_protocol::protocol::TokenUsage {
    fn from(val: &AnthropicUsage) -> Self {
        codex_protocol::protocol::TokenUsage {
            input_tokens: val.input_tokens,
            cached_input_tokens: 0,
            output_tokens: val.output_tokens,
            reasoning_output_tokens: 0,
            total_tokens: val.input_tokens + val.output_tokens,
        }
    }
}

pub fn spawn_anthropic_stream(
    stream_response: StreamResponse,
    idle_timeout: Duration,
    telemetry: Option<Arc<dyn SseTelemetry>>,
    turn_state: Option<Arc<OnceLock<String>>>,
) -> ResponseStream {
    let upstream_request_id = stream_response
        .headers
        .get("x-request-id")
        .and_then(|value| value.to_str().ok())
        .map(str::to_string);

    if let Some(turn_state) = turn_state.as_ref()
        && let Some(header_value) = stream_response
            .headers
            .get("x-codex-turn-state")
            .and_then(|v| v.to_str().ok())
    {
        let _ = turn_state.set(header_value.to_string());
    }

    let (tx_event, rx_event) = mpsc::channel::<Result<ResponseEvent, ApiError>>(1600);
    tokio::spawn(async move {
        process_anthropic_sse(stream_response.bytes, tx_event, idle_timeout, telemetry).await;
    });

    ResponseStream {
        rx_event,
        upstream_request_id,
    }
}

pub async fn process_anthropic_sse(
    stream: ByteStream,
    tx_event: mpsc::Sender<Result<ResponseEvent, ApiError>>,
    idle_timeout: Duration,
    telemetry: Option<Arc<dyn SseTelemetry>>,
) {
    let mut stream = stream.eventsource();
    let mut response_id: Option<String> = None;
    let mut accumulated_text = String::new();
    let mut tool_call_buffer: Option<(usize, String, String, String)> = None;
    let mut final_usage: Option<codex_protocol::protocol::TokenUsage> = None;
    let mut stop_reason: Option<String> = None;
    let mut event_type = String::new();

    loop {
        let start = Instant::now();
        let response = timeout(idle_timeout, stream.next()).await;
        if let Some(t) = telemetry.as_ref() {
            t.on_sse_poll(&response, start.elapsed());
        }

        let sse = match response {
            Ok(Some(Ok(sse))) => sse,
            Ok(Some(Err(e))) => {
                debug!("Anthropic SSE Error: {e:#}");
                let _ = tx_event.send(Err(ApiError::Stream(e.to_string()))).await;
                return;
            }
            Ok(None) => {
                let _ = tx_event.send(Err(ApiError::Stream("stream closed unexpectedly".into()))).await;
                return;
            }
            Err(_) => {
                let _ = tx_event.send(Err(ApiError::Stream("idle timeout waiting for SSE".into()))).await;
                return;
            }
        };

        trace!("Anthropic SSE event: {}", &sse.data);

        let event: AnthropicStreamEvent = match serde_json::from_str(&sse.data) {
            Ok(event) => event,
            Err(e) => {
                debug!("Failed to parse Anthropic SSE event: {e}, data: {}", &sse.data);
                continue;
            }
        };

        event_type = event.event.unwrap_or_default();

        match event_type.as_str() {
            "message_start" => {
                if let Some(data) = event.data {
                    if let Ok(msg_start) = serde_json::from_str::<AnthropicMessageStart>(&data) {
                        response_id = msg_start.message.id.clone();
                        let _ = tx_event.send(Ok(ResponseEvent::Created)).await;
                        if let Some(ref usage) = msg_start.message.usage {
                            final_usage = Some(codex_protocol::protocol::TokenUsage::from(usage));
                        }
                    }
                }
            }
            "content_block_start" => {
                if let Some(data) = event.data {
                    if let Ok(cb_start) = serde_json::from_str::<AnthropicContentBlockStart>(&data) {
                        if cb_start.content_block.r#type == "tool_use" {
                            let name = cb_start.content_block.name.unwrap_or_default();
                            tool_call_buffer = Some((cb_start.index, String::new(), String::new(), name));
                        } else if cb_start.content_block.r#type == "text" {
                            if !accumulated_text.is_empty() {
                                let message_item = codex_protocol::models::ResponseItem::Message {
                                    id: None,
                                    role: "assistant".to_string(),
                                    content: vec![codex_protocol::models::ContentItem::OutputText {
                                        text: accumulated_text.clone(),
                                    }],
                                    phase: None,
                                };
                                let _ = tx_event.send(Ok(ResponseEvent::OutputItemDone(message_item))).await;
                                accumulated_text.clear();
                            }
                        }
                    }
                }
            }
            "content_block_delta" => {
                if let Some(data) = event.data {
                    if let Ok(cb_delta) = serde_json::from_str::<AnthropicContentBlockDelta>(&data) {
                        match cb_delta.delta.delta_type.as_str() {
                            "text_delta" => {
                                if let Some(text) = cb_delta.delta.text {
                                    accumulated_text.push_str(&text);
                                    let _ = tx_event.send(Ok(ResponseEvent::OutputTextDelta(text))).await;
                                }
                            }
                            "input_json_delta" => {
                                if let Some(ref mut buffered) = tool_call_buffer {
                                    if let Some(partial) = cb_delta.delta.partial_json {
                                        buffered.2.push_str(&partial);
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
            "message_delta" => {
                if let Some(data) = event.data {
                    if let Ok(msg_delta) = serde_json::from_str::<AnthropicMessageDelta>(&data) {
                        stop_reason = msg_delta.delta.stop_reason.clone();
                        if let Some(usage) = msg_delta.usage {
                            final_usage = Some(codex_protocol::protocol::TokenUsage::from(&usage));
                        }
                    }
                }
            }
            "message_stop" => {
                if !accumulated_text.is_empty() {
                    let message_item = codex_protocol::models::ResponseItem::Message {
                        id: None,
                        role: "assistant".to_string(),
                        content: vec![codex_protocol::models::ContentItem::OutputText {
                            text: accumulated_text.clone(),
                        }],
                        phase: None,
                    };
                    let _ = tx_event.send(Ok(ResponseEvent::OutputItemDone(message_item))).await;
                    accumulated_text.clear();
                }

                if let Some((_index, _id, arguments, name)) = tool_call_buffer.take() {
                    if !arguments.is_empty() || !name.is_empty() {
                        let tool_call_id = format!("call_{}", response_id.as_deref().unwrap_or("0"));
                        let function_call_item = codex_protocol::models::ResponseItem::FunctionCall {
                            id: None,
                            name,
                            namespace: None,
                            arguments,
                            call_id: tool_call_id,
                        };
                        let _ = tx_event.send(Ok(ResponseEvent::OutputItemDone(function_call_item))).await;
                    }
                }

                let _ = tx_event.send(Ok(ResponseEvent::Completed {
                    response_id: response_id.unwrap_or_else(|| "unknown".to_string()),
                    token_usage: final_usage.take(),
                    end_turn: Some(true),
                })).await;
                return;
            }
            "error" => {
                if let Some(data) = event.data {
                    debug!("Anthropic error event: {}", data);
                    let _ = tx_event.send(Err(ApiError::Stream(data))).await;
                }
                return;
            }
            _ => {
                trace!("unhandled anthropic event: {}", event_type);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;
    use futures::stream;
    use http::HeaderMap;
    use http::StatusCode;

    async fn run_anthropic_sse(events: Vec<(&str, &str)>) -> Vec<ResponseEvent> {
        let mut body = String::new();
        for (event_type, data) in events {
            body.push_str(&format!("event: {}\n", event_type));
            body.push_str(&format!("data: {}\n", data));
            body.push_str("\n");
        }

        let (tx, mut rx) = mpsc::channel::<Result<ResponseEvent, ApiError>>(8);
        let bytes = stream::iter(vec![Ok(Bytes::from(body))]);
        let stream_response = StreamResponse {
            status: StatusCode::OK,
            headers: HeaderMap::new(),
            bytes: Box::pin(bytes),
        };

        tokio::spawn(process_anthropic_sse(
            stream_response.bytes,
            tx,
            Duration::from_secs(1),
            None,
        ));

        let mut out = Vec::new();
        while let Some(ev) = rx.recv().await {
            out.push(ev.expect("channel closed"));
        }
        out
    }

    #[tokio::test]
    async fn parses_anthropic_text_stream() {
        let events = run_anthropic_sse(vec![
            ("message_start", r#"{"type":"message_start","message":{"id":"msg_123","type":"message","role":"assistant","content":[],"model":"claude-3-5-sonnet-20241022"}}"#),
            ("content_block_start", r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#),
            ("content_block_delta", r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}"#),
            ("content_block_delta", r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"!"}}"#),
            ("content_block_stop", r#"{"type":"content_block_stop","index":0}"#),
            ("message_delta", r#"{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":10}}"#),
            ("message_stop", r#"{"type":"message_stop"}"#),
        ])
        .await;

        assert!(!events.is_empty());
        assert_matches!(events.first(), Some(ResponseEvent::Created));
    }
}
