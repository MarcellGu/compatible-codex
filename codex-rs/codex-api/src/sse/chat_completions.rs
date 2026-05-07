use crate::common::ResponseEvent;
use crate::common::ResponseStream;
use crate::error::ApiError;
use crate::rate_limits::parse_all_rate_limits;
use crate::telemetry::SseTelemetry;
use codex_client::ByteStream;
use codex_client::StreamResponse;
use codex_client::TransportError;
use codex_protocol::models::ResponseItem;
use codex_protocol::protocol::TokenUsage;
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

const OPENAI_MODEL_HEADER: &str = "openai-model";
const REQUEST_ID_HEADER: &str = "x-request-id";

#[derive(Debug, Serialize, Deserialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub choices: Vec<ChatCompletionChoice>,
    pub model: Option<String>,
    pub usage: Option<ChatCompletionUsage>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ChatCompletionChoice {
    pub index: usize,
    pub delta: ChatCompletionDelta,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ChatCompletionDelta {
    #[serde(default)]
    pub role: Option<String>,
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub tool_calls: Option<Vec<ChatCompletionToolCall>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ChatCompletionToolCall {
    pub index: Option<usize>,
    pub id: Option<String>,
    #[serde(rename = "type")]
    pub tool_type: Option<String>,
    pub function: Option<ChatCompletionFunctionCall>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ChatCompletionFunctionCall {
    pub name: Option<String>,
    pub arguments: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ChatCompletionUsage {
    pub prompt_tokens: Option<i64>,
    pub completion_tokens: Option<i64>,
    pub total_tokens: Option<i64>,
    pub prompt_tokens_details: Option<PromptTokensDetails>,
    pub completion_tokens_details: Option<CompletionTokensDetails>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PromptTokensDetails {
    pub cached_tokens: Option<i64>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CompletionTokensDetails {
    pub reasoning_tokens: Option<i64>,
}

impl From<&ChatCompletionUsage> for TokenUsage {
    fn from(val: &ChatCompletionUsage) -> Self {
        TokenUsage {
            input_tokens: val.prompt_tokens.unwrap_or(0),
            cached_input_tokens: val.prompt_tokens_details.as_ref().map(|d| d.cached_tokens.unwrap_or(0)).unwrap_or(0),
            output_tokens: val.completion_tokens.unwrap_or(0),
            reasoning_output_tokens: val.completion_tokens_details.as_ref().map(|d| d.reasoning_tokens.unwrap_or(0)).unwrap_or(0),
            total_tokens: val.total_tokens.unwrap_or(0),
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct ChatCompletionStreamEvent {
    #[serde(rename = "type")]
    pub event_type: Option<String>,
    pub id: Option<String>,
    pub model: Option<String>,
    pub choices: Option<Vec<ChatCompletionChoice>>,
    pub usage: Option<ChatCompletionUsage>,
    pub delta: Option<ChatCompletionDelta>,
    #[serde(default)]
    pub error: Option<ChatCompletionError>,
}

#[derive(Debug, Deserialize)]
pub struct ChatCompletionError {
    pub code: Option<String>,
    pub message: Option<String>,
    pub param: Option<String>,
    pub r#type: Option<String>,
}

impl ChatCompletionStreamEvent {
    pub fn parse_chunk(&self) -> Option<ChatCompletionChunk> {
        Some(ChatCompletionChunk {
            id: self.id.clone()?,
            choices: self.choices.clone()?,
            model: self.model.clone(),
            usage: self.usage.clone(),
        })
    }
}

pub fn spawn_chat_completion_stream(
    stream_response: StreamResponse,
    idle_timeout: Duration,
    telemetry: Option<Arc<dyn SseTelemetry>>,
    turn_state: Option<Arc<OnceLock<String>>>,
) -> ResponseStream {
    let rate_limit_snapshots = parse_all_rate_limits(&stream_response.headers);
    let server_model = stream_response
        .headers
        .get(OPENAI_MODEL_HEADER)
        .and_then(|v| v.to_str().ok())
        .map(ToString::to_string);
    let upstream_request_id = stream_response
        .headers
        .get(REQUEST_ID_HEADER)
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
        if let Some(model) = server_model {
            let _ = tx_event.send(Ok(ResponseEvent::ServerModel(model))).await;
        }
        for snapshot in rate_limit_snapshots {
            let _ = tx_event.send(Ok(ResponseEvent::RateLimits(snapshot))).await;
        }
        process_chat_completion_sse(stream_response.bytes, tx_event, idle_timeout, telemetry).await;
    });

    ResponseStream {
        rx_event,
        upstream_request_id,
    }
}

pub async fn process_chat_completion_sse(
    stream: ByteStream,
    tx_event: mpsc::Sender<Result<ResponseEvent, ApiError>>,
    idle_timeout: Duration,
    telemetry: Option<Arc<dyn SseTelemetry>>,
) {
    let mut stream = stream.eventsource();
    let mut response_error: Option<ApiError> = None;
    let mut last_server_model: Option<String> = None;
    let mut accumulated_content = String::new();
    let mut tool_call_buffer: Option<ChatCompletionToolCall> = None;
    let mut response_id: Option<String> = None;
    let mut final_usage: Option<TokenUsage> = None;

    loop {
        let start = Instant::now();
        let response = timeout(idle_timeout, stream.next()).await;
        if let Some(t) = telemetry.as_ref() {
            t.on_sse_poll(&response, start.elapsed());
        }

        let sse = match response {
            Ok(Some(Ok(sse))) => sse,
            Ok(Some(Err(e))) => {
                debug!("SSE Error: {e:#}");
                let _ = tx_event.send(Err(ApiError::Stream(e.to_string()))).await;
                return;
            }
            Ok(None) => {
                let error = response_error.unwrap_or(ApiError::Stream(
                    "stream closed before completion".into(),
                ));
                let _ = tx_event.send(Err(error)).await;
                return;
            }
            Err(_) => {
                let _ = tx_event
                    .send(Err(ApiError::Stream("idle timeout waiting for SSE".into())))
                    .await;
                return;
            }
        };

        trace!("Chat Completion SSE event: {}", &sse.data);

        let event: ChatCompletionStreamEvent = match serde_json::from_str(&sse.data) {
            Ok(event) => event,
            Err(e) => {
                debug!("Failed to parse Chat Completion SSE event: {e}, data: {}", &sse.data);
                continue;
            }
        };

        if let Some(model) = event.model.clone()
            && last_server_model.as_deref() != Some(model.as_str())
        {
            if tx_event.send(Ok(ResponseEvent::ServerModel(model.clone()))).await.is_err() {
                return;
            }
            last_server_model = Some(model);
        }

        if let Some(ref error) = event.error {
            let message = error.message.clone().unwrap_or_else(|| "Unknown error".to_string());
            let api_error = if error.code.as_deref() == Some("context_length_exceeded") {
                ApiError::ContextWindowExceeded
            } else if error.code.as_deref() == Some("rate_limit_exceeded") {
                ApiError::RateLimit(message)
            } else {
                ApiError::Stream(message)
            };
            let _ = tx_event.send(Err(api_error)).await;
            return;
        }

        if let Some(chunk) = event.parse_chunk() {
            if response_id.is_none() {
                response_id = Some(chunk.id.clone());
                let _ = tx_event.send(Ok(ResponseEvent::Created)).await;
            }

            for choice in chunk.choices {
                if let Some(delta) = choice.delta.content {
                    accumulated_content.push_str(&delta);
                    let _ = tx_event.send(Ok(ResponseEvent::OutputTextDelta(delta))).await;
                }

                if let Some(tool_calls) = delta.tool_calls {
                    for tc in tool_calls {
                        if tool_call_buffer.is_none() {
                            tool_call_buffer = Some(tc);
                        } else if let Some(ref mut buffered) = tool_call_buffer {
                            if let Some(ref func) = tc.function {
                                if let Some(ref buffered_func) = buffered.function {
                                    let name = tc.function.as_ref().and_then(|f| f.name.clone())
                                        .unwrap_or_else(|| buffered_func.name.clone().unwrap_or_default());
                                    let args = format!(
                                        "{}{}",
                                        buffered_func.arguments.clone().unwrap_or_default(),
                                        func.arguments.clone().unwrap_or_default()
                                    );
                                    *buffered = ChatCompletionToolCall {
                                        index: tc.index.or(buffered.index),
                                        id: tc.id.or(buffered.id.clone()),
                                        tool_type: tc.tool_type.or(buffered.tool_type.clone()),
                                        function: Some(ChatCompletionFunctionCall {
                                            name: Some(name),
                                            arguments: Some(args),
                                        }),
                                    };
                                }
                            }
                        }
                    }
                }

                if choice.finish_reason.is_some() {
                    if !accumulated_content.is_empty() {
                        let message_item = ResponseItem::Message {
                            id: None,
                            role: "assistant".to_string(),
                            content: vec![codex_protocol::models::ContentItem::OutputText {
                                text: accumulated_content.clone(),
                            }],
                            phase: None,
                        };
                        let _ = tx_event.send(Ok(ResponseEvent::OutputItemDone(message_item))).await;
                        accumulated_content.clear();
                    }

                    if let Some(mut buffered_tc) = tool_call_buffer.take() {
                        let tool_call_id = buffered_tc.id.clone().unwrap_or_else(|| format!("call_{}", response_id.as_deref().unwrap_or("0")));
                        let arguments = buffered_tc.function.as_ref()
                            .and_then(|f| f.arguments.clone())
                            .unwrap_or_default();
                        let name = buffered_tc.function.as_ref()
                            .and_then(|f| f.name.clone())
                            .unwrap_or_default();

                        let function_call_item = ResponseItem::FunctionCall {
                            id: None,
                            name,
                            namespace: None,
                            arguments,
                            call_id: tool_call_id,
                        };
                        let _ = tx_event.send(Ok(ResponseEvent::OutputItemDone(function_call_item))).await;
                    }
                }
            }

            if let Some(usage) = chunk.usage {
                final_usage = Some(TokenUsage::from(&usage));
            }
        }

        let event_type = event.event_type.as_deref().unwrap_or("");
        if event_type == "done" || (event.choices.is_none() && event.delta.is_none() && event.usage.is_some()) {
            if let Some(usage) = final_usage.take() {
                let _ = tx_event.send(Ok(ResponseEvent::Completed {
                    response_id: response_id.unwrap_or_else(|| "unknown".to_string()),
                    token_usage: Some(usage),
                    end_turn: Some(true),
                })).await;
            }
            return;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;
    use futures::stream;
    use http::HeaderMap;
    use http::HeaderValue;
    use http::StatusCode;

    async fn run_chat_sse(events: Vec<serde_json::Value>) -> Vec<ResponseEvent> {
        let mut body = String::new();
        for e in events {
            body.push_str("data: ");
            body.push_str(&e.to_string());
            body.push_str("\n\n");
        }
        body.push_str("data: [DONE]\n\n");

        let (tx, mut rx) = mpsc::channel::<Result<ResponseEvent, ApiError>>(8);
        let bytes = stream::iter(vec![Ok(Bytes::from(body))]);
        let stream_response = StreamResponse {
            status: StatusCode::OK,
            headers: HeaderMap::new(),
            bytes: Box::pin(bytes),
        };

        tokio::spawn(process_chat_completion_sse(
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
    async fn parses_chat_completion_stream() {
        let events = run_chat_sse(vec![
            json!({
                "id": "chatcmpl-123",
                "object": "chat.completion.chunk",
                "created": 1677652288,
                "model": "gpt-4",
                "choices": [{
                    "index": 0,
                    "delta": {
                        "content": "Hello"
                    },
                    "finish_reason": null
                }]
            }),
            json!({
                "id": "chatcmpl-123",
                "object": "chat.completion.chunk",
                "created": 1677652288,
                "model": "gpt-4",
                "choices": [{
                    "index": 0,
                    "delta": {
                        "content": "!"
                    },
                    "finish_reason": null
                }]
            }),
            json!({
                "id": "chatcmpl-123",
                "object": "chat.completion.chunk",
                "created": 1677652288,
                "model": "gpt-4",
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "total_tokens": 12
                }
            }),
        ])
        .await;

        assert!(!events.is_empty());
        assert_matches!(events.first(), Some(ResponseEvent::Created));
    }
}
