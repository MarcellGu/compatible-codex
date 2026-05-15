use crate::common::ResponseEvent;
use crate::common::ResponseStream;
use crate::error::ApiError;
use crate::telemetry::SseTelemetry;
use codex_client::ByteStream;
use codex_client::StreamResponse;
use codex_protocol::models::ContentItem;
use codex_protocol::models::ResponseItem;
use codex_protocol::protocol::TokenUsage;
use eventsource_stream::Eventsource;
use futures::StreamExt;
use serde::Deserialize;
use serde_json::Value;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio::time::Instant;
use tokio::time::timeout;
use tracing::debug;

const REQUEST_ID_HEADER: &str = "request-id";

pub fn spawn_anthropic_stream(
    stream_response: StreamResponse,
    idle_timeout: Duration,
    telemetry: Option<Arc<dyn SseTelemetry>>,
) -> ResponseStream {
    let upstream_request_id = stream_response
        .headers
        .get(REQUEST_ID_HEADER)
        .and_then(|value| value.to_str().ok())
        .map(str::to_string);
    let (tx_event, rx_event) = mpsc::channel::<Result<ResponseEvent, ApiError>>(1600);
    tokio::spawn(process_anthropic_sse(
        stream_response.bytes,
        tx_event,
        idle_timeout,
        telemetry,
    ));

    ResponseStream {
        rx_event,
        upstream_request_id,
    }
}

#[derive(Debug, Deserialize)]
struct AnthropicStreamEvent {
    #[serde(rename = "type")]
    kind: String,
    message: Option<AnthropicMessageStart>,
    index: Option<usize>,
    content_block: Option<AnthropicContentBlockStart>,
    delta: Option<AnthropicDelta>,
    usage: Option<AnthropicUsageDelta>,
    error: Option<AnthropicError>,
}

#[derive(Debug, Deserialize)]
struct AnthropicMessageStart {
    id: String,
    usage: Option<AnthropicUsage>,
}

#[derive(Debug, Deserialize)]
struct AnthropicUsage {
    input_tokens: i64,
    output_tokens: i64,
}

#[derive(Debug, Deserialize)]
struct AnthropicUsageDelta {
    output_tokens: Option<i64>,
}

#[derive(Debug, Deserialize)]
struct AnthropicError {
    message: String,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum AnthropicContentBlockStart {
    #[serde(rename = "text")]
    Text { text: Option<String> },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: Option<Value>,
    },
    #[serde(other)]
    Other,
}

#[derive(Debug, Deserialize)]
struct AnthropicDelta {
    #[serde(rename = "type")]
    kind: Option<String>,
    text: Option<String>,
    partial_json: Option<String>,
    thinking: Option<String>,
    stop_reason: Option<String>,
}

#[derive(Default)]
struct AnthropicStreamState {
    response_id: Option<String>,
    blocks: BTreeMap<usize, AnthropicBlockState>,
    input_tokens: Option<i64>,
    output_tokens: Option<i64>,
    stop_reason: Option<String>,
}

enum AnthropicBlockState {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        initial_input: Option<Value>,
        partial_json: String,
    },
    Thinking {
        text: String,
    },
    Other,
}

impl AnthropicStreamState {
    fn response_id(&self) -> String {
        self.response_id
            .clone()
            .unwrap_or_else(|| "anthropic-message".to_string())
    }

    fn text_item(&self, index: usize, text: String) -> ResponseItem {
        ResponseItem::Message {
            id: Some(format!("{}:text:{index}", self.response_id())),
            role: "assistant".to_string(),
            content: if text.is_empty() {
                Vec::new()
            } else {
                vec![ContentItem::OutputText { text }]
            },
            phase: None,
        }
    }

    fn token_usage(&self) -> Option<TokenUsage> {
        let input_tokens = self.input_tokens?;
        let output_tokens = self.output_tokens.unwrap_or(0);
        Some(TokenUsage {
            input_tokens,
            cached_input_tokens: 0,
            output_tokens,
            reasoning_output_tokens: 0,
            total_tokens: input_tokens + output_tokens,
        })
    }
}

pub async fn process_anthropic_sse(
    stream: ByteStream,
    tx_event: mpsc::Sender<Result<ResponseEvent, ApiError>>,
    idle_timeout: Duration,
    telemetry: Option<Arc<dyn SseTelemetry>>,
) {
    let mut stream = stream.eventsource();
    let mut state = AnthropicStreamState::default();

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
                let _ = tx_event
                    .send(Err(ApiError::Stream(
                        "stream closed before anthropic message_stop".to_string(),
                    )))
                    .await;
                return;
            }
            Err(_) => {
                let _ = tx_event
                    .send(Err(ApiError::Stream("idle timeout waiting for SSE".into())))
                    .await;
                return;
            }
        };

        let event: AnthropicStreamEvent = match serde_json::from_str(&sse.data) {
            Ok(event) => event,
            Err(e) => {
                debug!(
                    "Failed to parse Anthropic SSE event: {e}, data: {}",
                    &sse.data
                );
                continue;
            }
        };

        match event.kind.as_str() {
            "message_start" => {
                if let Some(message) = event.message {
                    state.response_id = Some(message.id);
                    if let Some(usage) = message.usage {
                        state.input_tokens = Some(usage.input_tokens);
                        state.output_tokens = Some(usage.output_tokens);
                    }
                }
                if tx_event.send(Ok(ResponseEvent::Created)).await.is_err() {
                    return;
                }
            }
            "content_block_start" => {
                let Some(index) = event.index else {
                    continue;
                };
                match event.content_block {
                    Some(AnthropicContentBlockStart::Text { text }) => {
                        let text = text.unwrap_or_default();
                        let item = state.text_item(index, text.clone());
                        state
                            .blocks
                            .insert(index, AnthropicBlockState::Text { text });
                        if tx_event
                            .send(Ok(ResponseEvent::OutputItemAdded(item)))
                            .await
                            .is_err()
                        {
                            return;
                        }
                    }
                    Some(AnthropicContentBlockStart::ToolUse { id, name, input }) => {
                        state.blocks.insert(
                            index,
                            AnthropicBlockState::ToolUse {
                                id,
                                name,
                                initial_input: input,
                                partial_json: String::new(),
                            },
                        );
                    }
                    Some(AnthropicContentBlockStart::Other) | None => {
                        state.blocks.insert(index, AnthropicBlockState::Other);
                    }
                }
            }
            "content_block_delta" => {
                let Some(index) = event.index else {
                    continue;
                };
                let Some(delta) = event.delta else {
                    continue;
                };
                match state.blocks.get_mut(&index) {
                    Some(AnthropicBlockState::Text { text: full_text })
                        if delta.kind.as_deref() == Some("text_delta") =>
                    {
                        let Some(text) = delta.text else {
                            continue;
                        };
                        full_text.push_str(&text);
                        if tx_event
                            .send(Ok(ResponseEvent::OutputTextDelta(text)))
                            .await
                            .is_err()
                        {
                            return;
                        }
                    }
                    Some(AnthropicBlockState::ToolUse { partial_json, .. })
                        if delta.kind.as_deref() == Some("input_json_delta") =>
                    {
                        if let Some(delta) = delta.partial_json {
                            partial_json.push_str(&delta);
                        }
                    }
                    Some(AnthropicBlockState::Thinking { text })
                        if delta.kind.as_deref() == Some("thinking_delta") =>
                    {
                        if let Some(thinking) = delta.thinking {
                            text.push_str(&thinking);
                        }
                    }
                    None if delta.kind.as_deref() == Some("thinking_delta") => {
                        if let Some(thinking) = delta.thinking {
                            state
                                .blocks
                                .insert(index, AnthropicBlockState::Thinking { text: thinking });
                        }
                    }
                    Some(_) | None => {}
                }
            }
            "content_block_stop" => {
                let Some(index) = event.index else {
                    continue;
                };
                let Some(block) = state.blocks.remove(&index) else {
                    continue;
                };
                if let Some(item) = finish_content_block(&state, index, block)
                    && tx_event
                        .send(Ok(ResponseEvent::OutputItemDone(item)))
                        .await
                        .is_err()
                {
                    return;
                }
            }
            "message_delta" => {
                if let Some(delta) = event.delta {
                    state.stop_reason = delta.stop_reason;
                }
                if let Some(usage) = event.usage
                    && let Some(output_tokens) = usage.output_tokens
                {
                    state.output_tokens = Some(output_tokens);
                }
            }
            "message_stop" => {
                let response_id = state.response_id();
                let token_usage = state.token_usage();
                let end_turn = Some(!matches!(
                    state.stop_reason.as_deref(),
                    Some("tool_use") | Some("max_tokens")
                ));
                let _ = tx_event
                    .send(Ok(ResponseEvent::Completed {
                        response_id,
                        token_usage,
                        end_turn,
                    }))
                    .await;
                return;
            }
            "error" => {
                let message = event
                    .error
                    .map(|error| error.message)
                    .unwrap_or_else(|| "Anthropic stream error".to_string());
                let _ = tx_event.send(Err(ApiError::Stream(message))).await;
                return;
            }
            _ => {}
        }
    }
}

fn finish_content_block(
    state: &AnthropicStreamState,
    index: usize,
    block: AnthropicBlockState,
) -> Option<ResponseItem> {
    match block {
        AnthropicBlockState::Text { text } => Some(state.text_item(index, text)),
        AnthropicBlockState::ToolUse {
            id,
            name,
            initial_input,
            partial_json,
        } => {
            let arguments = if partial_json.is_empty() {
                initial_input.map_or_else(|| "{}".to_string(), |input| input.to_string())
            } else {
                partial_json
            };
            Some(ResponseItem::FunctionCall {
                id: Some(id.clone()),
                name,
                namespace: None,
                arguments,
                call_id: id,
            })
        }
        AnthropicBlockState::Thinking { text } => {
            use codex_protocol::models::ReasoningItemReasoningSummary;
            Some(ResponseItem::Reasoning {
                id: format!("{}:thinking:{index}", state.response_id()),
                summary: vec![ReasoningItemReasoningSummary::SummaryText { text }],
                content: None,
                encrypted_content: None,
            })
        }
        AnthropicBlockState::Other => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;
    use futures::StreamExt;
    use serde_json::json;

    fn byte_stream(data: String) -> ByteStream {
        futures::stream::iter([Ok(Bytes::from(data))]).boxed()
    }

    fn sse(value: Value) -> String {
        format!("data: {value}\n\n")
    }

    #[tokio::test]
    async fn anthropic_stream_maps_text_tool_and_usage() {
        let data = [
            sse(json!({
                "type": "message_start",
                "message": {
                    "id": "msg_1",
                    "usage": {
                        "input_tokens": 12,
                        "output_tokens": 0
                    }
                }
            })),
            sse(json!({
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "text",
                    "text": ""
                }
            })),
            sse(json!({
                "type": "content_block_delta",
                "index": 0,
                "delta": {
                    "type": "text_delta",
                    "text": "checking"
                }
            })),
            sse(json!({
                "type": "content_block_stop",
                "index": 0
            })),
            sse(json!({
                "type": "content_block_start",
                "index": 1,
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "local_shell",
                    "input": {}
                }
            })),
            sse(json!({
                "type": "content_block_delta",
                "index": 1,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": "{\"command\":[\"pwd\"]}"
                }
            })),
            sse(json!({
                "type": "content_block_stop",
                "index": 1
            })),
            sse(json!({
                "type": "message_delta",
                "delta": {
                    "type": "message_delta",
                    "stop_reason": "tool_use"
                },
                "usage": {
                    "output_tokens": 7
                }
            })),
            sse(json!({
                "type": "message_stop"
            })),
        ]
        .concat();
        let (tx_event, mut rx_event) = mpsc::channel(16);

        process_anthropic_sse(
            byte_stream(data),
            tx_event,
            Duration::from_secs(1),
            /*telemetry*/ None,
        )
        .await;

        let mut events = Vec::new();
        while let Some(event) = rx_event.recv().await {
            events.push(event.expect("event should be ok"));
        }

        assert!(matches!(events[0], ResponseEvent::Created));
        assert!(matches!(
            events[1],
            ResponseEvent::OutputItemAdded(ResponseItem::Message { .. })
        ));
        assert!(matches!(
            &events[2],
            ResponseEvent::OutputTextDelta(delta) if delta == "checking"
        ));
        assert!(matches!(
            events[3],
            ResponseEvent::OutputItemDone(ResponseItem::Message { .. })
        ));
        assert!(matches!(
            &events[4],
            ResponseEvent::OutputItemDone(ResponseItem::FunctionCall {
                id: Some(id),
                name,
                namespace: None,
                arguments,
                call_id,
            }) if id == "toolu_1"
                && name == "local_shell"
                && arguments == "{\"command\":[\"pwd\"]}"
                && call_id == "toolu_1"
        ));
        assert!(matches!(
            &events[5],
            ResponseEvent::Completed {
                response_id,
                token_usage: Some(usage),
                end_turn: Some(false),
            } if response_id == "msg_1"
                && *usage == (TokenUsage {
                    input_tokens: 12,
                    cached_input_tokens: 0,
                    output_tokens: 7,
                    reasoning_output_tokens: 0,
                    total_tokens: 19,
                })
        ));
    }
}
