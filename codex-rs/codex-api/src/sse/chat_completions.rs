use crate::common::ResponseEvent;
use crate::common::ResponseStream;
use crate::error::ApiError;
use crate::telemetry::SseTelemetry;
use codex_client::ByteStream;
use codex_client::StreamResponse;
use codex_protocol::models::ContentItem;
use codex_protocol::models::ReasoningItemContent;
use codex_protocol::models::ResponseItem;
use codex_protocol::protocol::TokenUsage;
use eventsource_stream::Eventsource;
use futures::StreamExt;
use serde::Deserialize;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio::time::Instant;
use tokio::time::timeout;
use tracing::debug;

const REQUEST_ID_HEADER: &str = "x-request-id";
const THINK_OPEN_TAG: &str = "<think>";
const THINK_CLOSE_TAG: &str = "</think>";

pub fn spawn_chat_completions_stream(
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
    tokio::spawn(process_chat_completions_sse(
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
struct ChatCompletionChunk {
    id: Option<String>,
    model: Option<String>,
    #[serde(default)]
    choices: Vec<ChatCompletionChoice>,
    usage: Option<ChatCompletionUsage>,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionChoice {
    delta: ChatCompletionDelta,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionDelta {
    content: Option<String>,
    #[serde(default, alias = "reasoning")]
    reasoning_content: Option<String>,
    function_call: Option<ChatFunctionCallDelta>,
    tool_calls: Option<Vec<ChatToolCallDelta>>,
}

#[derive(Debug, Deserialize)]
struct ChatFunctionCallDelta {
    name: Option<String>,
    arguments: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ChatToolCallDelta {
    index: usize,
    id: Option<String>,
    function: Option<ChatFunctionCallDelta>,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionUsage {
    prompt_tokens: i64,
    completion_tokens: i64,
    total_tokens: i64,
    prompt_tokens_details: Option<ChatPromptTokensDetails>,
    completion_tokens_details: Option<ChatCompletionTokensDetails>,
}

#[derive(Debug, Deserialize)]
struct ChatPromptTokensDetails {
    cached_tokens: Option<i64>,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionTokensDetails {
    reasoning_tokens: Option<i64>,
}

impl From<ChatCompletionUsage> for TokenUsage {
    fn from(value: ChatCompletionUsage) -> Self {
        Self {
            input_tokens: value.prompt_tokens,
            cached_input_tokens: value
                .prompt_tokens_details
                .and_then(|details| details.cached_tokens)
                .unwrap_or(0),
            output_tokens: value.completion_tokens,
            reasoning_output_tokens: value
                .completion_tokens_details
                .and_then(|details| details.reasoning_tokens)
                .unwrap_or(0),
            total_tokens: value.total_tokens,
        }
    }
}

#[derive(Default)]
struct ChatCompletionStreamState {
    created: bool,
    response_id: Option<String>,
    reasoning_started: bool,
    reasoning_finished: bool,
    reasoning_text: String,
    message_started: bool,
    message_text: String,
    tool_calls: BTreeMap<usize, ChatToolCallState>,
    token_usage: Option<TokenUsage>,
    last_server_model: Option<String>,
    think_parser: ThinkTagParser,
}

#[derive(Default)]
struct ChatToolCallState {
    id: Option<String>,
    name: Option<String>,
    arguments: String,
}

impl ChatCompletionStreamState {
    fn response_id(&self) -> String {
        self.response_id
            .clone()
            .unwrap_or_else(|| "chatcmpl".to_string())
    }

    fn message_item(&self) -> ResponseItem {
        ResponseItem::Message {
            id: Some(format!("{}:message", self.response_id())),
            role: "assistant".to_string(),
            content: if self.message_text.is_empty() {
                Vec::new()
            } else {
                vec![ContentItem::OutputText {
                    text: self.message_text.clone(),
                }]
            },
            phase: None,
        }
    }

    fn reasoning_item(&self) -> ResponseItem {
        ResponseItem::Reasoning {
            id: format!("{}:reasoning", self.response_id()),
            summary: Vec::new(),
            content: if self.reasoning_text.is_empty() {
                None
            } else {
                Some(vec![ReasoningItemContent::ReasoningText {
                    text: self.reasoning_text.clone(),
                }])
            },
            encrypted_content: None,
        }
    }
}

#[derive(Debug, PartialEq)]
enum ThinkSegment {
    Visible(String),
    Reasoning(String),
}

#[derive(Default)]
struct ThinkTagParser {
    buffer: String,
    in_think: bool,
}

impl ThinkTagParser {
    fn push(&mut self, text: &str) -> Vec<ThinkSegment> {
        self.buffer.push_str(text);
        self.drain(/*flush*/ false)
    }

    fn finish(&mut self) -> Vec<ThinkSegment> {
        self.drain(/*flush*/ true)
    }

    fn drain(&mut self, flush: bool) -> Vec<ThinkSegment> {
        let mut segments = Vec::new();

        loop {
            if self.in_think {
                if let Some(index) = self.buffer.find(THINK_CLOSE_TAG) {
                    push_reasoning_segment(&mut segments, &self.buffer[..index]);
                    self.buffer.drain(..index + THINK_CLOSE_TAG.len());
                    self.in_think = false;
                    continue;
                }

                let drain_len = if flush {
                    self.buffer.len()
                } else {
                    safe_drain_len(&self.buffer, THINK_CLOSE_TAG)
                };
                if drain_len == 0 {
                    break;
                }
                push_reasoning_segment(&mut segments, &self.buffer[..drain_len]);
                self.buffer.drain(..drain_len);
                break;
            }

            if let Some(index) = self.buffer.find(THINK_OPEN_TAG) {
                push_visible_segment(&mut segments, &self.buffer[..index]);
                self.buffer.drain(..index + THINK_OPEN_TAG.len());
                self.in_think = true;
                continue;
            }

            let drain_len = if flush {
                self.buffer.len()
            } else {
                safe_drain_len(&self.buffer, THINK_OPEN_TAG)
            };
            if drain_len == 0 {
                break;
            }
            push_visible_segment(&mut segments, &self.buffer[..drain_len]);
            self.buffer.drain(..drain_len);
            break;
        }

        segments
    }
}

fn push_visible_segment(segments: &mut Vec<ThinkSegment>, text: &str) {
    if !text.is_empty() {
        segments.push(ThinkSegment::Visible(text.to_string()));
    }
}

fn push_reasoning_segment(segments: &mut Vec<ThinkSegment>, text: &str) {
    if !text.is_empty() {
        segments.push(ThinkSegment::Reasoning(text.to_string()));
    }
}

fn safe_drain_len(buffer: &str, tag: &str) -> usize {
    let keep_len = longest_suffix_prefix_len(buffer, tag);
    buffer.len().saturating_sub(keep_len)
}

fn longest_suffix_prefix_len(buffer: &str, tag: &str) -> usize {
    let max_len = buffer.len().min(tag.len().saturating_sub(1));
    (1..=max_len)
        .rev()
        .find(|len| buffer.ends_with(&tag[..*len]))
        .unwrap_or(0)
}

pub async fn process_chat_completions_sse(
    stream: ByteStream,
    tx_event: mpsc::Sender<Result<ResponseEvent, ApiError>>,
    idle_timeout: Duration,
    telemetry: Option<Arc<dyn SseTelemetry>>,
) {
    let mut stream = stream.eventsource();
    let mut state = ChatCompletionStreamState::default();

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
                        "stream closed before chat completion finished".to_string(),
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

        if sse.data.trim() == "[DONE]" {
            finish_chat_completion_stream(&tx_event, state).await;
            return;
        }

        let chunk: ChatCompletionChunk = match serde_json::from_str(&sse.data) {
            Ok(chunk) => chunk,
            Err(e) => {
                debug!(
                    "Failed to parse chat completion SSE event: {e}, data: {}",
                    &sse.data
                );
                continue;
            }
        };

        if state.response_id.is_none() {
            state.response_id = chunk.id;
        }
        if !state.created {
            state.created = true;
            if tx_event.send(Ok(ResponseEvent::Created)).await.is_err() {
                return;
            }
        }
        if let Some(model) = chunk.model
            && state.last_server_model.as_deref() != Some(model.as_str())
        {
            if tx_event
                .send(Ok(ResponseEvent::ServerModel(model.clone())))
                .await
                .is_err()
            {
                return;
            }
            state.last_server_model = Some(model);
        }
        if let Some(usage) = chunk.usage {
            state.token_usage = Some(usage.into());
        }

        for choice in chunk.choices {
            if let Some(reasoning_content) = choice.delta.reasoning_content
                && handle_chat_content_segment(
                    &tx_event,
                    &mut state,
                    ThinkSegment::Reasoning(reasoning_content),
                )
                .await
                .is_err()
            {
                return;
            }

            if let Some(content) = choice.delta.content {
                for segment in state.think_parser.push(&content) {
                    if handle_chat_content_segment(&tx_event, &mut state, segment)
                        .await
                        .is_err()
                    {
                        return;
                    }
                }
            }

            if let Some(function_call) = choice.delta.function_call {
                merge_function_delta(state.tool_calls.entry(0).or_default(), function_call);
            }

            if let Some(tool_calls) = choice.delta.tool_calls {
                for tool_call in tool_calls {
                    let state = state.tool_calls.entry(tool_call.index).or_default();
                    if let Some(id) = tool_call.id {
                        state.id = Some(id);
                    }
                    if let Some(function) = tool_call.function {
                        merge_function_delta(state, function);
                    }
                }
            }

            if choice.finish_reason.is_some() {
                finish_chat_completion_stream(&tx_event, state).await;
                return;
            }
        }
    }
}

fn merge_function_delta(state: &mut ChatToolCallState, delta: ChatFunctionCallDelta) {
    if let Some(name) = delta.name {
        state.name = Some(name);
    }
    if let Some(arguments) = delta.arguments {
        state.arguments.push_str(&arguments);
    }
}

async fn finish_chat_completion_stream(
    tx_event: &mpsc::Sender<Result<ResponseEvent, ApiError>>,
    mut state: ChatCompletionStreamState,
) {
    for segment in state.think_parser.finish() {
        if handle_chat_content_segment(tx_event, &mut state, segment)
            .await
            .is_err()
        {
            return;
        }
    }

    if finish_reasoning_item(tx_event, &mut state).await.is_err() {
        return;
    }

    let response_id = state.response_id();
    if state.message_started
        && tx_event
            .send(Ok(ResponseEvent::OutputItemDone(state.message_item())))
            .await
            .is_err()
    {
        return;
    }

    for (index, tool_call) in state.tool_calls {
        if let Some(name) = tool_call.name {
            let call_id = tool_call
                .id
                .unwrap_or_else(|| format!("{response_id}:tool_call:{index}"));
            let item = ResponseItem::FunctionCall {
                id: Some(call_id.clone()),
                name,
                namespace: None,
                arguments: tool_call.arguments,
                call_id,
            };
            if tx_event
                .send(Ok(ResponseEvent::OutputItemDone(item)))
                .await
                .is_err()
            {
                return;
            }
        }
    }

    let _ = tx_event
        .send(Ok(ResponseEvent::Completed {
            response_id,
            token_usage: state.token_usage,
            end_turn: Some(true),
        }))
        .await;
}

async fn handle_chat_content_segment(
    tx_event: &mpsc::Sender<Result<ResponseEvent, ApiError>>,
    state: &mut ChatCompletionStreamState,
    segment: ThinkSegment,
) -> Result<(), ()> {
    match segment {
        ThinkSegment::Visible(content) => {
            finish_reasoning_item(tx_event, state).await?;
            if !state.message_started {
                state.message_started = true;
                tx_event
                    .send(Ok(ResponseEvent::OutputItemAdded(state.message_item())))
                    .await
                    .map_err(|_| ())?;
            }
            state.message_text.push_str(&content);
            tx_event
                .send(Ok(ResponseEvent::OutputTextDelta(content)))
                .await
                .map_err(|_| ())?;
        }
        ThinkSegment::Reasoning(content) => {
            if state.reasoning_finished {
                state.reasoning_text.push_str(&content);
                return Ok(());
            }
            if !state.reasoning_started {
                state.reasoning_started = true;
                tx_event
                    .send(Ok(ResponseEvent::OutputItemAdded(state.reasoning_item())))
                    .await
                    .map_err(|_| ())?;
            }
            state.reasoning_text.push_str(&content);
            tx_event
                .send(Ok(ResponseEvent::ReasoningContentDelta {
                    delta: content,
                    content_index: 0,
                }))
                .await
                .map_err(|_| ())?;
        }
    }

    Ok(())
}

async fn finish_reasoning_item(
    tx_event: &mpsc::Sender<Result<ResponseEvent, ApiError>>,
    state: &mut ChatCompletionStreamState,
) -> Result<(), ()> {
    if !state.reasoning_started || state.reasoning_finished {
        return Ok(());
    }

    state.reasoning_finished = true;
    tx_event
        .send(Ok(ResponseEvent::OutputItemDone(state.reasoning_item())))
        .await
        .map_err(|_| ())
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;
    use futures::StreamExt;
    use pretty_assertions::assert_eq;
    use serde_json::Value;
    use serde_json::json;

    fn byte_stream(data: String) -> ByteStream {
        futures::stream::iter([Ok(Bytes::from(data))]).boxed()
    }

    fn sse(value: Value) -> String {
        format!("data: {value}\n\n")
    }

    #[test]
    fn think_tag_parser_handles_split_tags() {
        let mut parser = ThinkTagParser::default();

        assert_eq!(parser.push("<thi"), Vec::<ThinkSegment>::new());
        assert_eq!(
            parser.push("nk> 用户用中文</th"),
            vec![ThinkSegment::Reasoning(" 用户用中文".to_string())]
        );
        assert_eq!(
            parser.push("ink>\n你好"),
            vec![ThinkSegment::Visible("\n你好".to_string())]
        );
        assert_eq!(parser.finish(), Vec::<ThinkSegment>::new());
    }

    #[tokio::test]
    async fn chat_completion_stream_maps_text_tool_and_usage() {
        let data = [
            sse(json!({
                "id": "chatcmpl-1",
                "model": "gpt-test",
                "choices": [{
                    "delta": { "content": "hello" },
                    "finish_reason": null
                }]
            })),
            sse(json!({
                "id": "chatcmpl-1",
                "choices": [{
                    "delta": {
                        "tool_calls": [{
                            "index": 0,
                            "id": "call_1",
                            "function": {
                                "name": "local_shell",
                                "arguments": "{\"command\":[\"pwd\"]}"
                            }
                        }]
                    },
                    "finish_reason": "tool_calls"
                }],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 4,
                    "total_tokens": 14,
                    "prompt_tokens_details": { "cached_tokens": 3 },
                    "completion_tokens_details": { "reasoning_tokens": 0 }
                }
            })),
        ]
        .concat();
        let (tx_event, mut rx_event) = mpsc::channel(16);

        process_chat_completions_sse(
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
            &events[1],
            ResponseEvent::ServerModel(model) if model == "gpt-test"
        ));
        assert!(matches!(
            events[2],
            ResponseEvent::OutputItemAdded(ResponseItem::Message { .. })
        ));
        assert!(matches!(
            &events[3],
            ResponseEvent::OutputTextDelta(delta) if delta == "hello"
        ));
        assert!(matches!(
            events[4],
            ResponseEvent::OutputItemDone(ResponseItem::Message { .. })
        ));
        assert!(matches!(
            &events[5],
            ResponseEvent::OutputItemDone(ResponseItem::FunctionCall {
                id: Some(id),
                name,
                namespace: None,
                arguments,
                call_id,
            }) if id == "call_1"
                && name == "local_shell"
                && arguments == "{\"command\":[\"pwd\"]}"
                && call_id == "call_1"
        ));
        assert!(matches!(
            &events[6],
            ResponseEvent::Completed {
                response_id,
                token_usage: Some(usage),
                end_turn: Some(true),
            } if response_id == "chatcmpl-1"
                && *usage == (TokenUsage {
                    input_tokens: 10,
                    cached_input_tokens: 3,
                    output_tokens: 4,
                    reasoning_output_tokens: 0,
                    total_tokens: 14,
                })
        ));
    }

    #[tokio::test]
    async fn chat_completion_stream_maps_reasoning_content_with_tool_calls() {
        let data = [
            sse(json!({
                "id": "chatcmpl-deepseek",
                "model": "deepseek-v4-flash",
                "choices": [{
                    "delta": { "reasoning_content": "need a tool" },
                    "finish_reason": null
                }]
            })),
            sse(json!({
                "id": "chatcmpl-deepseek",
                "choices": [{
                    "delta": {
                        "tool_calls": [{
                            "index": 0,
                            "id": "call_1",
                            "function": {
                                "name": "local_shell",
                                "arguments": "{\"command\":[\"pwd\"]}"
                            }
                        }]
                    },
                    "finish_reason": "tool_calls"
                }],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 4,
                    "total_tokens": 14,
                    "completion_tokens_details": { "reasoning_tokens": 2 }
                }
            })),
        ]
        .concat();
        let (tx_event, mut rx_event) = mpsc::channel(16);

        process_chat_completions_sse(
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
            &events[1],
            ResponseEvent::ServerModel(model) if model == "deepseek-v4-flash"
        ));
        assert!(matches!(
            events[2],
            ResponseEvent::OutputItemAdded(ResponseItem::Reasoning { .. })
        ));
        assert!(matches!(
            &events[3],
            ResponseEvent::ReasoningContentDelta {
                delta,
                content_index: 0,
            } if delta == "need a tool"
        ));
        assert!(matches!(
            &events[4],
            ResponseEvent::OutputItemDone(ResponseItem::Reasoning {
                id,
                summary,
                content: Some(content),
                encrypted_content: None,
            }) if id == "chatcmpl-deepseek:reasoning"
                && summary.is_empty()
                && content == &vec![ReasoningItemContent::ReasoningText {
                    text: "need a tool".to_string()
                }]
        ));
        assert!(matches!(
            &events[5],
            ResponseEvent::OutputItemDone(ResponseItem::FunctionCall {
                id: Some(id),
                name,
                namespace: None,
                arguments,
                call_id,
            }) if id == "call_1"
                && name == "local_shell"
                && arguments == "{\"command\":[\"pwd\"]}"
                && call_id == "call_1"
        ));
        assert!(matches!(
            &events[6],
            ResponseEvent::Completed {
                response_id,
                token_usage: Some(usage),
                end_turn: Some(true),
            } if response_id == "chatcmpl-deepseek"
                && *usage == (TokenUsage {
                    input_tokens: 10,
                    cached_input_tokens: 0,
                    output_tokens: 4,
                    reasoning_output_tokens: 2,
                    total_tokens: 14,
                })
        ));
    }

    #[tokio::test]
    async fn chat_completion_stream_maps_think_tags_to_raw_reasoning() {
        let data = [
            sse(json!({
                "id": "chatcmpl-think",
                "model": "MiniMax-M2.7",
                "choices": [{
                    "delta": { "content": "<thi" },
                    "finish_reason": null
                }]
            })),
            sse(json!({
                "id": "chatcmpl-think",
                "choices": [{
                    "delta": { "content": "nk> 用户用中文" },
                    "finish_reason": null
                }]
            })),
            sse(json!({
                "id": "chatcmpl-think",
                "choices": [{
                    "delta": { "content": "打招呼 </think>\n你好！" },
                    "finish_reason": "stop"
                }]
            })),
        ]
        .concat();
        let (tx_event, mut rx_event) = mpsc::channel(16);

        process_chat_completions_sse(
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
            &events[1],
            ResponseEvent::ServerModel(model) if model == "MiniMax-M2.7"
        ));
        assert!(matches!(
            events[2],
            ResponseEvent::OutputItemAdded(ResponseItem::Reasoning { .. })
        ));
        assert!(matches!(
            &events[3],
            ResponseEvent::ReasoningContentDelta {
                delta,
                content_index: 0,
            } if delta == " 用户用中文"
        ));
        assert!(matches!(
            &events[4],
            ResponseEvent::ReasoningContentDelta {
                delta,
                content_index: 0,
            } if delta == "打招呼 "
        ));
        assert!(matches!(
            &events[5],
            ResponseEvent::OutputItemDone(ResponseItem::Reasoning {
                id,
                summary,
                content: Some(content),
                encrypted_content: None,
            }) if id == "chatcmpl-think:reasoning"
                && summary.is_empty()
                && content == &vec![ReasoningItemContent::ReasoningText {
                    text: " 用户用中文打招呼 ".to_string()
                }]
        ));
        assert!(matches!(
            events[6],
            ResponseEvent::OutputItemAdded(ResponseItem::Message { .. })
        ));
        assert!(matches!(
            &events[7],
            ResponseEvent::OutputTextDelta(delta) if delta == "\n你好！"
        ));
        assert!(matches!(
            &events[8],
            ResponseEvent::OutputItemDone(ResponseItem::Message {
                content,
                ..
            }) if content == &vec![ContentItem::OutputText {
                text: "\n你好！".to_string()
            }]
        ));
        assert!(matches!(
            &events[9],
            ResponseEvent::Completed {
                response_id,
                token_usage: None,
                end_turn: Some(true),
            } if response_id == "chatcmpl-think"
        ));
    }
}
