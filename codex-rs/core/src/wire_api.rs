use codex_api::AnthropicMessagesApiRequest;
use codex_api::ChatCompletionsApiRequest;
use codex_protocol::models::ContentItem;
use codex_protocol::models::FunctionCallOutputPayload;
use codex_protocol::models::LocalShellAction;
use codex_protocol::models::ReasoningItemContent;
use codex_protocol::models::ReasoningItemReasoningSummary;
use codex_protocol::models::ResponseItem;
use codex_protocol::openai_models::ModelInfo;
use codex_protocol::openai_models::ReasoningEffort as ReasoningEffortConfig;
use codex_tools::ResponsesApiNamespaceTool;
use codex_tools::ResponsesApiTool;
use codex_tools::ToolSpec;
use serde_json::Value;
use serde_json::json;

use crate::client_common::Prompt;

const ANTHROPIC_DEFAULT_MAX_TOKENS: u32 = 4096;

#[derive(Clone, Debug, Default)]
pub(crate) struct ChatCompletionsRequestOptions {
    pub(crate) service_tier: Option<String>,
    pub(crate) preserve_reasoning_content: bool,
}

pub(crate) fn build_chat_completions_request(
    prompt: &Prompt,
    model_info: &ModelInfo,
    effort: Option<ReasoningEffortConfig>,
    options: ChatCompletionsRequestOptions,
) -> codex_protocol::error::Result<ChatCompletionsApiRequest> {
    let mut builder = ChatMessageBuilder::new(
        /*preserve_reasoning_content*/ options.preserve_reasoning_content,
    );
    if !prompt.base_instructions.text.is_empty() {
        builder
            .system_messages
            .push(prompt.base_instructions.text.clone());
    }
    for item in prompt.get_formatted_input() {
        builder.append(item);
    }
    let (mut messages, system_messages) = builder.finish();
    if !system_messages.is_empty() {
        messages.insert(
            0,
            json!({
                "role": "system",
                "content": system_messages.join("\n\n"),
            }),
        );
    }

    let tools = chat_tools_for_specs(&prompt.tools)?;
    let response_format = prompt.output_schema.as_ref().map(|schema| {
        json!({
            "type": "json_schema",
            "json_schema": {
                "name": "codex_output_schema",
                "strict": prompt.output_schema_strict,
                "schema": schema,
            }
        })
    });

    Ok(ChatCompletionsApiRequest {
        model: model_info.slug.clone(),
        messages,
        tool_choice: None,
        parallel_tool_calls: (!tools.is_empty() && prompt.parallel_tool_calls).then_some(true),
        tools,
        stream: true,
        reasoning_effort: (!model_info.supported_reasoning_levels.is_empty())
            .then(|| effort.or(model_info.default_reasoning_level))
            .flatten(),
        response_format,
        service_tier: options.service_tier,
    })
}

pub(crate) fn build_anthropic_messages_request(
    prompt: &Prompt,
    model_info: &ModelInfo,
) -> codex_protocol::error::Result<AnthropicMessagesApiRequest> {
    let mut messages = Vec::new();
    for item in prompt.get_formatted_input() {
        append_anthropic_messages_for_item(&mut messages, item);
    }

    Ok(AnthropicMessagesApiRequest {
        model: model_info.slug.clone(),
        max_tokens: ANTHROPIC_DEFAULT_MAX_TOKENS,
        system: prompt.base_instructions.text.clone(),
        messages,
        tools: anthropic_tools_for_specs(&prompt.tools)?,
        stream: true,
    })
}

#[derive(Debug, Default)]
struct ChatMessageBuilder {
    messages: Vec<Value>,
    system_messages: Vec<String>,
    pending_reasoning_content: Option<String>,
    pending_tool_calls: Vec<Value>,
    preserve_reasoning_content: bool,
}

impl ChatMessageBuilder {
    fn new(preserve_reasoning_content: bool) -> Self {
        Self {
            preserve_reasoning_content,
            ..Default::default()
        }
    }

    fn append(&mut self, item: ResponseItem) {
        match item {
            ResponseItem::Message { role, content, .. } => {
                self.flush_pending_tool_calls();
                let role = chat_message_role(&role);
                if role == "system" {
                    if let Some(content) = chat_system_content_for_items(&content) {
                        self.system_messages.push(content);
                    }
                } else if let Some(content) = chat_content_for_items(&content) {
                    self.messages.push(json!({
                        "role": role,
                        "content": content,
                    }));
                }
            }
            ResponseItem::FunctionCall {
                name,
                arguments,
                call_id,
                ..
            }
            | ResponseItem::CustomToolCall {
                name,
                input: arguments,
                call_id,
                ..
            } => {
                self.pending_tool_calls.push(json!({
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": arguments,
                    }
                }));
            }
            ResponseItem::LocalShellCall {
                id,
                call_id,
                action,
                ..
            } => {
                let call_id = call_id.or(id).unwrap_or_else(|| "local_shell".to_string());
                self.pending_tool_calls.push(json!({
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": "local_shell",
                        "arguments": local_shell_arguments(action),
                    }
                }));
            }
            ResponseItem::FunctionCallOutput { call_id, output }
            | ResponseItem::CustomToolCallOutput {
                call_id, output, ..
            } => {
                self.flush_pending_tool_calls();
                self.messages.push(json!({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": tool_output_text(&output),
                }));
            }
            ResponseItem::Reasoning {
                summary, content, ..
            } => {
                if self.preserve_reasoning_content
                    && let Some(text) = chat_reasoning_text(summary, content)
                {
                    if !self.pending_tool_calls.is_empty() {
                        self.flush_pending_tool_calls();
                    }
                    self.append_pending_reasoning_content(&text);
                }
            }
            ResponseItem::ToolSearchCall { .. }
            | ResponseItem::ToolSearchOutput { .. }
            | ResponseItem::WebSearchCall { .. }
            | ResponseItem::ImageGenerationCall { .. }
            | ResponseItem::Compaction { .. }
            | ResponseItem::CompactionTrigger
            | ResponseItem::ContextCompaction { .. }
            | ResponseItem::Other => {}
        }
    }

    fn append_pending_reasoning_content(&mut self, text: &str) {
        match self.pending_reasoning_content.as_mut() {
            Some(existing) if !existing.is_empty() => {
                existing.push('\n');
                existing.push_str(text);
            }
            Some(existing) => existing.push_str(text),
            None => self.pending_reasoning_content = Some(text.to_string()),
        }
    }

    fn flush_pending_tool_calls(&mut self) {
        if self.pending_tool_calls.is_empty() {
            self.pending_reasoning_content = None;
            return;
        }

        let mut message = json!({
            "role": "assistant",
            "content": null,
            "tool_calls": std::mem::take(&mut self.pending_tool_calls),
        });
        if let Some(reasoning_content) = self.pending_reasoning_content.take()
            && !reasoning_content.is_empty()
        {
            message["reasoning_content"] = json!(reasoning_content);
        }
        self.messages.push(message);
    }

    fn finish(mut self) -> (Vec<Value>, Vec<String>) {
        self.flush_pending_tool_calls();
        (self.messages, self.system_messages)
    }
}

fn chat_message_role(role: &str) -> &str {
    match role {
        "developer" => "system",
        _ => role,
    }
}

fn chat_system_content_for_items(items: &[ContentItem]) -> Option<String> {
    let mut text = String::new();
    for item in items {
        match item {
            ContentItem::InputText { text: chunk } | ContentItem::OutputText { text: chunk } => {
                text.push_str(chunk);
            }
            ContentItem::InputImage { image_url, .. } => {
                if !text.is_empty() {
                    text.push('\n');
                }
                text.push_str(&format!("[Image omitted: {image_url}]"));
            }
        }
    }

    (!text.is_empty()).then_some(text)
}

fn chat_reasoning_text(
    summary: Vec<ReasoningItemReasoningSummary>,
    content: Option<Vec<ReasoningItemContent>>,
) -> Option<String> {
    let mut text = String::new();
    if let Some(content) = content {
        for item in content {
            match item {
                ReasoningItemContent::ReasoningText { text: chunk }
                | ReasoningItemContent::Text { text: chunk } => text.push_str(&chunk),
            }
        }
    }
    if text.is_empty() {
        for item in summary {
            match item {
                ReasoningItemReasoningSummary::SummaryText { text: chunk } => {
                    text.push_str(&chunk);
                }
            }
        }
    }

    (!text.is_empty()).then_some(text)
}

fn append_anthropic_messages_for_item(messages: &mut Vec<Value>, item: ResponseItem) {
    match item {
        ResponseItem::Message { role, content, .. } => {
            let role = if role == "assistant" {
                "assistant"
            } else {
                "user"
            };
            let blocks = anthropic_blocks_for_content_items(&content, role);
            push_anthropic_message(messages, role, blocks);
        }
        ResponseItem::FunctionCall {
            name,
            arguments,
            call_id,
            ..
        }
        | ResponseItem::CustomToolCall {
            name,
            input: arguments,
            call_id,
            ..
        } => {
            push_anthropic_message(
                messages,
                "assistant",
                vec![json!({
                    "type": "tool_use",
                    "id": call_id,
                    "name": name,
                    "input": parse_json_or_raw(arguments),
                })],
            );
        }
        ResponseItem::LocalShellCall {
            id,
            call_id,
            action,
            ..
        } => {
            push_anthropic_message(
                messages,
                "assistant",
                vec![json!({
                    "type": "tool_use",
                    "id": call_id.or(id).unwrap_or_else(|| "local_shell".to_string()),
                    "name": "local_shell",
                    "input": parse_json_or_raw(local_shell_arguments(action)),
                })],
            );
        }
        ResponseItem::FunctionCallOutput { call_id, output }
        | ResponseItem::CustomToolCallOutput {
            call_id, output, ..
        } => {
            push_anthropic_message(
                messages,
                "user",
                vec![json!({
                    "type": "tool_result",
                    "tool_use_id": call_id,
                    "content": tool_output_text(&output),
                })],
            );
        }
        ResponseItem::ToolSearchCall { .. }
        | ResponseItem::ToolSearchOutput { .. }
        | ResponseItem::WebSearchCall { .. }
        | ResponseItem::ImageGenerationCall { .. }
        | ResponseItem::Compaction { .. }
        | ResponseItem::CompactionTrigger
        | ResponseItem::ContextCompaction { .. }
        | ResponseItem::Reasoning { .. }
        | ResponseItem::Other => {}
    }
}

fn chat_content_for_items(items: &[ContentItem]) -> Option<Value> {
    let mut text = String::new();
    let mut parts = Vec::new();
    for item in items {
        match item {
            ContentItem::InputText { text: chunk } | ContentItem::OutputText { text: chunk } => {
                text.push_str(chunk);
                parts.push(json!({
                    "type": "text",
                    "text": chunk,
                }));
            }
            ContentItem::InputImage { image_url, detail } => {
                let mut image_url_value = json!({ "url": image_url });
                if let Some(detail) = detail {
                    image_url_value["detail"] = json!(detail);
                }
                parts.push(json!({
                    "type": "image_url",
                    "image_url": image_url_value,
                }));
            }
        }
    }

    if parts.is_empty() {
        None
    } else if parts
        .iter()
        .all(|part| part.get("type").and_then(Value::as_str) == Some("text"))
    {
        Some(Value::String(text))
    } else {
        Some(Value::Array(parts))
    }
}

fn anthropic_blocks_for_content_items(items: &[ContentItem], role: &str) -> Vec<Value> {
    let mut blocks = Vec::new();
    for item in items {
        match item {
            ContentItem::InputText { text } | ContentItem::OutputText { text } => {
                blocks.push(json!({
                    "type": "text",
                    "text": text,
                }));
            }
            ContentItem::InputImage { image_url, .. } => {
                let text = if role == "assistant" {
                    format!("[Image omitted: {image_url}]")
                } else {
                    format!("[User provided image: {image_url}]")
                };
                blocks.push(json!({
                    "type": "text",
                    "text": text,
                }));
            }
        }
    }
    blocks
}

fn push_anthropic_message(messages: &mut Vec<Value>, role: &str, mut blocks: Vec<Value>) {
    if blocks.is_empty() {
        return;
    }
    if let Some(last) = messages.last_mut()
        && last.get("role").and_then(Value::as_str) == Some(role)
        && let Some(content) = last.get_mut("content").and_then(Value::as_array_mut)
    {
        content.append(&mut blocks);
        return;
    }
    messages.push(json!({
        "role": role,
        "content": blocks,
    }));
}

fn chat_tools_for_specs(tools: &[ToolSpec]) -> Result<Vec<Value>, serde_json::Error> {
    let mut out = Vec::new();
    for tool in tools {
        match tool {
            ToolSpec::Function(tool) => out.push(chat_function_tool(tool)?),
            ToolSpec::Namespace(namespace) => {
                for tool in &namespace.tools {
                    match tool {
                        ResponsesApiNamespaceTool::Function(tool) => {
                            out.push(chat_function_tool(tool)?);
                        }
                    }
                }
            }
            ToolSpec::ToolSearch { .. }
            | ToolSpec::ImageGeneration { .. }
            | ToolSpec::WebSearch { .. }
            | ToolSpec::Freeform(_) => {}
        }
    }
    Ok(out)
}

fn anthropic_tools_for_specs(tools: &[ToolSpec]) -> Result<Vec<Value>, serde_json::Error> {
    let mut out = Vec::new();
    for tool in tools {
        match tool {
            ToolSpec::Function(tool) => out.push(anthropic_function_tool(tool)?),
            ToolSpec::Namespace(namespace) => {
                for tool in &namespace.tools {
                    match tool {
                        ResponsesApiNamespaceTool::Function(tool) => {
                            out.push(anthropic_function_tool(tool)?);
                        }
                    }
                }
            }
            ToolSpec::ToolSearch { .. }
            | ToolSpec::ImageGeneration { .. }
            | ToolSpec::WebSearch { .. }
            | ToolSpec::Freeform(_) => {}
        }
    }
    Ok(out)
}

fn chat_function_tool(tool: &ResponsesApiTool) -> Result<Value, serde_json::Error> {
    Ok(json!({
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": serde_json::to_value(&tool.parameters)?,
        }
    }))
}

fn anthropic_function_tool(tool: &ResponsesApiTool) -> Result<Value, serde_json::Error> {
    Ok(json!({
        "name": tool.name,
        "description": tool.description,
        "input_schema": serde_json::to_value(&tool.parameters)?,
    }))
}

fn local_shell_arguments(action: LocalShellAction) -> String {
    match action {
        LocalShellAction::Exec(exec) => json!({
            "command": exec.command,
            "workdir": exec.working_directory,
            "timeout_ms": exec.timeout_ms,
        })
        .to_string(),
    }
}

fn parse_json_or_raw(raw: String) -> Value {
    serde_json::from_str(&raw).unwrap_or_else(|_| json!({ "raw": raw }))
}

fn tool_output_text(output: &FunctionCallOutputPayload) -> String {
    output.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use codex_tools::JsonSchema;
    use pretty_assertions::assert_eq;

    #[test]
    fn chat_message_role_maps_developer_to_system() {
        assert_eq!(chat_message_role("developer"), "system");
        assert_eq!(chat_message_role("user"), "user");
        assert_eq!(chat_message_role("assistant"), "assistant");
    }

    #[test]
    fn chat_messages_use_system_role_for_developer_items() {
        let mut builder = ChatMessageBuilder::new(/*preserve_reasoning_content*/ false);

        builder.append(ResponseItem::Message {
            id: None,
            role: "developer".to_string(),
            content: vec![ContentItem::InputText {
                text: "follow the rules".to_string(),
            }],
            phase: None,
        });
        let (messages, system_messages) = builder.finish();

        assert_eq!(messages, Vec::<serde_json::Value>::new());
        assert_eq!(system_messages, vec!["follow the rules".to_string()]);
    }

    #[test]
    fn build_chat_request_coalesces_system_messages() {
        let mut prompt = Prompt::default();
        prompt.base_instructions.text = "base".to_string();
        prompt.input = vec![
            ResponseItem::Message {
                id: None,
                role: "developer".to_string(),
                content: vec![ContentItem::InputText {
                    text: "developer".to_string(),
                }],
                phase: None,
            },
            ResponseItem::Message {
                id: None,
                role: "user".to_string(),
                content: vec![ContentItem::InputText {
                    text: "hello".to_string(),
                }],
                phase: None,
            },
        ];
        let model_info = codex_models_manager::model_info::model_info_from_slug("test-model");

        let request = build_chat_completions_request(
            &prompt,
            &model_info,
            None,
            ChatCompletionsRequestOptions::default(),
        )
        .unwrap();

        assert_eq!(
            request.messages,
            vec![
                serde_json::json!({
                    "role": "system",
                    "content": "base\n\ndeveloper",
                }),
                serde_json::json!({
                    "role": "user",
                    "content": "hello",
                }),
            ]
        );
    }

    #[test]
    fn build_chat_request_preserves_deepseek_reasoning_for_tool_calls() {
        let mut prompt = Prompt::default();
        prompt.base_instructions.text.clear();
        prompt.input = vec![
            ResponseItem::Message {
                id: None,
                role: "user".to_string(),
                content: vec![ContentItem::InputText {
                    text: "run both".to_string(),
                }],
                phase: None,
            },
            ResponseItem::Reasoning {
                id: "reasoning-1".to_string(),
                summary: Vec::new(),
                content: Some(vec![ReasoningItemContent::ReasoningText {
                    text: "need two tool calls".to_string(),
                }]),
                encrypted_content: None,
            },
            ResponseItem::FunctionCall {
                id: Some("call_1".to_string()),
                name: "first".to_string(),
                namespace: None,
                arguments: "{\"a\":1}".to_string(),
                call_id: "call_1".to_string(),
            },
            ResponseItem::FunctionCall {
                id: Some("call_2".to_string()),
                name: "second".to_string(),
                namespace: None,
                arguments: "{\"b\":2}".to_string(),
                call_id: "call_2".to_string(),
            },
            ResponseItem::FunctionCallOutput {
                call_id: "call_1".to_string(),
                output: FunctionCallOutputPayload::from_text("ok 1".to_string()),
            },
            ResponseItem::FunctionCallOutput {
                call_id: "call_2".to_string(),
                output: FunctionCallOutputPayload::from_text("ok 2".to_string()),
            },
        ];
        let model_info = codex_models_manager::model_info::model_info_from_slug("deepseek-v4");

        let request = build_chat_completions_request(
            &prompt,
            &model_info,
            None,
            ChatCompletionsRequestOptions {
                preserve_reasoning_content: true,
                ..Default::default()
            },
        )
        .unwrap();

        assert_eq!(
            request.messages,
            vec![
                serde_json::json!({
                    "role": "user",
                    "content": "run both",
                }),
                serde_json::json!({
                    "role": "assistant",
                    "content": null,
                    "reasoning_content": "need two tool calls",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "first",
                                "arguments": "{\"a\":1}",
                            }
                        },
                        {
                            "id": "call_2",
                            "type": "function",
                            "function": {
                                "name": "second",
                                "arguments": "{\"b\":2}",
                            }
                        }
                    ],
                }),
                serde_json::json!({
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": "ok 1",
                }),
                serde_json::json!({
                    "role": "tool",
                    "tool_call_id": "call_2",
                    "content": "ok 2",
                }),
            ]
        );
    }

    #[test]
    fn build_chat_request_omits_optional_compat_fields_by_default() {
        let mut prompt = Prompt::default();
        prompt.base_instructions.text.clear();
        prompt.tools = vec![ToolSpec::Function(ResponsesApiTool {
            name: "do_work".to_string(),
            description: "Do work.".to_string(),
            strict: false,
            defer_loading: None,
            parameters: JsonSchema::object(
                Default::default(),
                Some(Vec::new()),
                Some(false.into()),
            ),
            output_schema: None,
        })];
        let model_info = codex_models_manager::model_info::model_info_from_slug("test-model");

        let request = build_chat_completions_request(
            &prompt,
            &model_info,
            Some(ReasoningEffortConfig::High),
            ChatCompletionsRequestOptions::default(),
        )
        .unwrap();
        let body = serde_json::to_value(&request).unwrap();

        assert_eq!(request.tool_choice, None);
        assert_eq!(request.parallel_tool_calls, None);
        assert_eq!(request.reasoning_effort, None);
        assert!(body.get("tool_choice").is_none());
        assert!(body.get("parallel_tool_calls").is_none());
        assert!(body.get("reasoning_effort").is_none());

        prompt.parallel_tool_calls = true;
        let request = build_chat_completions_request(
            &prompt,
            &model_info,
            None,
            ChatCompletionsRequestOptions::default(),
        )
        .unwrap();

        assert_eq!(request.parallel_tool_calls, Some(true));
    }
}
