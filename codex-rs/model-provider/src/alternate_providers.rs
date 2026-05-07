use std::fmt;
use std::path::PathBuf;
use std::sync::Arc;

use codex_api::ChatCompletionMessage;
use codex_api::ChatCompletionsApiRequest;
use codex_api::ChatCompletionsClient;
use codex_api::AnthropicClient;
use codex_api::AnthropicContentBlock;
use codex_api::AnthropicMessage;
use codex_api::AnthropicMessagesRequest;
use codex_api::Provider;
use codex_api::SharedAuthProvider;
use codex_login::AuthManager;
use codex_login::CodexAuth;
use codex_model_provider_info::ModelProviderInfo;
use codex_models_manager::manager::SharedModelsManager;
use codex_protocol::error::Result as CodexResult;
use codex_protocol::models::ContentItem;
use codex_protocol::models::ResponseItem;
use codex_protocol::openai_models::ModelsResponse;
use serde_json::Value;

use crate::auth::auth_manager_for_provider;
use crate::auth::resolve_provider_auth;
use crate::ProviderCapabilities;

pub const ANTHROPIC_PROVIDER_NAME: &str = "Anthropic";
pub const ANTHROPIC_PROVIDER_ID: &str = "anthropic";
pub const ANTHROPIC_DEFAULT_BASE_URL: &str = "https://api.anthropic.com/v1";
pub const ANTHROPIC_API_KEY_ENV_VAR: &str = "ANTHROPIC_API_KEY";

pub const CHAT_COMPLETIONS_PROVIDER_NAME: &str = "Chat Completions";

pub const DEFAULT_ANTHROPIC_MAX_TOKENS: u32 = 4096;

pub fn create_anthropic_provider_info(
    base_url: Option<String>,
    api_key: Option<String>,
) -> ModelProviderInfo {
    let env_key = api_key.or_else(|| std::env::var(ANTHROPIC_API_KEY_ENV_VAR).ok());
    ModelProviderInfo {
        name: ANTHROPIC_PROVIDER_NAME.to_string(),
        base_url: base_url.or_else(|| Some(ANTHROPIC_DEFAULT_BASE_URL.to_string())),
        env_key,
        env_key_instructions: Some(format!(
            "Set the {} environment variable with your Anthropic API key.",
            ANTHROPIC_API_KEY_ENV_VAR
        )),
        experimental_bearer_token: None,
        auth: None,
        aws: None,
        wire_api: codex_model_provider_info::WireApi::Anthropic,
        query_params: None,
        http_headers: Some(
            [("anthropic-version".to_string(), "2023-06-01".to_string())]
                .into_iter()
                .collect(),
        ),
        env_http_headers: None,
        request_max_retries: Some(4),
        stream_max_retries: Some(5),
        stream_idle_timeout_ms: Some(300_000),
        websocket_connect_timeout_ms: None,
        requires_openai_auth: false,
        supports_websockets: false,
    }
}

pub fn create_chat_completions_provider_info(
    base_url: Option<String>,
    api_key: Option<String>,
) -> ModelProviderInfo {
    ModelProviderInfo {
        name: CHAT_COMPLETIONS_PROVIDER_NAME.to_string(),
        base_url,
        env_key: api_key,
        env_key_instructions: None,
        experimental_bearer_token: None,
        auth: None,
        aws: None,
        wire_api: codex_model_provider_info::WireApi::ChatCompletions,
        query_params: None,
        http_headers: None,
        env_http_headers: None,
        request_max_retries: Some(4),
        stream_max_retries: Some(5),
        stream_idle_timeout_ms: Some(300_000),
        websocket_connect_timeout_ms: None,
        requires_openai_auth: false,
        supports_websockets: false,
    }
}

pub fn is_anthropic_provider(provider: &ModelProviderInfo) -> bool {
    provider.wire_api == codex_model_provider_info::WireApi::Anthropic
}

pub fn is_chat_completions_provider(provider: &ModelProviderInfo) -> bool {
    provider.wire_api == codex_model_provider_info::WireApi::ChatCompletions
}

pub fn convert_response_items_to_chat_messages(
    instructions: &str,
    input: &[ResponseItem],
) -> Vec<ChatCompletionMessage> {
    let mut messages = Vec::new();

    if !instructions.is_empty() {
        messages.push(ChatCompletionMessage {
            role: "system".to_string(),
            content: instructions.to_string(),
            name: None,
            tool_calls: None,
        });
    }

    for item in input {
        match item {
            ResponseItem::Message { role, content, .. } => {
                let content_str = content
                    .iter()
                    .filter_map(|c| match c {
                        ContentItem::InputText { text } => Some(text.clone()),
                        ContentItem::OutputText { text } => Some(text.clone()),
                        ContentItem::InputImage { .. } => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n");

                if !content_str.is_empty() {
                    messages.push(ChatCompletionMessage {
                        role: role.clone(),
                        content: content_str,
                        name: None,
                        tool_calls: None,
                    });
                }
            }
            ResponseItem::FunctionCall { name, arguments, call_id, .. } => {
                let tool_calls = vec![serde_json::json!({
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": arguments,
                    }
                })];
                messages.push(ChatCompletionMessage {
                    role: "assistant".to_string(),
                    content: String::new(),
                    name: None,
                    tool_calls: Some(tool_calls),
                });
            }
            ResponseItem::FunctionCallOutput { call_id, output, .. } => {
                let content = output.to_text().unwrap_or_default();
                messages.push(ChatCompletionMessage {
                    role: "tool".to_string(),
                    content: format!("{}: {}", call_id, content),
                    name: None,
                    tool_calls: None,
                });
            }
            ResponseItem::CustomToolCall { call_id, name, input, .. } => {
                messages.push(ChatCompletionMessage {
                    role: "assistant".to_string(),
                    content: String::new(),
                    name: None,
                    tool_calls: Some(vec![serde_json::json!({
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": input,
                        }
                    })]),
                });
            }
            ResponseItem::CustomToolCallOutput { call_id, name, output, .. } => {
                let content = output.to_text().unwrap_or_default();
                messages.push(ChatCompletionMessage {
                    role: "tool".to_string(),
                    content: format!("{}: {}", name.as_deref().unwrap_or(call_id), content),
                    name: None,
                    tool_calls: None,
                });
            }
            _ => {}
        }
    }

    messages
}

pub fn convert_response_items_to_anthropic_messages(
    instructions: &str,
    input: &[ResponseItem],
) -> (Option<String>, Vec<AnthropicMessage>) {
    let system = if instructions.is_empty() {
        None
    } else {
        Some(instructions.to_string())
    };

    let mut messages = Vec::new();

    for item in input {
        match item {
            ResponseItem::Message { role, content, .. } => {
                let content_blocks: Vec<AnthropicContentBlock> = content
                    .iter()
                    .filter_map(|c| match c {
                        ContentItem::InputText { text } => Some(AnthropicContentBlock::text(text.clone())),
                        ContentItem::OutputText { text } => Some(AnthropicContentBlock::text(text.clone())),
                        ContentItem::InputImage { .. } => None,
                    })
                    .collect();

                if !content_blocks.is_empty() {
                    let anthropic_role = match role.as_str() {
                        "user" => "user",
                        "assistant" => "assistant",
                        _ => continue,
                    };

                    messages.push(AnthropicMessage {
                        role: anthropic_role.to_string(),
                        content: content_blocks,
                    });
                }
            }
            ResponseItem::FunctionCall { name, arguments, call_id, .. } => {
                messages.push(AnthropicMessage {
                    role: "assistant".to_string(),
                    content: vec![AnthropicContentBlock::tool_use(
                        call_id.clone(),
                        name.clone(),
                        serde_json::from_str(arguments).unwrap_or(serde_json::Value::Null),
                    )],
                });
            }
            ResponseItem::FunctionCallOutput { call_id, output, .. } => {
                let content = output.to_text().unwrap_or_default();
                messages.push(AnthropicMessage {
                    role: "user".to_string(),
                    content: vec![AnthropicContentBlock::tool_result(
                        call_id.clone(),
                        serde_json::json!({"type": "text", "text": content}),
                    )],
                });
            }
            _ => {}
        }
    }

    (system, messages)
}

pub fn convert_tools_to_chat_tools(tools: &[Value]) -> Option<Vec<serde_json::Value>> {
    let chat_tools: Vec<serde_json::Value> = tools
        .iter()
        .filter_map(|tool| {
            let obj = tool.as_object()?;
            if obj.contains_key("function") {
                Some(tool.clone())
            } else {
                Some(serde_json::json!({
                    "type": "function",
                    "function": tool,
                }))
            }
        })
        .collect();

    if chat_tools.is_empty() {
        None
    } else {
        Some(chat_tools)
    }
}

pub fn convert_tools_to_anthropic_tools(tools: &[Value]) -> Vec<codex_api::AnthropicTool> {
    tools
        .iter()
        .filter_map(|tool| {
            let obj = tool.as_object()?;
            let function = obj.get("function")?;
            let function_obj = function.as_object()?;

            Some(codex_api::AnthropicTool {
                name: function_obj.get("name")?.as_str()?.to_string(),
                description: function_obj.get("description").and_then(|v| v.as_str()).map(String::from),
                input_schema: function_obj.get("parameters").cloned().unwrap_or(serde_json::json!({"type": "object", "properties": {}})),
            })
        })
        .collect()
}
