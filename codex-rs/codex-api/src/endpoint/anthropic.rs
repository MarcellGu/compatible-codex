use crate::auth::SharedAuthProvider;
use crate::common::ResponseStream;
use crate::endpoint::session::EndpointSession;
use crate::error::ApiError;
use crate::provider::Provider;
use crate::requests::Compression;
use crate::requests::headers::build_session_headers;
use crate::sse::spawn_anthropic_stream;
use crate::telemetry::SseTelemetry;
use codex_client::HttpTransport;
use codex_client::RequestCompression;
use codex_client::RequestTelemetry;
use http::HeaderMap;
use http::HeaderValue;
use http::Method;
use serde::Deserialize;
use serde::Serialize;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::OnceLock;

pub struct AnthropicClient<T: HttpTransport> {
    session: EndpointSession<T>,
    sse_telemetry: Option<Arc<dyn SseTelemetry>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AnthropicMessage {
    pub role: String,
    pub content: Vec<AnthropicContentBlock>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum AnthropicContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        content: Value,
        is_error: Option<bool>,
    },
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AnthropicTool {
    pub name: String,
    pub description: Option<String>,
    pub input_schema: Value,
}

#[derive(Debug, Clone, Default)]
pub struct AnthropicOptions {
    pub extra_headers: HeaderMap,
    pub compression: Compression,
    pub turn_state: Option<Arc<OnceLock<String>>>,
}

impl<T: HttpTransport> AnthropicClient<T> {
    pub fn new(transport: T, provider: Provider, auth: SharedAuthProvider) -> Self {
        Self {
            session: EndpointSession::new(transport, provider, auth),
            sse_telemetry: None,
        }
    }

    pub fn with_telemetry(
        self,
        request: Option<Arc<dyn RequestTelemetry>>,
        sse: Option<Arc<dyn SseTelemetry>>,
    ) -> Self {
        Self {
            session: self.session.with_request_telemetry(request),
            sse_telemetry: sse,
        }
    }

    pub async fn stream_request(
        &self,
        request: AnthropicMessagesRequest,
        options: AnthropicOptions,
    ) -> Result<ResponseStream, ApiError> {
        let AnthropicOptions {
            extra_headers,
            compression,
            turn_state,
        } = options;

        let mut headers = extra_headers;
        headers.extend(build_session_headers(None, None));

        let body = serde_json::to_value(&request)
            .map_err(|e| ApiError::Stream(format!("failed to encode anthropic request: {e}")))?;

        self.stream(body, headers, compression, turn_state).await
    }

    fn path() -> &'static str {
        "messages"
    }

    pub async fn stream(
        &self,
        body: Value,
        extra_headers: HeaderMap,
        compression: Compression,
        turn_state: Option<Arc<OnceLock<String>>>,
    ) -> Result<ResponseStream, ApiError> {
        let request_compression = match compression {
            Compression::None => RequestCompression::None,
            Compression::Zstd => RequestCompression::Zstd,
        };

        let mut request_headers = extra_headers;
        request_headers.insert(
            http::header::CONTENT_TYPE,
            HeaderValue::from_static("application/json"),
        );
        request_headers.insert(
            "anthropic-version",
            HeaderValue::from_static("2023-06-01"),
        );
        request_headers.insert(
            http::header::ACCEPT,
            HeaderValue::from_static("text/event-stream"),
        );

        let stream_response = self
            .session
            .stream_with(
                Method::POST,
                Self::path(),
                request_headers,
                Some(body),
                |req| {
                    req.compression = request_compression;
                },
            )
            .await?;

        Ok(spawn_anthropic_stream(
            stream_response,
            self.session.provider().stream_idle_timeout,
            self.sse_telemetry.clone(),
            turn_state,
        ))
    }
}

#[derive(Debug, Serialize)]
pub struct AnthropicMessagesRequest {
    pub model: String,
    pub messages: Vec<AnthropicMessage>,
    pub max_tokens: u32,
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<AnthropicTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<AnthropicToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<AnthropicThinkingConfig>,
}

#[derive(Debug, Serialize)]
pub struct AnthropicToolChoice {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub name: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct AnthropicThinkingConfig {
    #[serde(rename = "type")]
    pub thinking_type: String,
    pub budget_tokens: u32,
}

impl AnthropicMessagesRequest {
    pub fn new(model: String, messages: Vec<AnthropicMessage>, max_tokens: u32) -> Self {
        Self {
            model,
            messages,
            max_tokens,
            stream: true,
            system: None,
            tools: None,
            tool_choice: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            thinking: None,
        }
    }
}

impl AnthropicContentBlock {
    pub fn text(content: String) -> Self {
        AnthropicContentBlock::Text { text: content }
    }

    pub fn tool_use(id: String, name: String, input: Value) -> Self {
        AnthropicContentBlock::ToolUse { id, name, input }
    }

    pub fn tool_result(tool_use_id: String, content: Value) -> Self {
        AnthropicContentBlock::ToolResult {
            tool_use_id,
            content,
            is_error: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn creates_basic_request() {
        let messages = vec![AnthropicMessage {
            role: "user".to_string(),
            content: vec![AnthropicContentBlock::text("Hello!".to_string())],
        }];

        let request = AnthropicMessagesRequest::new(
            "claude-3-5-sonnet-20241022".to_string(),
            messages,
            1024,
        );

        assert_eq!(request.model, "claude-3-5-sonnet-20241022");
        assert_eq!(request.messages.len(), 1);
        assert_eq!(request.max_tokens, 1024);
        assert!(request.stream);
    }

    #[test]
    fn creates_text_content_block() {
        let block = AnthropicContentBlock::text("Hello".to_string());
        assert!(matches!(block, AnthropicContentBlock::Text { .. }));
    }
}
