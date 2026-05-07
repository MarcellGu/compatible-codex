use crate::auth::SharedAuthProvider;
use crate::common::ResponseStream;
use crate::endpoint::session::EndpointSession;
use crate::error::ApiError;
use crate::provider::Provider;
use crate::requests::Compression;
use crate::requests::headers::build_session_headers;
use crate::requests::headers::insert_header;
use crate::sse::spawn_chat_completion_stream;
use crate::telemetry::SseTelemetry;
use codex_client::HttpTransport;
use codex_client::RequestCompression;
use codex_client::RequestTelemetry;
use http::HeaderMap;
use http::Method;
use serde::Deserialize;
use serde::Serialize;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::OnceLock;

pub struct ChatCompletionsClient<T: HttpTransport> {
    session: EndpointSession<T>,
    sse_telemetry: Option<Arc<dyn SseTelemetry>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ChatCompletionMessage {
    pub role: String,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ChatCompletionToolCall>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ChatCompletionToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: ChatCompletionFunction,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ChatCompletionFunction {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ChatCompletionTool {
    pub r#type: String,
    pub function: ChatCompletionFunctionDefinition,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ChatCompletionFunctionDefinition {
    pub name: String,
    pub description: Option<String>,
    pub parameters: Option<Value>,
}

#[derive(Debug, Clone, Default)]
pub struct ChatCompletionsOptions {
    pub extra_headers: HeaderMap,
    pub compression: Compression,
    pub turn_state: Option<Arc<OnceLock<String>>>,
}

impl<T: HttpTransport> ChatCompletionsClient<T> {
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
        request: ChatCompletionsApiRequest,
        options: ChatCompletionsOptions,
    ) -> Result<ResponseStream, ApiError> {
        let ChatCompletionsOptions {
            extra_headers,
            compression,
            turn_state,
        } = options;

        let mut headers = extra_headers;
        headers.extend(build_session_headers(None, None));

        let body = serde_json::to_value(&request)
            .map_err(|e| ApiError::Stream(format!("failed to encode chat completions request: {e}")))?;

        self.stream(body, headers, compression, turn_state).await
    }

    fn path() -> &'static str {
        "chat/completions"
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

        let stream_response = self
            .session
            .stream_with(
                Method::POST,
                Self::path(),
                extra_headers,
                Some(body),
                |req| {
                    req.headers.insert(
                        http::header::ACCEPT,
                        http::HeaderValue::from_static("text/event-stream"),
                    );
                    req.compression = request_compression;
                },
            )
            .await?;

        Ok(spawn_chat_completion_stream(
            stream_response,
            self.session.provider().stream_idle_timeout,
            self.sse_telemetry.clone(),
            turn_state,
        ))
    }
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionsApiRequest {
    pub model: String,
    pub messages: Vec<ChatCompletionMessage>,
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ChatCompletionTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ChatCompletionReasoning>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionReasoning {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effort: Option<String>,
}

impl ChatCompletionsApiRequest {
    pub fn new(model: String, messages: Vec<ChatCompletionMessage>) -> Self {
        Self {
            model,
            messages,
            stream: true,
            max_tokens: None,
            temperature: None,
            top_p: None,
            tools: None,
            tool_choice: None,
            parallel_tool_calls: None,
            reasoning: None,
            stop: None,
            presence_penalty: None,
            frequency_penalty: None,
            user: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn creates_basic_request() {
        let messages = vec![
            ChatCompletionMessage {
                role: "system".to_string(),
                content: "You are a helpful assistant.".to_string(),
                name: None,
                tool_calls: None,
            },
            ChatCompletionMessage {
                role: "user".to_string(),
                content: "Hello!".to_string(),
                name: None,
                tool_calls: None,
            },
        ];

        let request = ChatCompletionsApiRequest::new("gpt-4".to_string(), messages);

        assert_eq!(request.model, "gpt-4");
        assert_eq!(request.messages.len(), 2);
        assert!(request.stream);
    }
}
