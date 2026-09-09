use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use super::transformation::AnthropicMessagesProviderConfig;

pub struct MessagesRequest {
    pub model: String,
    pub body: AnthropicMessagesRequest,
    pub api_key: Option<String>,
    pub api_base: Option<String>,
    pub custom_llm_provider: Option<String>,
    pub extra_headers: Option<Map<String, Value>>,
    pub timeout: Option<Duration>,
}

pub struct ProviderMessagesRequest {
    pub(super) provider: String,
    pub(super) model: String,
    pub(super) config: &'static dyn AnthropicMessagesProviderConfig,
    pub(super) url: String,
    pub(super) http: crate::lifecycle::SettledHttpRequest,
    pub(super) timeout: Option<Duration>,
}

impl ProviderMessagesRequest {
    pub fn body(&self) -> &[u8] {
        self.http.body()
    }

    pub fn headers(&self) -> &[(String, String)] {
        self.http.headers()
    }
}

pub struct MessagesOptions {
    pub model: String,
    pub api_key: Option<String>,
    pub api_base: Option<String>,
    pub custom_llm_provider: Option<String>,
    pub extra_headers: Option<Map<String, Value>>,
    pub timeout: Option<Duration>,
}

pub struct MessagesEndpoint {
    pub(super) provider: String,
    pub(super) model: String,
    pub(super) config: &'static dyn AnthropicMessagesProviderConfig,
    pub(super) url: String,
    pub(super) headers: Vec<(String, String)>,
    pub(super) timeout: Option<Duration>,
}

impl MessagesEndpoint {
    pub fn request_body_policy(&self) -> crate::lifecycle::RequestBodyPolicy {
        self.config.request_body_policy()
    }

    pub fn url(&self) -> &str {
        &self.url
    }

    pub fn headers(&self) -> &[(String, String)] {
        &self.headers
    }

    pub fn capture_buffered_body(
        &self,
        mut body: Value,
    ) -> Result<crate::lifecycle::PreCallBody<Value>, crate::Error> {
        let callback = body.clone();
        let object = body.as_object_mut().ok_or_else(|| {
            crate::Error::InvalidRequest("messages body must be an object".into())
        })?;
        object.remove("stream");
        let authorized = self.capture_body(body)?;
        Ok(crate::lifecycle::PreCallBody::StructuredAtBuild {
            callback,
            authorized,
        })
    }

    pub fn capture_body(
        &self,
        body: Value,
    ) -> Result<crate::lifecycle::AuthorizedBody, crate::Error> {
        if self.request_body_policy() != crate::lifecycle::RequestBodyPolicy::StructuredAtBuild {
            return Err(crate::Error::Unsupported("messages request body policy"));
        }
        let wire = crate::lifecycle::WireBody::encode(&body, "messages request")?;
        Ok(crate::lifecycle::AuthorizedBody::new(
            wire,
            self.headers.clone(),
        ))
    }

    pub fn settle(
        self,
        body: crate::lifecycle::AuthorizedBody,
        headers: Vec<(String, String)>,
    ) -> ProviderMessagesRequest {
        ProviderMessagesRequest {
            provider: self.provider,
            model: self.model,
            config: self.config,
            url: self.url,
            http: body.settle_headers(headers),
            timeout: self.timeout,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum SystemPrompt {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ContentBlock {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct CacheControl {
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub cache_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ttl: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scope: Option<String>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AnthropicMessage {
    pub role: String,
    pub content: MessageContent,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AnthropicMessagesRequest {
    pub model: String,
    pub messages: Vec<AnthropicMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<SystemPrompt>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub container: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mcp_servers: Option<Vec<Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_management: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_format: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_config: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speed: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inference_geo: Option<String>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AnthropicMessagesResponse {
    pub id: String,
    #[serde(rename = "type")]
    pub message_type: String,
    pub role: String,
    pub model: String,
    pub content: Vec<Value>,
    // Anthropic always includes stop_reason / stop_sequence, null until the turn
    // ends; serialize them even when None so callers see the same shape as Python.
    pub stop_reason: Option<String>,
    pub stop_sequence: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub container: Option<Value>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}
