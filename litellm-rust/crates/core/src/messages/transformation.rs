use crate::error::{CoreError, CoreResult};
use serde_json::Value;

use super::types::{AnthropicMessagesRequest, AnthropicMessagesResponse};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MessagesAuthStrategy {
    Bearer,
    Header(&'static str),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MessagesAuthKind {
    ApiKey {
        strategy: MessagesAuthStrategy,
        accepts_bearer: bool,
    },
    AwsSigV4 {
        region: String,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MessagesStreaming {
    Unsupported,
    SsePassthrough,
    BedrockEventStream,
}

impl MessagesAuthStrategy {
    pub fn header_name(self) -> &'static str {
        match self {
            Self::Bearer => "authorization",
            Self::Header(header_name) => header_name,
        }
    }
}

pub trait AnthropicMessagesProviderConfig: Sync {
    fn complete_url(
        &self,
        api_base: Option<&str>,
        model: &str,
        stream: bool,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> CoreResult<String>;

    fn resolve_api_key(
        &self,
        api_key: Option<&str>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> CoreResult<String> {
        let _ = (api_key, env_lookup);
        Err(crate::error::CoreError::Auth(
            "provider does not use API key authentication".to_string(),
        ))
    }

    fn auth_kind(
        &self,
        _model: &str,
        _env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> CoreResult<MessagesAuthKind> {
        Ok(MessagesAuthKind::ApiKey {
            strategy: MessagesAuthStrategy::Header("x-api-key"),
            accepts_bearer: false,
        })
    }

    fn streaming(&self) -> MessagesStreaming {
        MessagesStreaming::Unsupported
    }

    fn default_headers(&self) -> &'static [(&'static str, &'static str)] {
        &[
            ("anthropic-version", "2023-06-01"),
            ("content-type", "application/json"),
        ]
    }

    fn transform_request(
        &self,
        request: AnthropicMessagesRequest,
    ) -> CoreResult<AnthropicMessagesRequest> {
        Ok(request)
    }

    fn upstream_body(&self, request: AnthropicMessagesRequest) -> CoreResult<Value> {
        serde_json::to_value(self.transform_request(request)?).map_err(|error| {
            CoreError::InvalidRequest(format!("failed to serialize messages request: {error}"))
        })
    }

    fn transform_response(
        &self,
        _model: &str,
        response: AnthropicMessagesResponse,
    ) -> CoreResult<AnthropicMessagesResponse> {
        Ok(response)
    }
}
