use crate::error::Error;

use super::types::{AnthropicMessagesRequest, AnthropicMessagesResponse};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MessagesAuthStrategy {
    Bearer,
    Header(&'static str),
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
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, Error>;

    fn resolve_api_key(
        &self,
        api_key: Option<&str>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, Error>;

    fn auth_strategy(&self) -> MessagesAuthStrategy {
        MessagesAuthStrategy::Header("x-api-key")
    }

    fn accepts_bearer_auth(&self) -> bool {
        false
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
    ) -> Result<AnthropicMessagesRequest, Error> {
        Ok(request)
    }

    fn transform_response(
        &self,
        _model: &str,
        response: AnthropicMessagesResponse,
    ) -> Result<AnthropicMessagesResponse, Error> {
        Ok(response)
    }
}
