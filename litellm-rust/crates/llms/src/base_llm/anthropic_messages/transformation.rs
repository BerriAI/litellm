use litellm_http::request::{has_bearer_auth, has_header};
use litellm_types::llms::anthropic_messages::{
    anthropic_request::AnthropicMessagesRequest, anthropic_response::AnthropicMessagesResponse,
};

use crate::{
    anthropic::experimental_pass_through::messages::thinking::ThinkingContext,
    base_llm::chat::transformation::Error,
};

pub type Headers = Vec<(String, String)>;

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

/// What a provider transformation knows about the call beyond the request body: the
/// model's capability flags and the caller's `drop_params` choice.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MessagesTransformContext {
    pub thinking: ThinkingContext,
    pub drop_params: bool,
}

pub trait BaseAnthropicMessagesConfig: Sync {
    fn get_complete_url(
        &self,
        api_base: Option<&str>,
        model: &str,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, Error>;

    fn transform_anthropic_messages_request(
        &self,
        request: AnthropicMessagesRequest,
        _context: &MessagesTransformContext,
    ) -> Result<AnthropicMessagesRequest, Error> {
        Ok(request)
    }

    fn transform_anthropic_messages_response(
        &self,
        _model: &str,
        response: AnthropicMessagesResponse,
    ) -> Result<AnthropicMessagesResponse, Error> {
        Ok(response)
    }

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

    /// The forwarded headers with the provider credential applied. A request that already
    /// carries the provider's auth header (or a bearer the provider accepts) is left alone.
    fn authenticate(
        &self,
        headers: Headers,
        api_key: Option<&str>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<Headers, Error> {
        let strategy = self.auth_strategy();
        if has_header(&headers, strategy.header_name())
            || (self.accepts_bearer_auth() && has_bearer_auth(&headers))
        {
            return Ok(headers);
        }
        let api_key = self.resolve_api_key(api_key, env_lookup)?;
        let auth_header = match strategy {
            MessagesAuthStrategy::Bearer => {
                ("authorization".to_string(), format!("Bearer {api_key}"))
            }
            MessagesAuthStrategy::Header(name) => (name.to_string(), api_key),
        };
        Ok(headers.into_iter().chain([auth_header]).collect())
    }

    fn default_headers(&self) -> &'static [(&'static str, &'static str)] {
        &[
            ("anthropic-version", "2023-06-01"),
            ("content-type", "application/json"),
        ]
    }

    /// Headers the transformed request's features call for, such as `anthropic-beta`.
    fn request_headers(&self, headers: Headers, _request: &AnthropicMessagesRequest) -> Headers {
        headers
    }
}
