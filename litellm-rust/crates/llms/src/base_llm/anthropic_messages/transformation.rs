use litellm_types::llms::anthropic_messages::{
    anthropic_request::AnthropicMessagesRequest, anthropic_response::AnthropicMessagesResponse,
};

pub use crate::base_llm::auth::{Headers, ValidatedEnvironment};
use crate::{
    Error, anthropic::messages::thinking::ThinkingContext,
    base_llm::anthropic_messages::streaming::StreamDecoder,
};

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

    fn complete_stream_url(
        &self,
        api_base: Option<&str>,
        model: &str,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, Error> {
        self.get_complete_url(api_base, model, env_lookup)
    }

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

    fn secret_names(&self) -> &'static [&'static str];

    /// Shapes the forwarded headers and names the credential, the way Python's
    /// `validate_environment` does, without applying it: `resolve_auth` does that once
    /// for every config.
    fn validate_environment(
        &self,
        headers: Headers,
        api_key: Option<&str>,
        model: &str,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<ValidatedEnvironment, Error>;

    /// `None` relays the upstream bytes untouched, which is right for every host that already
    /// speaks Anthropic SSE. A host on another wire returns the decoder that lifts its frames
    /// into Anthropic stream events, and the route re-encodes those as Anthropic SSE.
    fn stream_decoder(&self) -> Option<StreamDecoder> {
        None
    }

    fn default_headers(&self) -> &'static [(&'static str, &'static str)] {
        &[
            ("anthropic-version", "2023-06-01"),
            ("content-type", "application/json"),
        ]
    }

    fn request_headers(&self, headers: Headers, _request: &AnthropicMessagesRequest) -> Headers {
        headers
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base_llm::auth::AuthScheme;

    struct DefaultsConfig;

    impl BaseAnthropicMessagesConfig for DefaultsConfig {
        fn secret_names(&self) -> &'static [&'static str] {
            &[]
        }

        fn get_complete_url(
            &self,
            _api_base: Option<&str>,
            _model: &str,
            _env_lookup: &dyn Fn(&str) -> Option<String>,
        ) -> Result<String, Error> {
            Ok(String::new())
        }

        fn validate_environment(
            &self,
            headers: Headers,
            _api_key: Option<&str>,
            _model: &str,
            _env_lookup: &dyn Fn(&str) -> Option<String>,
        ) -> Result<ValidatedEnvironment, Error> {
            Ok(ValidatedEnvironment {
                headers,
                auth: AuthScheme::Forwarded,
            })
        }
    }

    #[test]
    fn default_request_headers_are_the_given_headers() {
        let request: AnthropicMessagesRequest = serde_json::from_value(serde_json::json!({
            "model": "claude",
            "max_tokens": 16,
            "speed": "fast",
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .unwrap();
        assert_eq!(
            DefaultsConfig.request_headers(headers(&[("x-api-key", "sk")]), &request),
            headers(&[("x-api-key", "sk")])
        );
    }

    fn headers(pairs: &[(&str, &str)]) -> Headers {
        pairs
            .iter()
            .map(|(name, value)| (name.to_string(), value.to_string()))
            .collect()
    }
}
