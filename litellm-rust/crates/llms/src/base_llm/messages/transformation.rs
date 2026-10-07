use litellm_llms_types::formats::messages::{MessagesRequest, MessagesResponse};

use super::context::MessagesTransformContext;
pub use crate::base_llm::auth::{Headers, ValidatedEnvironment};
use crate::{Error, base_llm::messages::streaming::StreamDecoder};

pub const MESSAGES_PATH_SUFFIX: &str = "/v1/messages";
const VERSION_PATH_SUFFIX: &str = "/v1";

pub struct MessagesWireRequest {
    pub body: serde_json::Value,
    pub headers: Headers,
}

pub(crate) fn messages_request_body(request: &MessagesRequest) -> Result<serde_json::Value, Error> {
    serde_json::to_value(request).map_err(|error| {
        Error::InvalidRequest(crate::ErrorDetail::invalid("Messages request body", error))
    })
}

/// The Messages endpoint under an Anthropic-compatible base. A base already ending in
/// `/v1/messages` is used as is, and one trailing `/v1` is dropped before the suffix goes on,
/// so the same base serves a provider's OpenAI-compatible routes too.
pub fn complete_messages_url(api_base: &str) -> String {
    let api_base = api_base.trim_end_matches('/');
    if api_base.ends_with(MESSAGES_PATH_SUFFIX) {
        return api_base.to_string();
    }
    let api_base = api_base
        .strip_suffix(VERSION_PATH_SUFFIX)
        .unwrap_or(api_base);
    format!("{api_base}{MESSAGES_PATH_SUFFIX}")
}

pub trait BaseMessagesConfig: Sync {
    fn shape_request(
        &self,
        request: MessagesRequest,
        _reasoning_auto_summary: bool,
    ) -> Result<MessagesRequest, Error> {
        Ok(request)
    }

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
        request: MessagesRequest,
        _context: &MessagesTransformContext,
    ) -> Result<MessagesRequest, Error> {
        Ok(request)
    }

    fn transform_anthropic_messages_response(
        &self,
        _model: &str,
        response: MessagesResponse,
    ) -> Result<MessagesResponse, Error> {
        Ok(response)
    }

    fn request_body(&self, request: &MessagesRequest) -> Result<serde_json::Value, Error> {
        messages_request_body(request)
    }

    fn prepare_wire_request(
        &self,
        request: &MessagesRequest,
        headers: Headers,
    ) -> Result<MessagesWireRequest, Error> {
        Ok(MessagesWireRequest {
            body: self.request_body(request)?,
            headers,
        })
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
        &[("content-type", "application/json")]
    }

    fn request_headers(&self, headers: Headers, _request: &MessagesRequest) -> Headers {
        headers
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;
    use crate::base_llm::auth::AuthScheme;

    #[rstest]
    #[case::bare_host("https://h.example", "https://h.example/v1/messages")]
    #[case::trailing_slash(
        "https://h.example/anthropic/",
        "https://h.example/anthropic/v1/messages"
    )]
    #[case::version_suffix("https://h.example/v1", "https://h.example/v1/messages")]
    #[case::version_suffix_with_slash("https://h.example/v1/", "https://h.example/v1/messages")]
    #[case::other_version_kept("https://h.example/v3", "https://h.example/v3/v1/messages")]
    #[case::complete_endpoint("https://h.example/v1/messages", "https://h.example/v1/messages")]
    #[case::complete_endpoint_with_slash(
        "https://h.example/x/v1/messages/",
        "https://h.example/x/v1/messages"
    )]
    fn complete_messages_url_appends_the_suffix_once(#[case] base: &str, #[case] expected: &str) {
        assert_eq!(complete_messages_url(base), expected);
    }

    struct DefaultsConfig;

    impl BaseMessagesConfig for DefaultsConfig {
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
        let request: MessagesRequest = serde_json::from_value(serde_json::json!({
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

    #[rstest]
    #[case::disabled(false)]
    #[case::enabled(true)]
    fn default_shaping_preserves_provider_policy_inputs(#[case] reasoning_auto_summary: bool) {
        let request: MessagesRequest = serde_json::from_value(serde_json::json!({
            "model": "test-model",
            "metadata": {"user_id": 7, "extra": "keep"},
            "thinking": {"type": "enabled", "budget_tokens": 64},
            "messages": [{"role": "system", "content": "context"}]
        }))
        .unwrap();
        assert_eq!(
            DefaultsConfig.shape_request(request.clone(), reasoning_auto_summary),
            Ok(request)
        );
        assert_eq!(
            DefaultsConfig.default_headers(),
            &[("content-type", "application/json")]
        );
    }

    fn headers(pairs: &[(&str, &str)]) -> Headers {
        pairs
            .iter()
            .map(|(name, value)| (name.to_string(), value.to_string()))
            .collect()
    }
}
