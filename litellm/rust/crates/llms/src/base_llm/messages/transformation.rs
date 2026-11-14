use litellm_host::interceptors::RetriedRequest;
use litellm_llms_types::formats::messages::{
    MessagesOptionalParams, MessagesRequest, MessagesResponse,
};
use litellm_router_types::LitellmParams;
use litellm_secret_value::SecretValue;
use serde_json::Value;

use super::{
    context::{MessagesShapeOptions, MessagesTransformContext},
    tools::ToolCustomField,
};

pub use crate::base_llm::auth::{Headers, ValidatedEnvironment};
use crate::{Error, base_llm::messages::streaming::StreamDecoder};

pub enum FederatedEnvironment {
    Validated(ValidatedEnvironment),
    Unauthenticated {
        headers: Headers,
        missing: Error,
        federate: fn(Headers, SecretValue) -> ValidatedEnvironment,
    },
}

pub trait BaseMessagesConfig: Sync {
    fn shape_request(
        &self,
        request: MessagesRequest,
        _options: MessagesShapeOptions,
    ) -> Result<MessagesRequest, Error> {
        Ok(request)
    }

    fn get_complete_url(
        &self,
        api_base: Option<&str>,
        model: &str,
        litellm_params: &LitellmParams,
        stream: bool,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, Error>;

    fn transform_anthropic_messages_request(
        &self,
        request: MessagesRequest,
        _context: &MessagesTransformContext,
    ) -> Result<MessagesRequest, Error> {
        Ok(request)
    }

    fn tool_custom_field(&self, _model: &str) -> ToolCustomField {
        ToolCustomField::Keep
    }

    fn transform_anthropic_messages_response(
        &self,
        _model: &str,
        response: MessagesResponse,
    ) -> Result<MessagesResponse, Error> {
        Ok(response)
    }

    fn reported_cost(&self, _response: &MessagesResponse) -> Option<serde_json::Number> {
        None
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
        optional_params: &MessagesOptionalParams,
        litellm_params: &LitellmParams,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<ValidatedEnvironment, Error>;

    fn validate_federated_environment(
        &self,
        headers: Headers,
        api_key: Option<&str>,
        model: &str,
        optional_params: &MessagesOptionalParams,
        litellm_params: &LitellmParams,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<FederatedEnvironment, Error> {
        self.validate_environment(
            headers,
            api_key,
            model,
            optional_params,
            litellm_params,
            env_lookup,
        )
        .map(FederatedEnvironment::Validated)
    }

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

    fn request_for_url(&self, request: MessagesRequest, _url: &str) -> MessagesRequest {
        request
    }

    fn headers_for_url(&self, headers: Headers, _url: &str) -> Headers {
        headers
    }

    fn request_body(
        &self,
        request: MessagesRequest,
        _headers: &Headers,
        _drop_params: bool,
    ) -> Result<MessagesRequest, Error> {
        Ok(request)
    }

    /// The JSON that goes on the wire, for a host whose body differs from the typed request:
    /// Python's configs `pop("model")` when the model is addressed by the URL.
    fn wire_body(&self, body: Value) -> Value {
        body
    }

    fn retry_after_http_error(
        &self,
        _status: u16,
        _error_text: &str,
        _request_body: &Value,
        _context: &MessagesTransformContext,
    ) -> Option<RetriedRequest> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base_llm::auth::AuthScheme;
    use rstest::rstest;

    struct DefaultsConfig;

    impl BaseMessagesConfig for DefaultsConfig {
        fn secret_names(&self) -> &'static [&'static str] {
            &[]
        }

        fn get_complete_url(
            &self,
            _api_base: Option<&str>,
            _model: &str,
            _litellm_params: &LitellmParams,
            _stream: bool,
            _env_lookup: &dyn Fn(&str) -> Option<String>,
        ) -> Result<String, Error> {
            Ok(String::new())
        }

        fn validate_environment(
            &self,
            headers: Headers,
            _api_key: Option<&str>,
            _model: &str,
            _optional_params: &MessagesOptionalParams,
            _litellm_params: &LitellmParams,
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
    fn default_shaping_preserves_provider_policy_inputs(
        #[values(false, true)] reasoning_auto_summary: bool,
        #[values(false, true)] presanitized: bool,
    ) {
        let request: MessagesRequest = serde_json::from_value(serde_json::json!({
            "model": "test-model",
            "metadata": {"user_id": 7, "extra": "keep"},
            "thinking": {"type": "enabled", "budget_tokens": 64},
            "messages": [{"role": "system", "content": "context"}]
        }))
        .unwrap();
        assert_eq!(
            DefaultsConfig.shape_request(
                request.clone(),
                MessagesShapeOptions {
                    reasoning_auto_summary,
                    presanitized
                }
            ),
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
