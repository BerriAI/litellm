use litellm_llms_types::formats::messages::{MessagesRequest, MessagesResponse};
use litellm_router_types::LitellmParams;
use serde_json::Value;

use super::context::MessagesTransformContext;

pub use crate::base_llm::auth::{Headers, ValidatedEnvironment};
use crate::{
    Error,
    base_llm::{
        connection::{ConnectionInputs, PreparedConnection},
        messages::streaming::StreamDecoder,
    },
};

/// A connection from a config's own `validate_environment` and `get_complete_url`, for
/// providers whose settings have no shared resolver yet.
pub fn prepare_from_parts<C: BaseMessagesConfig + ?Sized>(
    config: &C,
    inputs: ConnectionInputs<'_>,
    headers: Headers,
    model: &str,
    stream: bool,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Result<PreparedConnection, Error> {
    let validated = config.validate_environment(
        headers,
        inputs.api_key,
        model,
        inputs.litellm_params,
        env_lookup,
    )?;
    let url = config.get_complete_url(
        inputs.api_base,
        model,
        inputs.litellm_params,
        stream,
        env_lookup,
    )?;
    Ok(PreparedConnection {
        url,
        headers: validated.headers,
        auth: validated.auth,
        api_base_source: match inputs.api_base {
            Some(_) => litellm_core_utils::settings::SettingSource::Explicit,
            None => litellm_core_utils::settings::SettingSource::Derived,
        },
    })
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

    fn transform_anthropic_messages_response(
        &self,
        _model: &str,
        response: MessagesResponse,
    ) -> Result<MessagesResponse, Error> {
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
        litellm_params: &LitellmParams,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<ValidatedEnvironment, Error>;

    /// The URL, headers and auth plan for one deployment, built from one selection of the
    /// provider's connection settings.
    fn prepare_connection(
        &self,
        inputs: ConnectionInputs<'_>,
        headers: Headers,
        model: &str,
        stream: bool,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<PreparedConnection, Error>;

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

    /// The JSON that goes on the wire, for a host whose body differs from the typed request:
    /// Python's configs `pop("model")` when the model is addressed by the URL.
    fn wire_body(&self, body: Value) -> Value {
        body
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base_llm::auth::AuthScheme;
    use rstest::rstest;

    struct DefaultsConfig;

    impl BaseMessagesConfig for DefaultsConfig {
        fn prepare_connection(
            &self,
            inputs: crate::base_llm::connection::ConnectionInputs<'_>,
            headers: Headers,
            model: &str,
            stream: bool,
            env_lookup: &dyn Fn(&str) -> Option<String>,
        ) -> Result<crate::base_llm::connection::PreparedConnection, Error> {
            crate::base_llm::messages::transformation::prepare_from_parts(
                self, inputs, headers, model, stream, env_lookup,
            )
        }

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
