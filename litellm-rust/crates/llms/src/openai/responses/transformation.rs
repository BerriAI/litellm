use litellm_core_utils::url_utils::ApiUrl;
use litellm_types::responses::main::ResponsesApiResponse;
use litellm_types::responses::streaming_websocket::ResponsesWsEvent;
use serde_json::{Map, Value};
use url::Url;

use litellm_auth::{CredentialPlacement, SecretValue};

use crate::{
    Error,
    base_llm::{
        auth::{AuthScheme, ValidatedEnvironment},
        responses::transformation::{
            BaseResponsesApiConfig, ResponsesWebSocketProviderConfig, ResponsesWsTransformResult,
            enforce_model,
        },
    },
};

pub const OPENAI_RESPONSES_DEFAULT_API_BASE: &str = "https://api.openai.com/v1";
pub const OPENAI_RESPONSES_PATH: &str = "/responses";

pub struct OpenAiResponsesApiConfig;

pub const OPENAI_RESPONSES_WS_CONFIG: OpenAiResponsesApiConfig = OpenAiResponsesApiConfig;

impl ResponsesWebSocketProviderConfig for OpenAiResponsesApiConfig {
    fn supports_native_websocket(&self) -> bool {
        true
    }

    fn complete_websocket_url(&self, api_base: Option<&str>, model: &str) -> Result<Url, Error> {
        complete_websocket_url(api_base, model)
    }

    fn transform_ws_request(
        &self,
        event: &ResponsesWsEvent,
        model: &str,
    ) -> Result<ResponsesWsTransformResult, Error> {
        Ok(ResponsesWsTransformResult::passthrough(enforce_model(
            event, model,
        )))
    }

    fn transform_ws_response(
        &self,
        event: &ResponsesWsEvent,
        _model: &str,
    ) -> Result<ResponsesWsTransformResult, Error> {
        Ok(ResponsesWsTransformResult::passthrough(event.clone()))
    }
}

fn complete_websocket_url(api_base: Option<&str>, model: &str) -> Result<Url, Error> {
    let base = api_base
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(OPENAI_RESPONSES_DEFAULT_API_BASE);
    let mut url = ApiUrl::parse(base)?
        .complete_path(&["responses"])?
        .into_url();
    let scheme = match url.scheme() {
        "https" | "wss" => "wss",
        "http" | "ws" => "ws",
        _ => return Err(Error::InvalidRequest("invalid WebSocket scheme".into())),
    };
    url.set_scheme(scheme)
        .map_err(|()| Error::InvalidRequest("invalid WebSocket scheme".into()))?;
    if !url.query_pairs().any(|(name, _)| name == "model") {
        url.query_pairs_mut().append_pair("model", model);
    }
    Ok(url)
}

impl BaseResponsesApiConfig for OpenAiResponsesApiConfig {
    fn secret_names(
        &self,
        api_key: Option<&str>,
        api_base: Option<&str>,
    ) -> &'static [&'static str] {
        match (
            api_key.is_some_and(|key| !key.is_empty()),
            api_base.is_some_and(|base| !base.is_empty()),
        ) {
            (true, true) => &[],
            (true, false) => &["OPENAI_BASE_URL", "OPENAI_API_BASE"],
            (false, true) => &["OPENAI_API_KEY"],
            (false, false) => &["OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_API_BASE"],
        }
    }

    fn validate_environment(
        &self,
        headers: Vec<(String, String)>,
        api_key: Option<&str>,
        lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<ValidatedEnvironment, Error> {
        let key = api_key
            .filter(|key| !key.is_empty())
            .map(str::to_owned)
            .or_else(|| lookup("OPENAI_API_KEY"))
            .filter(|key| !key.is_empty())
            .ok_or(litellm_auth::Error::MissingApiKey {
                provider: "OpenAI",
                environment_variable: "OPENAI_API_KEY",
            })?;
        Ok(ValidatedEnvironment {
            headers: litellm_http::request::with_default_headers(
                headers,
                &[("content-type", "application/json")],
            ),
            auth: AuthScheme::Credential {
                placement: CredentialPlacement::Bearer,
                secret: SecretValue::new(key),
            },
        })
    }

    fn get_complete_url(
        &self,
        api_base: Option<&str>,
        lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<Url, Error> {
        let base = api_base
            .filter(|base| !base.is_empty())
            .map(str::to_owned)
            .or_else(|| lookup("OPENAI_BASE_URL"))
            .or_else(|| lookup("OPENAI_API_BASE"))
            .unwrap_or_else(|| OPENAI_RESPONSES_DEFAULT_API_BASE.into());
        Ok(ApiUrl::parse(&base)?
            .complete_path(&["responses"])?
            .into_url())
    }

    fn transform_responses_api_request(
        &self,
        model: &str,
        input: Value,
        params: Map<String, Value>,
    ) -> Result<Value, Error> {
        if !input.is_string() && !input.is_array() {
            return Err(Error::InvalidRequest(
                "responses input must be a string or an array".into(),
            ));
        }
        if params
            .get("stream")
            .is_some_and(|stream| !stream.is_boolean())
        {
            return Err(Error::InvalidRequest("stream must be a boolean".into()));
        }
        Ok(Value::Object(
            params
                .into_iter()
                .chain([
                    ("model".into(), Value::String(model.into())),
                    ("input".into(), input),
                ])
                .collect(),
        ))
    }

    fn transform_response_api_response(&self, body: Value) -> Result<ResponsesApiResponse, Error> {
        serde_json::from_value(body)
            .map_err(|error| Error::InvalidResponse(error.to_string().into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[rstest::rstest]
    #[case::default(None)]
    #[case::blank(Some(" "))]
    fn default_endpoint_belongs_to_openai(#[case] api_base: Option<&str>) {
        let expected_base = OPENAI_RESPONSES_DEFAULT_API_BASE.replacen("https://", "wss://", 1);
        assert_eq!(
            OPENAI_RESPONSES_WS_CONFIG
                .complete_websocket_url(api_base, "test-model")
                .unwrap()
                .as_str(),
            format!("{expected_base}{OPENAI_RESPONSES_PATH}?model=test-model")
        );
    }

    #[rstest::rstest]
    #[case::http(
        "http://localhost:8080/",
        "ws://localhost:8080/responses?model=test+model"
    )]
    #[case::query(
        "https://example.test/v1?foo=bar",
        "wss://example.test/v1/responses?foo=bar&model=test+model"
    )]
    #[case::existing_model(
        "https://example.test?model=existing",
        "wss://example.test/responses?model=existing"
    )]
    fn provider_url_preserves_query_and_encodes_model(#[case] base: &str, #[case] expected: &str) {
        assert_eq!(
            OPENAI_RESPONSES_WS_CONFIG
                .complete_websocket_url(Some(base), "test model")
                .unwrap()
                .as_str(),
            expected
        );
    }

    #[rstest::rstest]
    fn openai_config_is_native_and_enforces_model() {
        let event: ResponsesWsEvent =
            serde_json::from_value(serde_json::json!({"type":"response.create"}))
                .expect("valid event");
        let result = OPENAI_RESPONSES_WS_CONFIG
            .transform_ws_request(&event, "gpt-5")
            .expect("valid transform");
        assert_eq!(result.events[0].model(), Some("gpt-5"));
        assert!(OPENAI_RESPONSES_WS_CONFIG.supports_native_websocket());
    }
    #[rstest::rstest]
    #[case::complete(
        "https://example.test/prefix/responses?x=a#fragment",
        "wss://example.test/prefix/responses?x=a&model=a%2Fb%25%3F%23#fragment"
    )]
    #[case::decoded_key(
        "https://example.test/v1?%6Dodel=existing#f",
        "wss://example.test/v1/responses?%6Dodel=existing#f"
    )]
    #[case::ipv6(
        "http://[::1]:8080/v1",
        "ws://[::1]:8080/v1/responses?model=a%2Fb%25%3F%23"
    )]
    fn websocket_completion_uses_url_components(#[case] base: &str, #[case] expected: &str) {
        assert_eq!(
            complete_websocket_url(Some(base), "a/b%?#")
                .unwrap()
                .as_str(),
            expected
        );
    }

    #[rstest::rstest]
    #[case::base(
        "https://example.test/prefix/v1?tenant=a#f",
        "https://example.test/prefix/v1/responses?tenant=a#f"
    )]
    #[case::complete(
        "https://example.test/prefix/responses?tenant=a#f",
        "https://example.test/prefix/responses?tenant=a#f"
    )]
    fn http_completion_preserves_queries(#[case] base: &str, #[case] expected: &str) {
        assert_eq!(
            OpenAiResponsesApiConfig
                .get_complete_url(Some(base), &|_| None)
                .unwrap()
                .as_str(),
            expected
        );
    }

    #[rstest::rstest]
    #[case::invalid("relative/path")]
    #[case::unsupported("ftp://example.test/v1")]
    fn invalid_endpoints_fail_before_dispatch(#[case] base: &str) {
        assert!(complete_websocket_url(Some(base), "model").is_err());
        assert!(
            OpenAiResponsesApiConfig
                .get_complete_url(Some(base), &|_| None)
                .is_err()
        );
    }
}
