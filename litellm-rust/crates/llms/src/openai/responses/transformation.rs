use litellm_llms_types::responses::{ResponsesApiResponse, streaming_websocket::ResponsesWsEvent};
use serde_json::{Map, Value};

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

    fn complete_websocket_url(&self, api_base: Option<&str>, model: &str) -> String {
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

fn complete_websocket_url(api_base: Option<&str>, model: &str) -> String {
    let base = api_base
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(OPENAI_RESPONSES_DEFAULT_API_BASE);
    let (base_without_query, query) = base
        .split_once('?')
        .map_or((base, None), |(value, query)| (value, Some(query)));
    let response_url = format!(
        "{}{}",
        base_without_query.trim_end_matches('/'),
        OPENAI_RESPONSES_PATH
    );
    let scheme_flipped = if let Some(rest) = response_url.strip_prefix("https://") {
        format!("wss://{rest}")
    } else if let Some(rest) = response_url.strip_prefix("http://") {
        format!("ws://{rest}")
    } else {
        response_url
    };
    let url = query.map_or(scheme_flipped.clone(), |value| {
        format!("{scheme_flipped}?{value}")
    });
    if query.is_some_and(|value| {
        value
            .split('&')
            .any(|part| part.split('=').next() == Some("model"))
    }) {
        return url;
    }
    format!(
        "{url}{}model={}",
        if query.is_some() { "&" } else { "?" },
        percent_encode(model)
    )
}

fn percent_encode(value: &str) -> String {
    value
        .bytes()
        .map(|byte| {
            if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'.' | b'_' | b'~') {
                format!("{}", byte as char)
            } else {
                format!("%{byte:02X}")
            }
        })
        .collect()
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
    ) -> String {
        let base = api_base
            .filter(|base| !base.is_empty())
            .map(str::to_owned)
            .or_else(|| lookup("OPENAI_BASE_URL"))
            .or_else(|| lookup("OPENAI_API_BASE"))
            .unwrap_or_else(|| OPENAI_RESPONSES_DEFAULT_API_BASE.into());
        format!("{}/responses", base.trim_end_matches('/'))
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
            OPENAI_RESPONSES_WS_CONFIG.complete_websocket_url(api_base, "test-model"),
            format!("{expected_base}{OPENAI_RESPONSES_PATH}?model=test-model")
        );
    }

    #[rstest::rstest]
    #[case::http(
        "http://localhost:8080/",
        "ws://localhost:8080/responses?model=test%20model"
    )]
    #[case::query(
        "https://example.test/v1?foo=bar",
        "wss://example.test/v1/responses?foo=bar&model=test%20model"
    )]
    #[case::existing_model(
        "https://example.test?model=existing",
        "wss://example.test/responses?model=existing"
    )]
    fn provider_url_preserves_query_and_encodes_model(#[case] base: &str, #[case] expected: &str) {
        assert_eq!(
            OPENAI_RESPONSES_WS_CONFIG.complete_websocket_url(Some(base), "test model"),
            expected
        );
    }

    #[test]
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
}
