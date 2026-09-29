use crate::base_llm::endpoint::{ProviderEndpoint, ResolvedEndpoint};
use crate::openai::endpoints::{OpenAiEndpoint, legacy_responses_target};
use litellm_core_utils::url_utils::{Complete, WebSocketUrl};
use litellm_llms_types::formats::responses::{
    ResponsesApiResponse, ResponsesInput, ResponsesRequest, streaming_websocket::ResponsesWsEvent,
};
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

use crate::openai::endpoints::DEFAULT_API_BASE as OPENAI_RESPONSES_DEFAULT_API_BASE;

pub struct OpenAiResponsesApiConfig;

pub const OPENAI_RESPONSES_WS_CONFIG: OpenAiResponsesApiConfig = OpenAiResponsesApiConfig;

impl ResponsesWebSocketProviderConfig for OpenAiResponsesApiConfig {
    fn supports_native_websocket(&self) -> bool {
        true
    }

    fn complete_websocket_url(
        &self,
        api_base: Option<&str>,
        model: &str,
    ) -> Result<WebSocketUrl<Complete>, Error> {
        crate::openai::endpoints::resolve_websocket(
            api_base.unwrap_or(OPENAI_RESPONSES_DEFAULT_API_BASE),
            model,
        )
        .map_err(|error| Error::InvalidRequest(crate::ErrorDetail::invalid("api_base", error)))
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
    ) -> Result<ResolvedEndpoint, Error> {
        let base = api_base
            .filter(|base| !base.is_empty())
            .map(str::to_owned)
            .or_else(|| lookup("OPENAI_BASE_URL"))
            .or_else(|| lookup("OPENAI_API_BASE"))
            .unwrap_or_else(|| OPENAI_RESPONSES_DEFAULT_API_BASE.into());
        let target = legacy_responses_target(&base).map_err(|error| {
            Error::InvalidRequest(crate::ErrorDetail::invalid("api_base", error))
        })?;
        OpenAiEndpoint::Responses
            .resolve(&target)
            .map_err(|error| Error::InvalidRequest(crate::ErrorDetail::invalid("api_base", error)))
    }

    fn transform_responses_api_request(
        &self,
        model: &str,
        input: Value,
        params: Map<String, Value>,
    ) -> Result<Value, Error> {
        let input: ResponsesInput = serde_json::from_value(input).map_err(|_| {
            Error::InvalidRequest("responses input must be a string or an array".into())
        })?;
        if params
            .get("stream")
            .is_some_and(|stream| !stream.is_boolean())
        {
            return Err(Error::InvalidRequest("stream must be a boolean".into()));
        }
        Ok(serde_json::json!(ResponsesRequest {
            model: model.into(),
            input,
            extra: params
                .into_iter()
                .filter(|(key, _)| !matches!(key.as_str(), "model" | "input"))
                .collect(),
        }))
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
                .as_url()
                .as_str(),
            format!(
                "{expected_base}/{}?model=test-model",
                OpenAiEndpoint::Responses
                    .path()
                    .segments()
                    .collect::<Vec<_>>()
                    .join("/")
            )
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
                .as_url()
                .as_str(),
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
