use crate::base_llm::auth::{AuthScheme, ValidatedEnvironment, with_default_headers};
use crate::base_llm::responses::transformation::{
    BaseResponsesApiConfig, OPENAI_RESPONSES_DEFAULT_API_BASE,
};
use litellm_auth::{CredentialPlacement, SecretValue};
use litellm_types::responses::main::ResponsesApiResponse;
use litellm_types::responses::streaming_websocket::{ResponsesWsEvent, ResponsesWsTransformResult};
use serde_json::{Map, Value};

use crate::{
    Error,
    base_llm::responses::transformation::{ResponsesWebSocketProviderConfig, enforce_model},
};

pub struct OpenAiResponsesApiConfig;

pub const OPENAI_RESPONSES_WS_CONFIG: OpenAiResponsesApiConfig = OpenAiResponsesApiConfig;

impl ResponsesWebSocketProviderConfig for OpenAiResponsesApiConfig {
    fn supports_native_websocket(&self) -> bool {
        true
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
            headers: with_default_headers(headers, &[("content-type", "application/json")]),
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
        serde_json::from_value(body).map_err(|error| Error::InvalidResponse(error.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
