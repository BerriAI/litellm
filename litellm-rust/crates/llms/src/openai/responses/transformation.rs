use litellm_auth::{CredentialPlacement, SecretValue};
use litellm_http::request::has_bearer_auth;
use litellm_types::responses::streaming_websocket::{ResponsesWsEvent, ResponsesWsTransformResult};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::{
    Error,
    base_llm::auth::{AuthScheme, Headers, ValidatedEnvironment},
    base_llm::responses::transformation::{ResponsesWebSocketProviderConfig, enforce_model},
};

pub struct OpenAiResponsesApiConfig;

pub const OPENAI_RESPONSES_WS_CONFIG: OpenAiResponsesApiConfig = OpenAiResponsesApiConfig;

#[derive(Debug, Deserialize, Serialize)]
pub struct ResponsesResponse {
    pub id: String,
    pub object: String,
    pub output: Vec<Value>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

impl OpenAiResponsesApiConfig {
    pub fn validate_environment(
        &self,
        headers: Headers,
        api_key: Option<&str>,
    ) -> Result<ValidatedEnvironment, Error> {
        let auth = match api_key.filter(|key| !key.is_empty()) {
            Some(key) => AuthScheme::Credential {
                placement: CredentialPlacement::Bearer,
                secret: SecretValue::new(key.to_owned()),
            },
            None if has_bearer_auth(&headers) => AuthScheme::Forwarded,
            None => {
                return Err(litellm_auth::Error::MissingApiKey {
                    provider: "OpenAI",
                    environment_variable: "OPENAI_API_KEY",
                }
                .into());
            }
        };
        Ok(ValidatedEnvironment { headers, auth })
    }

    pub fn get_complete_url(&self, api_base: Option<&str>) -> Result<String, Error> {
        let mut url = url::Url::parse(api_base.unwrap_or("https://api.openai.com/v1"))
            .map_err(|error| Error::InvalidRequest(format!("invalid api_base: {error}")))?;
        if !matches!(url.scheme(), "http" | "https") {
            return Err(Error::InvalidRequest(
                "api_base must use HTTP or HTTPS".into(),
            ));
        }
        let path = url.path().trim_end_matches('/');
        let endpoint = if path.ends_with("/responses") {
            path.to_owned()
        } else {
            format!("{path}/responses")
        };
        url.set_path(&endpoint);
        Ok(url.into())
    }

    pub fn transform_request(
        &self,
        model: &str,
        body: Map<String, Value>,
    ) -> Result<Map<String, Value>, Error> {
        if model.trim().is_empty() {
            return Err(Error::InvalidRequest("model is required".into()));
        }
        if let Some(input) = body.get("input")
            && !matches!(input, Value::String(_) | Value::Array(_) | Value::Null)
        {
            return Err(Error::InvalidRequest(
                "input must be a string or array".into(),
            ));
        }
        if let Some(stream) = body.get("stream")
            && !matches!(stream, Value::Bool(_) | Value::Null)
        {
            return Err(Error::InvalidRequest("stream must be a boolean".into()));
        }
        Ok(body
            .into_iter()
            .filter(|(name, _)| name != "model")
            .chain([("model".into(), Value::from(model))])
            .collect())
    }

    pub fn transform_response(&self, body: &[u8]) -> Result<ResponsesResponse, Error> {
        let response: ResponsesResponse = serde_json::from_slice(body).map_err(|error| {
            Error::InvalidResponse(format!("invalid Responses response: {error}"))
        })?;
        if response.object != "response" {
            return Err(Error::InvalidResponse("expected a response object".into()));
        }
        Ok(response)
    }
}

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
