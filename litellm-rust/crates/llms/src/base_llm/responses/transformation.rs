use litellm_llms_types::responses::{ResponsesApiResponse, streaming_websocket::ResponsesWsEvent};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::{Error, base_llm::auth::ValidatedEnvironment};

pub trait BaseResponsesApiConfig: Sync {
    fn secret_names(
        &self,
        api_key: Option<&str>,
        api_base: Option<&str>,
    ) -> &'static [&'static str];

    fn validate_environment(
        &self,
        headers: Vec<(String, String)>,
        api_key: Option<&str>,
        lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<ValidatedEnvironment, Error>;

    fn get_complete_url(
        &self,
        api_base: Option<&str>,
        lookup: &dyn Fn(&str) -> Option<String>,
    ) -> String;

    fn transform_responses_api_request(
        &self,
        model: &str,
        input: Value,
        params: Map<String, Value>,
    ) -> Result<Value, Error>;

    fn transform_response_api_response(&self, body: Value) -> Result<ResponsesApiResponse, Error>;
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResponsesWsTransformResult {
    pub events: Vec<ResponsesWsEvent>,
}

impl ResponsesWsTransformResult {
    pub fn passthrough(event: ResponsesWsEvent) -> Self {
        Self {
            events: vec![event],
        }
    }
}

pub trait ResponsesWebSocketProviderConfig: Sync {
    fn supports_native_websocket(&self) -> bool {
        false
    }

    fn complete_websocket_url(&self, api_base: Option<&str>, model: &str) -> String;

    fn transform_ws_request(
        &self,
        event: &ResponsesWsEvent,
        model: &str,
    ) -> Result<ResponsesWsTransformResult, Error>;

    fn transform_ws_response(
        &self,
        event: &ResponsesWsEvent,
        model: &str,
    ) -> Result<ResponsesWsTransformResult, Error>;
}

pub fn enforce_model(event: &ResponsesWsEvent, model: &str) -> ResponsesWsEvent {
    if !event.is_response_create() {
        return event.clone();
    }
    let mut enforced = event.clone();
    let has_flat_model = enforced.data.contains_key("model");
    if let Some(response) = enforced
        .data
        .get_mut("response")
        .and_then(serde_json::Value::as_object_mut)
    {
        response.insert(
            "model".to_string(),
            serde_json::Value::String(model.to_string()),
        );
        if has_flat_model {
            enforced.data.insert(
                "model".to_string(),
                serde_json::Value::String(model.to_string()),
            );
        }
    } else {
        enforced.data.insert(
            "model".to_string(),
            serde_json::Value::String(model.to_string()),
        );
    }
    enforced
}

#[cfg(test)]
mod tests {
    use super::*;

    fn event(value: serde_json::Value) -> ResponsesWsEvent {
        serde_json::from_value(value).expect("valid event")
    }

    #[test]
    fn enforce_model_overrides_flat_and_nested_values() {
        let flat = enforce_model(
            &event(serde_json::json!({"type":"response.create","model":"wrong"})),
            "gpt-5",
        );
        assert_eq!(flat.model(), Some("gpt-5"));
        let nested = enforce_model(
            &event(serde_json::json!({
                "type":"response.create",
                "model":"wrong",
                "response":{"model":"also-wrong"}
            })),
            "gpt-5",
        );
        assert_eq!(nested.model(), Some("gpt-5"));
        assert_eq!(
            nested
                .data
                .get("response")
                .and_then(|value| value.get("model")),
            Some(&serde_json::json!("gpt-5"))
        );
        let nested_without_flat = enforce_model(
            &event(serde_json::json!({
                "type":"response.create",
                "response":{"model":"also-wrong"}
            })),
            "gpt-5",
        );
        assert!(!nested_without_flat.data.contains_key("model"));
    }
}
