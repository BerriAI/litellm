use litellm_llms_types::formats::responses::{
    ResponsesApiResponse, streaming_websocket::ResponsesWsEvent,
};
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
    use litellm_llms_types::recognized::Recognized;
    let nested = event.response.as_ref().and_then(Recognized::known);
    ResponsesWsEvent {
        model: if event.model.is_some() || nested.is_none() {
            Some(Recognized::Known(model.to_string()))
        } else {
            None
        },
        response: match nested {
            Some(response) => Some(Recognized::Known(litellm_llms_types::formats::responses::streaming_websocket::ResponsesEventResponse {
                model: Some(Recognized::Known(model.to_string())),
                ..response.clone()
            })),
            None => event.response.clone(),
        },
        ..event.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn event(value: serde_json::Value) -> ResponsesWsEvent {
        serde_json::from_value(value).expect("valid event")
    }

    #[rstest::rstest]
    #[case::flat(
        serde_json::json!({"type":"response.create","model":"wrong"}),
        serde_json::json!({"type":"response.create","model":"configured-model"})
    )]
    #[case::both(
        serde_json::json!({"type":"response.create","model":null,"response":{"model":"wrong","future":null}}),
        serde_json::json!({"type":"response.create","model":"configured-model","response":{"model":"configured-model","future":null}})
    )]
    #[case::nested_only(
        serde_json::json!({"type":"response.create","response":{"model":false}}),
        serde_json::json!({"type":"response.create","response":{"model":"configured-model"}})
    )]
    #[case::malformed_response(
        serde_json::json!({"type":"response.create","response":17}),
        serde_json::json!({"type":"response.create","model":"configured-model","response":17})
    )]
    #[case::not_create(
        serde_json::json!({"type":"response.created","response":{"model":"provider-model"}}),
        serde_json::json!({"type":"response.created","response":{"model":"provider-model"}})
    )]
    fn enforce_model_preserves_shapes_and_extensions(
        #[case] wire: serde_json::Value,
        #[case] expected: serde_json::Value,
    ) {
        let enforced = enforce_model(&event(wire), "configured-model");
        assert_eq!(serde_json::to_value(enforced).unwrap(), expected);
    }
}
