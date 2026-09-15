use crate::Error;
use crate::responses::types::{ResponsesWsEvent, ResponsesWsTransformResult};
use crate::responses::websocket::{ResponsesWebSocketProviderConfig, enforce_model};

pub struct OpenAIResponsesWsConfig;

pub const OPENAI_RESPONSES_WS_CONFIG: OpenAIResponsesWsConfig = OpenAIResponsesWsConfig;

impl ResponsesWebSocketProviderConfig for OpenAIResponsesWsConfig {
    fn supports_native_websocket(&self) -> bool {
        true
    }

    fn transform_ws_request(
        &self,
        event: &ResponsesWsEvent,
        model: &str,
    ) -> Result<ResponsesWsTransformResult, Error> {
        let mut event = event.clone();
        event.data = event.data.into_provider_body()?.into();
        Ok(ResponsesWsTransformResult::passthrough(enforce_model(
            &event, model,
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
    fn request_extensions_round_trip_with_overrides_and_resolved_model() {
        let event = serde_json::from_value(serde_json::json!({
            "type":"response.create", "future":{"nested":[null,false,0]},
            "extra_body":{"model":"wrong","provider_option":null}
        }))
        .unwrap();
        let result = OPENAI_RESPONSES_WS_CONFIG
            .transform_ws_request(&event, "resolved")
            .unwrap();
        assert_eq!(
            serde_json::to_value(&result.events[0]).unwrap(),
            serde_json::json!({
                "type":"response.create", "model":"resolved", "future":{"nested":[null,false,0]}, "provider_option":null
            })
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
