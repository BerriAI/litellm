use crate::Error;
use crate::responses::types::{ResponsesWsEvent, ResponsesWsTransformResult};
use crate::responses::websocket::{ResponsesWebSocketProviderConfig, enforce_model};

pub struct OpenAiResponsesWsConfig;

pub const OPENAI_RESPONSES_WS_CONFIG: OpenAiResponsesWsConfig = OpenAiResponsesWsConfig;

impl ResponsesWebSocketProviderConfig for OpenAiResponsesWsConfig {
    fn transform_request(
        &self,
        event: &ResponsesWsEvent,
        model: &str,
    ) -> Result<ResponsesWsTransformResult, Error> {
        Ok(ResponsesWsTransformResult::passthrough(enforce_model(
            event, model,
        )))
    }

    fn transform_response(
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
            .transform_request(&event, "gpt-5")
            .expect("valid transform");
        assert_eq!(result.events[0].model(), Some("gpt-5"));
    }
}
