use crate::Error;
use crate::constants::{OPENAI_RESPONSES_DEFAULT_API_BASE, OPENAI_RESPONSES_PATH};
use crate::providers::openai::percent_encode;
use crate::responses::types::{ResponsesWsEvent, ResponsesWsTransformResult};
use crate::responses::websocket::{ResponsesWebSocketProviderConfig, enforce_model};

pub struct OpenAIResponsesWsConfig;

pub const OPENAI_RESPONSES_WS_CONFIG: OpenAIResponsesWsConfig = OpenAIResponsesWsConfig;

impl ResponsesWebSocketProviderConfig for OpenAIResponsesWsConfig {
    fn resolve_api_key(
        &self,
        api_key: Option<&str>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, Error> {
        crate::providers::openai::auth::resolve_api_key(api_key, env_lookup, "Responses WebSocket")
    }

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

pub fn complete_websocket_url(api_base: Option<&str>, model: &str, model_in_url: bool) -> String {
    let base = api_base
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(OPENAI_RESPONSES_DEFAULT_API_BASE);
    let (base, query) = base
        .split_once('?')
        .map_or((base, None), |(base, query)| (base, Some(query)));
    let response_url = format!("{}{}", base.trim_end_matches('/'), OPENAI_RESPONSES_PATH);
    let response_url = response_url
        .strip_prefix("https://")
        .map(|rest| format!("wss://{rest}"))
        .or_else(|| {
            response_url
                .strip_prefix("http://")
                .map(|rest| format!("ws://{rest}"))
        })
        .unwrap_or(response_url);
    let url = query.map_or_else(
        || response_url.clone(),
        |query| format!("{response_url}?{query}"),
    );
    if !model_in_url
        || query.is_some_and(|query| {
            query
                .split('&')
                .any(|part| part.split('=').next() == Some("model"))
        })
    {
        return url;
    }
    format!(
        "{url}{}model={}",
        if query.is_some() { "&" } else { "?" },
        percent_encode(model)
    )
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
