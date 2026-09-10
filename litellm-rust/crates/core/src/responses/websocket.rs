use crate::Error;
use crate::constants::OPENAI_RESPONSES_DEFAULT_API_BASE;
use crate::responses::types::{ResponsesWsEvent, ResponsesWsEventType, ResponsesWsTransformResult};
use crate::url_utils::ApiUrl;

pub trait ResponsesWebSocketProviderConfig: Sync {
    fn model_in_websocket_url(&self) -> bool {
        true
    }

    fn complete_url(&self, api_base: Option<&str>, model: &str) -> Result<String, Error> {
        complete_url(api_base, model, self.model_in_websocket_url())
    }

    fn transform_request(
        &self,
        event: &ResponsesWsEvent,
        model: &str,
    ) -> Result<ResponsesWsTransformResult, Error>;

    fn transform_response(
        &self,
        event: &ResponsesWsEvent,
        model: &str,
    ) -> Result<ResponsesWsTransformResult, Error>;
}

pub fn complete_url(
    api_base: Option<&str>,
    model: &str,
    model_in_websocket_url: bool,
) -> Result<String, Error> {
    let base = api_base
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(OPENAI_RESPONSES_DEFAULT_API_BASE);
    let url = ApiUrl::parse(base)
        .and_then(|url| match url.scheme() {
            "https" => url.with_scheme("wss"),
            "http" => url.with_scheme("ws"),
            _ => Ok(url),
        })
        .and_then(|url| url.complete_path(&["responses"]))
        .map_err(|error| Error::InvalidRequest(format!("invalid api_base: {error}")))?;
    if !model_in_websocket_url || url.has_query_key("model") {
        return Ok(url.into_string());
    }
    Ok(url.append_query_pair("model", model).into_string())
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

pub fn is_terminal_event(event_type: &ResponsesWsEventType) -> bool {
    matches!(
        event_type,
        ResponsesWsEventType::ResponseCreated
            | ResponsesWsEventType::ResponseCompleted
            | ResponsesWsEventType::ResponseFailed
            | ResponsesWsEventType::ResponseIncomplete
            | ResponsesWsEventType::Error
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn event(value: serde_json::Value) -> ResponsesWsEvent {
        serde_json::from_value(value).expect("valid event")
    }

    #[test]
    fn url_construction_matches_python_defaults_and_query_behavior() {
        assert_eq!(
            complete_url(None, "gpt-5", true).expect("url builds"),
            "wss://api.openai.com/v1/responses?model=gpt-5"
        );
        assert_eq!(
            complete_url(Some("http://localhost:8080/"), "gpt 5", true).expect("url builds"),
            "ws://localhost:8080/responses?model=gpt+5"
        );
        assert_eq!(
            complete_url(Some("https://example.test/v1?foo=bar"), "gpt-5", true)
                .expect("url builds"),
            "wss://example.test/v1/responses?foo=bar&model=gpt-5"
        );
        assert_eq!(
            complete_url(Some("https://example.test?model=existing"), "gpt-5", true)
                .expect("url builds"),
            "wss://example.test/responses?model=existing"
        );
        assert_eq!(
            complete_url(
                Some("https://example.test/v1/responses?foo=bar"),
                "gpt-5",
                true,
            )
            .expect("url builds"),
            "wss://example.test/v1/responses?foo=bar&model=gpt-5"
        );
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
