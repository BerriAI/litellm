use crate::constants::{OPENAI_DEFAULT_API_BASE, OPENAI_RESPONSES_PATH};
use crate::responses::types::{ResponsesWsEvent, ResponsesWsEventType, ResponsesWsTransformResult};
use crate::CoreResult;

pub trait ResponsesWebSocketProviderConfig: Sync {
    fn supports_native_websocket(&self) -> bool {
        false
    }

    fn model_in_websocket_url(&self) -> bool {
        true
    }

    fn complete_websocket_url(&self, api_base: Option<&str>, model: &str) -> String {
        complete_websocket_url(api_base, model, self.model_in_websocket_url())
    }

    fn transform_ws_request(
        &self,
        event: &ResponsesWsEvent,
        model: &str,
    ) -> CoreResult<ResponsesWsTransformResult>;

    fn transform_ws_response(
        &self,
        event: &ResponsesWsEvent,
        model: &str,
    ) -> CoreResult<ResponsesWsTransformResult>;
}

pub fn complete_websocket_url(
    api_base: Option<&str>,
    model: &str,
    model_in_websocket_url: bool,
) -> String {
    let base = api_base
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(OPENAI_DEFAULT_API_BASE);
    let scheme_flipped = if let Some(rest) = base.strip_prefix("https://") {
        format!("wss://{rest}")
    } else if let Some(rest) = base.strip_prefix("http://") {
        format!("ws://{rest}")
    } else {
        base.to_string()
    };
    let (host, query) = scheme_flipped
        .split_once('?')
        .map_or((scheme_flipped.as_str(), None), |(host, query)| {
            (host, Some(query))
        });
    let url = format!(
        "{}{}{}",
        host.trim_end_matches('/'),
        OPENAI_RESPONSES_PATH,
        query.map_or(String::new(), |value| format!("?{value}"))
    );
    if !model_in_websocket_url
        || query.is_some_and(|value| {
            value
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
            complete_websocket_url(None, "gpt-5", true),
            "wss://api.openai.com/v1/responses?model=gpt-5"
        );
        assert_eq!(
            complete_websocket_url(Some("http://localhost:8080/"), "gpt 5", true),
            "ws://localhost:8080/v1/responses?model=gpt%205"
        );
        assert_eq!(
            complete_websocket_url(Some("https://example.test/v1?foo=bar"), "gpt-5", true),
            "wss://example.test/v1/v1/responses?foo=bar&model=gpt-5"
        );
        assert_eq!(
            complete_websocket_url(Some("https://example.test?model=existing"), "gpt-5", true),
            "wss://example.test/v1/responses?model=existing"
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
