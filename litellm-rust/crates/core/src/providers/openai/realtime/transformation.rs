use crate::realtime::transformation::RealtimeProviderConfig;
use crate::realtime::types::{RealtimeEvent, RealtimeTransformResult};
use crate::CoreResult;

/// Default OpenAI API base, used when the caller does not override `api_base`.
pub const OPENAI_REALTIME_DEFAULT_API_BASE: &str = "https://api.openai.com";

/// Path appended to the resolved host base to reach the realtime endpoint.
pub const OPENAI_REALTIME_PATH: &str = "/v1/realtime";

/// Percent-encode a query value, escaping any char outside the RFC 3986
/// unreserved set (`A-Za-z0-9-._~`). Keeps us dependency-free; common realtime
/// model slugs have no special chars, but this stays correct for the rest.
fn percent_encode(value: &str) -> String {
    let mut encoded = String::with_capacity(value.len());
    for byte in value.bytes() {
        let unreserved = byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'.' | b'_' | b'~');
        if unreserved {
            encoded.push(byte as char);
        } else {
            encoded.push('%');
            encoded.push_str(&format!("{byte:02X}"));
        }
    }
    encoded
}

/// Build the realtime WebSocket URL, porting Python's `OpenAIRealtime._construct_url`.
///
/// Blank/whitespace `api_base` is treated as absent (guard at resolution time),
/// falling back to the default. The scheme is swapped to its WebSocket
/// equivalent (`https://`→`wss://`, `http://`→`ws://`); bases already using
/// `ws`/`wss` are left untouched. A bare host or unrecognized scheme defaults to
/// secure `wss://` so we never hand a scheme-less URL to the connector (this is
/// a deliberate hardening over Python's `_construct_url`, which would emit a
/// scheme-less URL here). A trailing `/` is trimmed before the path and
/// `?model=<encoded>` are appended.
pub fn complete_url(api_base: Option<&str>, model: &str) -> String {
    let base = api_base
        .map(str::trim)
        .filter(|base| !base.is_empty())
        .unwrap_or(OPENAI_REALTIME_DEFAULT_API_BASE);

    let base = if let Some(rest) = base.strip_prefix("https://") {
        format!("wss://{rest}")
    } else if let Some(rest) = base.strip_prefix("http://") {
        format!("ws://{rest}")
    } else if base.starts_with("wss://") || base.starts_with("ws://") {
        base.to_string()
    } else {
        format!("wss://{base}")
    };

    let base = base.trim_end_matches('/');

    format!(
        "{base}{OPENAI_REALTIME_PATH}?model={}",
        percent_encode(model)
    )
}

pub struct OpenAiRealtimeConfig;

pub const OPENAI_REALTIME_CONFIG: OpenAiRealtimeConfig = OpenAiRealtimeConfig;

impl RealtimeProviderConfig for OpenAiRealtimeConfig {
    fn complete_url(&self, api_base: Option<&str>, model: &str) -> String {
        complete_url(api_base, model)
    }

    fn transform_realtime_request(
        &self,
        event: &RealtimeEvent,
        _model: &str,
    ) -> CoreResult<RealtimeTransformResult> {
        Ok(RealtimeTransformResult::passthrough(event.clone()))
    }

    fn transform_realtime_response(
        &self,
        event: &RealtimeEvent,
        _model: &str,
    ) -> CoreResult<RealtimeTransformResult> {
        Ok(RealtimeTransformResult::passthrough(event.clone()))
    }
}

pub fn transform_realtime_request(
    event: &RealtimeEvent,
    model: &str,
) -> CoreResult<RealtimeTransformResult> {
    OPENAI_REALTIME_CONFIG.transform_realtime_request(event, model)
}

pub fn transform_realtime_response(
    event: &RealtimeEvent,
    model: &str,
) -> CoreResult<RealtimeTransformResult> {
    OPENAI_REALTIME_CONFIG.transform_realtime_response(event, model)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complete_url_defaults_to_openai_wss() {
        assert_eq!(
            complete_url(None, "gpt-4o-realtime-preview"),
            "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview"
        );
    }

    #[test]
    fn complete_url_blank_base_uses_default() {
        assert_eq!(
            complete_url(Some("   "), "gpt-4o-realtime-preview"),
            "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview"
        );
    }

    #[test]
    fn complete_url_swaps_http_to_ws() {
        assert_eq!(
            complete_url(Some("http://localhost:8080"), "gpt-4o-realtime-preview"),
            "ws://localhost:8080/v1/realtime?model=gpt-4o-realtime-preview"
        );
    }

    #[test]
    fn complete_url_dedupes_trailing_slash() {
        assert_eq!(
            complete_url(Some("https://api.openai.com/"), "gpt-4o-realtime-preview"),
            "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview"
        );
    }

    #[test]
    fn complete_url_custom_base() {
        assert_eq!(
            complete_url(Some("https://oai.azure.example"), "gpt-4o-realtime-preview"),
            "wss://oai.azure.example/v1/realtime?model=gpt-4o-realtime-preview"
        );
    }

    #[test]
    fn complete_url_preserves_existing_wss_scheme() {
        assert_eq!(
            complete_url(Some("wss://api.openai.com"), "gpt-realtime"),
            "wss://api.openai.com/v1/realtime?model=gpt-realtime"
        );
    }

    #[test]
    fn complete_url_bare_host_defaults_to_wss() {
        assert_eq!(
            complete_url(Some("api.openai.com"), "gpt-realtime"),
            "wss://api.openai.com/v1/realtime?model=gpt-realtime"
        );
    }

    #[test]
    fn complete_url_percent_encodes_model_space() {
        assert_eq!(
            complete_url(None, "gpt 4o"),
            "wss://api.openai.com/v1/realtime?model=gpt%204o"
        );
    }

    #[test]
    fn transform_realtime_request_passthrough_preserves_event() {
        let event: RealtimeEvent =
            serde_json::from_str(r#"{"type":"session.update","session":{"voice":"alloy"}}"#)
                .expect("valid event");
        let result =
            transform_realtime_request(&event, "gpt-realtime").expect("passthrough is infallible");
        assert_eq!(result.events, vec![event]);
    }

    #[test]
    fn transform_realtime_response_passthrough_preserves_event() {
        let event: RealtimeEvent =
            serde_json::from_str(r#"{"type":"response.output_audio.delta","delta":"abc=="}"#)
                .expect("valid event");
        let result =
            transform_realtime_response(&event, "gpt-realtime").expect("passthrough is infallible");
        assert_eq!(result.events, vec![event]);
    }
}
