use crate::Error;
use crate::realtime::transformation::RealtimeProviderConfig;
use crate::realtime::types::{RealtimeEvent, RealtimeTransformResult};
use crate::url_utils::ApiUrl;

/// Default OpenAI API base, used when the caller does not override `api_base`.
pub const OPENAI_REALTIME_DEFAULT_API_BASE: &str = "https://api.openai.com";

/// Path appended to the resolved host base to reach the realtime endpoint.
pub const OPENAI_REALTIME_PATH: &str = "/v1/realtime";

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
pub fn complete_url(api_base: Option<&str>, model: &str) -> Result<String, Error> {
    let base = api_base
        .map(str::trim)
        .filter(|base| !base.is_empty())
        .unwrap_or(OPENAI_REALTIME_DEFAULT_API_BASE);

    let url = ApiUrl::parse_with_default_scheme(base, "wss")
        .and_then(|url| match url.scheme() {
            "https" => url.with_scheme("wss"),
            "http" => url.with_scheme("ws"),
            "wss" | "ws" => Ok(url),
            _ => url.with_scheme("wss"),
        })
        .and_then(|url| url.complete_path(&["v1", "realtime"]))
        .map_err(|error| Error::InvalidRequest(format!("invalid api_base: {error}")))?;
    Ok(url.append_query_pair("model", model).into_string())
}

pub struct OpenAiRealtimeConfig;

pub const OPENAI_REALTIME_CONFIG: OpenAiRealtimeConfig = OpenAiRealtimeConfig;

impl RealtimeProviderConfig for OpenAiRealtimeConfig {
    fn complete_url(&self, api_base: Option<&str>, model: &str) -> Result<String, Error> {
        complete_url(api_base, model)
    }

    fn transform_request(
        &self,
        event: &RealtimeEvent,
        _model: &str,
    ) -> Result<RealtimeTransformResult, Error> {
        Ok(RealtimeTransformResult::passthrough(event.clone()))
    }

    fn transform_response(
        &self,
        event: &RealtimeEvent,
        _model: &str,
    ) -> Result<RealtimeTransformResult, Error> {
        Ok(RealtimeTransformResult::passthrough(event.clone()))
    }
}

pub fn transform_request(
    event: &RealtimeEvent,
    model: &str,
) -> Result<RealtimeTransformResult, Error> {
    OPENAI_REALTIME_CONFIG.transform_request(event, model)
}

pub fn transform_response(
    event: &RealtimeEvent,
    model: &str,
) -> Result<RealtimeTransformResult, Error> {
    OPENAI_REALTIME_CONFIG.transform_response(event, model)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complete_url_defaults_to_openai_wss() {
        assert_eq!(
            complete_url(None, "gpt-4o-realtime-preview").expect("url builds"),
            "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview"
        );
    }

    #[test]
    fn complete_url_blank_base_uses_default() {
        assert_eq!(
            complete_url(Some("   "), "gpt-4o-realtime-preview").expect("url builds"),
            "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview"
        );
    }

    #[test]
    fn complete_url_swaps_http_to_ws() {
        assert_eq!(
            complete_url(Some("http://localhost:8080"), "gpt-4o-realtime-preview")
                .expect("url builds"),
            "ws://localhost:8080/v1/realtime?model=gpt-4o-realtime-preview"
        );
    }

    #[test]
    fn complete_url_dedupes_trailing_slash() {
        assert_eq!(
            complete_url(Some("https://api.openai.com/"), "gpt-4o-realtime-preview")
                .expect("url builds"),
            "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview"
        );
    }

    #[test]
    fn complete_url_custom_base() {
        assert_eq!(
            complete_url(Some("https://oai.azure.example"), "gpt-4o-realtime-preview")
                .expect("url builds"),
            "wss://oai.azure.example/v1/realtime?model=gpt-4o-realtime-preview"
        );
    }

    #[test]
    fn complete_url_preserves_existing_wss_scheme() {
        assert_eq!(
            complete_url(Some("wss://api.openai.com"), "gpt-realtime").expect("url builds"),
            "wss://api.openai.com/v1/realtime?model=gpt-realtime"
        );
    }

    #[test]
    fn complete_url_bare_host_defaults_to_wss() {
        assert_eq!(
            complete_url(Some("api.openai.com"), "gpt-realtime").expect("url builds"),
            "wss://api.openai.com/v1/realtime?model=gpt-realtime"
        );
    }

    #[test]
    fn complete_url_percent_encodes_model_space() {
        assert_eq!(
            complete_url(None, "gpt 4o").expect("url builds"),
            "wss://api.openai.com/v1/realtime?model=gpt+4o"
        );
    }

    #[test]
    fn transform_realtime_request_passthrough_preserves_event() {
        let event: RealtimeEvent =
            serde_json::from_str(r#"{"type":"session.update","session":{"voice":"alloy"}}"#)
                .expect("valid event");
        let result = transform_request(&event, "gpt-realtime").expect("passthrough is infallible");
        assert_eq!(result.events, vec![event]);
    }

    #[test]
    fn transform_realtime_response_passthrough_preserves_event() {
        let event: RealtimeEvent =
            serde_json::from_str(r#"{"type":"response.output_audio.delta","delta":"abc=="}"#)
                .expect("valid event");
        let result = transform_response(&event, "gpt-realtime").expect("passthrough is infallible");
        assert_eq!(result.events, vec![event]);
    }
}
