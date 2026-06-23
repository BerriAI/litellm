//! End-to-end OpenAI realtime invocation.
//!
//! The host-facing entry point, mirroring `providers::ocr::run_ocr`: open the
//! WebSocket to OpenAI, drive typed events through the pure
//! `OPENAI_REALTIME_CONFIG` transforms, and collect the response events.
//! Network, auth header, key resolution, and wire (de)serialization live here so
//! the `transformation` module stays pure and typed.

use std::time::Duration;

use futures_util::{SinkExt, StreamExt};
use litellm_core::error::CoreError;
use litellm_core::realtime::transformation::RealtimeProviderConfig;
use litellm_core::realtime::types::RealtimeEvent;
use litellm_core::CoreResult;
use tokio_tungstenite::connect_async;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::http::header::AUTHORIZATION;
use tokio_tungstenite::tungstenite::http::HeaderValue;
use tokio_tungstenite::tungstenite::Message;

use crate::openai::realtime::transformation::OPENAI_REALTIME_CONFIG;

/// Environment variable holding the OpenAI API key (last-resort fallback).
const OPENAI_API_KEY_ENV: &str = "OPENAI_API_KEY";

/// Default overall ceiling for a single realtime invocation.
const DEFAULT_TIMEOUT_SECS: u64 = 60;

const MISSING_KEY_MESSAGE: &str = "Missing OpenAI API Key - a realtime call is being made but no key was passed via params or the OPENAI_API_KEY environment variable";

/// Resolve the OpenAI API key from the explicit param or the environment.
///
/// Blank/whitespace values are treated as absent (guard at resolution time).
fn resolve_api_key(api_key: Option<&str>) -> CoreResult<String> {
    api_key
        .map(str::trim)
        .filter(|key| !key.is_empty())
        .map(str::to_string)
        .or_else(|| {
            std::env::var(OPENAI_API_KEY_ENV)
                .ok()
                .filter(|key| !key.trim().is_empty())
        })
        .ok_or_else(|| CoreError::Auth(MISSING_KEY_MESSAGE.to_string()))
}

/// True for events that end a realtime turn: a completed response or an error.
fn is_terminal_event(event: &RealtimeEvent) -> bool {
    event.event_type == "response.done" || event.event_type == "error"
}

/// Invoke the OpenAI realtime API end to end over a WebSocket.
///
/// Sends each `input_events` entry after passing it through
/// `transform_realtime_request`, then collects backend events — each passed
/// through `transform_realtime_response` — until a terminal event
/// (`response.done` / `error`) arrives, the socket closes, or the `timeout`
/// elapses. Returns the transformed backend events in arrival order.
///
/// Mirrors `run_ocr`: pure transforms come from `core`/`providers`; the network,
/// auth header, key resolution, and JSON (de)serialization are owned here.
pub async fn realtime(
    model: &str,
    input_events: Vec<RealtimeEvent>,
    api_key: Option<&str>,
    api_base: Option<&str>,
    timeout: Option<Duration>,
) -> CoreResult<Vec<RealtimeEvent>> {
    let config = &OPENAI_REALTIME_CONFIG;
    let api_key = resolve_api_key(api_key)?;
    let url = config.complete_url(api_base, model);

    let mut request = url
        .as_str()
        .into_client_request()
        .map_err(|err| CoreError::Network(err.to_string()))?;
    // GA realtime API: only Authorization is needed. The legacy
    // `OpenAI-Beta: realtime=v1` header opts into the now-removed beta request
    // shape and triggers `beta_api_shape_disabled`, so we do not send it.
    request.headers_mut().insert(
        AUTHORIZATION,
        HeaderValue::from_str(&format!("Bearer {api_key}"))
            .map_err(|err| CoreError::Auth(err.to_string()))?,
    );

    let (mut ws, _response) = connect_async(request)
        .await
        .map_err(|err| CoreError::Network(err.to_string()))?;

    for event in &input_events {
        for outbound in config.transform_realtime_request(event, model)?.events {
            let payload = serde_json::to_string(&outbound)
                .map_err(|err| CoreError::InvalidResponse(err.to_string()))?;
            ws.send(Message::Text(payload))
                .await
                .map_err(|err| CoreError::Network(err.to_string()))?;
        }
    }

    let deadline = timeout.unwrap_or_else(|| Duration::from_secs(DEFAULT_TIMEOUT_SECS));
    let mut received: Vec<RealtimeEvent> = Vec::new();

    let collect = async {
        while let Some(message) = ws.next().await {
            match message.map_err(|err| CoreError::Network(err.to_string()))? {
                Message::Text(text) => {
                    let event: RealtimeEvent = serde_json::from_str(text.as_str())
                        .map_err(|err| CoreError::InvalidResponse(err.to_string()))?;
                    for outbound in config.transform_realtime_response(&event, model)?.events {
                        let terminal = is_terminal_event(&outbound);
                        received.push(outbound);
                        if terminal {
                            return Ok::<(), CoreError>(());
                        }
                    }
                }
                Message::Close(_) => return Ok(()),
                _ => {}
            }
        }
        Ok(())
    };

    tokio::time::timeout(deadline, collect)
        .await
        .map_err(|_| CoreError::Network("realtime call timed out".to_string()))??;

    Ok(received)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn event(raw: &str) -> RealtimeEvent {
        serde_json::from_str(raw).expect("valid event json")
    }

    #[test]
    fn resolve_api_key_prefers_param_then_blank_falls_through() {
        assert_eq!(resolve_api_key(Some("sk-test")).unwrap(), "sk-test");
        // A blank param with no env set should error.
        if std::env::var(OPENAI_API_KEY_ENV).is_err() {
            assert!(resolve_api_key(Some("   ")).is_err());
        }
    }

    #[test]
    fn is_terminal_event_matches_done_and_error_only() {
        assert!(is_terminal_event(&event(r#"{"type":"response.done"}"#)));
        assert!(is_terminal_event(&event(r#"{"type":"error","error":{}}"#)));
        assert!(!is_terminal_event(&event(
            r#"{"type":"response.output_text.delta"}"#
        )));
    }

    /// Live end-to-end check against OpenAI. Ignored by default (CI never runs
    /// it); run explicitly with `OPENAI_API_KEY` set:
    ///   `cargo test -p litellm-providers realtime_invokes_openai -- --ignored --nocapture`
    #[tokio::test]
    #[ignore = "hits the live OpenAI realtime API; needs OPENAI_API_KEY"]
    async fn realtime_invokes_openai_and_responds() {
        let key =
            std::env::var(OPENAI_API_KEY_ENV).expect("set OPENAI_API_KEY to run this ignored test");

        let response_create = event(
            r#"{"type":"response.create","response":{"output_modalities":["text"],"instructions":"Respond with exactly: hello world"}}"#,
        );

        let events = realtime(
            "gpt-realtime",
            vec![response_create],
            Some(&key),
            None,
            Some(Duration::from_secs(30)),
        )
        .await
        .expect("realtime call should succeed");

        let types: Vec<&str> = events.iter().map(|e| e.event_type.as_str()).collect();
        eprintln!("received {} events: {:?}", events.len(), types);

        assert!(
            types.contains(&"response.done"),
            "expected a response.done event, got: {types:?}"
        );
        assert!(
            types.contains(&"response.output_text.delta"),
            "expected streamed text output, got: {types:?}"
        );
    }
}
