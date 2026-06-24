//! End-to-end OpenAI realtime invocation.
//!
//! The host-facing entry point, mirroring `providers::ocr::run_ocr`: open the
//! WebSocket to OpenAI, then splice a client realtime stream to the upstream,
//! driving typed events through the pure `OPENAI_REALTIME_CONFIG` transforms.
//! Network, auth header, key resolution, and wire (de)serialization live here so
//! the `transformation` module stays pure and typed.

use std::time::Duration;

use futures_util::{Sink, SinkExt, Stream, StreamExt};
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

const MISSING_KEY_MESSAGE: &str = "Missing OpenAI API Key - a realtime call is being made but no key was passed via params or the OPENAI_API_KEY environment variable";

/// Default **idle** timeout: if neither side sends a frame for this long, the
/// session is reaped. It resets on any activity, so it does not cap a healthy
/// (continuously streaming) session — it only frees a stalled one (e.g. a
/// half-open upstream that keeps the socket open but stops sending).
const DEFAULT_IDLE_TIMEOUT_SECS: u64 = 300;

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

/// Splice a client realtime stream to OpenAI: forward client events upstream
/// (via `transform_realtime_request`) and backend events downstream (via
/// `transform_realtime_response`). Returns when either side closes.
///
/// Generic over the client transport (typed events) so this crate stays
/// framework-agnostic; the gateway adapts its axum socket to these.
pub async fn realtime<In, Out>(
    model: &str,
    api_key: Option<&str>,
    api_base: Option<&str>,
    idle_timeout: Option<Duration>,
    mut client_in: In,
    mut client_out: Out,
) -> CoreResult<()>
where
    In: Stream<Item = RealtimeEvent> + Unpin + Send,
    Out: Sink<RealtimeEvent> + Unpin + Send,
    <Out as Sink<RealtimeEvent>>::Error: std::fmt::Display,
{
    let config = &OPENAI_REALTIME_CONFIG;
    let api_key = resolve_api_key(api_key)?;
    let url = config.complete_url(api_base, model);

    let mut request = url
        .as_str()
        .into_client_request()
        .map_err(|err| CoreError::Network(err.to_string()))?;
    // GA realtime: only Authorization. The legacy OpenAI-Beta header triggers
    // beta_api_shape_disabled, so we do not send it.
    request.headers_mut().insert(
        AUTHORIZATION,
        HeaderValue::from_str(&format!("Bearer {api_key}"))
            .map_err(|err| CoreError::Auth(err.to_string()))?,
    );

    let (upstream, _response) = connect_async(request)
        .await
        .map_err(|err| CoreError::Network(err.to_string()))?;
    let (mut upstream_tx, mut upstream_rx) = upstream.split();

    let idle = idle_timeout.unwrap_or(Duration::from_secs(DEFAULT_IDLE_TIMEOUT_SECS));

    // One loop forwarding both directions. The `sleep(idle)` arm is rebuilt every
    // iteration, so any frame (either way) resets it — it fires only when the
    // session has been fully idle for `idle`, reaping a stalled connection
    // (task + upstream TCP socket) instead of leaking it.
    loop {
        tokio::select! {
            // client -> upstream
            client_event = client_in.next() => {
                let Some(event) = client_event else { break }; // client disconnected
                for outbound in config.transform_realtime_request(&event, model)?.events {
                    let payload = serde_json::to_string(&outbound)
                        .map_err(|err| CoreError::InvalidResponse(err.to_string()))?;
                    upstream_tx
                        .send(Message::Text(payload))
                        .await
                        .map_err(|err| CoreError::Network(err.to_string()))?;
                }
            }
            // upstream -> client
            upstream_message = upstream_rx.next() => {
                let Some(message) = upstream_message else { break }; // upstream closed
                match message.map_err(|err| CoreError::Network(err.to_string()))? {
                    Message::Text(text) => {
                        let event: RealtimeEvent = serde_json::from_str(&text)
                            .map_err(|err| CoreError::InvalidResponse(err.to_string()))?;
                        for outbound in config.transform_realtime_response(&event, model)?.events {
                            client_out
                                .send(outbound)
                                .await
                                .map_err(|err| CoreError::Network(err.to_string()))?;
                        }
                    }
                    Message::Close(_) => break,
                    _ => {}
                }
            }
            // idle timeout: no activity from either side within `idle`
            _ = tokio::time::sleep(idle) => break,
        }
    }
    Ok(())
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

    /// Live end-to-end check against OpenAI. Ignored by default (CI never runs
    /// it); run explicitly with `OPENAI_API_KEY` set:
    ///   `cargo test -p litellm-providers realtime_invokes_openai -- --ignored --nocapture`
    #[tokio::test]
    #[ignore = "hits the live OpenAI realtime API; needs OPENAI_API_KEY"]
    async fn realtime_invokes_openai_and_responds() {
        use futures_channel::mpsc;

        let key =
            std::env::var(OPENAI_API_KEY_ENV).expect("set OPENAI_API_KEY to run this ignored test");

        // client -> provider (we hold `client_tx` to push events upstream)
        let (mut client_tx, client_in) = mpsc::unbounded::<RealtimeEvent>();
        // provider -> client (we hold `backend_rx` to read backend events)
        let (client_out, mut backend_rx) = mpsc::unbounded::<RealtimeEvent>();

        // Clone the key so the spawned task owns its `String` (no borrow across await).
        let key_owned = key.clone();
        let call = tokio::spawn(async move {
            realtime(
                "gpt-realtime",
                Some(&key_owned),
                None,
                None,
                client_in,
                client_out,
            )
            .await
        });

        // 1. First backend event should be session.created.
        let first = tokio::time::timeout(Duration::from_secs(30), backend_rx.next())
            .await
            .expect("timed out waiting for session.created")
            .expect("backend stream closed before session.created");
        assert_eq!(
            first.event_type, "session.created",
            "expected session.created, got: {}",
            first.event_type
        );

        // 2. Ask for a short audio response.
        client_tx
            .send(event(
                r#"{"type":"conversation.item.create","item":{"type":"message","role":"user","content":[{"type":"input_text","text":"Say hi."}]}}"#,
            ))
            .await
            .expect("send conversation.item.create");
        client_tx
            .send(event(r#"{"type":"response.create"}"#))
            .await
            .expect("send response.create");

        // 3. Read backend events; require a non-empty audio delta, then response.done.
        let mut saw_audio_delta = false;
        let mut saw_done = false;
        for _ in 0..500 {
            let next = tokio::time::timeout(Duration::from_secs(30), backend_rx.next()).await;
            let event = match next {
                Ok(Some(event)) => event,
                Ok(None) => break,
                Err(_) => panic!("timed out waiting for backend events"),
            };
            match event.event_type.as_str() {
                "response.output_audio.delta" => {
                    let delta = event
                        .data
                        .get("delta")
                        .and_then(|value| value.as_str())
                        .unwrap_or("");
                    if !delta.is_empty() {
                        saw_audio_delta = true;
                    }
                }
                "response.done" => {
                    saw_done = true;
                    break;
                }
                _ => {}
            }
        }

        assert!(
            saw_audio_delta,
            "expected a response.output_audio.delta with non-empty delta"
        );
        assert!(saw_done, "expected a response.done event");

        // Drop the client sender so the provider's to_upstream side finishes.
        drop(client_tx);
        let _ = call.await;
    }
}
