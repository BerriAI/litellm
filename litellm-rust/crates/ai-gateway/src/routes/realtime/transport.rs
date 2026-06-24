//! Transport adapter: bridges the axum WebSocket to the typed-event
//! `Stream`/`Sink` the service expects. Keeps axum types out of the service.

use std::sync::Arc;

use axum::extract::ws::{Message, WebSocket};
use futures_util::{SinkExt, StreamExt};
use litellm_core::realtime::types::RealtimeEvent;
use litellm_core::router::Router;

/// Split the socket, adapt text frames ⇄ typed events, and run the splice.
pub async fn bridge(socket: WebSocket, router: Arc<Router>, model: String) {
    let (ws_sink, ws_stream) = socket.split();

    // client text frames -> typed events (non-text / parse failures are skipped)
    let client_in = ws_stream.filter_map(|message| async move {
        match message {
            Ok(Message::Text(text)) => serde_json::from_str::<RealtimeEvent>(&text).ok(),
            _ => None,
        }
    });

    // typed events -> client text frames
    let client_out = ws_sink.with(|event: RealtimeEvent| async move {
        Ok::<Message, axum::Error>(Message::Text(
            serde_json::to_string(&event).unwrap_or_default(),
        ))
    });

    futures_util::pin_mut!(client_in, client_out);
    let _ = super::service::run(&router, &model, None, client_in, client_out).await;
}
