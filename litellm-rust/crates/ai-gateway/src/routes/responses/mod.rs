mod service;

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::Response;
use axum::routing::get;
use axum::Router;
use futures_util::{Sink, SinkExt, StreamExt};
use litellm_core::responses::types::{ResponsesErrorFrame, ResponsesWsEvent, ResponsesWsEventType};
use litellm_core::router::Router as ModelRouter;
use serde::Deserialize;

use crate::auth::RequireMasterKey;
use crate::integrations::custom_logger::CustomLogger;
use crate::integrations::types::RequestMetadata;
use crate::responses::streaming::{ResponseSessionStatus, ResponsesWsStreaming};
use crate::state::AppState;

static CALL_SEQ: AtomicU64 = AtomicU64::new(0);

fn new_call_id() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    let sequence = CALL_SEQ.fetch_add(1, Ordering::Relaxed);
    format!("respws-{nanos:x}-{sequence:x}")
}

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/v1/responses", get(handle))
        .route("/responses", get(handle))
}

#[derive(Debug, Deserialize)]
struct ResponsesQuery {
    model: Option<String>,
}

async fn handle(
    _auth: RequireMasterKey,
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
    Query(query): Query<ResponsesQuery>,
) -> Result<Response, (StatusCode, String)> {
    if let Some(model) = query.model.as_deref() {
        validate_model(&state.router, model)?;
    }
    let router = state.router.clone();
    let loggers = state.loggers.clone();
    let master_key = state.master_key.clone();
    Ok(ws.on_upgrade(move |socket| bridge(socket, router, loggers, master_key, query.model)))
}

fn validate_model(router: &ModelRouter, model: &str) -> Result<(), (StatusCode, String)> {
    if model.trim().is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            "missing 'model' query param".to_string(),
        ));
    }
    if !router.has_deployment(model) {
        return Err((
            StatusCode::NOT_FOUND,
            format!("no deployment for model '{model}'"),
        ));
    }
    Ok(())
}

async fn send_error_and_close<S>(sink: &mut S, message: String)
where
    S: futures_util::Sink<Message> + Unpin,
    S::Error: std::fmt::Display,
{
    if let Ok(payload) = serde_json::to_string(&ResponsesErrorFrame::invalid_request(message)) {
        let _ = sink.send(Message::Text(payload)).await;
    }
    let _ = sink
        .send(Message::Close(Some(axum::extract::ws::CloseFrame {
            code: 1008,
            reason: "Pre-call error".into(),
        })))
        .await;
    let _ = sink.close().await;
}

struct ResponseClientSink {
    sink: futures_util::stream::SplitSink<WebSocket, Message>,
}

impl Sink<ResponsesWsEvent> for ResponseClientSink {
    type Error = axum::Error;

    fn poll_ready(
        mut self: std::pin::Pin<&mut Self>,
        context: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Result<(), Self::Error>> {
        std::pin::Pin::new(&mut self.sink).poll_ready(context)
    }

    fn start_send(
        mut self: std::pin::Pin<&mut Self>,
        item: ResponsesWsEvent,
    ) -> Result<(), Self::Error> {
        let payload = serde_json::to_string(&item).map_err(axum::Error::new)?;
        std::pin::Pin::new(&mut self.sink).start_send(Message::Text(payload))
    }

    fn poll_flush(
        mut self: std::pin::Pin<&mut Self>,
        context: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Result<(), Self::Error>> {
        std::pin::Pin::new(&mut self.sink).poll_flush(context)
    }

    fn poll_close(
        mut self: std::pin::Pin<&mut Self>,
        context: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Result<(), Self::Error>> {
        std::pin::Pin::new(&mut self.sink).poll_close(context)
    }
}

impl ResponseClientSink {
    async fn close_with_code(&mut self, code: u16, reason: &'static str) {
        let _ = self
            .sink
            .send(Message::Close(Some(axum::extract::ws::CloseFrame {
                code,
                reason: reason.into(),
            })))
            .await;
        let _ = self.sink.close().await;
    }
}

async fn bridge(
    socket: WebSocket,
    router: Arc<ModelRouter>,
    loggers: Arc<Vec<Arc<dyn CustomLogger>>>,
    master_key: Option<Arc<str>>,
    requested_model: Option<String>,
) {
    let (mut ws_sink, ws_stream) = socket.split();
    let (model, first_frame, stream) = if let Some(model) = requested_model {
        (model, None, ws_stream)
    } else {
        let mut stream = ws_stream;
        let first = match stream.next().await {
            Some(Ok(Message::Text(text))) => {
                match serde_json::from_str::<ResponsesWsEvent>(&text) {
                    Ok(event) => event,
                    Err(_) => {
                        send_error_and_close(
                            &mut ws_sink,
                            "Invalid JSON in response.create event".to_string(),
                        )
                        .await;
                        return;
                    }
                }
            }
            _ => {
                send_error_and_close(&mut ws_sink, "Missing response.create event".to_string())
                    .await;
                return;
            }
        };
        let Some(model) = first.model().filter(|value| !value.trim().is_empty()) else {
            send_error_and_close(
                &mut ws_sink,
                "Missing model in response.create event".to_string(),
            )
            .await;
            return;
        };
        if first.event_type != ResponsesWsEventType::ResponseCreate {
            send_error_and_close(
                &mut ws_sink,
                "First frame must be a response.create event".to_string(),
            )
            .await;
            return;
        }
        (model.to_string(), Some(first), stream)
    };
    if let Err((status, message)) = validate_model(&router, &model) {
        let _ = status;
        let _ = message;
        send_error_and_close(&mut ws_sink, "Unknown model deployment".to_string()).await;
        return;
    }

    let metadata = RequestMetadata {
        user_api_key_hash: master_key.as_deref().map(crate::auth::hash_token),
        ..RequestMetadata::default()
    };
    let mut collector = ResponsesWsStreaming::new(
        loggers.as_ref().clone(),
        new_call_id(),
        model.clone(),
        metadata,
    );
    let client_in = Box::pin(stream.filter_map(|message| async move {
        match message {
            Ok(Message::Text(text)) => serde_json::from_str::<ResponsesWsEvent>(&text).ok(),
            _ => None,
        }
    }));
    let mut client_out = ResponseClientSink { sink: ws_sink };
    let result = service::run(
        &router,
        &model,
        first_frame,
        None,
        |event| collector.observe(event),
        client_in,
        &mut client_out,
    )
    .await;
    if result.is_err() {
        client_out
            .close_with_code(1011, "Internal server error")
            .await;
    }
    collector
        .log_messages(if result.is_ok() {
            ResponseSessionStatus::Success
        } else {
            ResponseSessionStatus::Failure
        })
        .await;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::realtime_pool::RealtimePool;
    use crate::state::AppState;
    use axum::body::Body;
    use axum::http::Request;
    use litellm_core::router::Router as ModelRouter;
    use serde_json::json;
    use std::pin::Pin;
    use std::sync::Arc;
    use std::task::{Context, Poll};
    use tower::ServiceExt;

    struct RecordingSink {
        messages: Vec<Message>,
    }

    impl Sink<Message> for RecordingSink {
        type Error = std::convert::Infallible;

        fn poll_ready(
            self: Pin<&mut Self>,
            _context: &mut Context<'_>,
        ) -> Poll<Result<(), Self::Error>> {
            Poll::Ready(Ok(()))
        }

        fn start_send(mut self: Pin<&mut Self>, item: Message) -> Result<(), Self::Error> {
            self.messages.push(item);
            Ok(())
        }

        fn poll_flush(
            self: Pin<&mut Self>,
            _context: &mut Context<'_>,
        ) -> Poll<Result<(), Self::Error>> {
            Poll::Ready(Ok(()))
        }

        fn poll_close(
            self: Pin<&mut Self>,
            _context: &mut Context<'_>,
        ) -> Poll<Result<(), Self::Error>> {
            Poll::Ready(Ok(()))
        }
    }

    #[tokio::test]
    async fn pre_call_error_matches_python_frame_and_close() {
        let mut sink = RecordingSink {
            messages: Vec::new(),
        };
        send_error_and_close(&mut sink, "missing model".to_string()).await;
        let Message::Text(payload) = &sink.messages[0] else {
            panic!("expected error text frame");
        };
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(payload).expect("error json"),
            json!({
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": "missing model"
                }
            })
        );
        assert_eq!(
            sink.messages[1],
            Message::Close(Some(axum::extract::ws::CloseFrame {
                code: 1008,
                reason: "Pre-call error".into(),
            }))
        );
    }

    fn state() -> AppState {
        AppState {
            router: Arc::new(ModelRouter::default()),
            master_key: Some(Arc::from("master-key")),
            loggers: Arc::new(Vec::new()),
            realtime_pool: RealtimePool::disabled(),
        }
    }

    #[tokio::test]
    async fn auth_rejects_responses_upgrade_before_handler() {
        let request = Request::builder()
            .uri("/responses?model=known")
            .body(Body::empty())
            .expect("request");
        let response = router()
            .with_state(state())
            .oneshot(request)
            .await
            .expect("response");
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[test]
    fn unknown_query_model_is_rejected_before_upgrade() {
        assert_eq!(
            validate_model(&ModelRouter::default(), "unknown").expect_err("unknown model"),
            (
                StatusCode::NOT_FOUND,
                "no deployment for model 'unknown'".to_string()
            )
        );
    }
}
