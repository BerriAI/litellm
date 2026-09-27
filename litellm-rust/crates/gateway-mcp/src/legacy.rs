use std::{sync::Arc, time::Duration};

use axum::{
    Json, Router,
    extract::{Query, Request, State},
    http::{StatusCode, request::Parts},
    response::{
        IntoResponse, Response, Sse,
        sse::{Event, KeepAlive},
    },
    routing::{get, post},
};
use futures_util::{StreamExt, sink, stream};
use moka::future::Cache;
use rmcp::{ServiceExt, model::*};
use serde::Deserialize;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::{Context, HttpConfig, McpServer, Operations, sessions::fingerprint};

#[derive(Clone)]
struct Session {
    sender: mpsc::Sender<ClientJsonRpcMessage>,
    owner: [u8; 32],
    cancellation: CancellationToken,
}

#[derive(Clone)]
struct Legacy {
    operations: Arc<dyn Operations>,
    server: McpServer,
    sessions: Cache<String, Session>,
    shutdown: CancellationToken,
    hosts: Arc<[String]>,
    origins: Arc<[String]>,
}

pub(crate) fn router(
    operations: Arc<dyn Operations>,
    server: McpServer,
    config: &HttpConfig,
) -> Router {
    let state = Legacy {
        operations,
        server,
        sessions: Cache::builder()
            .max_capacity(10_000)
            .time_to_idle(Duration::from_secs(300))
            .eviction_listener(|_, session: Session, _| session.cancellation.cancel())
            .build(),
        shutdown: config.cancellation_token.clone(),
        hosts: config.allowed_hosts.clone().into(),
        origins: config.allowed_origins.clone().into(),
    };
    Router::new()
        .route("/mcp/sse", get(connect))
        .route("/mcp/sse/", get(connect))
        .route("/mcp/sse/messages", post(message))
        .route("/mcp/sse/messages/", post(message))
        .with_state(state)
}

impl Legacy {
    fn validate(&self, parts: &Parts) -> Result<(), StatusCode> {
        let host = parts
            .headers
            .get("host")
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.parse::<http::uri::Authority>().ok())
            .ok_or(StatusCode::FORBIDDEN)?;
        let allowed = self.hosts.iter().any(|allowed| {
            host.as_str().eq_ignore_ascii_case(allowed)
                || host
                    .host()
                    .trim_matches(['[', ']'])
                    .eq_ignore_ascii_case(allowed)
        });
        if !allowed {
            return Err(StatusCode::FORBIDDEN);
        }
        if let Some(origin) = parts.headers.get("origin") {
            let origin = origin.to_str().map_err(|_| StatusCode::FORBIDDEN)?;
            if !self
                .origins
                .iter()
                .any(|allowed| same_origin(origin, allowed))
            {
                return Err(StatusCode::FORBIDDEN);
            }
        }
        Ok(())
    }
}

fn same_origin(value: &str, allowed: &str) -> bool {
    if value == "null" {
        return allowed == "null";
    }
    match (url::Url::parse(value), url::Url::parse(allowed)) {
        (Ok(value), Ok(allowed)) => {
            value.path() == "/"
                && value.query().is_none()
                && value.fragment().is_none()
                && value.username().is_empty()
                && value.password().is_none()
                && value.origin() == allowed.origin()
        }
        _ => false,
    }
}

async fn connect(State(state): State<Legacy>, request: Request) -> Response {
    let owner = fingerprint(&request);
    let (parts, _) = request.into_parts();
    if let Err(status) = state.validate(&parts) {
        return status.into_response();
    }
    if let Err(error) = state
        .operations
        .authorize(Context {
            parts,
            server: None,
            mcp: None,
        })
        .await
    {
        return error.into_response();
    }
    let id = Uuid::new_v4().to_string();
    let cancellation = state.shutdown.child_token();
    let (input, receiver) = mpsc::channel(16);
    let (sender, output) = mpsc::channel::<ServerJsonRpcMessage>(16);
    state
        .sessions
        .insert(
            id.clone(),
            Session {
                sender: input,
                owner,
                cancellation: cancellation.clone(),
            },
        )
        .await;
    let sink = Box::pin(sink::unfold(sender, |sender, message| async {
        sender.send(message).await.map_err(std::io::Error::other)?;
        Ok::<_, std::io::Error>(sender)
    }));
    let input = Box::pin(stream::unfold(receiver, |mut receiver| async {
        receiver.recv().await.map(|message| (message, receiver))
    }));
    let worker_id = id.clone();
    let worker_cancellation = cancellation.clone();
    tokio::spawn(async move {
        if let Ok(service) = state
            .server
            .serve_with_ct((sink, input), worker_cancellation)
            .await
        {
            let _ = service.waiting().await;
        }
        state.sessions.invalidate(&worker_id).await;
    });
    let first = stream::once(async move {
        Ok::<_, std::io::Error>(
            Event::default()
                .event("endpoint")
                .data(format!("/mcp/sse/messages?session_id={id}")),
        )
    });
    let messages = stream::unfold(
        (output, cancellation.drop_guard()),
        |(mut output, guard)| async {
            output.recv().await.map(|message| {
                let event = Event::default()
                    .event("message")
                    .json_data(message)
                    .map_err(std::io::Error::other);
                (event, (output, guard))
            })
        },
    );
    Sse::new(first.chain(messages))
        .keep_alive(KeepAlive::default())
        .into_response()
}

#[derive(Deserialize)]
struct SessionQuery {
    session_id: String,
}

async fn message(
    State(state): State<Legacy>,
    Query(query): Query<SessionQuery>,
    request: Request,
) -> Response {
    let owner = fingerprint(&request);
    let (parts, body) = request.into_parts();
    if let Err(status) = state.validate(&parts) {
        return status.into_response();
    }
    let Some(session) = state.sessions.get(&query.session_id).await else {
        return StatusCode::NOT_FOUND.into_response();
    };
    if session.owner != owner {
        return StatusCode::FORBIDDEN.into_response();
    }
    let request = Request::from_parts(parts.clone(), body);
    let Json(mut message) =
        match <Json<ClientJsonRpcMessage> as axum::extract::FromRequest<()>>::from_request(
            request,
            &(),
        )
        .await
        {
            Ok(message) => message,
            Err(error) => return error.into_response(),
        };
    match &mut message {
        ClientJsonRpcMessage::Request(request) => {
            request.request.extensions_mut().insert(parts);
        }
        ClientJsonRpcMessage::Notification(notification) => {
            notification.notification.extensions_mut().insert(parts);
        }
        _ => (),
    }
    match session.sender.try_send(message) {
        Ok(()) => (StatusCode::ACCEPTED, "Accepted").into_response(),
        Err(mpsc::error::TrySendError::Full(_)) => StatusCode::TOO_MANY_REQUESTS.into_response(),
        Err(mpsc::error::TrySendError::Closed(_)) => StatusCode::NOT_FOUND.into_response(),
    }
}
