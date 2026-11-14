use std::{sync::Arc, time::Duration};

use axum::{
    Router,
    extract::{FromRequest, Request, State},
    http::{StatusCode, request::Parts},
    response::{
        IntoResponse, Response, Sse,
        sse::{Event, KeepAlive},
    },
    routing::{get, post},
};
use futures_util::{Stream, StreamExt, sink, stream};
use moka::future::Cache;
use rmcp::{ServiceExt, model::*};
use serde::Deserialize;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::{
    Context, Error, HttpConfig, McpServer, Operations,
    request::{Json, Query},
    sessions::fingerprint,
};

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
    fn validate(&self, parts: &Parts) -> Result<(), Error> {
        let host = parts
            .headers
            .get("host")
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.parse::<http::uri::Authority>().ok())
            .ok_or(Error::HostNotAllowed)?;
        let allowed = self.hosts.iter().any(|allowed| {
            host.as_str().eq_ignore_ascii_case(allowed)
                || host
                    .host()
                    .trim_matches(['[', ']'])
                    .eq_ignore_ascii_case(allowed)
        });
        if !allowed {
            return Err(Error::HostNotAllowed);
        }
        if let Some(origin) = parts.headers.get("origin")
            && !origin.to_str().is_ok_and(|origin| {
                self.origins
                    .iter()
                    .any(|allowed| same_origin(origin, allowed))
            })
        {
            return Err(Error::OriginNotAllowed);
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

async fn connect(State(state): State<Legacy>, request: Request) -> Result<Response, Error> {
    let owner = fingerprint(&request);
    let (parts, _) = request.into_parts();
    state.validate(&parts)?;
    state
        .operations
        .authorize(Context {
            parts,
            server: None,
            mcp: None,
        })
        .await?;
    let connection = start_session(state, owner).await;
    Ok(Sse::new(connection.events())
        .keep_alive(KeepAlive::default())
        .into_response())
}

struct Connection {
    id: String,
    output: mpsc::Receiver<ServerJsonRpcMessage>,
    cancellation: CancellationToken,
}

async fn start_session(state: Legacy, owner: [u8; 32]) -> Connection {
    let id = Uuid::new_v4().to_string();
    let cancellation = state.shutdown.child_token();
    let (input_sender, input_receiver) = mpsc::channel(16);
    let (output_sender, output_receiver) = mpsc::channel(16);
    state
        .sessions
        .insert(
            id.clone(),
            Session {
                sender: input_sender,
                owner,
                cancellation: cancellation.clone(),
            },
        )
        .await;
    let connection = Connection {
        id: id.clone(),
        output: output_receiver,
        cancellation: cancellation.clone(),
    };
    tokio::spawn(async move {
        let sink = Box::pin(sink::unfold(output_sender, |sender, message| async move {
            sender.send(message).await.map_err(std::io::Error::other)?;
            Ok::<_, std::io::Error>(sender)
        }));
        let input = Box::pin(stream::unfold(input_receiver, |mut receiver| async move {
            receiver.recv().await.map(|message| (message, receiver))
        }));
        if let Ok(service) = state
            .server
            .serve_with_ct((sink, input), cancellation)
            .await
        {
            let _ = service.waiting().await;
        }
        state.sessions.invalidate(&id).await;
    });
    connection
}

impl Connection {
    fn events(self) -> impl Stream<Item = Result<Event, std::io::Error>> {
        let endpoint = Event::default()
            .event("endpoint")
            .data(format!("/mcp/sse/messages?session_id={}", self.id));
        let first = stream::once(async move { Ok(endpoint) });
        let messages = stream::unfold(
            (self.output, self.cancellation.drop_guard()),
            |(mut receiver, guard)| async move {
                receiver
                    .recv()
                    .await
                    .map(|message| (message_event(message), (receiver, guard)))
            },
        );
        first.chain(messages)
    }
}

fn message_event(message: ServerJsonRpcMessage) -> Result<Event, std::io::Error> {
    Event::default()
        .event("message")
        .json_data(message)
        .map_err(std::io::Error::other)
}

#[derive(Deserialize)]
struct SessionQuery {
    session_id: String,
}

async fn message(
    State(state): State<Legacy>,
    Query(query): Query<SessionQuery>,
    request: Request,
) -> Result<Response, Error> {
    let owner = fingerprint(&request);
    let (parts, body) = request.into_parts();
    state.validate(&parts)?;
    let session = state
        .sessions
        .get(&query.session_id)
        .await
        .ok_or(Error::UnknownSession)?;
    if session.owner != owner {
        return Err(Error::ForeignSession);
    }
    let Json(mut message) =
        Json::<ClientJsonRpcMessage>::from_request(Request::from_parts(parts.clone(), body), &())
            .await?;
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
        Ok(()) => Ok((StatusCode::ACCEPTED, "Accepted").into_response()),
        Err(mpsc::error::TrySendError::Full(_)) => Err(Error::SessionBusy),
        Err(mpsc::error::TrySendError::Closed(_)) => Err(Error::UnknownSession),
    }
}
