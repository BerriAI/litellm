use std::{sync::Arc, time::Duration};

use axum::{
    Router,
    body::{Body, to_bytes},
    extract::{Request, State},
    http::{Method, StatusCode},
    response::{IntoResponse, Response},
    routing::any,
};
use moka::future::Cache;
use rmcp::{
    model::{ClientJsonRpcMessage, ClientRequest},
    transport::streamable_http_server::{
        StreamableHttpServerConfig, StreamableHttpService,
        session::{SessionManager, local::LocalSessionManager},
    },
};
use sha2::{Digest, Sha256};
use tower::ServiceExt;

use crate::{McpServer, server::ServerScope};

#[derive(Clone)]
pub struct SessionOwner(pub String);

#[derive(Clone)]
struct Transport {
    stateful: Router,
    stateless: Router,
    owners: Cache<String, [u8; 32]>,
}

pub(crate) fn transport(server: McpServer, config: StreamableHttpServerConfig) -> Router {
    let manager = Arc::new(LocalSessionManager::default());
    let cleanup = manager.clone();
    let owners = Cache::builder()
        .max_capacity(10_000)
        .time_to_idle(Duration::from_secs(300))
        .async_eviction_listener(move |id: Arc<String>, _, _| {
            let manager = cleanup.clone();
            Box::pin(async move {
                let _ = manager.close_session(&id.as_str().into()).await;
            })
        })
        .build();
    let stateful_server = server.clone();
    let stateful = StreamableHttpService::new(
        move || Ok(stateful_server.clone()),
        manager,
        config.clone().with_legacy_session_mode(true),
    );
    let stateless = StreamableHttpService::new(
        move || Ok(server.clone()),
        Arc::new(LocalSessionManager::default()),
        config.with_legacy_session_mode(false),
    );
    Router::new().fallback(any(dispatch)).with_state(Transport {
        stateful: Router::new().fallback_service(stateful),
        stateless: Router::new().fallback_service(stateless),
        owners,
    })
}

async fn dispatch(State(transport): State<Transport>, request: Request) -> Response {
    let owner = fingerprint(&request);
    let session = request
        .headers()
        .get("mcp-session-id")
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned);
    if let Some(session) = &session {
        match transport.owners.get(session).await {
            Some(expected) if expected == owner => (),
            Some(_) => return StatusCode::FORBIDDEN.into_response(),
            None => return StatusCode::NOT_FOUND.into_response(),
        }
    }
    let method = request.method().clone();
    let (request, initialize) = if method == Method::POST && session.is_none() {
        let (parts, body) = request.into_parts();
        let bytes = match to_bytes(body, 4 * 1024 * 1024).await {
            Ok(bytes) => bytes,
            Err(_) => return StatusCode::PAYLOAD_TOO_LARGE.into_response(),
        };
        let initialize = matches!(serde_json::from_slice::<ClientJsonRpcMessage>(&bytes),
            Ok(ClientJsonRpcMessage::Request(request)) if matches!(request.request, ClientRequest::InitializeRequest(_)));
        (Request::from_parts(parts, Body::from(bytes)), initialize)
    } else {
        (request, false)
    };
    let service = if initialize || session.is_some() {
        transport.stateful
    } else {
        transport.stateless
    };
    let response = match service.oneshot(request).await {
        Ok(response) => response,
        Err(never) => match never {},
    };
    if initialize
        && response.status().is_success()
        && let Some(id) = response
            .headers()
            .get("mcp-session-id")
            .and_then(|value| value.to_str().ok())
    {
        transport.owners.insert(id.to_owned(), owner).await;
    }
    if method == Method::DELETE
        && response.status().is_success()
        && let Some(session) = session
    {
        transport.owners.invalidate(&session).await;
    }
    response
}

pub(crate) fn fingerprint(request: &Request) -> [u8; 32] {
    let scope = request
        .extensions()
        .get::<ServerScope>()
        .map(|scope| scope.0.as_str())
        .unwrap_or_default();
    let mut digest = Sha256::new();
    match request.extensions().get::<SessionOwner>() {
        Some(owner) => {
            digest.update(b"principal");
            digest.update(owner.0.len().to_be_bytes());
            digest.update(owner.0.as_bytes());
        }
        None => {
            digest.update(b"credentials");
            for name in ["authorization", "x-litellm-api-key"] {
                let value = request
                    .headers()
                    .get(name)
                    .map(|value| value.as_bytes())
                    .unwrap_or_default();
                digest.update(value.len().to_be_bytes());
                digest.update(value);
            }
        }
    }
    digest.update(scope.len().to_be_bytes());
    digest.update(scope.as_bytes());
    digest.update(
        request
            .headers()
            .get("x-mcp-servers")
            .map(|value| value.as_bytes())
            .unwrap_or_default(),
    );
    digest.finalize().into()
}
