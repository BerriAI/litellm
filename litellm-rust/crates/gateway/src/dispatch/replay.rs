use std::net::SocketAddr;

use axum::{
    body::{Body, Bytes},
    extract::{ConnectInfo, OriginalUri, Request, State},
    http::request::Parts,
    middleware::Next,
    response::Response,
};
use bytes::BytesMut;
use futures_util::StreamExt;
use litellm_core_utils::decline::Decline;

use super::Forward;

#[derive(Clone)]
pub(super) struct Replay {
    pub forward: Forward,
    pub record_limit: usize,
}

/// Records the request before any handler reads it, and forwards the recording when the
/// response says the request was declined.
pub(super) async fn replay(
    State(Replay {
        forward,
        record_limit,
    }): State<Replay>,
    request: Request,
    next: Next,
) -> Response {
    let (parts, body) = request.into_parts();
    let body = match buffer(body, record_limit).await {
        Ok(body) => body,
        Err(body) => return next.run(Request::from_parts(parts, body)).await,
    };
    let recorded = record(&parts, body.clone());
    let response = next.run(Request::from_parts(parts, Body::from(body))).await;
    let Some(&decline) = response.extensions().get::<Decline>() else {
        return response;
    };
    forward.send(recorded.map(Body::from), decline).await
}

/// The whole body, or, when it is too large or unreadable, a body that yields the same bytes
/// and the same failure, so the handler reports it in its own error envelope.
async fn buffer(body: Body, limit: usize) -> Result<Bytes, Body> {
    let mut stream = body.into_data_stream();
    let mut seen = BytesMut::new();
    while let Some(chunk) = stream.next().await {
        match chunk {
            Ok(chunk) if seen.len() + chunk.len() <= limit => {
                seen.extend_from_slice(&chunk);
            }
            failed => {
                let replayed = futures_util::stream::iter([Ok(seen.freeze()), failed]);
                return Err(Body::from_stream(replayed.chain(stream)));
            }
        }
    }
    Ok(seen.freeze())
}

fn record(parts: &Parts, body: Bytes) -> Request<Bytes> {
    let uri = parts
        .extensions
        .get::<OriginalUri>()
        .map_or_else(|| parts.uri.clone(), |original| original.0.clone());
    let mut recorded = Request::new(body);
    *recorded.method_mut() = parts.method.clone();
    *recorded.uri_mut() = uri;
    *recorded.version_mut() = parts.version;
    *recorded.headers_mut() = parts.headers.clone();
    if let Some(&client) = parts.extensions.get::<ConnectInfo<SocketAddr>>() {
        recorded.extensions_mut().insert(client);
    }
    recorded
}
