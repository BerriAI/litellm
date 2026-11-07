use std::{io, time::Duration};

use axum::{
    body::Body,
    extract::Request,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use http_body_util::{BodyExt, Limited, StreamBody};
use hyper_util::{
    client::legacy::{Client as HyperClient, connect::HttpConnector},
    rt::TokioExecutor,
};

use crate::{App, clean_headers, header_number};

pub type Client = HyperClient<HttpConnector, Body>;

pub fn client() -> Client {
    let mut connector = HttpConnector::new();
    connector.set_nodelay(true);
    connector.set_connect_timeout(Some(Duration::from_secs(1)));
    HyperClient::builder(TokioExecutor::new())
        .pool_max_idle_per_host(500)
        .pool_idle_timeout(Duration::from_secs(15))
        .retry_canceled_requests(false)
        .build(connector)
}

pub async fn forward(app: &App, request: Request) -> Response {
    if header_number(request.headers(), "content-length", 0) > 8 * 1024 * 1024 {
        return StatusCode::PAYLOAD_TOO_LARGE.into_response();
    }
    let target = if request.uri().path() != "/v1/messages"
        || request.headers().contains_key("x-bench-python")
    {
        &app.python
    } else {
        &app.backend
    };
    let uri = match format!("{target}{}", request.uri()).parse() {
        Ok(uri) => uri,
        Err(_) => return StatusCode::BAD_GATEWAY.into_response(),
    };
    let (mut parts, body) = request.into_parts();
    parts.uri = uri;
    parts.headers = clean_headers(&parts.headers);
    let outbound = Request::from_parts(parts, Body::new(Limited::new(body, 8 * 1024 * 1024)));
    let deadline = tokio::time::Instant::now() + Duration::from_secs(30);
    let result = tokio::time::timeout_at(deadline, app.raw_client.request(outbound)).await;
    let upstream = match result {
        Ok(Ok(response)) => response,
        _ => return StatusCode::BAD_GATEWAY.into_response(),
    };
    let (mut parts, mut body) = upstream.into_parts();
    parts.headers = clean_headers(&parts.headers);
    let frames = async_stream::stream! {
        loop {
            match tokio::time::timeout_at(deadline, body.frame()).await {
                Ok(Some(Ok(frame))) => yield Ok::<_, io::Error>(frame),
                Ok(Some(Err(error))) => { yield Err(io::Error::other(error)); break; }
                Ok(None) => break,
                Err(error) => { yield Err(io::Error::new(io::ErrorKind::TimedOut, error)); break; }
            }
        }
    };
    Response::from_parts(parts, Body::new(StreamBody::new(frames)))
}
