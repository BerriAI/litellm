mod streaming;

use std::{
    convert::Infallible,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};

use axum::{
    Router,
    body::{Body, to_bytes},
    extract::{Request, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::any,
};
use bytes::Bytes;
use clap::{Parser, Subcommand};
use futures_util::TryStreamExt;
use serde_json::json;
use sha2::{Digest, Sha256};

#[derive(Parser)]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    Serve {
        #[arg(long)]
        port: u16,
        #[arg(long, default_value = "backend")]
        role: String,
        #[arg(long, default_value = "http://127.0.0.1:19400")]
        backend: String,
        #[arg(long, default_value = "http://127.0.0.1:19402")]
        python: String,
    },
}

#[derive(Default)]
struct Counters {
    requests: AtomicU64,
    upload_bytes: AtomicU64,
    active: AtomicU64,
    completed: AtomicU64,
    cancelled: AtomicU64,
}

struct StreamGuard {
    counters: Arc<Counters>,
    completed: bool,
}

impl Drop for StreamGuard {
    fn drop(&mut self) {
        self.counters.active.fetch_sub(1, Ordering::Relaxed);
        if self.completed {
            self.counters.completed.fetch_add(1, Ordering::Relaxed);
        } else {
            self.counters.cancelled.fetch_add(1, Ordering::Relaxed);
        }
    }
}

#[derive(Clone)]
struct App {
    role: String,
    backend: String,
    python: String,
    raw_client: streaming::Client,
    counters: Arc<Counters>,
}

fn clean_headers(headers: &HeaderMap) -> HeaderMap {
    let connection_tokens = headers
        .get("connection")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    headers
        .iter()
        .filter(|(name, _)| {
            !matches!(
                name.as_str(),
                "host"
                    | "connection"
                    | "keep-alive"
                    | "proxy-authenticate"
                    | "proxy-authorization"
                    | "te"
                    | "trailer"
                    | "transfer-encoding"
                    | "upgrade"
            ) && !connection_tokens
                .split(',')
                .any(|token| token.trim().eq_ignore_ascii_case(name.as_str()))
        })
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect()
}

fn header_number(headers: &HeaderMap, key: &str, fallback: u64) -> u64 {
    headers
        .get(key)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse().ok())
        .unwrap_or(fallback)
}

async fn backend(app: App, headers: HeaderMap, data: Bytes) -> Response {
    app.counters.requests.fetch_add(1, Ordering::Relaxed);
    let value = match serde_json::from_slice::<serde_json::Value>(&data) {
        Ok(value) => value,
        Err(_) => return StatusCode::BAD_REQUEST.into_response(),
    };
    let status = StatusCode::from_u16(header_number(&headers, "x-bench-status", 200) as u16)
        .unwrap_or(StatusCode::BAD_REQUEST);
    let digest = format!("{:x}", Sha256::digest(&data));
    if let Some(expected) = headers
        .get("x-bench-expected-sha256")
        .and_then(|value| value.to_str().ok())
        && expected != digest
    {
        return StatusCode::BAD_REQUEST.into_response();
    }
    let common = [
        ("x-body-sha256", digest),
        (
            "x-bench-id",
            headers
                .get("x-bench-id")
                .and_then(|v| v.to_str().ok())
                .unwrap_or("")
                .to_owned(),
        ),
    ];
    if status != StatusCode::OK {
        return (status, common, "injected failure").into_response();
    }
    let message = json!({"id":"msg_fixture", "type":"message", "role":"assistant", "model":"fixture-model", "content":[{"type":"text","text":"x".repeat(1024)}], "stop_reason":"end_turn", "stop_sequence":null, "usage":{"input_tokens":10,"output_tokens":8}});
    if value.get("stream").and_then(|v| v.as_bool()) != Some(true) {
        return (common, axum::Json(message)).into_response();
    }
    let delay = header_number(&headers, "x-bench-delay-ms", 5);
    let count = header_number(&headers, "x-bench-events", 8);
    app.counters.active.fetch_add(1, Ordering::Relaxed);
    let guard = StreamGuard {
        counters: app.counters.clone(),
        completed: false,
    };
    let stream = async_stream::stream! {
        let mut guard = guard;
        let start = json!({"type":"message_start","message":{"id":"msg_fixture","type":"message","role":"assistant","model":"fixture-model","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":0}}});
        yield Ok::<Bytes, Infallible>(Bytes::from(format!("event: message_start\ndata: {start}\n\n")));
        yield Ok(Bytes::from_static(b"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n"));
        for _ in 0..count {
            tokio::time::sleep(Duration::from_millis(delay)).await;
            let event = json!({"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"x".repeat(128)}});
            yield Ok(Bytes::from(format!("event: content_block_delta\ndata: {event}\n\n")));
        }
        yield Ok(Bytes::from_static(b"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\nevent: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":8}}\n\nevent: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"));
        guard.completed = true;
    };
    let mut response = (common, Body::from_stream(stream)).into_response();
    response
        .headers_mut()
        .insert("content-type", "text/event-stream".parse().unwrap());
    response
}

async fn handle(State(app): State<App>, request: Request) -> Response {
    if request.uri().path() == "/metrics" {
        return axum::Json(json!({
            "requests": app.counters.requests.load(Ordering::Relaxed),
            "upload_bytes": app.counters.upload_bytes.load(Ordering::Relaxed),
            "active": app.counters.active.load(Ordering::Relaxed),
            "completed": app.counters.completed.load(Ordering::Relaxed),
            "cancelled": app.counters.cancelled.load(Ordering::Relaxed),
        }))
        .into_response();
    }
    if app.role == "stream" {
        return streaming::forward(&app, request).await;
    }
    let (parts, body) = request.into_parts();
    let body = if app.role == "backend" && parts.headers.contains_key("x-bench-upload-probe") {
        let counters = app.counters.clone();
        Body::from_stream(body.into_data_stream().inspect_ok(move |chunk| {
            counters
                .upload_bytes
                .fetch_add(chunk.len() as u64, Ordering::Relaxed);
        }))
    } else {
        body
    };
    let data = match to_bytes(body, 8 * 1024 * 1024).await {
        Ok(data) => data,
        Err(_) => return StatusCode::PAYLOAD_TOO_LARGE.into_response(),
    };
    if app.role == "backend" {
        return backend(app, parts.headers, data).await;
    }
    StatusCode::NOT_FOUND.into_response()
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    match Args::parse().command {
        Command::Serve {
            port,
            role,
            backend,
            python,
        } => {
            let app = App {
                role,
                backend,
                python,
                raw_client: streaming::client(),
                counters: Arc::new(Counters::default()),
            };
            let listener = tokio::net::TcpListener::bind(("127.0.0.1", port)).await?;
            axum::serve(
                listener,
                Router::new().fallback(any(handle)).with_state(app),
            )
            .await?;
        }
    }
    Ok(())
}
