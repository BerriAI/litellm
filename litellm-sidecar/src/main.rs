use bytes::Bytes;
use dashmap::DashMap;
use futures_util::StreamExt;
use http_body_util::{combinators::BoxBody, BodyExt, Full, StreamBody};
use hyper::body::Frame;
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper::{Request, Response, StatusCode};
use hyper_util::rt::TokioIo;
use reqwest::Client;
use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tokio::net::TcpListener;

struct Sidecar {
    pools: DashMap<String, Client>,
    total_requests: AtomicU64,
    total_errors: AtomicU64,
    total_latency_us: AtomicU64,
}

impl Sidecar {
    fn new() -> Self {
        Self {
            pools: DashMap::new(),
            total_requests: AtomicU64::new(0),
            total_errors: AtomicU64::new(0),
            total_latency_us: AtomicU64::new(0),
        }
    }

    fn get_or_create_client(&self, host: &str) -> Client {
        self.pools
            .entry(host.to_string())
            .or_insert_with(|| {
                Client::builder()
                    .pool_max_idle_per_host(200)
                    .pool_idle_timeout(std::time::Duration::from_secs(90))
                    .tcp_keepalive(std::time::Duration::from_secs(60))
                    .tcp_nodelay(true)
                    .build()
                    .expect("Failed to build reqwest client")
            })
            .clone()
    }
}

async fn handle_request(
    req: Request<hyper::body::Incoming>,
    sidecar: Arc<Sidecar>,
) -> Result<Response<BoxBody<Bytes, Infallible>>, Infallible> {
    let start = Instant::now();
    sidecar.total_requests.fetch_add(1, Ordering::Relaxed);

    let path = req.uri().path();
    if path == "/health" {
        let total = sidecar.total_latency_us.load(Ordering::Relaxed);
        let count = sidecar.total_requests.load(Ordering::Relaxed);
        let avg = if count > 0 { total / count } else { 0 };
        let stats = format!(
            r#"{{"status":"ok","requests":{},"errors":{},"avg_latency_us":{}}}"#,
            count,
            sidecar.total_errors.load(Ordering::Relaxed),
            avg,
        );
        return Ok(Response::builder()
            .status(200)
            .header("content-type", "application/json")
            .body(Full::new(Bytes::from(stats)).map_err(|e| match e {}).boxed())
            .unwrap());
    }

    // Extract routing metadata from X-LiteLLM-* headers
    let provider_url = match req.headers().get("x-litellm-provider-url") {
        Some(v) => v.to_str().unwrap_or("").to_string(),
        None => {
            sidecar.total_errors.fetch_add(1, Ordering::Relaxed);
            return Ok(Response::builder()
                .status(400)
                .body(
                    Full::new(Bytes::from(r#"{"error":"Missing X-LiteLLM-Provider-URL"}"#))
                        .map_err(|e| match e {})
                        .boxed(),
                )
                .unwrap());
        }
    };

    let request_path = req
        .headers()
        .get("x-litellm-path")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("/chat/completions")
        .to_string();

    let api_key = req
        .headers()
        .get("x-litellm-api-key")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();

    let timeout_secs: u64 = req
        .headers()
        .get("x-litellm-timeout")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse().ok())
        .unwrap_or(300);

    let is_stream = req
        .headers()
        .get("x-litellm-stream")
        .and_then(|v| v.to_str().ok())
        .map(|v| v == "true")
        .unwrap_or(false);

    let method_str = req
        .headers()
        .get("x-litellm-method")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("POST")
        .to_uppercase();

    let content_type = req
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("application/json")
        .to_string();

    // Collect forwarded headers (X-LiteLLM-Fwd-*)
    let mut fwd_headers: Vec<(String, String)> = Vec::new();
    for (name, value) in req.headers() {
        let name_str = name.as_str();
        if let Some(orig_name) = name_str.strip_prefix("x-litellm-fwd-") {
            if let Ok(val) = value.to_str() {
                fwd_headers.push((orig_name.to_string(), val.to_string()));
            }
        }
    }

    let body_bytes = match req.collect().await {
        Ok(collected) => collected.to_bytes(),
        Err(_) => {
            sidecar.total_errors.fetch_add(1, Ordering::Relaxed);
            return Ok(Response::builder()
                .status(400)
                .body(
                    Full::new(Bytes::from(r#"{"error":"Failed to read request body"}"#))
                        .map_err(|e| match e {})
                        .boxed(),
                )
                .unwrap());
        }
    };

    let host = url::Url::parse(&provider_url)
        .map(|u| u.host_str().unwrap_or("unknown").to_string())
        .unwrap_or_else(|_| "unknown".to_string());

    let client = sidecar.get_or_create_client(&host);
    let full_url = format!("{}{}", provider_url.trim_end_matches('/'), request_path);

    let mut req_builder = match method_str.as_str() {
        "GET" => client.get(&full_url),
        "PUT" => client.put(&full_url),
        "PATCH" => client.patch(&full_url),
        "DELETE" => client.delete(&full_url),
        _ => client.post(&full_url),
    };

    req_builder = req_builder
        .header("content-type", &content_type)
        .timeout(std::time::Duration::from_secs(timeout_secs));

    if method_str != "GET" {
        req_builder = req_builder.body(body_bytes.to_vec());
    }

    if !api_key.is_empty() {
        req_builder = req_builder.header("authorization", format!("Bearer {}", api_key));
    }

    for (name, value) in &fwd_headers {
        req_builder = req_builder.header(name.as_str(), value.as_str());
    }

    let provider_response = match req_builder.send().await {
        Ok(resp) => resp,
        Err(e) => {
            sidecar.total_errors.fetch_add(1, Ordering::Relaxed);
            let elapsed = start.elapsed().as_micros() as u64;
            sidecar.total_latency_us.fetch_add(elapsed, Ordering::Relaxed);
            let error_body = format!(r#"{{"error":"Provider request failed: {}"}}"#, e);
            return Ok(Response::builder()
                .status(502)
                .header("content-type", "application/json")
                .body(Full::new(Bytes::from(error_body)).map_err(|e| match e {}).boxed())
                .unwrap());
        }
    };

    let status = provider_response.status();
    let status_code = StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);

    // Forward response headers from provider
    let mut resp_builder = Response::builder().status(status_code);
    for (name, value) in provider_response.headers() {
        resp_builder = resp_builder.header(name.as_str(), value.as_bytes());
    }

    if is_stream {
        let stream = provider_response.bytes_stream().map(|result| {
            match result {
                Ok(bytes) => Ok::<_, Infallible>(Frame::data(bytes)),
                Err(e) => {
                    tracing::warn!("Stream chunk error: {}", e);
                    Ok(Frame::data(Bytes::new()))
                }
            }
        });
        let elapsed = start.elapsed().as_micros() as u64;
        sidecar.total_latency_us.fetch_add(elapsed, Ordering::Relaxed);
        Ok(resp_builder
            .body(BodyExt::boxed(StreamBody::new(stream)))
            .unwrap())
    } else {
        let resp_bytes = match provider_response.bytes().await {
            Ok(b) => b,
            Err(e) => {
                sidecar.total_errors.fetch_add(1, Ordering::Relaxed);
                let elapsed = start.elapsed().as_micros() as u64;
                sidecar.total_latency_us.fetch_add(elapsed, Ordering::Relaxed);
                let error_body = format!(r#"{{"error":"Failed to read response: {}"}}"#, e);
                return Ok(Response::builder()
                    .status(502)
                    .body(Full::new(Bytes::from(error_body)).map_err(|e| match e {}).boxed())
                    .unwrap());
            }
        };
        let elapsed = start.elapsed().as_micros() as u64;
        sidecar.total_latency_us.fetch_add(elapsed, Ordering::Relaxed);
        Ok(resp_builder
            .body(Full::new(resp_bytes).map_err(|e| match e {}).boxed())
            .unwrap())
    }
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "litellm_sidecar=info".parse().unwrap()),
        )
        .init();

    let port: u16 = std::env::var("SIDECAR_PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(8787);

    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    let sidecar = Arc::new(Sidecar::new());

    tracing::info!("LiteLLM Sidecar starting on {}", addr);

    let listener = TcpListener::bind(addr).await.expect("Failed to bind");

    loop {
        let (stream, _) = listener.accept().await.expect("Failed to accept");
        let io = TokioIo::new(stream);
        let sidecar = sidecar.clone();

        tokio::task::spawn(async move {
            let service = service_fn(move |req| {
                let sidecar = sidecar.clone();
                handle_request(req, sidecar)
            });
            if let Err(err) = http1::Builder::new().serve_connection(io, service).await {
                if !err.is_incomplete_message() {
                    tracing::error!("Connection error: {}", err);
                }
            }
        });
    }
}
