//! Thin core for the plugin-arch POC.
//!
//! The core knows nothing about OCR, LLM payloads, or any specific provider.
//! It only:
//!   1. Reads a routing config that maps `model` -> plugin HTTP address.
//!   2. On every request, peeks at the `model` field in the JSON body, looks
//!      up the plugin address, and forwards the raw body bytes to the
//!      plugin's `/handle` endpoint.
//!   3. Returns the plugin's response bytes and status verbatim.
//!
//! Anything provider-specific lives behind the plugin boundary.

use std::{collections::HashMap, net::SocketAddr, path::PathBuf, sync::Arc};

use axum::{
    body::Bytes,
    extract::State,
    http::{HeaderMap, HeaderName, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use clap::Parser;
use serde::{Deserialize, Serialize};

#[derive(Parser, Debug)]
#[command(name = "plugin_arch_core")]
struct Cli {
    #[arg(long, default_value = "routes.toml")]
    config: PathBuf,
    #[arg(long, default_value = "127.0.0.1:8080")]
    bind: SocketAddr,
}

#[derive(Debug, Deserialize)]
struct RawConfig {
    plugin: Vec<RawPlugin>,
}

#[derive(Debug, Deserialize)]
struct RawPlugin {
    name: String,
    address: String,
    models: Vec<String>,
}

#[derive(Debug, Clone)]
struct PluginEntry {
    name: String,
    address: String,
}

#[derive(Debug)]
struct AppState {
    by_model: HashMap<String, PluginEntry>,
    plugins: Vec<PluginEntry>,
    http: reqwest::Client,
}

#[derive(Debug, Serialize)]
struct ErrorBody<'a> {
    error: ErrorInner<'a>,
}

#[derive(Debug, Serialize)]
struct ErrorInner<'a> {
    code: &'a str,
    message: String,
    r#type: &'a str,
}

fn error_response(status: StatusCode, code: &str, kind: &str, message: impl Into<String>) -> Response {
    let body = ErrorBody {
        error: ErrorInner {
            code,
            message: message.into(),
            r#type: kind,
        },
    };
    (status, Json(body)).into_response()
}

fn load_config(path: &PathBuf) -> Result<AppState, String> {
    let raw = std::fs::read_to_string(path).map_err(|e| format!("read {:?}: {}", path, e))?;
    let cfg: RawConfig = toml::from_str(&raw).map_err(|e| format!("parse {:?}: {}", path, e))?;

    let plugins: Vec<PluginEntry> = cfg
        .plugin
        .iter()
        .map(|p| PluginEntry {
            name: p.name.clone(),
            address: p.address.trim_end_matches('/').to_string(),
        })
        .collect();

    let by_model: HashMap<String, PluginEntry> = cfg
        .plugin
        .iter()
        .flat_map(|p| {
            let entry = PluginEntry {
                name: p.name.clone(),
                address: p.address.trim_end_matches('/').to_string(),
            };
            p.models.iter().map(move |m| (m.clone(), entry.clone()))
        })
        .collect();

    let http = reqwest::Client::builder()
        .build()
        .map_err(|e| format!("reqwest build: {e}"))?;

    Ok(AppState { by_model, plugins, http })
}

fn extract_model(body: &[u8]) -> Option<String> {
    let v: serde_json::Value = serde_json::from_slice(body).ok()?;
    v.get("model")?.as_str().map(|s| s.to_string())
}

async fn handle(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let Some(model) = extract_model(&body) else {
        return error_response(
            StatusCode::BAD_REQUEST,
            "missing_model",
            "invalid_request",
            "request body must be JSON with a top-level `model` string",
        );
    };

    let Some(plugin) = state.by_model.get(&model).cloned() else {
        return error_response(
            StatusCode::NOT_FOUND,
            "no_plugin_for_model",
            "routing_error",
            format!("no plugin is registered for model `{model}`"),
        );
    };

    let url = format!("{}/handle", plugin.address);

    let forward_ct = headers
        .get(http::header::CONTENT_TYPE)
        .cloned()
        .unwrap_or_else(|| HeaderValue::from_static("application/json"));

    let upstream = state
        .http
        .post(&url)
        .header(http::header::CONTENT_TYPE, forward_ct)
        .body(body)
        .send()
        .await;

    let resp = match upstream {
        Ok(r) => r,
        Err(e) => {
            return error_response(
                StatusCode::BAD_GATEWAY,
                "plugin_unreachable",
                "plugin_transport_error",
                format!("could not reach plugin `{}` at {}: {}", plugin.name, url, e),
            );
        }
    };

    let status = resp.status();
    let resp_headers = resp.headers().clone();
    let body_bytes = match resp.bytes().await {
        Ok(b) => b,
        Err(e) => {
            return error_response(
                StatusCode::BAD_GATEWAY,
                "plugin_body_error",
                "plugin_transport_error",
                format!("error reading plugin body: {e}"),
            );
        }
    };

    let mut out = Response::builder().status(status);
    if let Some(h) = out.headers_mut() {
        if let Some(ct) = resp_headers.get(http::header::CONTENT_TYPE) {
            h.insert(http::header::CONTENT_TYPE, ct.clone());
        }
        if let Ok(name) = HeaderName::try_from("x-plugin") {
            if let Ok(v) = HeaderValue::try_from(plugin.name.as_str()) {
                h.insert(name, v);
            }
        }
    }
    out.body(axum::body::Body::from(body_bytes))
        .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response())
}

#[derive(Debug, Serialize, Deserialize, Default)]
struct Capabilities {
    #[serde(default)]
    models: Vec<String>,
    #[serde(default)]
    endpoints: Vec<String>,
}

#[derive(Debug, Serialize)]
struct AggregatedCapabilities {
    plugins: Vec<PluginCapabilitiesEntry>,
}

#[derive(Debug, Serialize)]
struct PluginCapabilitiesEntry {
    name: String,
    capabilities: Capabilities,
    reachable: bool,
}

async fn capabilities(State(state): State<Arc<AppState>>) -> Response {
    let mut entries: Vec<PluginCapabilitiesEntry> = Vec::with_capacity(state.plugins.len());
    for p in &state.plugins {
        let url = format!("{}/capabilities", p.address);
        match state.http.get(&url).send().await {
            Ok(r) if r.status().is_success() => match r.json::<Capabilities>().await {
                Ok(c) => entries.push(PluginCapabilitiesEntry {
                    name: p.name.clone(),
                    capabilities: c,
                    reachable: true,
                }),
                Err(_) => entries.push(PluginCapabilitiesEntry {
                    name: p.name.clone(),
                    capabilities: Capabilities::default(),
                    reachable: false,
                }),
            },
            _ => entries.push(PluginCapabilitiesEntry {
                name: p.name.clone(),
                capabilities: Capabilities::default(),
                reachable: false,
            }),
        }
    }
    Json(AggregatedCapabilities { plugins: entries }).into_response()
}

async fn health() -> &'static str {
    "ok"
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let state = Arc::new(load_config(&cli.config)?);
    tracing::info!(
        "loaded {} plugin(s), {} model route(s)",
        state.plugins.len(),
        state.by_model.len()
    );

    let app = Router::new()
        .route("/v1/handle", post(handle))
        .route("/v1/capabilities", get(capabilities))
        .route("/healthz", get(health))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(cli.bind).await?;
    tracing::info!("core listening on http://{}", cli.bind);
    axum::serve(listener, app).await?;
    Ok(())
}
