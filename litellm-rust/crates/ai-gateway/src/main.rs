//! LiteLLM AI Gateway — a minimal Axum server fronting the Rust router.
//!
//! Flow: client → `POST /v1/realtime` → `router.realtime()` selects a deployment
//! (simple-shuffle) → `providers::realtime::realtime()` invokes OpenAI. The
//! server owns transport + config; routing lives in the `router` crate.

mod auth;
mod constants;
mod gil;
mod integrations;
#[cfg(feature = "python-config")]
mod python;
mod realtime;
mod routes;
mod state;

use std::sync::Arc;

use litellm_core::router::{Deployment, LiteLLMParams, Router};

use crate::integrations::custom_logger::CustomLogger;
use crate::integrations::litellm_python_proxy_api::LiteLLMPythonProxyAPILogger;
use crate::state::AppState;

/// Bind to localhost by default so the gateway is not a public, unauthenticated
/// provider proxy out of the box. Override with `HOST` (e.g. `0.0.0.0`).
const DEFAULT_HOST: &str = "127.0.0.1";
const DEFAULT_PORT: u16 = 4001;

#[tokio::main]
async fn main() {
    // Trim before storing so it matches the trimmed bearer token in `auth`
    // (avoids a silent auth failure when the env var has surrounding whitespace).
    let master_key: Option<Arc<str>> = std::env::var("LITELLM_MASTER_KEY")
        .ok()
        .map(|key| key.trim().to_string())
        .filter(|key| !key.is_empty())
        .map(Arc::from);
    if master_key.is_none() {
        eprintln!(
            "warning: LITELLM_MASTER_KEY is not set; /v1/realtime will reject all requests (fail closed)"
        );
    }

    // Spawn the realtime-logging worker (drains a channel → POSTs batches to the
    // Python proxy's /v1/callbacks/logs). Built here so the spawn lands on the
    // tokio runtime. `from_env` reads LITELLM_PROXY_BASE_URL + LITELLM_MASTER_KEY.
    let proxy_logger = LiteLLMPythonProxyAPILogger::from_env();
    let loggers: Vec<Arc<dyn CustomLogger>> = vec![proxy_logger];

    let state = AppState {
        router: Arc::new(build_router()),
        master_key,
        loggers: Arc::new(loggers),
    };

    let host = std::env::var("HOST").unwrap_or_else(|_| DEFAULT_HOST.to_string());
    let port = resolve_port();

    let listener = tokio::net::TcpListener::bind((host.as_str(), port))
        .await
        .expect("failed to bind listener");
    eprintln!("litellm-ai-gateway listening on {host}:{port}");
    axum::serve(listener, routes::app(state))
        .await
        .expect("server error");
}

/// Resolve `PORT`, warning (rather than silently defaulting) on an invalid value.
fn resolve_port() -> u16 {
    match std::env::var("PORT") {
        Ok(raw) => raw.parse().unwrap_or_else(|_| {
            eprintln!("warning: PORT={raw:?} is not a valid port; using {DEFAULT_PORT}");
            DEFAULT_PORT
        }),
        Err(_) => DEFAULT_PORT,
    }
}

/// Build the router. With the `python-config` feature and `LITELLM_CONFIG_PATH`
/// set, load the resolved `model_list` from the proxy config via the embedded
/// Python reader (load time only). Otherwise fall back to the env stand-in.
fn build_router() -> Router {
    #[cfg(feature = "python-config")]
    if let Ok(config_path) = std::env::var("LITELLM_CONFIG_PATH") {
        match python::config::load_router_from_config(&config_path) {
            Ok(router) => {
                eprintln!("loaded model_list from {config_path} via python config reader");
                return router;
            }
            Err(err) => {
                eprintln!("config load failed ({err}); falling back to env deployment");
            }
        }
    }
    build_router_from_env()
}

/// Build a minimal single-deployment `model_list` from the environment.
///
/// A real deployment loads `model_list` from config; this is the minimal stand-in
/// so the gateway has one OpenAI deployment to route to.
fn build_router_from_env() -> Router {
    let model =
        std::env::var("OPENAI_REALTIME_MODEL").unwrap_or_else(|_| "gpt-realtime".to_string());
    let api_key = std::env::var("OPENAI_API_KEY").ok();
    if api_key.is_none() {
        eprintln!(
            "warning: OPENAI_API_KEY is not set; realtime requests will fail with auth errors"
        );
    }
    let deployment = Deployment {
        model_name: model.clone(),
        litellm_params: LiteLLMParams {
            model,
            api_key,
            api_base: None,
        },
    };
    Router::new(vec![deployment])
}
