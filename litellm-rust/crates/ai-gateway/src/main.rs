//! LiteLLM AI Gateway — a minimal Axum server fronting the Rust router.
//!
//! Flow: client → `POST /v1/realtime` → `router.realtime()` selects a deployment
//! (simple-shuffle) → `providers::realtime::realtime()` invokes OpenAI. The
//! server owns transport + config; routing lives in the `router` crate.

mod routes;
mod state;

use std::sync::Arc;

use litellm_router::{Deployment, LiteLLMParams, Router};

use crate::state::AppState;

#[tokio::main]
async fn main() {
    let state = AppState {
        router: Arc::new(build_router_from_env()),
    };

    let port: u16 = std::env::var("PORT")
        .ok()
        .and_then(|raw| raw.parse().ok())
        .unwrap_or(4001);

    let listener = tokio::net::TcpListener::bind(("0.0.0.0", port))
        .await
        .expect("failed to bind listener");
    eprintln!("litellm-ai-gateway listening on 0.0.0.0:{port}");
    axum::serve(listener, routes::app(state))
        .await
        .expect("server error");
}

/// Build a minimal single-deployment `model_list` from the environment.
///
/// A real deployment loads `model_list` from config; this is the minimal stand-in
/// so the gateway has one OpenAI deployment to route to.
fn build_router_from_env() -> Router {
    let model =
        std::env::var("OPENAI_REALTIME_MODEL").unwrap_or_else(|_| "gpt-realtime".to_string());
    let deployment = Deployment {
        model_name: model.clone(),
        litellm_params: LiteLLMParams {
            model,
            api_key: std::env::var("OPENAI_API_KEY").ok(),
            api_base: None,
        },
    };
    Router::new(vec![deployment])
}
