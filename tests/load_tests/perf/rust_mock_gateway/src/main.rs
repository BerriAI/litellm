// Minimal OpenAI-compatible mock gateway in Rust.
// Returns a canned mock response for POST /v1/chat/completions.
// Purpose: measure the theoretical ceiling of a tokio+hyper+axum gateway
// on the same machine / hardware / driver as the LiteLLM benchmarks.

use axum::{
    http::StatusCode,
    response::Json,
    routing::{get, post},
    Router,
};
use serde::Deserialize;
use serde_json::{json, Value};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Deserialize)]
#[allow(dead_code)]
struct ChatRequest {
    model: String,
    messages: Vec<Value>,
}

async fn liveness() -> (StatusCode, &'static str) {
    (StatusCode::OK, "ok")
}

async fn chat_completions(Json(req): Json<ChatRequest>) -> Json<Value> {
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    Json(json!({
        "id": format!("chatcmpl-{}", uuid::Uuid::new_v4()),
        "object": "chat.completion",
        "created": created,
        "model": req.model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "hello this is a mock response"
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30
        }
    }))
}

#[tokio::main(flavor = "multi_thread")]
async fn main() {
    let port: u16 = std::env::var("PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4002);

    let app = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/health/liveness", get(liveness));

    let addr = format!("127.0.0.1:{}", port);
    let listener = tokio::net::TcpListener::bind(&addr).await.expect("bind failed");
    eprintln!("rust_mock_gateway listening on http://{}", addr);
    axum::serve(listener, app).await.expect("serve failed");
}
