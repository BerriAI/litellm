#![forbid(unsafe_code)]

use axum::{routing::post, Json, Router};
use serde_json::{json, Value};

async fn chat_completions() -> Json<Value> {
    Json(json!({
        "id": "chatcmpl-gwbench",
        "object": "chat.completion",
        "created": 0,
        "model": "mock-upstream",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "stub response"},
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 1,
            "completion_tokens": 2,
            "total_tokens": 3
        }
    }))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let port = std::env::var("GWBENCH_MOCK_PORT")
        .ok()
        .and_then(|value| value.parse::<u16>().ok())
        .unwrap_or(8080);
    let app = Router::new().route("/v1/chat/completions", post(chat_completions));
    let listener = tokio::net::TcpListener::bind(("0.0.0.0", port)).await?;
    println!("mock upstream listening on {listener:?}");
    // Streaming and configurable response timing are intentionally deferred.
    axum::serve(listener, app).await?;
    Ok(())
}
