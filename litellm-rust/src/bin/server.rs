//! Standalone axum HTTP server: `litellm-rust-server`.
//!
//! - `GET  /health` -> `{"status":"ok"}`
//! - `POST /v1/ocr`  -> body `{model, document, api_key?, api_base?, ...params}`
//!
//! The async OCR path reuses the SAME pure transforms as the Python extension
//! (via `pipeline::build_call` / `pipeline::parse_response`). No PyO3.

use std::net::SocketAddr;

use axum::{
    extract::Json,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Router,
};
use serde_json::{json, Map, Value};

use litellm_rust::error::OcrError;
use litellm_rust::llms::base_llm::ocr::transformation::OcrRequest;
use litellm_rust::llms::mistral::ocr::transformation::MistralOcrConfig;
use litellm_rust::pipeline::{build_call, parse_response};

/// Default port if `PORT` is unset.
const DEFAULT_PORT: u16 = 8088;

#[tokio::main]
async fn main() {
    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/ocr", post(ocr_handler));

    let port: u16 = std::env::var("PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(DEFAULT_PORT);
    let addr = SocketAddr::from(([0, 0, 0, 0], port));

    println!("litellm-rust-server listening on http://{addr}");

    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .expect("failed to bind listener");
    axum::serve(listener, app).await.expect("server error");
}

async fn health() -> impl IntoResponse {
    Json(json!({ "status": "ok" }))
}

/// Build an `OcrRequest` from a free-form JSON body.
///
/// `model` and `document` are required; `api_key` / `api_base` are optional;
/// every other top-level key is treated as a raw OCR param.
fn request_from_body(mut body: Map<String, Value>) -> Result<OcrRequest, String> {
    let model = match body.remove("model") {
        Some(Value::String(s)) => s,
        Some(_) => return Err("'model' must be a string".to_string()),
        None => return Err("Missing required field 'model'".to_string()),
    };

    let document = body
        .remove("document")
        .ok_or_else(|| "Missing required field 'document'".to_string())?;

    let api_key = match body.remove("api_key") {
        Some(Value::String(s)) => Some(s),
        Some(Value::Null) | None => None,
        Some(_) => return Err("'api_key' must be a string".to_string()),
    };

    let api_base = match body.remove("api_base") {
        Some(Value::String(s)) => Some(s),
        Some(Value::Null) | None => None,
        Some(_) => return Err("'api_base' must be a string".to_string()),
    };

    // Remaining keys are raw OCR params.
    Ok(OcrRequest {
        model,
        document,
        api_key,
        api_base,
        optional_params: body,
    })
}

/// Map an `OcrError` to an HTTP status + JSON error body.
fn ocr_error_response(err: OcrError) -> Response {
    let (status, message) = match &err {
        OcrError::Auth(msg) => (StatusCode::UNAUTHORIZED, msg.clone()),
        OcrError::Transform(msg) => (StatusCode::BAD_REQUEST, msg.clone()),
        OcrError::Http { status, .. } => (
            StatusCode::from_u16(*status).unwrap_or(StatusCode::BAD_GATEWAY),
            err.to_string(),
        ),
        OcrError::Network(_) | OcrError::Parse(_) => (StatusCode::BAD_GATEWAY, err.to_string()),
    };
    (status, Json(json!({ "error": message }))).into_response()
}

async fn ocr_handler(Json(body): Json<Value>) -> Response {
    let body_map = match body {
        Value::Object(map) => map,
        _ => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({ "error": "request body must be a JSON object" })),
            )
                .into_response()
        }
    };

    let req = match request_from_body(body_map) {
        Ok(req) => req,
        Err(msg) => {
            return (StatusCode::BAD_REQUEST, Json(json!({ "error": msg }))).into_response()
        }
    };

    match perform_ocr(req).await {
        Ok(value) => Json(value).into_response(),
        Err(err) => ocr_error_response(err),
    }
}

/// Async OCR call reusing the shared pure transforms.
async fn perform_ocr(req: OcrRequest) -> Result<Value, OcrError> {
    let config = MistralOcrConfig::new();
    let (url, headers, request_body) = build_call(&config, &req)?;

    let client = reqwest::Client::new();
    let mut builder = client.post(&url).json(&request_body);
    for (k, v) in &headers {
        builder = builder.header(k, v);
    }

    let resp = builder
        .send()
        .await
        .map_err(|e| OcrError::Network(e.to_string()))?;

    let status = resp.status();
    let text = resp
        .text()
        .await
        .map_err(|e| OcrError::Network(e.to_string()))?;

    if !status.is_success() {
        return Err(OcrError::Http {
            status: status.as_u16(),
            body: text,
        });
    }

    let response_json: Value =
        serde_json::from_str(&text).map_err(|e| OcrError::Parse(e.to_string()))?;

    let response = parse_response(&config, &req.model, &response_json)?;

    serde_json::to_value(&response).map_err(|e| OcrError::Parse(e.to_string()))
}
