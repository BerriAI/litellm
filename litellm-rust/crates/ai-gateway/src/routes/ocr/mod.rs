mod service;
mod transport;

use axum::body::{to_bytes, Body};
use axum::extract::{DefaultBodyLimit, FromRequest, Multipart, Request, State};
use axum::http::header::CONTENT_TYPE;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};
use litellm_core::error::CoreError;
use litellm_core::CoreResult;
use serde_json::{json, Value};

use crate::auth::RequireMasterKey;
use crate::constants::MAX_OCR_REQUEST_BYTES;
use crate::state::AppState;

use transport::OcrCall;

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/v1/ocr", post(handle))
        .route("/ocr", post(handle))
        .layer(DefaultBodyLimit::max(MAX_OCR_REQUEST_BYTES))
}

async fn handle(
    _auth: RequireMasterKey,
    State(state): State<AppState>,
    request: Request,
) -> Response {
    let call = match parse_request(request, &state).await {
        Ok(call) => call,
        Err(err) => return error_response(&err),
    };
    match service::run_ocr(&state.router, call).await {
        Ok(value) => (StatusCode::OK, Json(value)).into_response(),
        Err(err) => error_response(&err),
    }
}

fn is_multipart(request: &Request) -> bool {
    request
        .headers()
        .get(CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .map(|value| value.to_ascii_lowercase().contains("multipart/form-data"))
        .unwrap_or(false)
}

async fn parse_request(request: Request, state: &AppState) -> CoreResult<OcrCall> {
    if is_multipart(&request) {
        parse_multipart(request, state).await
    } else {
        parse_json(request).await
    }
}

async fn parse_json(request: Request) -> CoreResult<OcrCall> {
    let bytes = read_body(request.into_body()).await?;
    transport::parse_json_body(&bytes)
}

async fn read_body(body: Body) -> CoreResult<Vec<u8>> {
    to_bytes(body, MAX_OCR_REQUEST_BYTES)
        .await
        .map(|bytes| bytes.to_vec())
        .map_err(|err| CoreError::InvalidRequest(format!("could not read request body: {err}")))
}

async fn parse_multipart(request: Request, state: &AppState) -> CoreResult<OcrCall> {
    let mut multipart = Multipart::from_request(request, state)
        .await
        .map_err(|err| {
            CoreError::InvalidRequest(format!("could not read multipart form: {err}"))
        })?;

    let mut file: Option<(Vec<u8>, Option<String>, Option<String>)> = None;
    let mut text_fields: Vec<(String, String)> = Vec::new();

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|err| CoreError::InvalidRequest(format!("invalid multipart field: {err}")))?
    {
        let name = field.name().map(str::to_string);
        match name.as_deref() {
            Some("file") => {
                let filename = field.file_name().map(str::to_string);
                let content_type = field.content_type().map(str::to_string);
                let bytes = field.bytes().await.map_err(|err| {
                    CoreError::InvalidRequest(format!("could not read uploaded file: {err}"))
                })?;
                file = Some((bytes.to_vec(), filename, content_type));
            }
            Some(name) => {
                let name = name.to_string();
                let text = field.text().await.map_err(|err| {
                    CoreError::InvalidRequest(format!("could not read form field '{name}': {err}"))
                })?;
                text_fields.push((name, text));
            }
            None => {}
        }
    }

    let (bytes, filename, content_type) = file.ok_or_else(|| {
        CoreError::InvalidRequest(
            "multipart OCR request must include a 'file' field with the document to process"
                .to_string(),
        )
    })?;
    let document =
        transport::build_upload_document(bytes, filename.as_deref(), content_type.as_deref())?;
    transport::assemble_multipart_call(document, &text_fields)
}

fn error_response(error: &CoreError) -> Response {
    let (status, error_type, message) = match error {
        CoreError::InvalidRequest(_)
        | CoreError::InvalidType { .. }
        | CoreError::MissingField(_)
        | CoreError::InvalidProvider(_) => (
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            error.to_string(),
        ),
        CoreError::Auth(_) => (
            StatusCode::UNAUTHORIZED,
            "authentication_error",
            error.to_string(),
        ),
        CoreError::Routing(_) => (StatusCode::NOT_FOUND, "not_found_error", error.to_string()),
        CoreError::Http { status, .. } => {
            let status = StatusCode::from_u16(*status).unwrap_or(StatusCode::BAD_GATEWAY);
            (
                status,
                "upstream_error",
                format!(
                    "the OCR provider returned an error (status {})",
                    status.as_u16()
                ),
            )
        }
        CoreError::Network(_) => (
            StatusCode::BAD_GATEWAY,
            "upstream_error",
            "the OCR provider could not be reached".to_string(),
        ),
        CoreError::InvalidResponse(_) => (
            StatusCode::BAD_GATEWAY,
            "upstream_error",
            "the OCR provider returned an unexpected response".to_string(),
        ),
    };
    (status, Json(error_body(&message, error_type))).into_response()
}

fn error_body(message: &str, error_type: &str) -> Value {
    json!({
        "error": {
            "message": message,
            "type": error_type,
        }
    })
}

#[cfg(test)]
mod tests {
    use std::net::SocketAddr;
    use std::sync::Arc;

    use litellm_core::router::{Deployment, LiteLLMParams, Router as ModelRouter};
    use serde_json::Value;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::{TcpListener, TcpStream};

    use crate::io::realtime_pool::RealtimePool;
    use crate::state::AppState;

    const MASTER_KEY: &str = "sk-master-test";

    async fn read_http_request(socket: &mut TcpStream) -> String {
        let mut request = Vec::new();
        let mut buffer = [0_u8; 2048];
        loop {
            let n = socket.read(&mut buffer).await.expect("reads request");
            if n == 0 {
                break;
            }
            request.extend_from_slice(&buffer[..n]);
            if request.windows(4).any(|window| window == b"\r\n\r\n") {
                break;
            }
        }
        String::from_utf8_lossy(&request).into_owned()
    }

    async fn spawn_mock_upstream() -> (String, tokio::task::JoinHandle<String>) {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("binds upstream");
        let addr = listener.local_addr().expect("upstream addr");
        let handle = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.expect("accepts one request");
            let request = read_http_request(&mut socket).await;
            let body = r#"{"pages":[{"index":0,"markdown":"hello ocr"}],"model":"mistral-ocr-latest","usage_info":{"pages_processed":1}}"#;
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            socket
                .write_all(response.as_bytes())
                .await
                .expect("writes response");
            request
        });
        (format!("http://{addr}"), handle)
    }

    async fn spawn_mock_upstream_error() -> String {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("binds upstream");
        let addr = listener.local_addr().expect("upstream addr");
        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.expect("accepts one request");
            let _ = read_http_request(&mut socket).await;
            let body = r#"{"error":"invalid_api_key: sk-leaked-secret-value"}"#;
            let response = format!(
                "HTTP/1.1 403 Forbidden\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            socket
                .write_all(response.as_bytes())
                .await
                .expect("writes response");
        });
        format!("http://{addr}")
    }

    fn app_with_deployment(api_base: &str) -> axum::Router {
        let router = ModelRouter::new(vec![Deployment {
            model_name: "rust-ocr-mistral".to_string(),
            litellm_params: LiteLLMParams {
                model: "mistral/mistral-ocr-latest".to_string(),
                api_key: Some("sk-upstream".to_string()),
                api_base: Some(api_base.to_string()),
            },
        }]);
        let state = AppState {
            router: Arc::new(router),
            master_key: Some(Arc::from(MASTER_KEY)),
            loggers: Arc::new(Vec::new()),
            realtime_pool: RealtimePool::disabled(),
        };
        crate::routes::app(state)
    }

    async fn serve(app: axum::Router) -> SocketAddr {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("binds gateway");
        let addr = listener.local_addr().expect("gateway addr");
        tokio::spawn(async move {
            axum::serve(listener, app).await.expect("serves");
        });
        addr
    }

    #[tokio::test]
    async fn json_document_url_returns_normalized_ocr_on_both_paths() {
        for path in ["/v1/ocr", "/ocr"] {
            let (upstream, upstream_handle) = spawn_mock_upstream().await;
            let addr = serve(app_with_deployment(&upstream)).await;

            let response = reqwest::Client::new()
                .post(format!("http://{addr}{path}"))
                .bearer_auth(MASTER_KEY)
                .json(&serde_json::json!({
                    "model": "rust-ocr-mistral",
                    "document": {"type": "document_url", "document_url": "https://example.com/doc.pdf"}
                }))
                .send()
                .await
                .expect("request sent");

            assert_eq!(response.status(), reqwest::StatusCode::OK, "path {path}");
            let body: Value = response.json().await.expect("json body");
            assert_eq!(body["object"], "ocr", "path {path}");
            assert_eq!(body["model"], "mistral-ocr-latest", "path {path}");
            assert_eq!(body["pages"][0]["markdown"], "hello ocr", "path {path}");

            let upstream_request = upstream_handle.await.expect("upstream served");
            assert!(
                upstream_request.contains("authorization: Bearer sk-upstream")
                    || upstream_request.contains("Authorization: Bearer sk-upstream"),
                "upstream must receive the deployment credential: {upstream_request}"
            );
        }
    }

    #[tokio::test]
    async fn multipart_upload_returns_normalized_ocr() {
        let (upstream, upstream_handle) = spawn_mock_upstream().await;
        let addr = serve(app_with_deployment(&upstream)).await;

        let form = reqwest::multipart::Form::new()
            .text("model", "rust-ocr-mistral")
            .part(
                "file",
                reqwest::multipart::Part::bytes(b"%PDF-1.4 minimal".to_vec())
                    .file_name("doc.pdf")
                    .mime_str("application/pdf")
                    .expect("mime"),
            );

        let response = reqwest::Client::new()
            .post(format!("http://{addr}/v1/ocr"))
            .bearer_auth(MASTER_KEY)
            .multipart(form)
            .send()
            .await
            .expect("request sent");

        assert_eq!(response.status(), reqwest::StatusCode::OK);
        let body: Value = response.json().await.expect("json body");
        assert_eq!(body["object"], "ocr");
        assert_eq!(body["pages"][0]["markdown"], "hello ocr");

        let upstream_request = upstream_handle.await.expect("upstream served");
        assert!(upstream_request.starts_with("POST"), "{upstream_request}");
    }

    #[tokio::test]
    async fn upstream_error_status_is_propagated_without_leaking_provider_body() {
        let upstream = spawn_mock_upstream_error().await;
        let addr = serve(app_with_deployment(&upstream)).await;

        let response = reqwest::Client::new()
            .post(format!("http://{addr}/v1/ocr"))
            .bearer_auth(MASTER_KEY)
            .json(&serde_json::json!({
                "model": "rust-ocr-mistral",
                "document": {"type": "document_url", "document_url": "https://example.com/doc.pdf"}
            }))
            .send()
            .await
            .expect("request sent");

        assert_eq!(response.status(), reqwest::StatusCode::FORBIDDEN);
        let body: Value = response.json().await.expect("json body");
        assert_eq!(body["error"]["type"], "upstream_error");
        let message = body["error"]["message"].as_str().expect("message string");
        assert!(
            !message.contains("sk-leaked-secret-value") && !message.contains("invalid_api_key"),
            "provider body must not leak into the public error: {message}"
        );
    }

    #[tokio::test]
    async fn missing_master_key_is_unauthorized() {
        let addr = serve(app_with_deployment("http://127.0.0.1:1")).await;
        let response = reqwest::Client::new()
            .post(format!("http://{addr}/v1/ocr"))
            .json(&serde_json::json!({
                "model": "rust-ocr-mistral",
                "document": {"type": "document_url", "document_url": "https://example.com/doc.pdf"}
            }))
            .send()
            .await
            .expect("request sent");
        assert_eq!(response.status(), reqwest::StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn unknown_model_is_not_found() {
        let addr = serve(app_with_deployment("http://127.0.0.1:1")).await;
        let response = reqwest::Client::new()
            .post(format!("http://{addr}/v1/ocr"))
            .bearer_auth(MASTER_KEY)
            .json(&serde_json::json!({
                "model": "does-not-exist",
                "document": {"type": "document_url", "document_url": "https://example.com/doc.pdf"}
            }))
            .send()
            .await
            .expect("request sent");
        assert_eq!(response.status(), reqwest::StatusCode::NOT_FOUND);
        let body: Value = response.json().await.expect("json body");
        assert_eq!(body["error"]["type"], "not_found_error");
    }

    #[tokio::test]
    async fn file_document_over_json_is_rejected() {
        let addr = serve(app_with_deployment("http://127.0.0.1:1")).await;
        let response = reqwest::Client::new()
            .post(format!("http://{addr}/v1/ocr"))
            .bearer_auth(MASTER_KEY)
            .json(&serde_json::json!({
                "model": "rust-ocr-mistral",
                "document": {"type": "file", "file": "/etc/passwd"}
            }))
            .send()
            .await
            .expect("request sent");
        assert_eq!(response.status(), reqwest::StatusCode::BAD_REQUEST);
        let body: Value = response.json().await.expect("json body");
        assert_eq!(body["error"]["type"], "invalid_request_error");
    }
}
