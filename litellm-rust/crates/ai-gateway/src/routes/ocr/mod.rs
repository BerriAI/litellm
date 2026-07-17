mod service;
mod transport;

#[cfg(test)]
mod tests;

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
        .map_err(|_| CoreError::InvalidRequest("could not read the multipart form".to_string()))?;

    let mut file: Option<(Vec<u8>, Option<String>, Option<String>)> = None;
    let mut text_fields: Vec<(String, String)> = Vec::new();

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|_| CoreError::InvalidRequest("could not read a multipart field".to_string()))?
    {
        let name = field.name().map(str::to_string);
        match name.as_deref() {
            Some("file") => {
                if file.is_some() {
                    return Err(CoreError::InvalidRequest(
                        "the 'file' field must appear at most once in a multipart OCR request"
                            .to_string(),
                    ));
                }
                let filename = field.file_name().map(str::to_string);
                let content_type = field.content_type().map(str::to_string);
                let bytes = field.bytes().await.map_err(|_| {
                    CoreError::InvalidRequest("could not read the uploaded file".to_string())
                })?;
                file = Some((bytes.to_vec(), filename, content_type));
            }
            Some(name) => {
                let name = name.to_string();
                let text = field.text().await.map_err(|_| {
                    CoreError::InvalidRequest("could not read a multipart form field".to_string())
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
            "authentication failed".to_string(),
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
