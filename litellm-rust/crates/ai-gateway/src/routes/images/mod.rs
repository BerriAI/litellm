//! `POST /v1/images/generations` and `POST /v1/images/edits`, the OpenAI-compatible images HTTP surface.

mod service;
#[cfg(test)]
mod tests;

use axum::Router;
use axum::extract::{Json, State};
use axum::http::StatusCode;
use axum::http::header::HeaderMap;
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use litellm_core::CoreError;
use serde_json::{Map, Value};

use crate::auth::key_auth::RequireValidKey;
use crate::constants::{IMAGES_HEADERS_NOT_FORWARDED, IMAGES_ROUTE_PATH_GENERATIONS, IMAGES_ROUTE_PATH_EDITS};
use crate::state::AppState;

/// This route's contribution to the app router.
pub fn router() -> Router<AppState> {
    Router::new()
        .route(IMAGES_ROUTE_PATH_GENERATIONS, post(handle_generation))
        .route(IMAGES_ROUTE_PATH_EDITS, post(handle_edit))
}

async fn handle_generation(
    auth: RequireValidKey,
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Result<Response, ImagesRouteError> {
    let extra_headers = forwarded_headers(&headers)?;
    match service::run_generation(&state, body, extra_headers, &auth.key_object, &auth.hashed_token)
        .await
        .map_err(ImagesRouteError::from)?
    {
        service::ImagesResponseEnum::Json(body) => Ok(Json(body).into_response()),
    }
}

async fn handle_edit(
    auth: RequireValidKey,
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Result<Response, ImagesRouteError> {
    let extra_headers = forwarded_headers(&headers)?;
    match service::run_edit(&state, body, extra_headers, &auth.key_object, &auth.hashed_token)
        .await
        .map_err(ImagesRouteError::from)?
    {
        service::ImagesResponseEnum::Json(body) => Ok(Json(body).into_response()),
    }
}

fn forwarded_headers(headers: &HeaderMap) -> Result<Option<Map<String, Value>>, CoreError> {
    let forwarded = headers
        .iter()
        .filter(|(name, _)| {
            !IMAGES_HEADERS_NOT_FORWARDED
                .iter()
                .any(|excluded| name.as_str().eq_ignore_ascii_case(excluded))
        })
        .map(|(name, value)| {
            let value = value.to_str().map_err(|_| {
                CoreError::InvalidRequest(format!("invalid value for header {}", name.as_str()))
            })?;
            Ok((name.to_string(), Value::String(value.to_string())))
        })
        .collect::<Result<Map<_, _>, CoreError>>()?;
    Ok((!forwarded.is_empty()).then_some(forwarded))
}

#[derive(Debug)]
struct ImagesRouteError(CoreError);

impl From<CoreError> for ImagesRouteError {
    fn from(error: CoreError) -> Self {
        Self(error)
    }
}

impl IntoResponse for ImagesRouteError {
    fn into_response(self) -> Response {
        let (status, message) = match self.0 {
            CoreError::InvalidRequest(message) => (StatusCode::BAD_REQUEST, message),
            CoreError::InvalidProvider(_) | CoreError::Routing(_) => (
                StatusCode::NOT_FOUND,
                "no images deployment is configured for this model".to_string(),
            ),
            CoreError::Auth(_) => (
                StatusCode::BAD_GATEWAY,
                "images provider authentication failed".to_string(),
            ),
            CoreError::Http { .. }
            | CoreError::Network(_)
            | CoreError::Connect(_)
            | CoreError::InvalidResponse(_)
            | CoreError::InvalidType { .. }
            | CoreError::MissingField(_) => (
                StatusCode::BAD_GATEWAY,
                "images provider request failed".to_string(),
            ),
            CoreError::Unsupported(reason) => (
                StatusCode::BAD_REQUEST,
                format!("images request is not supported: {reason}"),
            ),
            CoreError::Timeout(message) => (
                StatusCode::GATEWAY_TIMEOUT,
                format!("images request timed out: {message}"),
            ),
        };
        (
            status,
            Json(serde_json::json!({"error": {"message": message}})),
        )
            .into_response()
    }
}
