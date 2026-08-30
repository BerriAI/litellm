//! `POST /v1/embeddings`, the OpenAI-compatible embeddings HTTP surface.

mod service;

use axum::Router;
use axum::extract::{Json, State};
use axum::http::StatusCode;
use axum::http::header::HeaderMap;
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use litellm_core::CoreError;
use serde_json::{Map, Value};

use crate::auth::key_auth::RequireValidKey;
use crate::constants::{EMBEDDINGS_HEADERS_NOT_FORWARDED, EMBEDDINGS_ROUTE_PATH};
use crate::state::AppState;

/// This route's contribution to the app router.
pub fn router() -> Router<AppState> {
    Router::new().route(EMBEDDINGS_ROUTE_PATH, post(handle))
}

async fn handle(
    auth: RequireValidKey,
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Result<Response, EmbeddingsRouteError> {
    let extra_headers = forwarded_headers(&headers)?;
    match service::run(&state, body, extra_headers, &auth.key_object, &auth.hashed_token)
        .await
        .map_err(EmbeddingsRouteError::from)?
    {
        service::EmbeddingsResponseEnum::Json(body) => Ok(Json(body).into_response()),
    }
}

fn forwarded_headers(headers: &HeaderMap) -> Result<Option<Map<String, Value>>, CoreError> {
    let forwarded = headers
        .iter()
        .filter(|(name, _)| {
            !EMBEDDINGS_HEADERS_NOT_FORWARDED
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
struct EmbeddingsRouteError(CoreError);

impl From<CoreError> for EmbeddingsRouteError {
    fn from(error: CoreError) -> Self {
        Self(error)
    }
}

impl IntoResponse for EmbeddingsRouteError {
    fn into_response(self) -> Response {
        let (status, message) = match self.0 {
            CoreError::InvalidRequest(message) => (StatusCode::BAD_REQUEST, message),
            CoreError::InvalidProvider(_) | CoreError::Routing(_) => (
                StatusCode::NOT_FOUND,
                "no embeddings deployment is configured for this model".to_string(),
            ),
            CoreError::Auth(_) => (
                StatusCode::BAD_GATEWAY,
                "embeddings provider authentication failed".to_string(),
            ),
            CoreError::Http { .. }
            | CoreError::Network(_)
            | CoreError::Connect(_)
            | CoreError::InvalidResponse(_)
            | CoreError::InvalidType { .. }
            | CoreError::MissingField(_) => (
                StatusCode::BAD_GATEWAY,
                "embeddings provider request failed".to_string(),
            ),
            CoreError::Unsupported(reason) => (
                StatusCode::BAD_REQUEST,
                format!("embeddings request is not supported: {reason}"),
            ),
        };
        (
            status,
            Json(serde_json::json!({"error": {"message": message}})),
        )
            .into_response()
    }
}
