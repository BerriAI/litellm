//! Audio routes: speech (TTS) and transcription (STT).

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
use crate::constants::{AUDIO_HEADERS_NOT_FORWARDED, AUDIO_ROUTE_PATH_SPEECH, AUDIO_ROUTE_PATH_TRANSCRIPTIONS};
use crate::state::AppState;

/// This route's contribution to the app router.
pub fn router() -> Router<AppState> {
    Router::new()
        .route(AUDIO_ROUTE_PATH_SPEECH, post(handle_speech))
        .route(AUDIO_ROUTE_PATH_TRANSCRIPTIONS, post(handle_transcription))
}

async fn handle_speech(
    auth: RequireValidKey,
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Result<Response, AudioRouteError> {
    let extra_headers = forwarded_headers(&headers)?;
    match service::run_speech(&state, body, extra_headers, &auth.key_object, &auth.hashed_token)
        .await
        .map_err(AudioRouteError::from)?
    {
        service::AudioResponseEnum::Binary { data, content_type } => {
            Ok((
                [(axum::http::header::CONTENT_TYPE, content_type)],
                data,
            ).into_response())
        }
        service::AudioResponseEnum::Json(_) => {
            Err(AudioRouteError(CoreError::InvalidResponse(
                "speech endpoint should return binary data, not JSON".to_string(),
            )))
        }
    }
}

async fn handle_transcription(
    auth: RequireValidKey,
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Result<Response, AudioRouteError> {
    let extra_headers = forwarded_headers(&headers)?;
    match service::run_transcription(&state, body, extra_headers, &auth.key_object, &auth.hashed_token)
        .await
        .map_err(AudioRouteError::from)?
    {
        service::AudioResponseEnum::Json(body) => Ok(Json(body).into_response()),
        service::AudioResponseEnum::Binary { .. } => {
            Err(AudioRouteError(CoreError::InvalidResponse(
                "transcription endpoint should return JSON, not binary data".to_string(),
            )))
        }
    }
}

fn forwarded_headers(headers: &HeaderMap) -> Result<Option<Map<String, Value>>, CoreError> {
    let forwarded = headers
        .iter()
        .filter(|(name, _)| {
            !AUDIO_HEADERS_NOT_FORWARDED
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
struct AudioRouteError(CoreError);

impl From<CoreError> for AudioRouteError {
    fn from(error: CoreError) -> Self {
        Self(error)
    }
}

impl IntoResponse for AudioRouteError {
    fn into_response(self) -> Response {
        let (status, message) = match self.0 {
            CoreError::InvalidRequest(message) => (StatusCode::BAD_REQUEST, message),
            CoreError::InvalidProvider(_) | CoreError::Routing(_) => (
                StatusCode::NOT_FOUND,
                "no audio deployment is configured for this model".to_string(),
            ),
            CoreError::Auth(_) => (
                StatusCode::BAD_GATEWAY,
                "audio provider authentication failed".to_string(),
            ),
            CoreError::Http { .. }
            | CoreError::Network(_)
            | CoreError::Connect(_)
            | CoreError::InvalidResponse(_)
            | CoreError::InvalidType { .. }
            | CoreError::MissingField(_) => (
                StatusCode::BAD_GATEWAY,
                "audio provider request failed".to_string(),
            ),
            CoreError::Unsupported(reason) => (
                StatusCode::BAD_REQUEST,
                format!("audio request is not supported: {reason}"),
            ),
        };
        (
            status,
            Json(serde_json::json!({"error": {"message": message}})),
        )
            .into_response()
    }
}
