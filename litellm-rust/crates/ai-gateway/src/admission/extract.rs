//! Axum extractor running [`Admission`] on the raw request body.

use axum::body::to_bytes;
use axum::extract::{FromRequest, Request};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};

use crate::auth::bearer_token;
use crate::state::AppState;

use super::{Admission, Admitted, Rejection};

/// Handler argument that yields the admitted request, or the rejection response.
pub struct Admit(pub Admitted);

#[axum::async_trait]
impl FromRequest<AppState> for Admit {
    type Rejection = Response;

    async fn from_request(request: Request, state: &AppState) -> Result<Self, Self::Rejection> {
        let (parts, body) = request.into_parts();
        let admission: &Admission = &state.admission;
        let raw = to_bytes(body, admission.max_request_bytes())
            .await
            .map_err(|error| (StatusCode::PAYLOAD_TOO_LARGE, error.to_string()).into_response())?;
        admission
            .admit(bearer_token(&parts.headers), &raw)
            .await
            .map(Admit)
            .map_err(|rejection| reject(&rejection))
    }
}

fn reject(rejection: &Rejection) -> Response {
    let status = match rejection {
        Rejection::Unauthorized => StatusCode::UNAUTHORIZED,
        Rejection::IdentityUnavailable(_) => StatusCode::SERVICE_UNAVAILABLE,
        Rejection::ModelNotAllowed(_) => StatusCode::FORBIDDEN,
        Rejection::RequestTooLarge { .. } => StatusCode::PAYLOAD_TOO_LARGE,
        Rejection::ContextTooLarge { .. } | Rejection::InvalidRequest(_) => StatusCode::BAD_REQUEST,
        Rejection::LimitExceeded(_) => StatusCode::TOO_MANY_REQUESTS,
        Rejection::Tokenizer(_) => StatusCode::INTERNAL_SERVER_ERROR,
    };
    (
        status,
        axum::Json(serde_json::json!({"error": {"message": rejection.to_string()}})),
    )
        .into_response()
}
