//! `POST /ocr`, the OCR endpoint.

mod service;

use axum::Router;
use axum::extract::{Json, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use litellm_core::Error;
use serde_json::Value;

use crate::auth::RequireMasterKey;
use crate::state::AppState;

const OCR_ROUTE_PATH: &str = "/ocr";

/// This route's contribution to the app router.
#[tracing::instrument(name = "ocr_router")]
pub fn router() -> Router<AppState> {
    Router::new().route(OCR_ROUTE_PATH, post(handle))
}

#[tracing::instrument(name = "ocr_handler", skip(state, body))]
async fn handle(
    _auth: RequireMasterKey,
    State(state): State<AppState>,
    Json(body): Json<Value>,
) -> Result<Response, OcrRouteError> {
    let response = service::run(&state.router, body)
        .await
        .map_err(OcrRouteError::from)?;
    Ok(Json(response).into_response())
}

#[derive(Debug)]
struct OcrRouteError(Error);

impl From<Error> for OcrRouteError {
    fn from(error: Error) -> Self {
        Self(error)
    }
}

impl IntoResponse for OcrRouteError {
    fn into_response(self) -> Response {
        let (status, message) = match self.0 {
            Error::InvalidRequest(message) => (StatusCode::BAD_REQUEST, message),
            Error::InvalidProvider(_) | Error::Routing(_) => (
                StatusCode::NOT_FOUND,
                "no OCR deployment is configured for this model".to_string(),
            ),
            Error::Auth(_) => (
                StatusCode::BAD_GATEWAY,
                "OCR provider authentication failed".to_string(),
            ),
            Error::Http { .. }
            | Error::Network(_)
            | Error::Connect(_)
            | Error::InvalidResponse(_)
            | Error::InvalidType { .. }
            | Error::MissingField(_) => (
                StatusCode::BAD_GATEWAY,
                "OCR provider request failed".to_string(),
            ),
            Error::Unsupported(reason) => (
                StatusCode::BAD_REQUEST,
                format!("OCR request is not supported: {reason}"),
            ),
        };
        (
            status,
            Json(serde_json::json!({"error": {"message": message}})),
        )
            .into_response()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use axum::body::Body;
    use axum::http::Request;
    use axum::http::StatusCode;
    use litellm_core::router::{Deployment, LiteLLMParams, Router as ModelRouter};
    use serde_json::json;
    use tower::ServiceExt;

    use super::super::app;
    use crate::io::realtime_pool::RealtimePool;
    use crate::state::AppState;

    fn test_state(model: &str, api_base: String, master_key: Option<&str>) -> AppState {
        AppState {
            router: Arc::new(ModelRouter::new(vec![Deployment {
                model_name: model.to_string(),
                litellm_params: LiteLLMParams {
                    model: format!("mistral/{model}"),
                    api_key: Some("test-key".to_string()),
                    api_base: Some(api_base),
                },
            }])),
            master_key: master_key.map(Arc::from),
            loggers: Arc::new(Vec::new()),
            realtime_pool: RealtimePool::disabled(),
        }
    }

    #[tokio::test]
    async fn route_rejects_missing_master_key() {
        let app = app(test_state(
            "mistral-ocr",
            "http://127.0.0.1:1".to_string(),
            Some("master-key"),
        ));
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/ocr")
                    .header("content-type", "application/json")
                    .body(Body::from("{}"))
                    .expect("request builds"),
            )
            .await
            .expect("route responds");
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn route_rejects_invalid_master_key() {
        let app = app(test_state(
            "mistral-ocr",
            "http://127.0.0.1:1".to_string(),
            Some("master-key"),
        ));
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/ocr")
                    .header("authorization", "Bearer wrong-key")
                    .header("content-type", "application/json")
                    .body(Body::from("{}"))
                    .expect("request builds"),
            )
            .await
            .expect("route responds");
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }
}
