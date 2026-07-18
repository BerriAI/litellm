mod service;

use axum::body::Body;
use axum::extract::{Json, State};
use axum::http::header::CONTENT_TYPE;
use axum::http::{Response, StatusCode};
use axum::response::IntoResponse;
use axum::routing::post;
use axum::Router;
use serde_json::Value;

use crate::auth::RequireMasterKey;
use crate::state::AppState;

pub fn router() -> Router<AppState> {
    Router::new().route("/v1/messages", post(handle))
}

async fn handle(
    _auth: RequireMasterKey,
    State(state): State<AppState>,
    Json(body): Json<Value>,
) -> Result<Response<Body>, (StatusCode, String)> {
    match service::run(&state.router, body).await.map_err(map_error)? {
        service::MessagesResponse::Json(body) => {
            Ok((StatusCode::OK, axum::Json(body)).into_response())
        }
        service::MessagesResponse::Stream(response) => {
            let content_type = response
                .headers()
                .get(CONTENT_TYPE)
                .cloned()
                .unwrap_or_else(|| axum::http::HeaderValue::from_static("text/event-stream"));
            let mut result = Response::new(Body::from_stream(response.bytes_stream()));
            result.headers_mut().insert(CONTENT_TYPE, content_type);
            Ok(result)
        }
    }
}

fn map_error(error: litellm_core::CoreError) -> (StatusCode, String) {
    match error {
        litellm_core::CoreError::Http { status, body } => (
            StatusCode::from_u16(status).unwrap_or(StatusCode::BAD_GATEWAY),
            body,
        ),
        litellm_core::CoreError::InvalidRequest(message)
        | litellm_core::CoreError::InvalidProvider(message)
        | litellm_core::CoreError::Routing(message) => (StatusCode::BAD_REQUEST, message),
        litellm_core::CoreError::Network(message) => (StatusCode::BAD_GATEWAY, message),
        error => (StatusCode::BAD_REQUEST, error.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use litellm_core::router::{Deployment, LiteLLMParams, Router as ModelRouter};
    use serde_json::json;
    use tower::ServiceExt;

    use super::router;
    use crate::io::realtime_pool::RealtimePool;
    use crate::state::AppState;

    fn app() -> axum::Router {
        let state = AppState {
            router: Arc::new(ModelRouter::new(vec![Deployment {
                model_name: "rust-model".to_string(),
                litellm_params: LiteLLMParams {
                    model: "azure_ai/claude-loadtest".to_string(),
                    api_key: Some("sk-upstream".to_string()),
                    api_base: Some("http://127.0.0.1:1".to_string()),
                },
            }])),
            master_key: Some(Arc::from("sk-1234")),
            loggers: Arc::new(Vec::new()),
            realtime_pool: RealtimePool::disabled(),
        };
        router().with_state(state)
    }

    #[tokio::test]
    async fn requires_authentication() {
        let request = Request::builder()
            .method("POST")
            .uri("/v1/messages")
            .header("content-type", "application/json")
            .body(Body::from(json!({"model": "rust-model"}).to_string()))
            .expect("request builds");
        let response = app().oneshot(request).await.expect("response");
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn rejects_unknown_model() {
        let request = Request::builder()
            .method("POST")
            .uri("/v1/messages")
            .header("authorization", "Bearer sk-1234")
            .header("content-type", "application/json")
            .body(Body::from(json!({"model": "missing"}).to_string()))
            .expect("request builds");
        let response = app().oneshot(request).await.expect("response");
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }
}
