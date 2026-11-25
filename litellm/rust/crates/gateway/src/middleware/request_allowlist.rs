//! The proxy's route allowlist, which it checks before it reads a credential.

use std::sync::Arc;

use axum::{
    Json,
    extract::{Request, State},
    http::StatusCode,
    middleware::Next,
    response::{IntoResponse, Response},
};
use litellm_config::GeneralSettings;
use litellm_gateway_auth_core::{Allowlists, RequestDenial, check_request, client_address::Origin};
use serde_json::json;

/// `general_settings.allowed_routes`, as `gateway-auth-core` reads it. `allowed_ips` sends every
/// request to Python in `admit`, so no address reaches the check here.
#[derive(Debug, Default)]
pub struct Settings {
    pub allowlists: Allowlists,
}

impl Settings {
    pub fn from_general(general: &GeneralSettings) -> Self {
        Self {
            allowlists: Allowlists::new(general),
        }
    }
}

/// `pre_db_read_auth_checks`' route check, which Python's authentication runs, so
/// `middleware::allowlisted` puts this outside the native routes' `authenticate` and on no other
/// route. A preflight never reaches it, since CORS answers one first. The route is the path the
/// router matched, with `root_path` taken off as Python's `get_request_route` takes it off. Python
/// matches on the percent-decoded path and axum on the path as sent, so a path that spells a
/// native route with an escape never reaches this. The body is the OpenAI one the proxy's auth
/// error handler writes, on every route.
pub(super) async fn refuse_unlisted(
    State(allowlists): State<Arc<Allowlists>>,
    request: Request,
    next: Next,
) -> Response {
    match checked(&allowlists, request.uri().path()) {
        Ok(()) => next.run(request).await,
        Err(denial) => (
            StatusCode::FORBIDDEN,
            Json(json!({"error": {
                "message": denial.to_string(),
                "type": "auth_error",
                "param": null,
                "code": "403",
            }})),
        )
            .into_response(),
    }
}

/// `admit` asks the same question first, so a request this refuses is never read further.
pub(crate) fn checked(allowlists: &Allowlists, route: &str) -> Result<(), RequestDenial> {
    check_request(route, Origin::default(), allowlists)
}

#[cfg(test)]
mod tests {
    use axum::{Router, body::Body, middleware::from_fn_with_state, routing::post};
    use litellm_config::Config;
    use rstest::rstest;
    use tower::ServiceExt;

    use super::*;

    async fn answered(yaml: &str, path: &str) -> (StatusCode, serde_json::Value) {
        let config = Config::from_yaml(yaml).unwrap();
        let allowlists = Arc::new(Settings::from_general(&config.general_settings).allowlists);
        let app = Router::new()
            .route("/v1/messages", post(|| async { Json(json!("served")) }))
            .route("/health", post(|| async { Json(json!("served")) }))
            .route_layer(from_fn_with_state(allowlists, refuse_unlisted));
        let request = Request::post(path).body(Body::empty()).unwrap();
        let response = app.oneshot(request).await.unwrap();
        let status = response.status();
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        (status, serde_json::from_slice(&body).unwrap_or_default())
    }

    fn refused(message: &str) -> (StatusCode, serde_json::Value) {
        (
            StatusCode::FORBIDDEN,
            json!({"error": {
                "message": message,
                "type": "auth_error",
                "param": null,
                "code": "403",
            }}),
        )
    }

    fn served() -> (StatusCode, serde_json::Value) {
        (StatusCode::OK, json!("served"))
    }

    const ROUTES: &str = "general_settings:\n  allowed_routes: [/health]\n";

    #[rstest]
    #[case::no_allowlist("model_list: []\n", "/v1/messages", served())]
    #[case::a_listed_route(ROUTES, "/health", served())]
    #[case::an_unlisted_route(
        ROUTES,
        "/v1/messages",
        refused("Access forbidden: Route /v1/messages not allowed")
    )]
    #[tokio::test]
    async fn a_request_the_allowlists_leave_out_is_refused_before_its_handler(
        #[case] yaml: &str,
        #[case] path: &str,
        #[case] expected: (StatusCode, serde_json::Value),
    ) {
        assert_eq!(answered(yaml, path).await, expected);
    }
}
