//! The middleware that records each request counting toward the gateway request totals with the
//! status it answered. Which requests count, and under which route, is
//! `litellm_core_utils::billable_route`, the table the proxy classifies with too.

use std::sync::Arc;

use axum::{
    extract::{Request, State},
    http::Method,
    middleware::Next,
    response::Response,
};
use litellm_core_utils::billable_route;

use crate::gateway_requests::GatewayRequests;

/// `BillableRequestMetricsMiddleware` with only its gateway request sink: a request that counts is
/// recorded with the status its response answered, a 2xx as successful and anything else as
/// failed. A handler that panics records nothing, as an exception that escapes the app records
/// nothing in Python. The gateway installs it only with a database, as Python builds the sink only
/// with one.
///
/// Accepted differences: Python records once the body is sent and this records when the response
/// starts, which moves only the flush a long stream lands in. The built-in EU JEV decision route
/// is not classified, since it counts in Python only while no pass-through endpoint claims it and
/// the gateway never serves it. The enterprise billing recorder is not ported.
pub(super) async fn count(
    State(requests): State<Arc<GatewayRequests>>,
    method: Method,
    request: Request,
    next: Next,
) -> Response {
    let Some((category, route)) = billable_route::classify(request.uri().path(), method.as_str())
    else {
        return next.run(request).await;
    };
    let response = next.run(request).await;
    requests.record(category, route, response.status());
    response
}

#[cfg(test)]
mod tests {
    use axum::{Router, body::Body, http::StatusCode, routing::any};
    use rstest::rstest;
    use tower::ServiceExt;

    use super::*;

    async fn counted(path: &str, status: StatusCode) -> String {
        let requests = Arc::new(GatewayRequests::new());
        let app = Router::new()
            .route("/{*path}", any(move || async move { status }))
            .layer(axum::middleware::from_fn_with_state(
                requests.clone(),
                count,
            ));
        let response = app
            .oneshot(Request::post(path).body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), status);
        requests.summary()
    }

    #[rstest]
    #[case::success("/v1/messages", StatusCode::OK, "llm /v1/messages 200: 1 ok 0 failed")]
    #[case::failure(
        "/v1/messages",
        StatusCode::BAD_GATEWAY,
        "llm /v1/messages 502: 0 ok 1 failed"
    )]
    #[case::not_counted("/key/generate", StatusCode::OK, "")]
    #[tokio::test]
    async fn a_counted_request_is_recorded_with_its_status(
        #[case] path: &str,
        #[case] status: StatusCode,
        #[case] expected: &str,
    ) {
        assert_eq!(counted(path, status).await, expected);
    }
}
