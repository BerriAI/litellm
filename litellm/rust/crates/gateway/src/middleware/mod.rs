//! [`global`] wraps the whole app in the proxy's FastAPI middleware (`litellm/proxy/proxy_server.py`),
//! [`inference`] adds what only inference and MCP routes get, and [`root_path`] goes outside the
//! router.

pub mod admission_control;
pub mod body_limit;
mod call_id;
pub mod cors;
pub mod decompression;
mod in_flight;
mod panic;
mod request_log;
pub mod root_path;
pub mod security_headers;

use std::sync::Arc;

use axum::{
    Extension, Router,
    extract::DefaultBodyLimit,
    middleware::{from_fn, from_fn_with_state},
};
use litellm_license::Entitlements;
use tower::ServiceBuilder;

pub use in_flight::InFlight;

#[derive(Debug, Default)]
pub struct Settings {
    pub admission_control: Option<admission_control::Settings>,
    pub security_headers: security_headers::Settings,
    pub cors: cors::Settings,
    pub body_limit: body_limit::Settings,
    pub decompression: decompression::Settings,
}

/// Outermost first, in the order of the proxy's FastAPI middleware, with the panic handler where
/// Starlette answers an unhandled exception, outside every middleware. Python caps a body only through its size limit middleware,
/// so axum's own 2 MB extractor default is lifted and a router that sets its own keeps it. Every
/// middleware and route inside reads the license as `Extension<Arc<Entitlements>>`.
pub fn global(
    router: Router,
    settings: &Settings,
    entitlements: Arc<Entitlements>,
    in_flight: &InFlight,
) -> Router {
    let size_limits = Arc::new(body_limit::SizeLimits::new(&settings.body_limit));
    let admission = settings.admission_control.map(|settings| {
        from_fn_with_state(
            Arc::new(admission_control::Admission::new(settings)),
            admission_control::admit,
        )
    });
    router.layer(
        ServiceBuilder::new()
            .layer(panic::layer())
            .layer(Extension(entitlements))
            .option_layer(admission)
            .layer(from_fn_with_state(
                size_limits.clone(),
                body_limit::limit_body,
            ))
            .layer(DefaultBodyLimit::disable())
            .layer(security_headers::layer(&settings.security_headers))
            .layer(in_flight::layer(in_flight))
            .layer(cors::layer(&settings.cors))
            .layer(from_fn_with_state(
                Arc::new(decompression::Decompression::new(
                    settings.decompression,
                    size_limits,
                )),
                decompression::decompress,
            )),
    )
}

/// Request logging skips UI routes because their bodies carry credentials, and Python resolves the
/// call id per inference handler. The call id wraps the log so logged requests carry it.
pub(crate) fn inference(router: Router) -> Router {
    router
        .layer(from_fn(request_log::log_request))
        .layer(call_id::layer())
}

#[cfg(test)]
mod tests {
    use axum::{
        body::{Body, Bytes},
        extract::Request,
        http::{HeaderMap, StatusCode, header},
        routing::post,
    };
    use rstest::rstest;
    use serde_json::{Value, json};
    use tower::ServiceExt;

    use super::*;

    fn premium() -> Arc<Entitlements> {
        Arc::new(Entitlements::ServerVerified)
    }

    fn app(max_request_size_mb: f64) -> Router {
        global(
            Router::new().route("/v1/messages", post(|_: Bytes| async { "ok" })),
            &Settings {
                body_limit: body_limit::Settings {
                    max_request_size_mb: Some(max_request_size_mb),
                },
                ..Settings::default()
            },
            premium(),
            &InFlight::new(),
        )
    }

    fn assert_shell_headers(headers: &HeaderMap) {
        assert_eq!(headers[header::X_FRAME_OPTIONS], "DENY");
    }

    #[rstest]
    #[case::unknown_path(Request::get("/unknown").body(Body::empty()).unwrap(), StatusCode::NOT_FOUND)]
    #[case::cors_preflight(
        Request::options("/v1/messages")
            .header(header::ORIGIN, "https://site.example")
            .header(header::ACCESS_CONTROL_REQUEST_METHOD, "POST")
            .body(Body::empty())
            .unwrap(),
        StatusCode::OK
    )]
    #[tokio::test]
    async fn every_response_passes_through_the_shell(
        #[case] request: Request,
        #[case] status: StatusCode,
    ) {
        let response = app(1.0).oneshot(request).await.unwrap();
        assert_eq!(response.status(), status);
        assert_shell_headers(response.headers());
    }

    async fn answered(response: axum::response::Response) -> (StatusCode, Value) {
        let status = response.status();
        let body = axum::body::to_bytes(response.into_body(), 4096)
            .await
            .unwrap();
        (status, serde_json::from_slice(&body).unwrap())
    }

    #[rstest]
    #[tokio::test]
    async fn a_body_over_the_limit_is_rejected() {
        let request = Request::post("/v1/messages")
            .header(header::CONTENT_LENGTH, 2 * 1024 * 1024)
            .body(Body::empty())
            .unwrap();
        assert_eq!(
            answered(app(1.0).oneshot(request).await.unwrap()).await,
            (
                StatusCode::PAYLOAD_TOO_LARGE,
                json!({"error": "Request size is too large. Max size is 1 MB"})
            )
        );
    }

    #[rstest]
    #[tokio::test]
    async fn a_handler_panic_is_an_internal_server_error() {
        async fn panics() -> &'static str {
            panic!("handler bug")
        }
        let app = global(
            Router::new().route("/v1/messages", post(panics)),
            &Settings::default(),
            premium(),
            &InFlight::new(),
        );
        let request = Request::post("/v1/messages").body(Body::empty()).unwrap();
        assert_eq!(
            answered(app.oneshot(request).await.unwrap()).await,
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                json!({"error": {"message": "Internal server error", "type": "internal_server_error"}})
            )
        );
    }

    #[rstest]
    #[tokio::test]
    async fn a_request_over_capacity_is_shed() {
        let app = global(
            Router::new().route("/v1/messages", post(|| async { "ok" })),
            &Settings {
                admission_control: admission_control::Settings::from_general(
                    &litellm_config::GeneralSettings {
                        max_in_flight_requests_per_worker: Some(1),
                        max_queued_requests_per_worker: Some(0),
                        ..Default::default()
                    },
                ),
                ..Settings::default()
            },
            premium(),
            &InFlight::new(),
        );
        let request = || Request::post("/v1/messages").body(Body::empty()).unwrap();
        let _unfinished = app.clone().oneshot(request()).await.unwrap();
        assert_eq!(
            answered(app.oneshot(request()).await.unwrap()).await,
            (
                StatusCode::SERVICE_UNAVAILABLE,
                json!({"error": {
                    "message": "Worker at capacity: 1 in-flight, 0 queued requests. Retry later.",
                    "type": "overloaded_error",
                    "code": "503",
                }})
            )
        );
    }

    #[rstest]
    #[tokio::test]
    async fn a_limit_above_axums_default_is_the_one_enforced() {
        let request = Request::post("/v1/messages")
            .body(Body::from(vec![b'a'; 3 * 1024 * 1024]))
            .unwrap();
        let response = app(4.0).oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }
}
