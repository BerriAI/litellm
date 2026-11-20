use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use axum::{
    Extension, Json,
    body::Body,
    extract::{Request, State},
    http::{StatusCode, header},
    middleware::Next,
    response::{IntoResponse, Response},
};
use http_body_util::{BodyExt, LengthLimitError, Limited};
use litellm_config::GeneralSettings;
use litellm_license::Entitlements;
use serde_json::json;

#[derive(Debug, Default)]
pub struct Settings {
    pub max_request_size_mb: Option<f64>,
}

impl Settings {
    pub fn from_general(general: &GeneralSettings) -> Self {
        Self {
            max_request_size_mb: general.max_request_size_mb,
        }
    }
}

/// `_mb_to_bytes`, Python's `int(megabytes * 1024 * 1024)`.
pub(super) fn bytes(megabytes: f64) -> u64 {
    (megabytes * 1024.0 * 1024.0) as u64
}

/// `effective_max_request_size_mb`: the configured limit applies only with a premium license.
pub(super) struct SizeLimits {
    configured: Option<f64>,
}

impl SizeLimits {
    pub(super) fn new(settings: &Settings) -> Self {
        Self {
            configured: settings.max_request_size_mb,
        }
    }

    pub(super) fn megabytes(&self, entitlements: &Entitlements) -> Option<f64> {
        self.configured
            .filter(|limit| *limit > 0.0 && entitlements.is_enterprise())
    }
}

#[derive(Default)]
struct Reading {
    overflowed: AtomicBool,
    by_middleware: AtomicBool,
}

/// A request extension tracking the limited body. Python answers an overflow by where the
/// body was being read: FastAPI's `HTTPException` handler when a route was reading, and
/// `RequestSizeLimitMiddleware` itself when a middleware ahead of FastAPI was.
#[derive(Clone, Default)]
pub(super) struct Overflow(Arc<Reading>);

impl Overflow {
    pub(super) fn read_by_middleware(&self) {
        self.0.by_middleware.store(true, Ordering::Relaxed);
    }

    fn limit(self, body: Body, max_bytes: u64) -> Body {
        let max_bytes = usize::try_from(max_bytes).unwrap_or(usize::MAX);
        Body::new(Limited::new(body, max_bytes).map_err(move |error| {
            if error.is::<LengthLimitError>() {
                self.0.overflowed.store(true, Ordering::Relaxed);
            }
            error
        }))
    }

    fn rejection(&self, megabytes: f64) -> Option<Response> {
        if !self.0.overflowed.load(Ordering::Relaxed) {
            return None;
        }
        Some(if self.0.by_middleware.load(Ordering::Relaxed) {
            too_large(megabytes)
        } else {
            too_large_as(megabytes, "detail")
        })
    }
}

/// `RequestSizeLimitMiddleware`. `tower_http::limit` answers in plain text, so this wraps the body
/// in the same `http_body_util::Limited` itself and replaces the response to a body that overflows
/// before the response starts, see [`Overflow`]. Python's `_ROUTE_MAX_REQUEST_SIZE_MB` upload limits
/// are not ported because those routes have no native route and are always served by Python.
pub(super) async fn limit_body(
    State(limits): State<Arc<SizeLimits>>,
    Extension(entitlements): Extension<Arc<Entitlements>>,
    mut request: Request,
    next: Next,
) -> Response {
    let Some(megabytes) = limits.megabytes(&entitlements) else {
        return next.run(request).await;
    };
    let max_bytes = bytes(megabytes);
    if content_length(&request).is_some_and(|length| length > max_bytes) {
        return too_large(megabytes);
    }
    let overflow = Overflow::default();
    request.extensions_mut().insert(overflow.clone());
    let request = request.map(|body| overflow.clone().limit(body, max_bytes));
    let response = next.run(request).await;
    overflow.rejection(megabytes).unwrap_or(response)
}

pub(super) fn content_length(request: &Request) -> Option<u64> {
    request
        .headers()
        .get(header::CONTENT_LENGTH)?
        .to_str()
        .ok()?
        .parse()
        .ok()
}

/// `send_request_too_large`, whose OpenAI body is a bare message.
pub(super) fn too_large(megabytes: f64) -> Response {
    too_large_as(megabytes, "error")
}

fn too_large_as(megabytes: f64, openai_key: &str) -> Response {
    let message = format!("Request size is too large. Max size is {megabytes} MB");
    (
        StatusCode::PAYLOAD_TOO_LARGE,
        Json(json!({openai_key: message})),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use std::convert::Infallible;

    use axum::{
        Router, body::Bytes, extract::DefaultBodyLimit, middleware::from_fn_with_state,
        routing::post,
    };
    use futures_util::stream;
    use rstest::{fixture, rstest};
    use serde_json::Value;
    use tower::ServiceExt;

    use super::*;

    const MB: usize = 1024 * 1024;
    const PREMIUM: Entitlements = Entitlements::ServerVerified;

    fn limits(configured: Option<f64>) -> SizeLimits {
        SizeLimits::new(&Settings {
            max_request_size_mb: configured,
        })
    }

    #[rstest]
    #[case::community_ignores_the_setting(Some(1.0), Entitlements::Community, None)]
    #[case::premium_applies_the_setting(Some(1.0), PREMIUM, Some(1.0))]
    #[case::fractional_setting(Some(0.5), PREMIUM, Some(0.5))]
    #[case::zero_means_no_limit(Some(0.0), PREMIUM, None)]
    #[case::negative_means_no_limit(Some(-1.0), PREMIUM, None)]
    #[case::unset(None, PREMIUM, None)]
    fn the_configured_limit_needs_a_premium_license(
        #[case] configured: Option<f64>,
        #[case] entitlements: Entitlements,
        #[case] expected: Option<f64>,
    ) {
        assert_eq!(limits(configured).megabytes(&entitlements), expected);
    }

    #[fixture]
    fn app(#[default(Some(1.0))] configured: Option<f64>) -> Router {
        let length = |body: Bytes| async move { body.len().to_string() };
        Router::new()
            .route("/v1/messages", post(length))
            .route(
                "/echo",
                post(|request: Request| async move { Response::new(request.into_body()) }),
            )
            .layer(DefaultBodyLimit::disable())
            .layer(from_fn_with_state(Arc::new(limits(configured)), limit_body))
            .layer(Extension(Arc::new(PREMIUM)))
    }

    fn streamed(bytes: usize) -> Body {
        let chunk = Bytes::from(vec![b'a'; 64 * 1024]);
        Body::from_stream(stream::iter(
            std::iter::repeat_n(chunk, bytes.div_ceil(64 * 1024)).map(Ok::<_, Infallible>),
        ))
    }

    async fn assert_too_large(response: Response, megabytes: &str, openai_key: &str) {
        let message = format!("Request size is too large. Max size is {megabytes} MB");
        assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
        let body = axum::body::to_bytes(response.into_body(), 4096)
            .await
            .unwrap();
        assert_eq!(
            serde_json::from_slice::<Value>(&body).unwrap(),
            json!({openai_key: message})
        );
    }

    #[rstest]
    #[tokio::test]
    async fn a_declared_length_over_the_limit_is_rejected_unread(app: Router) {
        let request = Request::post("/v1/messages")
            .header(header::CONTENT_LENGTH, MB + 1)
            .body(Body::from_stream(stream::pending::<
                Result<Bytes, Infallible>,
            >()))
            .unwrap();
        assert_too_large(app.oneshot(request).await.unwrap(), "1", "error").await;
    }

    #[rstest]
    #[tokio::test]
    async fn a_fractional_limit_is_enforced_and_named(#[with(Some(0.5))] app: Router) {
        let request = Request::post("/v1/messages")
            .header(header::CONTENT_LENGTH, MB / 2 + 1)
            .body(Body::empty())
            .unwrap();
        assert_too_large(app.oneshot(request).await.unwrap(), "0.5", "error").await;
    }

    #[rstest]
    #[tokio::test]
    async fn a_body_a_route_reads_past_the_limit_gets_fastapis_rejection(app: Router) {
        let request = Request::post("/v1/messages")
            .body(streamed(2 * MB))
            .unwrap();
        assert_too_large(app.oneshot(request).await.unwrap(), "1", "detail").await;
    }

    #[rstest]
    #[tokio::test]
    async fn a_body_at_the_limit_passes(app: Router) {
        let request = Request::post("/v1/messages").body(streamed(MB)).unwrap();
        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), 64)
            .await
            .unwrap();
        assert_eq!(body, MB.to_string());
    }

    #[rstest]
    #[tokio::test]
    async fn an_overflow_after_the_response_started_keeps_that_response(app: Router) {
        let request = Request::post("/echo").body(streamed(2 * MB)).unwrap();
        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert!(
            axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .is_err()
        );
    }
}
