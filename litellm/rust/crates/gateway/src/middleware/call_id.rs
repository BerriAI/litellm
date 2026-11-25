use std::convert::Infallible;

use axum::{
    extract::{FromRequestParts, Request},
    http::{HeaderName, request::Parts},
};
use tower::util::MapRequestLayer;
use tower_http::request_id::{
    MakeRequestUuid, PropagateRequestIdLayer, RequestId, SetRequestIdLayer,
};

const CALL_ID: HeaderName = HeaderName::from_static("x-litellm-call-id");
const MAX_CALL_ID_LENGTH: usize = 256;

type DropUnusableCallId = MapRequestLayer<fn(Request) -> Request>;

/// `resolve_litellm_call_id`: a client's `x-litellm-call-id` of 1 to 256 characters is kept,
/// anything else is replaced by a fresh UUID, and the response echoes the id.
pub(super) fn layer() -> (
    DropUnusableCallId,
    SetRequestIdLayer<MakeRequestUuid>,
    PropagateRequestIdLayer,
) {
    (
        MapRequestLayer::new(drop_unusable_call_id as fn(Request) -> Request),
        SetRequestIdLayer::new(CALL_ID, MakeRequestUuid),
        PropagateRequestIdLayer::new(CALL_ID),
    )
}

/// The call id [`layer`] resolved, `None` on a route that does not wear it.
pub(super) struct CallId(pub(super) Option<String>);

impl<S: Send + Sync> FromRequestParts<S> for CallId {
    type Rejection = Infallible;

    async fn from_request_parts(parts: &mut Parts, _: &S) -> Result<Self, Infallible> {
        Ok(Self(
            parts
                .extensions
                .get::<RequestId>()
                .and_then(|id| id.header_value().to_str().ok())
                .map(str::to_owned),
        ))
    }
}

fn drop_unusable_call_id(mut request: Request) -> Request {
    let usable = request
        .headers()
        .get(CALL_ID)
        .is_none_or(|value| (1..=MAX_CALL_ID_LENGTH).contains(&value.len()));
    if !usable {
        request.headers_mut().remove(CALL_ID);
    }
    request
}

#[cfg(test)]
mod tests {
    use axum::{Router, body::Body, routing::get};
    use rstest::rstest;
    use tower::ServiceExt;
    use uuid::Uuid;

    use super::*;

    async fn call(call_id: Option<&str>) -> (String, String) {
        let app = Router::new()
            .route(
                "/",
                get(|CallId(call_id): CallId| async move { call_id.unwrap() }),
            )
            .layer(layer());
        let request = call_id
            .into_iter()
            .fold(Request::get("/"), |request, id| request.header(CALL_ID, id))
            .body(Body::empty())
            .unwrap();
        let response = app.oneshot(request).await.unwrap();
        let echoed = response.headers()[CALL_ID].to_str().unwrap().to_owned();
        let seen = axum::body::to_bytes(response.into_body(), 1024)
            .await
            .unwrap();
        (String::from_utf8(seen.to_vec()).unwrap(), echoed)
    }

    #[rstest]
    #[case::short("call-1")]
    #[case::longest(&"a".repeat(MAX_CALL_ID_LENGTH))]
    #[tokio::test]
    async fn a_usable_client_call_id_is_kept_and_echoed(#[case] call_id: &str) {
        assert_eq!(call(Some(call_id)).await, (call_id.into(), call_id.into()));
    }

    #[rstest]
    #[case::missing(None)]
    #[case::empty(Some(String::new()))]
    #[case::too_long(Some("a".repeat(MAX_CALL_ID_LENGTH + 1)))]
    #[tokio::test]
    async fn an_unusable_call_id_is_replaced_by_a_uuid(#[case] call_id: Option<String>) {
        let (seen, echoed) = call(call_id.as_deref()).await;
        assert_eq!(seen, echoed);
        assert!(Uuid::parse_str(&seen).is_ok(), "{seen}");
    }
}
