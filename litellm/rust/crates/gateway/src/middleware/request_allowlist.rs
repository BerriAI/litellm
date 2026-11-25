//! The proxy's address and route allowlists, which it checks before it reads a credential.

use std::{net::SocketAddr, sync::Arc};

use axum::{
    Json,
    extract::{ConnectInfo, Request, State},
    http::{Extensions, HeaderMap, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
};
use litellm_config::GeneralSettings;
use litellm_gateway_auth_core::{Allowlists, RequestDenial, check_request, client_address::Origin};
use serde_json::json;

const FORWARDED_FOR: &str = "x-forwarded-for";

/// `general_settings.allowed_ips`, `use_x_forwarded_for` and `allowed_routes`, as
/// `gateway-auth-core` reads them.
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

/// `pre_db_read_auth_checks`' address and route checks, which Python's authentication runs, so
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
    match checked(
        &allowlists,
        request.uri().path(),
        request.headers(),
        request.extensions(),
    ) {
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
pub(crate) fn checked(
    allowlists: &Allowlists,
    route: &str,
    headers: &HeaderMap,
    extensions: &Extensions,
) -> Result<(), RequestDenial> {
    if !allowlists.client_addresses.reads_address() {
        return check_request(route, Origin::default(), allowlists);
    }
    let peer = extensions
        .get::<ConnectInfo<SocketAddr>>()
        .map(|ConnectInfo(peer)| peer.ip().to_string());
    let forwarded_for = headers
        .get(FORWARDED_FOR)
        .map(|value| latin1(value.as_bytes()));
    let origin = Origin {
        peer: peer.as_deref(),
        forwarded_for: forwarded_for.as_deref(),
    };
    check_request(route, origin, allowlists)
}

/// Starlette decodes a header's bytes as Latin-1.
fn latin1(bytes: &[u8]) -> String {
    bytes.iter().copied().map(char::from).collect()
}

#[cfg(test)]
mod tests {
    use axum::{Router, body::Body, middleware::from_fn_with_state, routing::post};
    use litellm_config::Config;
    use rstest::rstest;
    use tower::ServiceExt;

    use super::*;

    async fn answered(
        yaml: &str,
        path: &str,
        peer: &str,
        forwarded_for: Option<&[u8]>,
    ) -> (StatusCode, serde_json::Value) {
        let config = Config::from_yaml(yaml).unwrap();
        let allowlists = Arc::new(Settings::from_general(&config.general_settings).allowlists);
        let app = Router::new()
            .route("/v1/messages", post(|| async { Json(json!("served")) }))
            .route("/health", post(|| async { Json(json!("served")) }))
            .route_layer(from_fn_with_state(allowlists, refuse_unlisted));
        let request = forwarded_for
            .into_iter()
            .fold(Request::post(path), |request, value| {
                request.header(FORWARDED_FOR, value)
            })
            .extension(ConnectInfo(SocketAddr::new(peer.parse().unwrap(), 4000)))
            .body(Body::empty())
            .unwrap();
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
    const ADDRESSES: &str = "general_settings:\n  allowed_ips: [10.0.0.1, '::1']\n";
    const FORWARDED: &str =
        "general_settings:\n  allowed_ips: [10.0.0.9, 'caf\u{e9}']\n  use_x_forwarded_for: true\n";

    #[rstest]
    #[case::no_allowlist("model_list: []\n", "/v1/messages", "10.0.0.9", None, served())]
    #[case::a_listed_route(ROUTES, "/health", "10.0.0.9", None, served())]
    #[case::an_unlisted_route(
        ROUTES,
        "/v1/messages",
        "10.0.0.9",
        None,
        refused("Access forbidden: Route /v1/messages not allowed")
    )]
    #[case::a_listed_peer(ADDRESSES, "/v1/messages", "10.0.0.1", None, served())]
    #[case::a_listed_ipv6_peer(ADDRESSES, "/v1/messages", "::1", None, served())]
    #[case::an_unlisted_peer(
        ADDRESSES,
        "/v1/messages",
        "10.0.0.9",
        None,
        refused("Access forbidden: IP address 10.0.0.9 not allowed.")
    )]
    #[case::a_forwarded_header_read_when_the_operator_says_so(
        FORWARDED,
        "/v1/messages",
        "10.0.0.1",
        Some(b"10.0.0.9".as_slice()),
        served()
    )]
    #[case::a_forwarded_header_that_is_not_listed(
        FORWARDED,
        "/v1/messages",
        "10.0.0.9",
        Some(b"10.0.0.1".as_slice()),
        refused("Access forbidden: IP address 10.0.0.1 not allowed.")
    )]
    #[case::a_forwarded_header_read_as_latin1(
        FORWARDED,
        "/v1/messages",
        "10.0.0.1",
        Some(b"caf\xe9".as_slice()),
        served()
    )]
    #[tokio::test]
    async fn a_request_the_allowlists_leave_out_is_refused_before_its_handler(
        #[case] yaml: &str,
        #[case] path: &str,
        #[case] peer: &str,
        #[case] forwarded_for: Option<&[u8]>,
        #[case] expected: (StatusCode, serde_json::Value),
    ) {
        assert_eq!(answered(yaml, path, peer, forwarded_for).await, expected);
    }
}
