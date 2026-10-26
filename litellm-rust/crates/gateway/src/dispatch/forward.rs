use std::{net::SocketAddr, sync::Arc};

use axum::{
    body::Body,
    extract::{ConnectInfo, OriginalUri, Request},
    http::{
        HeaderMap, HeaderName, HeaderValue, Method, StatusCode, Uri, Version,
        header::{
            CONNECTION, PROXY_AUTHENTICATE, PROXY_AUTHORIZATION, TE, TRAILER, TRANSFER_ENCODING,
            UPGRADE,
        },
        uri::{Authority, InvalidUri, PathAndQuery, Scheme},
    },
    response::{IntoResponse, Response},
};
use hyper_util::{
    client::legacy::{Client, connect::HttpConnector},
    rt::TokioExecutor,
};
use litellm_core_utils::decline::Decline;

use crate::upgrades::{HANDSHAKE_TIMEOUT, Upgrades};

/// Python reads it to log why the gateway handed the request over.
const DECLINED_HEADER: HeaderName = HeaderName::from_static("x-litellm-gateway-declined");

const X_FORWARDED_FOR: HeaderName = HeaderName::from_static("x-forwarded-for");

const KEEP_ALIVE: HeaderName = HeaderName::from_static("keep-alive");

const HOP_BY_HOP: [HeaderName; 8] = [
    CONNECTION,
    KEEP_ALIVE,
    PROXY_AUTHENTICATE,
    PROXY_AUTHORIZATION,
    TE,
    TRAILER,
    TRANSFER_ENCODING,
    UPGRADE,
];

#[derive(Debug, thiserror::Error)]
#[error("the Python upstream must be an http:// origin, got {upstream}")]
pub struct InvalidUpstream {
    upstream: String,
    #[source]
    cause: Option<InvalidUri>,
}

/// Reverse proxy to the Python proxy on loopback, with its own client: provider clients carry
/// proxy and TLS settings that must not apply here.
#[derive(Clone)]
pub struct Forward {
    client: Arc<Client<HttpConnector, Body>>,
    authority: Authority,
    trusted_proxies: Option<crate::proxy_headers::TrustedProxies>,
}

impl Forward {
    pub fn new(upstream: &str) -> Result<Self, InvalidUpstream> {
        let invalid = |cause| InvalidUpstream {
            upstream: upstream.to_owned(),
            cause,
        };
        let uri: Uri = upstream.parse().map_err(|cause| invalid(Some(cause)))?;
        let origin_only = uri.path_and_query().is_none_or(|path| path.as_str() == "/");
        let authority = match (uri.scheme(), uri.authority()) {
            (Some(scheme), Some(authority)) if *scheme == Scheme::HTTP && origin_only => {
                authority.clone()
            }
            _ => return Err(invalid(None)),
        };
        Ok(Self {
            client: Arc::new(Client::builder(TokioExecutor::new()).build(HttpConnector::new())),
            authority,
            trusted_proxies: None,
        })
    }

    pub fn with_trusted_proxies(mut self, proxies: &str) -> Result<Self, crate::Error> {
        self.trusted_proxies = Some(crate::proxy_headers::TrustedProxies::parse(proxies)?);
        Ok(self)
    }

    pub(super) async fn send(&self, mut request: Request, decline: Decline) -> Response {
        let websocket = if request.method() == Method::GET
            && request.version() == Version::HTTP_11
            && is_websocket(request.headers())
        {
            let Some(lifecycle) = request.extensions().get::<Upgrades>().cloned() else {
                return (
                    StatusCode::SERVICE_UNAVAILABLE,
                    "WebSocket forwarding requires litellm_gateway::serve",
                )
                    .into_response();
            };
            if lifecycle.is_draining() {
                return StatusCode::SERVICE_UNAVAILABLE.into_response();
            }
            if request
                .extensions()
                .get::<hyper::upgrade::OnUpgrade>()
                .is_none()
            {
                return (
                    StatusCode::SERVICE_UNAVAILABLE,
                    "WebSocket forwarding requires an HTTP upgrade transport",
                )
                    .into_response();
            }
            Some((hyper::upgrade::on(&mut request), lifecycle))
        } else {
            None
        };
        let (mut parts, body) = request.into_parts();
        let path = parts
            .extensions
            .get::<OriginalUri>()
            .map_or(&parts.uri, |original| &original.0)
            .path_and_query()
            .cloned()
            .unwrap_or_else(|| PathAndQuery::from_static("/"));
        parts.uri = Uri::from_parts({
            let mut uri = axum::http::uri::Parts::default();
            uri.scheme = Some(Scheme::HTTP);
            uri.authority = Some(self.authority.clone());
            uri.path_and_query = Some(path);
            uri
        })
        .expect("scheme, authority and path make an absolute uri");
        parts.version = Version::HTTP_11;
        strip_hop_by_hop(&mut parts.headers);
        if websocket.is_some() {
            websocket_headers(&mut parts.headers);
        }
        let peer = parts
            .extensions
            .get::<ConnectInfo<SocketAddr>>()
            .map(|info| info.0);
        if let Some(policy) = &self.trusted_proxies {
            policy.apply(&mut parts.headers, peer);
        } else if let Some(client) = peer {
            append_forwarded_for(&mut parts.headers, client);
        }
        parts
            .headers
            .insert(DECLINED_HEADER, HeaderValue::from_static(decline.reason));
        tracing::info!(
            reason = decline.reason,
            path = parts.uri.path(),
            "forwarded to python"
        );
        let response = self.client.request(Request::from_parts(parts, body));
        let response = if let Some((_, lifecycle)) = &websocket {
            tokio::select! {
                _ = lifecycle.cancelled() => return StatusCode::SERVICE_UNAVAILABLE.into_response(),
                result = tokio::time::timeout(HANDSHAKE_TIMEOUT, response) => match result {
                    Ok(result) => result,
                    Err(_) => return StatusCode::GATEWAY_TIMEOUT.into_response(),
                }
            }
        } else {
            response.await
        };
        match response {
            Ok(mut response) => {
                if response.status() == StatusCode::SWITCHING_PROTOCOLS {
                    let Some((downstream, lifecycle)) = websocket else {
                        return StatusCode::BAD_GATEWAY.into_response();
                    };
                    if !is_websocket(response.headers()) {
                        return StatusCode::BAD_GATEWAY.into_response();
                    }
                    let upstream = hyper::upgrade::on(&mut response);
                    let upstream = tokio::select! {
                        _ = lifecycle.cancelled() => {
                            return StatusCode::SERVICE_UNAVAILABLE.into_response();
                        }
                        result = tokio::time::timeout(HANDSHAKE_TIMEOUT, upstream) => {
                            match result {
                                Ok(Ok(upstream)) => upstream,
                                Ok(Err(_)) => return StatusCode::BAD_GATEWAY.into_response(),
                                Err(_) => return StatusCode::GATEWAY_TIMEOUT.into_response(),
                            }
                        }
                    };
                    let (mut parts, _) = response.into_parts();
                    strip_hop_by_hop(&mut parts.headers);
                    websocket_headers(&mut parts.headers);
                    lifecycle.relay(downstream, upstream);
                    return Response::from_parts(parts, Body::empty());
                }
                let (mut parts, body) = response.into_parts();
                strip_hop_by_hop(&mut parts.headers);
                Response::from_parts(parts, Body::new(body))
            }
            Err(error) => {
                tracing::warn!(connect = error.is_connect(), "python upstream unreachable");
                StatusCode::BAD_GATEWAY.into_response()
            }
        }
    }
}

fn is_websocket(headers: &HeaderMap) -> bool {
    headers
        .get(UPGRADE)
        .is_some_and(|value| value.as_bytes().eq_ignore_ascii_case(b"websocket"))
        && headers.get_all(CONNECTION).iter().any(|value| {
            value.to_str().is_ok_and(|value| {
                value
                    .split(',')
                    .any(|token| token.trim().eq_ignore_ascii_case("upgrade"))
            })
        })
}

fn websocket_headers(headers: &mut HeaderMap) {
    headers.insert(CONNECTION, HeaderValue::from_static("upgrade"));
    headers.insert(UPGRADE, HeaderValue::from_static("websocket"));
}

/// Python sees the gateway as its peer, so the client's address goes on the end of the chain
/// any proxy in front of the gateway already started.
fn append_forwarded_for(headers: &mut HeaderMap, client: SocketAddr) {
    let client = client.ip().to_string();
    let chain = headers
        .get_all(&X_FORWARDED_FOR)
        .iter()
        .filter_map(|value| value.to_str().ok())
        .chain([client.as_str()])
        .collect::<Vec<_>>()
        .join(", ");
    if let Ok(chain) = HeaderValue::from_str(&chain) {
        headers.insert(X_FORWARDED_FOR, chain);
    }
}

fn strip_hop_by_hop(headers: &mut HeaderMap) {
    let named: Vec<HeaderName> = headers
        .get_all(CONNECTION)
        .iter()
        .filter_map(|value| value.to_str().ok())
        .flat_map(|value| value.split(','))
        .filter_map(|name| HeaderName::try_from(name.trim()).ok())
        .collect();
    for name in HOP_BY_HOP.iter().chain(&named) {
        headers.remove(name);
    }
}

#[cfg(test)]
mod tests {
    use axum::http::uri::InvalidUri;
    use rstest::rstest;

    use super::Forward;

    #[rstest]
    #[case::https("https://127.0.0.1:4000", false)]
    #[case::with_path("http://127.0.0.1:4000/v1", false)]
    #[case::no_scheme("127.0.0.1:4000", false)]
    #[case::unparsable("http://exa mple", true)]
    fn only_an_http_origin_is_a_python_upstream(#[case] upstream: &str, #[case] unparsable: bool) {
        let Err(error) = Forward::new(upstream) else {
            panic!("{upstream} must be rejected");
        };
        let report = litellm_error::Report::new(&error).to_string();
        assert!(report.contains(upstream), "{report}");
        assert_eq!(
            litellm_error::find::<InvalidUri>(&error).is_some(),
            unparsable
        );
    }

    #[rstest]
    #[case::bare("http://127.0.0.1:4000")]
    #[case::root("http://127.0.0.1:4000/")]
    fn an_http_origin_is_a_python_upstream(#[case] upstream: &str) {
        assert!(Forward::new(upstream).is_ok());
    }
}
