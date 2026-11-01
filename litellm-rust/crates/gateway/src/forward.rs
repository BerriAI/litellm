use std::net::{Ipv4Addr, SocketAddr};

use axum::{
    Router,
    body::Body,
    extract::{ConnectInfo, DefaultBodyLimit, Request, State},
    http::{HeaderMap, HeaderValue, StatusCode, Uri, header},
    response::Response,
};
use hyper_util::{
    client::legacy::{Client, connect::HttpConnector},
    rt::TokioExecutor,
};

use crate::Error;

#[derive(Clone)]
pub struct Forward {
    upstream: Uri,
    client: Client<HttpConnector, Body>,
}

impl Forward {
    pub fn new(upstream: &str) -> Result<Self, Error> {
        let upstream: Uri = upstream.parse().map_err(Error::ForwardUri)?;
        if upstream.scheme_str() != Some("http")
            || !upstream
                .host()
                .and_then(|host| host.parse::<Ipv4Addr>().ok())
                .is_some_and(|ip| ip.is_loopback())
            || upstream
                .authority()
                .is_some_and(|authority| authority.as_str().contains('@'))
            || upstream.path() != "/"
            || upstream.query().is_some()
        {
            return Err(Error::ForwardUpstream);
        }
        let client = Client::builder(TokioExecutor::new())
            .retry_canceled_requests(false)
            .build_http();
        Ok(Self { upstream, client })
    }

    pub fn router(self) -> Router {
        Router::new()
            .fallback(forward)
            .layer(DefaultBodyLimit::disable())
            .with_state(self)
    }
}

async fn forward(
    State(state): State<Forward>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    mut request: Request,
) -> Result<Response, StatusCode> {
    let mut uri = state.upstream.into_parts();
    uri.path_and_query = request.uri().path_and_query().cloned();
    *request.uri_mut() = Uri::from_parts(uri).map_err(|_| StatusCode::BAD_GATEWAY)?;
    *request.version_mut() = axum::http::Version::HTTP_11;
    let headers = request.headers_mut();
    strip_hop_headers(headers);
    let forwarded = headers
        .get_all("x-forwarded-for")
        .iter()
        .map(|value| value.as_bytes())
        .collect::<Vec<_>>();
    let mut forwarded = forwarded.join(b", ".as_slice());
    if !forwarded.is_empty() {
        forwarded.extend_from_slice(b", ");
    }
    forwarded.extend_from_slice(peer.ip().to_string().as_bytes());
    headers.insert(
        "x-forwarded-for",
        HeaderValue::from_bytes(&forwarded).map_err(|_| StatusCode::BAD_REQUEST)?,
    );
    if !headers.contains_key("x-forwarded-proto") {
        headers.insert("x-forwarded-proto", HeaderValue::from_static("http"));
    }
    let mut response = state
        .client
        .request(request)
        .await
        .map_err(|_| StatusCode::BAD_GATEWAY)?;
    strip_hop_headers(response.headers_mut());
    Ok(response.map(Body::new))
}

fn strip_hop_headers(headers: &mut HeaderMap) {
    let connection_headers: Vec<_> = headers
        .get_all(header::CONNECTION)
        .iter()
        .filter_map(|value| value.to_str().ok())
        .flat_map(|value| value.split(','))
        .filter_map(|name| header::HeaderName::from_bytes(name.trim().as_bytes()).ok())
        .collect();
    for name in connection_headers {
        headers.remove(name);
    }
    for name in [
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
        "proxy-connection",
    ] {
        headers.remove(name);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{body::Bytes, http::Version};
    use http_body_util::BodyExt;
    use rstest::rstest;
    use std::{convert::Infallible, time::Duration};
    use tokio::{net::TcpListener, sync::mpsc};
    use tower::ServiceExt;

    #[rstest]
    #[case::https("https://127.0.0.1:4000")]
    #[case::public("http://192.0.2.1:4000")]
    #[case::hostname("http://localhost:4000")]
    #[case::ipv6("http://[::1]:4000")]
    #[case::path("http://127.0.0.1:4000/api")]
    #[case::query("http://127.0.0.1:4000/?x=y")]
    #[case::credentials("http://user@127.0.0.1:4000")]
    fn rejects_invalid_upstream(#[case] upstream: &str) {
        assert!(Forward::new(upstream).is_err());
    }

    #[rstest]
    #[case::standard("http://127.0.0.1:4000")]
    #[case::loopback_range("http://127.0.0.2:4000/")]
    fn accepts_loopback_origin(#[case] upstream: &str) {
        assert!(Forward::new(upstream).is_ok());
    }

    #[rstest]
    #[tokio::test]
    async fn preserves_request_and_response_without_hop_headers() {
        let upstream = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let origin = format!("http://{}", upstream.local_addr().unwrap());
        let task = tokio::spawn(async move {
            axum::serve(
                upstream,
                Router::new().fallback(|request: Request| async {
                    assert_eq!(request.method(), "PATCH");
                    assert_eq!(request.uri(), "/v1/messages?x=a%2Fb&x=c");
                    assert_eq!(request.headers()["host"], "public.example");
                    assert_eq!(request.headers()["authorization"], "Bearer test");
                    assert_eq!(
                        request.headers()["x-forwarded-for"],
                        "192.0.2.1, 192.0.2.2, 198.51.100.5"
                    );
                    assert_eq!(request.headers()["x-forwarded-proto"], "https");
                    assert!(!request.headers().contains_key("x-remove"));
                    assert!(!request.headers().contains_key("proxy-authorization"));
                    assert_eq!(
                        request.into_body().collect().await.unwrap().to_bytes(),
                        "payload"
                    );
                    Response::builder()
                        .status(StatusCode::UNAUTHORIZED)
                        .header("connection", "x-upstream-hop")
                        .header("x-upstream-hop", "remove")
                        .header("x-response", "keep")
                        .body(Body::from("python error"))
                        .unwrap()
                }),
            )
            .await
            .unwrap();
        });
        let request = Request::builder()
            .method("PATCH")
            .version(Version::HTTP_2)
            .uri("/v1/messages?x=a%2Fb&x=c")
            .header("host", "public.example")
            .header("authorization", "Bearer test")
            .header("x-forwarded-for", "192.0.2.1")
            .header("x-forwarded-for", "192.0.2.2")
            .header("x-forwarded-proto", "https")
            .header("connection", "x-remove, keep-alive")
            .header("x-remove", "remove")
            .header("proxy-authorization", "remove")
            .extension(ConnectInfo(
                "198.51.100.5:1234".parse::<SocketAddr>().unwrap(),
            ))
            .body(Body::from("payload"))
            .unwrap();
        let response = Forward::new(&origin)
            .unwrap()
            .router()
            .oneshot(request)
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
        assert_eq!(response.headers()["x-response"], "keep");
        assert!(!response.headers().contains_key("connection"));
        assert!(!response.headers().contains_key("x-upstream-hop"));
        assert_eq!(
            response.into_body().collect().await.unwrap().to_bytes(),
            "python error"
        );
        task.abort();
    }

    #[rstest]
    #[case::health("/health/liveliness")]
    #[case::readiness("/health/readiness")]
    #[case::messages("/v1/messages")]
    #[tokio::test]
    async fn unreachable_upstream_is_502(#[case] path: &str) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let origin = format!("http://{}", listener.local_addr().unwrap());
        drop(listener);
        let response = Forward::new(&origin)
            .unwrap()
            .router()
            .oneshot(
                Request::builder()
                    .uri(path)
                    .extension(ConnectInfo(
                        "198.51.100.5:1234".parse::<SocketAddr>().unwrap(),
                    ))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    }

    #[rstest]
    #[tokio::test]
    async fn streams_upload_and_response_before_eof_without_body_limit() {
        let upstream = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let origin = format!("http://{}", upstream.local_addr().unwrap());
        let upstream_task = tokio::spawn(async move {
            axum::serve(
                upstream,
                Router::new().fallback(|request: Request| async {
                    assert_eq!(request.headers()["x-forwarded-for"], "127.0.0.1");
                    assert_eq!(request.headers()["x-forwarded-proto"], "http");
                    Response::new(request.into_body())
                }),
            )
            .await
            .unwrap();
        });
        let public = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let public_uri = format!("http://{}/chat/completions", public.local_addr().unwrap());
        let forward_task = tokio::spawn(async move {
            axum::serve(
                public,
                Forward::new(&origin)
                    .unwrap()
                    .router()
                    .into_make_service_with_connect_info::<SocketAddr>(),
            )
            .await
            .unwrap();
        });
        let (sender, receiver) = mpsc::channel::<Result<Bytes, Infallible>>(1);
        let stream = futures_util::stream::unfold(receiver, |mut receiver| async {
            receiver.recv().await.map(|chunk| (chunk, receiver))
        });
        sender.send(Ok(Bytes::from_static(b"first"))).await.unwrap();
        let client: Client<HttpConnector, Body> =
            Client::builder(TokioExecutor::new()).build_http();
        let mut response = tokio::time::timeout(
            Duration::from_secs(5),
            client.request(
                Request::post(public_uri)
                    .body(Body::from_stream(stream))
                    .unwrap(),
            ),
        )
        .await
        .unwrap()
        .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let frame = tokio::time::timeout(Duration::from_secs(5), response.body_mut().frame())
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(frame.into_data().unwrap(), "first");
        let large = Bytes::from(vec![b'x'; 3 * 1024 * 1024]);
        sender.send(Ok(large.clone())).await.unwrap();
        drop(sender);
        let rest = tokio::time::timeout(Duration::from_secs(5), response.into_body().collect())
            .await
            .unwrap()
            .unwrap()
            .to_bytes();
        assert_eq!(rest, large);
        upstream_task.abort();
        forward_task.abort();
    }
}
