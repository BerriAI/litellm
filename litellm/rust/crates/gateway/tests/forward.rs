use std::{fmt, io::Write, net::SocketAddr, time::Duration};

use axum::{
    Router,
    body::{Body, Bytes},
    extract::Request,
    http::{HeaderMap, Method, StatusCode, header},
    response::Response,
};
use flate2::{Compression, write::GzEncoder};
use http_body_util::BodyExt;
use hyper_util::{client::legacy::Client, rt::TokioExecutor};
use litellm_config::Config;
use litellm_gateway_forward::ANSWERED_BY_HEADER;
use rstest::rstest;
use tokio::{net::TcpListener, sync::mpsc};

struct Received {
    path: String,
    headers: HeaderMap,
    body: Bytes,
}

struct Refused;

impl fmt::Display for Refused {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("refused by the test")
    }
}

impl From<Refused> for &'static str {
    fn from(_: Refused) -> Self {
        "test"
    }
}

async fn python() -> (SocketAddr, mpsc::UnboundedReceiver<Received>) {
    let (sender, receiver) = mpsc::unbounded_channel();
    let app = Router::new().fallback(move |request: Request| {
        let sender = sender.clone();
        async move {
            let (parts, body) = request.into_parts();
            let body = body.collect().await.unwrap().to_bytes();
            sender
                .send(Received {
                    path: parts.uri.path().to_owned(),
                    headers: parts.headers,
                    body,
                })
                .unwrap();
            "python"
        }
    });
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (address, receiver)
}

/// The native gateway in front of `python`, admitting every request or refusing every one.
async fn gateway(python: SocketAddr, admits: bool) -> SocketAddr {
    let config = Config::from_yaml("model_list: []").unwrap();
    let native = litellm_gateway::router(
        litellm_gateway::AppState {
            inference: litellm_gateway::build_inference(
                &config,
                &litellm_gateway::Resources::build(&config).unwrap(),
            )
            .unwrap(),
            redis: litellm_gateway::RedisClients::default(),
            database: None,
        },
        &config,
    );
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(litellm_gateway_forward::serve(
        listener,
        format!("http://{python}").parse().unwrap(),
        Duration::from_secs(5),
        Duration::from_secs(1),
        native,
        move |request: Request| async move {
            if admits {
                Ok(request)
            } else {
                Err((Box::new(request), Refused))
            }
        },
        std::future::pending(),
    ));
    address
}

fn gzip(body: &[u8]) -> Vec<u8> {
    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(body).unwrap();
    encoder.finish().unwrap()
}

async fn send(gateway: SocketAddr, method: Method, path: &str, gzipped: &[u8]) -> Response {
    let request = Request::builder()
        .method(method)
        .uri(format!("http://{gateway}{path}"))
        .header(header::CONTENT_TYPE, "application/json")
        .header(header::CONTENT_ENCODING, "gzip")
        .body(Body::from(gzipped.to_vec()))
        .unwrap();
    let response = tokio::time::timeout(
        Duration::from_secs(5),
        Client::builder(TokioExecutor::new())
            .build_http()
            .request(request),
    )
    .await
    .unwrap()
    .unwrap();
    response.map(Body::new)
}

#[rstest]
#[case::no_native_route(Method::POST, "/only/python")]
#[case::wrong_method_on_a_native_route(Method::PUT, "/v1/chat/completions")]
#[case::native_route(Method::POST, "/v1/chat/completions")]
#[tokio::test]
async fn python_answers_a_refused_request_as_the_client_sent_it(
    #[case] method: Method,
    #[case] path: &str,
) {
    let (upstream, mut received) = python().await;
    let gateway = gateway(upstream, false).await;
    let gzipped = gzip(br#"{"model":"m"}"#);

    let response = send(gateway, method, path, &gzipped).await;

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers()[ANSWERED_BY_HEADER],
        "python; reason=test"
    );
    assert_eq!(
        response.into_body().collect().await.unwrap().to_bytes(),
        "python"
    );
    let python = received.recv().await.unwrap();
    assert_eq!(python.path, path);
    assert_eq!(python.headers[header::CONTENT_ENCODING], "gzip");
    assert_eq!(python.body, gzipped);
}

#[rstest]
#[tokio::test]
async fn an_admitted_request_is_answered_natively_and_never_reaches_python() {
    let (upstream, mut received) = python().await;
    let gateway = gateway(upstream, true).await;

    let response = send(
        gateway,
        Method::POST,
        "/v1/chat/completions",
        &gzip(br#"{"model":"m"}"#),
    )
    .await;

    assert_eq!(response.headers()[ANSWERED_BY_HEADER], "native");
    assert!(received.try_recv().is_err());
}

#[rstest]
#[tokio::test]
async fn an_admitted_request_gets_the_native_404_when_no_route_takes_it() {
    let (upstream, mut received) = python().await;
    let gateway = gateway(upstream, true).await;

    let response = send(
        gateway,
        Method::POST,
        "/only/python",
        &gzip(br#"{"model":"m"}"#),
    )
    .await;

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    assert_eq!(response.headers()[ANSWERED_BY_HEADER], "native");
    assert!(received.try_recv().is_err());
}
