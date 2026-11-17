use std::{fmt, io::Write, net::SocketAddr, sync::Arc, time::Duration};

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
use litellm_config::{Config, Environment};
use litellm_gateway::middleware;
use litellm_gateway_forward::ANSWERED_BY_HEADER;
use rstest::rstest;
use tokio::{net::TcpListener, sync::mpsc};
use tower::Layer;

const CALL_ID: &str = "x-litellm-call-id";

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

/// The native gateway under `root_path` with its global middleware, in front of `python`,
/// admitting every request or refusing every one.
async fn gateway(python: SocketAddr, admits: bool, root_path: &str) -> SocketAddr {
    let config = Config::from_yaml("model_list: []").unwrap();
    let native = middleware::root_path::layer(middleware::root_path::Settings::from_environment(
        &Environment {
            server_root_path: root_path.into(),
            ..Environment::default()
        },
    ))
    .layer(middleware::global(
        litellm_gateway::router(
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
        ),
        &middleware::Settings::default(),
        Arc::new(litellm_license::Entitlements::Community),
        &middleware::InFlight::new(),
    ));
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
#[case::no_native_route(Method::POST, "", "/only/python")]
#[case::wrong_method_on_a_native_route(Method::PUT, "", "/v1/chat/completions")]
#[case::native_route(Method::POST, "", "/v1/chat/completions")]
#[case::native_route_under_a_root_path(Method::POST, "/api", "/api/v1/chat/completions")]
#[tokio::test]
async fn python_answers_a_refused_request_as_the_client_sent_it(
    #[case] method: Method,
    #[case] root_path: &str,
    #[case] path: &str,
) {
    let (upstream, mut received) = python().await;
    let gateway = gateway(upstream, false, root_path).await;
    let gzipped = gzip(br#"{"model":"m"}"#);

    let response = send(gateway, method, path, &gzipped).await;

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers()[ANSWERED_BY_HEADER],
        "python; reason=test"
    );
    assert!(!response.headers().contains_key(CALL_ID));
    assert!(!response.headers().contains_key(header::X_FRAME_OPTIONS));
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
#[case::at_the_root("", "/v1/chat/completions")]
#[case::under_a_root_path("/api", "/api/v1/chat/completions")]
#[tokio::test]
async fn an_admitted_request_gets_the_native_layers_and_never_reaches_python(
    #[case] root_path: &str,
    #[case] path: &str,
) {
    let (upstream, mut received) = python().await;
    let gateway = gateway(upstream, true, root_path).await;

    let response = send(gateway, Method::POST, path, &gzip(br#"{"model":"m"}"#)).await;

    assert_eq!(response.headers()[ANSWERED_BY_HEADER], "native");
    assert!(response.headers().contains_key(CALL_ID));
    assert_eq!(response.headers()[header::X_FRAME_OPTIONS], "DENY");
    assert!(received.try_recv().is_err());
}

#[rstest]
#[tokio::test]
async fn an_admitted_request_gets_the_native_404_when_no_route_takes_it() {
    let (upstream, mut received) = python().await;
    let gateway = gateway(upstream, true, "").await;

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
