mod support;

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use axum::{body::Body, extract::Request, http::StatusCode};
use http_body_util::BodyExt;
use rstest::rstest;
use support::{Gateway, Raw, client, raw_upstream, unreachable_upstream, within};
use tokio::sync::{Notify, mpsc};

/// Python stops a provider stream when it sees the client go away, so a client disconnect must
/// close Rust's connection to Python at every stage of the exchange.
#[rstest]
#[case::waiting_for_headers(
    "POST /v1/chat/completions HTTP/1.1\r\nhost: x\r\ncontent-length: 0\r\n\r\n",
    false
)]
#[case::mid_stream(
    "POST /v1/chat/completions HTTP/1.1\r\nhost: x\r\ncontent-length: 0\r\n\r\n",
    true
)]
#[case::mid_upload(
    "POST /v1/files HTTP/1.1\r\nhost: x\r\ncontent-length: 1000000\r\n\r\npartial",
    false
)]
#[tokio::test]
async fn client_disconnect_closes_the_upstream_connection(
    #[case] request: &'static str,
    #[case] headers_sent: bool,
) {
    let (event, mut events) = mpsc::unbounded_channel();
    let upstream = raw_upstream(move |mut python: Raw| {
        let event = event.clone();
        async move {
            python.request_head().await.unwrap();
            if headers_sent {
                python
                    .send("HTTP/1.1 200 OK\r\ntransfer-encoding: chunked\r\n\r\n6\r\ndata: \r\n")
                    .await;
            }
            event.send("received").unwrap();
            python.closed().await;
            event.send("closed").unwrap();
        }
    })
    .await;
    let gateway = Gateway::start(upstream).await;
    let mut raw = gateway.connect().await;
    raw.send(request).await;
    assert_eq!(within(events.recv()).await, Some("received"));
    if headers_sent {
        assert_eq!(
            within(raw.response_head()).await.unwrap().status,
            StatusCode::OK
        );
    }
    drop(raw);
    assert_eq!(within(events.recv()).await, Some("closed"));
}

/// uvicorn closes a connection that sat idle past its keep-alive timeout. A request Rust sends on
/// that connection as uvicorn closes it is lost, so Rust must retire idle connections first.
#[rstest]
#[case::reused_while_fresh(Duration::from_millis(0))]
#[case::past_the_pool_timeout(Duration::from_millis(250))]
#[case::past_python_keepalive(Duration::from_millis(450))]
#[tokio::test]
async fn never_reuses_a_connection_python_is_closing_for_idleness(#[case] idle: Duration) {
    let keepalive = Duration::from_millis(400);
    let upstream = raw_upstream(move |mut python: Raw| async move {
        let mut idle_since = None::<Instant>;
        while python.request_head().await.is_some() {
            if idle_since.is_some_and(|since| since.elapsed() >= keepalive) {
                return;
            }
            python
                .send("HTTP/1.1 200 OK\r\ncontent-length: 2\r\n\r\nok")
                .await;
            idle_since = Some(Instant::now());
        }
    })
    .await;
    let gateway = Gateway::with_keepalive(upstream, keepalive).await;
    let client = client();
    let first = within(client.request(Request::get(gateway.url("/")).body(Body::empty()).unwrap()))
        .await
        .unwrap();
    first.into_body().collect().await.unwrap();
    tokio::time::sleep(idle).await;
    let second =
        within(client.request(Request::get(gateway.url("/")).body(Body::empty()).unwrap()))
            .await
            .unwrap();
    assert_eq!(second.status(), StatusCode::OK);
}

#[rstest]
#[case::health("/health/liveliness")]
#[case::messages("/v1/messages")]
#[tokio::test]
async fn unreachable_python_is_a_502(#[case] path: &str) {
    let gateway = Gateway::start(unreachable_upstream()).await;
    let response =
        within(client().request(Request::get(gateway.url(path)).body(Body::empty()).unwrap()))
            .await
            .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
}

#[rstest]
#[tokio::test]
async fn shutdown_lets_an_in_flight_stream_finish() {
    let release = Arc::new(Notify::new());
    let released = release.clone();
    let upstream = raw_upstream(move |mut python: Raw| {
        let released = released.clone();
        async move {
            python.request_head().await.unwrap();
            python
                .send("HTTP/1.1 200 OK\r\ntransfer-encoding: chunked\r\n\r\n5\r\nfirst\r\n")
                .await;
            released.notified().await;
            python.send("4\r\nlast\r\n0\r\n\r\n").await;
        }
    })
    .await;
    let gateway = Gateway::start(upstream).await;
    let response =
        within(client().request(Request::get(gateway.url("/")).body(Body::empty()).unwrap()))
            .await
            .unwrap();
    let server = gateway.shutdown();
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert!(!server.is_finished());
    release.notify_one();
    let body = within(response.into_body().collect())
        .await
        .unwrap()
        .to_bytes();
    assert_eq!(body, "firstlast");
    within(server).await.unwrap().unwrap();
}
