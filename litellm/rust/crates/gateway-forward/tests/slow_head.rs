mod support;

use std::time::Duration;

use axum::{body::Body, extract::Request, http::StatusCode};
use http_body_util::BodyExt;
use rstest::rstest;
use support::{Gateway, client, raw_upstream};
use tokio::time::Instant;

/// Reasoning models with `stream=false`, image and video generation, and model health checks send
/// no response headers until the provider finishes, which can take many minutes.
#[rstest]
#[tokio::test(start_paused = true)]
async fn waits_for_response_headers_as_long_as_python_takes() {
    let thinking = Duration::from_secs(30 * 60);
    let upstream = raw_upstream(move |mut python| async move {
        python.request_head().await.unwrap();
        tokio::time::sleep(thinking).await;
        python
            .send("HTTP/1.1 200 OK\r\ncontent-length: 5\r\n\r\nimage")
            .await;
    })
    .await;
    let gateway = Gateway::start(upstream).await;
    let started = Instant::now();

    let response = client()
        .request(
            Request::post(gateway.url("/v1/images/generations"))
                .body(Body::from("{}"))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    assert!(started.elapsed() >= thinking);
    let body = response.into_body().collect().await.unwrap().to_bytes();
    assert_eq!(body, "image");
}
