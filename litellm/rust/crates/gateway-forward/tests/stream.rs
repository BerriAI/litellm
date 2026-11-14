mod support;

use std::{
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

use axum::{body::Body, extract::Request, http::StatusCode};
use http_body_util::BodyExt;
use hyper::body::Incoming;
use rstest::rstest;
use support::{Gateway, client, raw_upstream, within};
use tokio::{
    io::AsyncWriteExt,
    sync::{Notify, mpsc},
};

fn chunk(bytes: &[u8]) -> Vec<u8> {
    [format!("{:x}\r\n", bytes.len()).as_bytes(), bytes, b"\r\n"].concat()
}

const EVENTS: [&str; 3] = ["data: 1\n\n", "data: 2\n\n", "data: [DONE]\n\n"];

#[rstest]
#[tokio::test]
async fn each_event_reaches_the_client_before_python_produces_the_next(
    #[values(true, false)] chunked: bool,
) {
    let produce = Arc::new(Notify::new());
    let next = produce.clone();
    let upstream = raw_upstream(move |mut python| {
        let next = next.clone();
        async move {
            python.request_head().await.unwrap();
            let framing = match chunked {
                true => "transfer-encoding: chunked".to_owned(),
                false => format!("content-length: {}", EVENTS.concat().len()),
            };
            python
                .send(format!(
                    "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n{framing}\r\n\r\n"
                ))
                .await;
            for event in EVENTS {
                match chunked {
                    true => python.send(chunk(event.as_bytes())).await,
                    false => python.send(event).await,
                }
                next.notified().await;
            }
            if chunked {
                python.send("0\r\n\r\n").await;
            }
        }
    })
    .await;
    let gateway = Gateway::start(upstream).await;
    let mut response = within(
        client().request(
            Request::post(gateway.url("/v1/chat/completions"))
                .body(Body::from(r#"{"stream":true}"#))
                .unwrap(),
        ),
    )
    .await
    .unwrap();

    for event in EVENTS {
        let frame = within(response.body_mut().frame()).await.unwrap().unwrap();
        assert_eq!(frame.into_data().unwrap(), event);
        produce.notify_one();
    }
    let rest = within(response.into_body().collect()).await.unwrap();
    assert!(rest.to_bytes().is_empty());
}

#[rstest]
#[tokio::test]
async fn relays_a_binary_framed_stream_unchanged() {
    let message: Vec<u8> = [
        &[
            0x00, 0x00, 0x00, 0x3d, 0x00, 0x00, 0x00, 0x20, 0x8a, 0x0d, 0x0a, 0xff,
        ][..],
        b"\x0b:event-type\x07\x00\x05chunk\x0d:content-type\x07\x00\x10application/json",
        b"0\r\n\r\n{\"bytes\":\"eyJ0eXBlIjoiY29udGVudCJ9\"}",
        &[0xc3, 0x28, 0x00, 0x0d],
    ]
    .concat();
    let sent = message.clone();
    let upstream = raw_upstream(move |mut python| {
        let sent = sent.clone();
        async move {
            python.request_head().await.unwrap();
            python
                .send("HTTP/1.1 200 OK\r\ncontent-type: application/vnd.amazon.eventstream\r\ntransfer-encoding: chunked\r\n\r\n")
                .await;
            for piece in sent.chunks(7) {
                python.send(chunk(piece)).await;
            }
            python.send("0\r\n\r\n").await;
        }
    })
    .await;
    let gateway = Gateway::start(upstream).await;
    let response = within(
        client().request(
            Request::post(gateway.url("/bedrock/model/m/converse-stream"))
                .body(Body::empty())
                .unwrap(),
        ),
    )
    .await
    .unwrap();

    assert_eq!(
        response.headers()["content-type"],
        "application/vnd.amazon.eventstream"
    );
    let body = within(response.into_body().collect())
        .await
        .unwrap()
        .to_bytes();
    assert_eq!(body, message);
}

#[rstest]
#[tokio::test]
async fn relays_a_body_delimited_by_the_upstream_closing() {
    let upstream = raw_upstream(|mut python| async move {
        python.request_head().await.unwrap();
        python
            .send("HTTP/1.1 200 OK\r\nconnection: close\r\n\r\nuntil eof")
            .await;
    })
    .await;
    let gateway = Gateway::start(upstream).await;
    let response = client().request(Request::get(gateway.url("/")).body(Body::empty()).unwrap());
    let body = within(response).await.unwrap().into_body().collect();
    assert_eq!(within(body).await.unwrap().to_bytes(), "until eof");
}

async fn drain(mut body: Incoming) -> (Vec<u8>, Result<(), hyper::Error>) {
    let mut delivered = Vec::new();
    loop {
        match within(body.frame()).await {
            None => return (delivered, Ok(())),
            Some(Err(error)) => return (delivered, Err(error)),
            Some(Ok(frame)) => delivered.extend_from_slice(&frame.into_data().unwrap()),
        }
    }
}

#[rstest]
#[case::after_the_head("transfer-encoding: chunked", "", "")]
#[case::between_events("transfer-encoding: chunked", "6\r\ndata: \r\n", "data: ")]
#[case::mid_event("transfer-encoding: chunked", "a\r\ndata:", "data:")]
#[case::short_of_its_content_length("content-length: 100", "data: 1\n\n", "data: 1\n\n")]
#[tokio::test]
async fn an_upstream_dying_mid_stream_fails_the_client_body_instead_of_ending_it(
    #[case] framing: &'static str,
    #[case] sent: &'static str,
    #[case] delivered: &'static str,
) {
    let upstream = raw_upstream(move |mut python| async move {
        python.request_head().await.unwrap();
        python
            .send(format!(
                "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n{framing}\r\n\r\n{sent}"
            ))
            .await;
    })
    .await;
    let gateway = Gateway::start(upstream).await;
    let response =
        within(client().request(Request::get(gateway.url("/")).body(Body::empty()).unwrap()))
            .await
            .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let (received, ended) = drain(response.into_body()).await;
    assert_eq!(received, delivered.as_bytes());
    assert!(ended.is_err());
}

#[rstest]
#[tokio::test]
async fn response_headers_reach_the_client_before_python_sends_any_body() {
    let release = Arc::new(Notify::new());
    let released = release.clone();
    let upstream = raw_upstream(move |mut python| {
        let released = released.clone();
        async move {
            python.request_head().await.unwrap();
            python
                .send("HTTP/1.1 200 OK\r\ncontent-type: text/event-stream; charset=utf-8\r\nx-accel-buffering: no\r\nx-litellm-call-id: call-1\r\ntransfer-encoding: chunked\r\n\r\n")
                .await;
            released.notified().await;
            python.send(chunk(b"data: 1\n\n")).await;
            python.send("0\r\n\r\n").await;
        }
    })
    .await;
    let gateway = Gateway::start(upstream).await;

    let response =
        within(client().request(Request::post(gateway.url("/")).body(Body::empty()).unwrap()))
            .await
            .unwrap();

    assert_eq!(response.headers()["x-litellm-call-id"], "call-1");
    assert_eq!(response.headers()["x-accel-buffering"], "no");
    release.notify_one();
    let body = within(response.into_body().collect())
        .await
        .unwrap()
        .to_bytes();
    assert_eq!(body, "data: 1\n\n");
}

/// A connection closed without any response looks retryable to client SDKs, which would re-run a
/// request Python already ran. Python dying right after its head races the relay, hence the loop.
#[rstest]
#[tokio::test]
async fn a_head_python_sent_right_before_dying_still_reaches_the_client() {
    let upstream = raw_upstream(|mut python| async move {
        python.request_head().await.unwrap();
        python
            .send("HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\n\r\n")
            .await;
    })
    .await;
    let gateway = Gateway::start(upstream).await;
    for _ in 0..20 {
        let mut raw = gateway.connect().await;
        raw.send("POST / HTTP/1.1\r\nhost: x\r\ncontent-length: 0\r\n\r\n")
            .await;
        let head = within(raw.response_head()).await;
        assert_eq!(head.map(|head| head.status), Some(StatusCode::OK));
    }
}

#[rstest]
#[tokio::test]
async fn a_client_that_stops_reading_stalls_python_instead_of_being_buffered_for() {
    let written = Arc::new(AtomicUsize::new(0));
    let counted = written.clone();
    let (failed, mut write_failed) = mpsc::unbounded_channel();
    let upstream = raw_upstream(move |mut python| {
        let counted = counted.clone();
        let failed = failed.clone();
        async move {
            python.request_head().await.unwrap();
            let mut python = python.into_stream();
            python
                .write_all(b"HTTP/1.1 200 OK\r\ntransfer-encoding: chunked\r\n\r\n")
                .await
                .unwrap();
            let piece = chunk(&[0x5a; 64 * 1024]);
            while python.write_all(&piece).await.is_ok() {
                counted.fetch_add(piece.len(), Ordering::SeqCst);
            }
            failed.send(()).unwrap();
        }
    })
    .await;
    let gateway = Gateway::start(upstream).await;
    let mut raw = gateway.connect().await;
    raw.send("POST / HTTP/1.1\r\nhost: x\r\ncontent-length: 0\r\n\r\n")
        .await;
    within(raw.response_head()).await.unwrap();

    tokio::time::sleep(Duration::from_secs(1)).await;
    let stalled = written.load(Ordering::SeqCst);
    tokio::time::sleep(Duration::from_millis(250)).await;

    assert_eq!(written.load(Ordering::SeqCst), stalled);
    assert!(stalled < 64 * 1024 * 1024, "{stalled} bytes in flight");
    drop(raw);
    within(write_failed.recv()).await.unwrap();
}

#[rstest]
#[tokio::test]
async fn an_http10_client_receives_a_streamed_body_delimited_by_close() {
    let upstream = raw_upstream(|mut python| async move {
        python.request_head().await.unwrap();
        python
            .send([
                &b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\n\r\n"[..],
                &chunk(b"data: 1\n"),
                &chunk(b"data: 2\n"),
                b"0\r\n\r\n",
            ]
            .concat())
            .await;
    })
    .await;
    let gateway = Gateway::start(upstream).await;
    let mut raw = gateway.connect().await;
    raw.send("GET /v1/stream HTTP/1.0\r\nhost: x\r\n\r\n").await;

    let response = String::from_utf8(within(raw.until_closed()).await).unwrap();
    let (head, body) = response.split_once("\r\n\r\n").unwrap();
    assert!(head.starts_with("HTTP/1.0 200 OK"), "{head}");
    assert!(
        !head.to_ascii_lowercase().contains("transfer-encoding"),
        "{head}"
    );
    assert_eq!(body, "data: 1\ndata: 2\n");
}
