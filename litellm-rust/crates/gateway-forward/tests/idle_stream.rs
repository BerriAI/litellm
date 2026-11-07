mod support;

use std::time::Duration;

use axum::{
    body::Body,
    extract::Request,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use http_body_util::BodyExt;
use rstest::rstest;
use support::{Gateway, client, raw_upstream, recording, within};
use tokio::{sync::mpsc, time::Instant};

const SSE_HEAD: &str = "HTTP/1.1 200 OK\r\n\
    date: Tue, 03 Nov 2026 00:00:00 GMT\r\n\
    server: uvicorn\r\n\
    cache-control: no-cache, no-transform\r\n\
    connection: keep-alive\r\n\
    content-type: text/event-stream\r\n\
    mcp-session-id: session-1\r\n\
    x-accel-buffering: no\r\n\
    transfer-encoding: chunked\r\n\r\n";

const PING: &str = ": ping - 2026-11-03 00:20:00.000000+00:00\r\n\r\n";

const MESSAGE: &str = "event: message\r\ndata: {\"jsonrpc\":\"2.0\",\"method\":\"notifications/tools/list_changed\"}\r\n\r\n";

fn chunk(data: &str) -> String {
    format!("{:x}\r\n{data}\r\n", data.len())
}

/// An MCP client holds a GET stream open for server-initiated messages, which may not come for
/// many minutes, so neither hop may close a stream for being idle.
#[rstest]
#[tokio::test(start_paused = true)]
async fn an_idle_mcp_stream_stays_open_in_both_directions() {
    let idle = Duration::from_secs(20 * 60);
    let (session, mut sessions) = mpsc::unbounded_channel();
    let upstream = raw_upstream(move |mut python| {
        let session = session.clone();
        async move {
            let head = python.request_head().await.unwrap();
            session
                .send(head.header("mcp-session-id").map(<[u8]>::to_vec))
                .unwrap();
            python.send(SSE_HEAD).await;
            tokio::time::sleep(idle).await;
            python.send(chunk(PING)).await;
            python.send(chunk(MESSAGE)).await;
            python.send("0\r\n\r\n").await;
        }
    })
    .await;
    let gateway = Gateway::start(upstream).await;
    let started = Instant::now();

    let response = client()
        .request(
            Request::get(gateway.url("/github/mcp"))
                .header("accept", "text/event-stream")
                .header("mcp-session-id", "session-1")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.headers()["mcp-session-id"], "session-1");
    assert_eq!(response.headers()["x-accel-buffering"], "no");
    assert!(!response.headers().contains_key("connection"));
    assert_eq!(
        sessions.recv().await.unwrap().as_deref(),
        Some(&b"session-1"[..])
    );
    let body = response.into_body().collect().await.unwrap().to_bytes();
    assert_eq!(body, [PING, MESSAGE].concat());
    assert!(started.elapsed() >= idle);
}

#[rstest]
#[tokio::test]
async fn ends_a_session_with_a_bodiless_delete() {
    let (upstream, mut received) = recording(|| -> Response {
        ([("mcp-session-id", "session-1")], StatusCode::NO_CONTENT).into_response()
    })
    .await;
    let gateway = Gateway::start(upstream).await;
    let mut raw = gateway.connect().await;
    raw.send("DELETE /mcp HTTP/1.1\r\nhost: x\r\nmcp-session-id: session-1\r\n\r\n")
        .await;

    let head = within(raw.response_head()).await.unwrap();
    assert_eq!(head.status, StatusCode::NO_CONTENT);
    assert_eq!(head.header("mcp-session-id"), Some(&b"session-1"[..]));
    let request = received.recv().await.unwrap();
    assert_eq!(request.method, "DELETE");
    assert_eq!(request.headers["mcp-session-id"], "session-1");
    assert!(!request.headers.contains_key("transfer-encoding"));
    assert!(request.body.is_empty());
}
