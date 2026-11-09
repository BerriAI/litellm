mod support;

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use axum::{Router, body::Body, extract::Request, http::StatusCode, response::Response};
use http_body_util::BodyExt;
use rstest::rstest;
use support::{
    DRAIN, Gateway, Raw, client, raw_upstream, recording, serve, unreachable_upstream, within,
};
use tokio::{
    net::{TcpListener, TcpStream},
    sync::{Notify, mpsc},
};

fn ok() -> Response {
    Response::new(Body::from("ok"))
}

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
#[case::waiting_for_websocket_handshake(
    "GET /v1/realtime HTTP/1.1\r\nhost: x\r\nconnection: Upgrade\r\nupgrade: websocket\r\nsec-websocket-version: 13\r\nsec-websocket-key: dGhlIHNhbXBsZSBub25jZQ==\r\n\r\n",
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

/// Python treats EOF on its socket as the client leaving, so Rust must not half-close the
/// upstream connection once it has sent the request body.
#[rstest]
#[tokio::test]
async fn python_sees_no_eof_while_its_response_is_pending() {
    let (verdict, mut verdicts) = mpsc::unbounded_channel();
    let upstream = raw_upstream(move |mut python: Raw| {
        let verdict = verdict.clone();
        async move {
            let head = python.request_head().await.unwrap();
            python.body(head.content_length().unwrap()).await;
            let eof = tokio::time::timeout(Duration::from_millis(200), python.closed())
                .await
                .is_ok();
            verdict.send(eof).unwrap();
            python
                .send("HTTP/1.1 200 OK\r\ncontent-length: 2\r\n\r\nok")
                .await;
        }
    })
    .await;
    let gateway = Gateway::start(upstream).await;

    let response = within(
        client().request(
            Request::post(gateway.url("/mcp"))
                .body(Body::from("{}"))
                .unwrap(),
        ),
    )
    .await
    .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(verdicts.recv().await, Some(false));
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
#[case::bad_status_line(b"HTTP/1.1 OK\r\ncontent-length: 0\r\n\r\n".to_vec())]
#[case::header_without_colon(b"HTTP/1.1 200 OK\r\nbroken header\r\ncontent-length: 0\r\n\r\n".to_vec())]
#[case::not_http(b"SSH-2.0-OpenSSH_9.6\r\n".to_vec())]
#[case::conflicting_lengths(
    b"HTTP/1.1 200 OK\r\ncontent-length: 2\r\ncontent-length: 3\r\n\r\nok".to_vec()
)]
#[case::oversized_head(
    [&b"HTTP/1.1 200 OK\r\nx-big: "[..], &vec![b'a'; 1024 * 1024], b"\r\n\r\n"].concat()
)]
#[case::closed_without_reply(Vec::new())]
#[tokio::test]
async fn an_upstream_that_breaks_http_is_a_502_and_the_client_connection_survives(
    #[case] reply: Vec<u8>,
) {
    let upstream = raw_upstream(move |mut python: Raw| {
        let reply = reply.clone();
        async move {
            python.request_head().await.unwrap();
            if reply.is_empty() {
                return;
            }
            python.send(reply).await;
            std::future::pending::<()>().await;
        }
    })
    .await;
    let gateway = Gateway::start(upstream).await;
    let mut raw = gateway.connect().await;

    for _ in 0..2 {
        raw.send("GET / HTTP/1.1\r\nhost: x\r\n\r\n").await;
        let head = within(raw.response_head()).await.unwrap();
        assert_eq!(head.status, StatusCode::BAD_GATEWAY);
        assert_eq!(head.content_length(), Some(0));
    }
}

#[rstest]
#[tokio::test]
async fn python_restarting_is_a_502_until_it_listens_again() {
    let address = unreachable_upstream();
    let gateway = Gateway::start(address).await;
    let get = || client().request(Request::get(gateway.url("/")).body(Body::empty()).unwrap());
    assert_eq!(
        within(get()).await.unwrap().status(),
        StatusCode::BAD_GATEWAY
    );

    let listener = TcpListener::bind(address).await.unwrap();
    let router = Router::new().fallback(|| async { "up" });
    tokio::spawn(async move { axum::serve(listener, router).await });

    assert_eq!(within(get()).await.unwrap().status(), StatusCode::OK);
}

/// A request Python already read may have run, so it is answered with a 502 and never replayed.
#[rstest]
#[tokio::test]
async fn a_request_python_read_before_closing_is_a_502_and_never_replayed() {
    let (event, mut seen) = mpsc::unbounded_channel();
    let upstream = raw_upstream(move |mut python: Raw| {
        let event = event.clone();
        async move {
            while let Some(head) = python.request_head().await {
                python.body(head.content_length().unwrap()).await;
                event.send(head.path.clone()).unwrap();
                if head.path == "/v1/chat/completions" {
                    return;
                }
                python
                    .send("HTTP/1.1 200 OK\r\ncontent-length: 2\r\n\r\nok")
                    .await;
            }
        }
    })
    .await;
    let gateway = Gateway::start(upstream).await;
    let mut raw = gateway.connect().await;
    raw.send("POST /warm HTTP/1.1\r\nhost: x\r\ncontent-length: 0\r\n\r\n")
        .await;
    let warm = within(raw.response_head()).await.unwrap();
    raw.body(warm.content_length().unwrap()).await;

    raw.send("POST /v1/chat/completions HTTP/1.1\r\nhost: x\r\ncontent-length: 2\r\n\r\n{}")
        .await;

    let head = within(raw.response_head()).await.unwrap();
    assert_eq!(head.status, StatusCode::BAD_GATEWAY);
    let paths: Vec<_> = std::iter::from_fn(|| seen.try_recv().ok()).collect();
    assert_eq!(paths, ["/warm", "/v1/chat/completions"]);
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
    within(server).await.unwrap();
}

#[rstest]
#[tokio::test]
async fn shutdown_stops_waiting_for_a_stream_still_open_after_the_drain() {
    let upstream = raw_upstream(|mut python: Raw| async move {
        python.request_head().await.unwrap();
        python
            .send("HTTP/1.1 200 OK\r\ntransfer-encoding: chunked\r\n\r\n5\r\nfirst\r\n")
            .await;
        std::future::pending::<()>().await;
    })
    .await;
    let gateway = Gateway::start(upstream).await;
    let response =
        within(client().request(Request::get(gateway.url("/")).body(Body::empty()).unwrap()))
            .await
            .unwrap();
    let started = std::time::Instant::now();

    within(gateway.shutdown()).await.unwrap();

    assert!(started.elapsed() >= DRAIN);
    drop(response);
}

#[rstest]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_keep_alive_connections_each_get_their_own_responses() {
    let upstream = serve(Router::new().fallback(|request: Request| async move {
        let index: usize = request.uri().query().unwrap().parse().unwrap();
        Response::builder()
            .header("etag", format!("\"{index}\""))
            .body(Body::from(index.to_string().repeat(1 + index * 1000)))
            .unwrap()
    }))
    .await;
    let gateway = Gateway::start(upstream).await;
    let connections = 6;

    let tasks: Vec<_> = (0..connections)
        .map(|first| {
            let address = gateway.address;
            tokio::spawn(async move {
                let mut raw = Raw::new(TcpStream::connect(address).await.unwrap());
                for index in (first..50).step_by(connections) {
                    raw.send(format!(
                        "GET /ui/chunk.js?{index} HTTP/1.1\r\nhost: x\r\n\r\n"
                    ))
                    .await;
                    let head = within(raw.response_head()).await.unwrap();
                    assert_eq!(head.header("etag"), Some(format!("\"{index}\"").as_bytes()));
                    let body = raw.body(head.content_length().unwrap()).await;
                    assert_eq!(
                        body,
                        index.to_string().repeat(1 + index * 1000).into_bytes()
                    );
                }
            })
        })
        .collect();

    for task in tasks {
        within(task).await.unwrap();
    }
}

// Python's `--keepalive_timeout` governed idle client connections before Rust took the public
// port, and operators tune it against their load balancer's idle timeout, so it still does.
// Without it, a client that vanishes without closing its connection holds a socket forever and
// the gateway eventually runs out of file descriptors.
const IDLE: Duration = Duration::from_millis(400);

#[rstest]
#[tokio::test]
async fn closes_a_client_connection_idle_past_python_keepalive() {
    let (upstream, _received) = recording(ok).await;
    let gateway = Gateway::with_keepalive(upstream, IDLE).await;
    let mut raw = gateway.connect().await;

    for _ in 0..2 {
        tokio::time::sleep(IDLE / 2).await;
        raw.send("GET /health/liveliness HTTP/1.1\r\nhost: x\r\n\r\n")
            .await;
        let head = within(raw.response_head()).await.unwrap();
        assert_eq!(head.status, StatusCode::OK);
        raw.body(head.content_length().unwrap()).await;
    }
    let idle_since = Instant::now();
    within(raw.closed()).await;

    assert!(idle_since.elapsed() >= IDLE * 3 / 4);
}

/// uvicorn would hold a connection that never sends a request, which lets a client park sockets
/// for free, so the same limit applies before the first request too.
#[rstest]
#[tokio::test]
async fn closes_a_client_connection_that_never_sends_a_request() {
    let (upstream, _received) = recording(ok).await;
    let gateway = Gateway::with_keepalive(upstream, IDLE).await;
    let mut raw = gateway.connect().await;

    within(raw.closed()).await;
}

/// The limit only covers the wait for the next request. A response that goes quiet for longer
/// belongs to Python, which decides how long it may take.
#[rstest]
#[tokio::test]
async fn keeps_a_connection_open_while_a_response_goes_quiet() {
    let upstream = raw_upstream(|mut python: Raw| async move {
        python.request_head().await.unwrap();
        python
            .send("HTTP/1.1 200 OK\r\ntransfer-encoding: chunked\r\n\r\n")
            .await;
        tokio::time::sleep(IDLE * 3).await;
        python.send("4\r\ndone\r\n0\r\n\r\n").await;
    })
    .await;
    let gateway = Gateway::with_keepalive(upstream, IDLE).await;
    let mut raw = gateway.connect().await;

    raw.send("GET /v1/chat/completions HTTP/1.1\r\nhost: x\r\n\r\n")
        .await;

    within(raw.response_head()).await.unwrap();
    assert_eq!(within(raw.chunked_body()).await, b"done");
}

/// Likewise for an upload that pauses longer than the limit before it finishes.
#[rstest]
#[tokio::test]
async fn keeps_a_connection_open_while_an_upload_goes_quiet() {
    let (upstream, mut received) = recording(ok).await;
    let gateway = Gateway::with_keepalive(upstream, IDLE).await;
    let mut raw = gateway.connect().await;

    raw.send("POST /v1/files HTTP/1.1\r\nhost: x\r\ncontent-length: 8\r\n\r\nhalf")
        .await;
    tokio::time::sleep(IDLE * 3).await;
    raw.send("done").await;

    assert_eq!(
        within(raw.response_head()).await.unwrap().status,
        StatusCode::OK
    );
    assert_eq!(received.recv().await.unwrap().body, "halfdone");
}

/// uvicorn answers a request it rejects (413, 401) as soon as it reads the head and then closes,
/// leaving the upload unread. The client must get that answer, not a 502 or a reset.
#[rstest]
#[case::sized("content-length: 16777216\r\n", vec![b'x'; 256 * 1024])]
#[case::chunked(
    "transfer-encoding: chunked\r\n",
    [&b"40000\r\n"[..], &[b'x'; 256 * 1024], b"\r\n"].concat()
)]
#[tokio::test]
async fn python_rejecting_an_upload_unread_and_closing_still_answers_the_client(
    #[case] framing: &'static str,
    #[case] partial_upload: Vec<u8>,
) {
    let upstream = raw_upstream(|mut python: Raw| async move {
        python.request_head().await.unwrap();
        python
            .send("HTTP/1.1 413 Payload Too Large\r\ncontent-type: application/json\r\ncontent-length: 26\r\nconnection: close\r\n\r\n{\"error\":\"file too large\"}")
            .await;
    })
    .await;
    let gateway = Gateway::start(upstream).await;
    let mut raw = gateway.connect().await;

    raw.send(format!(
        "POST /v1/files HTTP/1.1\r\nhost: x\r\n{framing}\r\n"
    ))
    .await;
    raw.send(partial_upload).await;

    let head = within(raw.response_head()).await.unwrap();
    assert_eq!(head.status, StatusCode::PAYLOAD_TOO_LARGE);
    assert_eq!(
        raw.body(head.content_length().unwrap()).await,
        b"{\"error\":\"file too large\"}"
    );
}
