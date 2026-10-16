//! Graceful drain end to end: a real listener, the ops routes and a streaming handler.

use std::{convert::Infallible, net::SocketAddr, sync::Arc, time::Duration};

use axum::{
    Router,
    body::{Body, Bytes},
    routing::post,
};
use futures_util::stream;
use litellm_config::Config;
use litellm_gateway::{
    drain::{self, DrainSettings, Outcome},
    health::Draining,
    metrics::Metrics,
};
use rstest::rstest;
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, TcpStream},
    sync::{Notify, mpsc},
    task::JoinHandle,
};
use tokio_util::sync::CancellationToken;

struct Server {
    address: SocketAddr,
    signals: mpsc::UnboundedSender<()>,
    /// Lets the streaming handler send its last chunk.
    release: Arc<Notify>,
    serving: JoinHandle<std::io::Result<Outcome>>,
}

/// Serves `POST /stream`, which sends `first`, waits for `release`, then sends `last`.
async fn start(settings: DrainSettings) -> Server {
    let release = Arc::new(Notify::new());
    let handler_release = release.clone();
    let app = Router::new().route(
        "/stream",
        post(move || {
            let release = handler_release.clone();
            async move {
                let chunks = stream::unfold(0, move |step| {
                    let release = release.clone();
                    async move {
                        match step {
                            0 => Some((Ok::<_, Infallible>(Bytes::from_static(b"first;")), 1)),
                            1 => {
                                release.notified().await;
                                Some((Ok(Bytes::from_static(b"last;")), 2))
                            }
                            _ => None,
                        }
                    }
                });
                Body::from_stream(chunks)
            }
        }),
    );
    let draining = Draining::default();
    let metrics = Metrics::new(&Config::from_yaml("{}").unwrap(), draining.flag());
    let app = litellm_gateway::with_ops(app, metrics, draining.clone(), None);
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let (signals, receiver) = mpsc::unbounded_channel();
    let receiver = Arc::new(tokio::sync::Mutex::new(receiver));
    let serving = tokio::spawn(drain::serve(
        listener,
        app,
        draining,
        settings,
        CancellationToken::new(),
        move || {
            let receiver = receiver.clone();
            async move {
                receiver.lock().await.recv().await;
            }
        },
    ));
    Server {
        address,
        signals,
        release,
        serving,
    }
}

fn settings(unready_delay: Duration, window: Duration) -> DrainSettings {
    DrainSettings {
        unready_delay,
        window,
    }
}

async fn send(address: SocketAddr, method: &str, path: &str) -> std::io::Result<TcpStream> {
    let mut connection = TcpStream::connect(address).await?;
    connection
        .write_all(
            format!("{method} {path} HTTP/1.1\r\nhost: test\r\ncontent-length: 0\r\nconnection: close\r\n\r\n")
                .as_bytes(),
        )
        .await?;
    Ok(connection)
}

async fn read_to_end(mut connection: TcpStream) -> String {
    let mut response = Vec::new();
    tokio::time::timeout(
        Duration::from_secs(10),
        connection.read_to_end(&mut response),
    )
    .await
    .unwrap()
    .unwrap();
    String::from_utf8(response).unwrap()
}

async fn read_until(connection: &mut TcpStream, needle: &str) -> String {
    let mut response = Vec::new();
    let mut buffer = [0; 1024];
    while !String::from_utf8_lossy(&response).contains(needle) {
        let read = tokio::time::timeout(Duration::from_secs(10), connection.read(&mut buffer))
            .await
            .unwrap()
            .unwrap();
        assert_ne!(read, 0, "connection closed before {needle:?}");
        response.extend_from_slice(&buffer[..read]);
    }
    String::from_utf8(response).unwrap()
}

async fn get(address: SocketAddr, path: &str) -> std::io::Result<String> {
    Ok(read_to_end(send(address, "GET", path).await?).await)
}

/// Retries until a connect is refused, since the listener closes asynchronously.
async fn wait_until_refused(address: SocketAddr) {
    tokio::time::timeout(Duration::from_secs(10), async {
        while TcpStream::connect(address).await.is_ok() {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("listener never closed");
}

#[rstest]
#[tokio::test]
async fn in_flight_stream_completes_after_shutdown_while_new_connections_are_refused() {
    let server = start(settings(Duration::ZERO, Duration::from_secs(30))).await;
    let mut stream = send(server.address, "POST", "/stream").await.unwrap();
    let head = read_until(&mut stream, "first;").await;
    assert!(head.starts_with("HTTP/1.1 200"));

    server.signals.send(()).unwrap();
    wait_until_refused(server.address).await;
    assert!(!server.serving.is_finished());
    server.release.notify_one();

    let rest = read_to_end(stream).await;
    assert!(rest.contains("last;"));
    assert_eq!(server.serving.await.unwrap().unwrap(), Outcome::Drained);
}

#[rstest]
#[tokio::test]
async fn readiness_reports_503_during_the_unready_delay() {
    let server = start(settings(Duration::from_secs(30), Duration::from_secs(30))).await;
    let mut stream = send(server.address, "POST", "/stream").await.unwrap();
    read_until(&mut stream, "first;").await;
    assert!(
        get(server.address, "/health/readiness")
            .await
            .unwrap()
            .starts_with("HTTP/1.1 200")
    );

    server.signals.send(()).unwrap();
    let readiness = tokio::time::timeout(Duration::from_secs(10), async {
        loop {
            let response = get(server.address, "/health/readiness").await.unwrap();
            if response.starts_with("HTTP/1.1 503") {
                break response;
            }
        }
    })
    .await
    .unwrap();

    assert!(readiness.ends_with(r#"{"status":"shutting_down"}"#));
    let metrics = get(server.address, "/metrics").await.unwrap();
    assert!(metrics.contains("litellm_gateway_draining 1\n"));
    assert!(metrics.contains("litellm_in_flight_requests 1\n"));
    assert!(
        get(server.address, "/health/liveliness")
            .await
            .unwrap()
            .starts_with("HTTP/1.1 200")
    );
    server.release.notify_one();
    read_to_end(stream).await;
}

#[rstest]
#[tokio::test]
async fn drain_window_bounds_how_long_a_stuck_stream_is_awaited() {
    let server = start(settings(Duration::ZERO, Duration::from_millis(200))).await;
    let mut stream = send(server.address, "POST", "/stream").await.unwrap();
    read_until(&mut stream, "first;").await;

    server.signals.send(()).unwrap();

    let outcome = tokio::time::timeout(Duration::from_secs(10), server.serving)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert_eq!(outcome, Outcome::WindowElapsed);
}

#[rstest]
#[tokio::test]
async fn a_second_signal_ends_the_drain() {
    let server = start(settings(Duration::ZERO, Duration::from_secs(30))).await;
    let mut stream = send(server.address, "POST", "/stream").await.unwrap();
    read_until(&mut stream, "first;").await;

    server.signals.send(()).unwrap();
    wait_until_refused(server.address).await;
    server.signals.send(()).unwrap();

    let outcome = tokio::time::timeout(Duration::from_secs(10), server.serving)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert_eq!(outcome, Outcome::Interrupted);
}
