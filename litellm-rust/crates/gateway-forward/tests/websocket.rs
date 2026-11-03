mod support;

use std::{
    net::SocketAddr,
    time::{Duration, Instant},
};

use axum::{
    Router,
    http::{HeaderValue, StatusCode},
    response::IntoResponse,
};
use futures_util::{SinkExt, StreamExt};
use litellm_gateway_forward::PEER_HEADER;
use rstest::rstest;
use support::{DRAIN, Gateway, raw_upstream, serve, within};
use tokio::net::TcpStream;
use tokio_tungstenite::{
    WebSocketStream, accept_hdr_async, client_async,
    tungstenite::{
        Error, Message,
        client::IntoClientRequest,
        handshake::server::{ErrorResponse, Request as Handshake, Response as Accepted},
        protocol::{CloseFrame, frame::coding::CloseCode},
    },
};

async fn connect(
    address: SocketAddr,
    connection: &'static str,
) -> Result<WebSocketStream<TcpStream>, Box<Error>> {
    let mut request = format!("ws://{address}/v1/realtime?model=m")
        .into_client_request()
        .unwrap();
    let headers = request.headers_mut();
    headers.insert("authorization", HeaderValue::from_static("Bearer test"));
    headers.insert(
        "sec-websocket-protocol",
        HeaderValue::from_static("realtime"),
    );
    headers.insert("connection", HeaderValue::from_static(connection));
    let stream = TcpStream::connect(address).await.unwrap();
    let (socket, response) = client_async(request, stream).await.map_err(Box::new)?;
    assert_eq!(response.headers()["sec-websocket-protocol"], "realtime");
    Ok(socket)
}

/// Greets, echoes every message, and closes with 4000 when told to "bye".
#[expect(
    clippy::result_large_err,
    reason = "tungstenite's handshake callback returns ErrorResponse"
)]
async fn echo() -> SocketAddr {
    raw_upstream(|python| async move {
        let callback = |request: &Handshake, mut response: Accepted| {
            assert_eq!(request.uri(), "/v1/realtime?model=m");
            assert_eq!(request.headers()["authorization"], "Bearer test");
            assert!(request.headers().contains_key(PEER_HEADER));
            response.headers_mut().insert(
                "sec-websocket-protocol",
                HeaderValue::from_static("realtime"),
            );
            Ok::<_, ErrorResponse>(response)
        };
        let mut socket = accept_hdr_async(python.into_stream(), callback)
            .await
            .unwrap();
        socket.send(Message::text("greeting")).await.unwrap();
        while let Some(Ok(message)) = socket.next().await {
            if message == Message::text("bye") {
                socket
                    .close(Some(CloseFrame {
                        code: CloseCode::Library(4000),
                        reason: "done".into(),
                    }))
                    .await
                    .unwrap();
                continue;
            }
            if message.is_close() {
                break;
            }
            socket.send(message).await.unwrap();
        }
    })
    .await
}

#[rstest]
#[tokio::test]
async fn relays_messages_both_ways(
    #[values("Upgrade", "keep-alive, Upgrade", "upgrade")] connection: &'static str,
    #[values(
        Message::text("hello"),
        Message::binary(vec![0, 1, 2]),
        Message::binary(vec![7; 4 * 1024 * 1024])
    )]
    message: Message,
) {
    let gateway = Gateway::start(echo().await).await;
    let mut socket = within(connect(gateway.address, connection)).await.unwrap();
    assert_eq!(
        socket.next().await.unwrap().unwrap(),
        Message::text("greeting")
    );
    socket.send(message.clone()).await.unwrap();
    assert_eq!(within(socket.next()).await.unwrap().unwrap(), message);
    socket.close(None).await.unwrap();
    while socket.next().await.is_some() {}
    drop(socket);
    within(gateway.shutdown()).await.unwrap().unwrap();
}

#[rstest]
#[tokio::test]
async fn relays_a_close_python_initiates() {
    let gateway = Gateway::start(echo().await).await;
    let mut socket = within(connect(gateway.address, "Upgrade")).await.unwrap();
    socket.next().await.unwrap().unwrap();
    socket.send(Message::text("bye")).await.unwrap();
    let Message::Close(Some(frame)) = within(socket.next()).await.unwrap().unwrap() else {
        panic!("expected python's close frame");
    };
    assert_eq!(frame.code, CloseCode::Library(4000));
    assert_eq!(frame.reason, "done");
}

#[rstest]
#[tokio::test]
async fn passes_an_upstream_rejection_through() {
    let upstream = serve(
        Router::new()
            .fallback(|| async { (StatusCode::FORBIDDEN, "origin not allowed").into_response() }),
    )
    .await;
    let gateway = Gateway::start(upstream).await;
    let error = within(connect(gateway.address, "Upgrade"))
        .await
        .unwrap_err();
    let Error::Http(response) = *error else {
        panic!("expected the upstream rejection, got {error}");
    };
    assert_eq!(response.status(), StatusCode::FORBIDDEN);
    assert_eq!(response.body().as_deref(), Some(&b"origin not allowed"[..]));
}

#[rstest]
#[tokio::test]
async fn shutdown_closes_open_sockets_after_drain() {
    let gateway = Gateway::start(echo().await).await;
    let mut socket = within(connect(gateway.address, "Upgrade")).await.unwrap();
    socket.next().await.unwrap().unwrap();
    let started = Instant::now();
    within(gateway.shutdown()).await.unwrap().unwrap();
    assert!(started.elapsed() >= DRAIN);
    assert!(!matches!(socket.next().await, Some(Ok(Message::Text(_)))));
}

#[rstest]
#[tokio::test]
async fn shutdown_relays_the_close_python_sends_while_draining() {
    let gateway = Gateway::start(echo().await).await;
    let mut socket = within(connect(gateway.address, "Upgrade")).await.unwrap();
    socket.next().await.unwrap().unwrap();
    let server = gateway.shutdown();
    tokio::time::sleep(Duration::from_millis(100)).await;
    socket.send(Message::text("bye")).await.unwrap();
    let Message::Close(Some(frame)) = within(socket.next()).await.unwrap().unwrap() else {
        panic!("expected python's close frame");
    };
    assert_eq!(frame.code, CloseCode::Library(4000));
    while socket.next().await.is_some() {}
    drop(socket);
    let started = Instant::now();
    within(server).await.unwrap().unwrap();
    assert!(started.elapsed() < DRAIN / 2);
}

#[rstest]
#[tokio::test]
async fn refuses_new_sockets_while_draining() {
    let gateway = Gateway::start(echo().await).await;
    let mut open = within(connect(gateway.address, "Upgrade")).await.unwrap();
    open.next().await.unwrap().unwrap();
    let address = gateway.address;
    let server = gateway.shutdown();
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert!(
        TcpStream::connect(address).await.is_err() || connect(address, "Upgrade").await.is_err()
    );
    open.send(Message::text("still open")).await.unwrap();
    assert_eq!(
        within(open.next()).await.unwrap().unwrap(),
        Message::text("still open")
    );
    drop(open);
    within(server).await.unwrap().unwrap();
}
