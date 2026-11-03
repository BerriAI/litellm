use std::time::Duration;

use axum::{
    body::Body,
    extract::Request,
    http::{HeaderMap, HeaderValue, Method, StatusCode, Version, header},
    response::Response,
};
use hyper::{
    body::Incoming,
    upgrade::{OnUpgrade, Upgraded},
};
use hyper_util::rt::TokioIo;
use tokio_util::{
    sync::{CancellationToken, DropGuard},
    task::TaskTracker,
};

const DRAIN_TIMEOUT: Duration = Duration::from_secs(5);

#[derive(Clone, Default)]
pub(super) struct Upgrades {
    tasks: TaskTracker,
    draining: CancellationToken,
    closed: CancellationToken,
}

impl Upgrades {
    pub fn begin_shutdown(&self) {
        self.draining.cancel();
        let closed = self.closed.clone();
        tokio::spawn(async move {
            tokio::time::sleep(DRAIN_TIMEOUT).await;
            closed.cancel();
        });
    }

    pub fn take_handshake(&self, request: &mut Request) -> Result<Option<OnUpgrade>, StatusCode> {
        if request.method() != Method::GET
            || request.version() != Version::HTTP_11
            || !is_handshake(request.headers())
        {
            return Ok(None);
        }
        if self.draining.is_cancelled() {
            return Err(StatusCode::SERVICE_UNAVAILABLE);
        }
        Ok(Some(hyper::upgrade::on(request)))
    }

    pub async fn until_closed<T>(&self, future: impl Future<Output = T>) -> Result<T, StatusCode> {
        tokio::select! {
            _ = self.closed.cancelled() => Err(StatusCode::SERVICE_UNAVAILABLE),
            output = future => Ok(output),
        }
    }

    pub async fn switch(
        &self,
        downstream: Option<OnUpgrade>,
        mut response: Response<Incoming>,
    ) -> Result<Response, StatusCode> {
        let Some(downstream) = downstream else {
            return Err(StatusCode::BAD_GATEWAY);
        };
        if !is_handshake(response.headers()) {
            return Err(StatusCode::BAD_GATEWAY);
        }
        let upstream = self
            .until_closed(hyper::upgrade::on(&mut response))
            .await?
            .map_err(|error| {
                tracing::warn!(%error, "python proxy websocket upgrade failed");
                StatusCode::BAD_GATEWAY
            })?;
        super::strip_hop_headers(response.headers_mut());
        restore_handshake_headers(response.headers_mut());
        self.relay(downstream, upstream);
        Ok(response.map(|_| Body::empty()))
    }

    pub fn close(&self) {
        self.closed.cancel();
    }

    pub fn close_on_drop(&self) -> DropGuard {
        self.closed.clone().drop_guard()
    }

    pub async fn wait(&self) {
        self.tasks.close();
        self.tasks.wait().await;
    }

    fn relay(&self, downstream: OnUpgrade, upstream: Upgraded) {
        let closed = self.closed.clone();
        self.tasks.spawn(async move {
            tokio::select! {
                _ = closed.cancelled() => {}
                _ = async {
                    let Ok(downstream) = downstream.await else {
                        tracing::debug!("websocket upgrade did not complete");
                        return;
                    };
                    let mut downstream = TokioIo::new(downstream);
                    let mut upstream = TokioIo::new(upstream);
                    if let Err(error) = tokio::io::copy_bidirectional(&mut downstream, &mut upstream).await {
                        tracing::debug!(%error, "websocket relay disconnected");
                    }
                } => {}
            }
        });
    }
}

fn is_handshake(headers: &HeaderMap) -> bool {
    headers
        .get(header::UPGRADE)
        .is_some_and(|value| value.as_bytes().eq_ignore_ascii_case(b"websocket"))
        && headers
            .get_all(header::CONNECTION)
            .iter()
            .filter_map(|value| value.to_str().ok())
            .flat_map(|value| value.split(','))
            .any(|token| token.trim().eq_ignore_ascii_case("upgrade"))
}

// Hop-by-hop stripping drops the handshake's Connection and Upgrade, so each hop gets them back.
pub(super) fn restore_handshake_headers(headers: &mut HeaderMap) {
    headers.insert(header::CONNECTION, HeaderValue::from_static("upgrade"));
    headers.insert(header::UPGRADE, HeaderValue::from_static("websocket"));
}

#[cfg(test)]
mod tests {
    use super::super::{Forward, PEER_HEADER};
    use axum::{Router, http::HeaderValue, http::StatusCode, response::IntoResponse};
    use futures_util::{SinkExt, StreamExt};
    use std::net::SocketAddr;
    use std::time::{Duration, Instant};
    use tokio::net::TcpListener;
    use tokio::{net::TcpStream, sync::oneshot};
    use tokio_tungstenite::{
        accept_hdr_async, client_async,
        tungstenite::{
            Message,
            client::IntoClientRequest,
            handshake::server::{ErrorResponse, Request as Handshake, Response as Accepted},
        },
    };

    async fn gateway(
        upstream: impl FnOnce(TcpListener) -> tokio::task::JoinHandle<()>,
    ) -> (SocketAddr, oneshot::Sender<()>, tokio::task::JoinHandle<()>) {
        let python = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let origin = format!("http://{}", python.local_addr().unwrap());
        upstream(python);
        let public = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = public.local_addr().unwrap();
        let (stop, stopped) = oneshot::channel::<()>();
        let server = tokio::spawn(async move {
            Forward::new(&origin)
                .unwrap()
                .serve(public, async {
                    stopped.await.ok();
                })
                .await
                .unwrap();
        });
        (address, stop, server)
    }

    async fn connect(
        address: SocketAddr,
    ) -> Result<
        tokio_tungstenite::WebSocketStream<TcpStream>,
        Box<tokio_tungstenite::tungstenite::Error>,
    > {
        let mut request = format!("ws://{address}/v1/realtime?model=m")
            .into_client_request()
            .unwrap();
        request
            .headers_mut()
            .insert("authorization", HeaderValue::from_static("Bearer test"));
        request.headers_mut().insert(
            "sec-websocket-protocol",
            HeaderValue::from_static("realtime"),
        );
        let stream = TcpStream::connect(address).await.unwrap();
        let (socket, response) = client_async(request, stream).await.map_err(Box::new)?;
        assert_eq!(response.headers()["sec-websocket-protocol"], "realtime");
        Ok(socket)
    }

    #[expect(
        clippy::result_large_err,
        reason = "tungstenite's handshake callback returns ErrorResponse"
    )]
    fn echo(listener: TcpListener) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            loop {
                let (stream, _) = listener.accept().await.unwrap();
                tokio::spawn(async move {
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
                    let mut socket = accept_hdr_async(stream, callback).await.unwrap();
                    socket.send(Message::text("greeting")).await.unwrap();
                    while let Some(Ok(message)) = socket.next().await {
                        if message.is_close() {
                            break;
                        }
                        socket.send(message).await.unwrap();
                    }
                });
            }
        })
    }

    #[tokio::test]
    async fn relays_messages_both_ways() {
        let (address, stop, server) = gateway(echo).await;
        let mut socket = connect(address).await.unwrap();
        assert_eq!(
            socket.next().await.unwrap().unwrap(),
            Message::text("greeting")
        );
        socket.send(Message::text("hello")).await.unwrap();
        assert_eq!(
            socket.next().await.unwrap().unwrap(),
            Message::text("hello")
        );
        socket.send(Message::binary(vec![0, 1, 2])).await.unwrap();
        assert_eq!(
            socket.next().await.unwrap().unwrap(),
            Message::binary(vec![0, 1, 2])
        );
        socket.close(None).await.unwrap();
        while socket.next().await.is_some() {}
        drop(socket);
        stop.send(()).unwrap();
        tokio::time::timeout(Duration::from_secs(4), server)
            .await
            .unwrap()
            .unwrap();
    }

    #[tokio::test]
    async fn passes_upstream_rejection_through() {
        let (address, _stop, _server) = gateway(|listener| {
            tokio::spawn(async move {
                axum::serve(
                    listener,
                    Router::new().fallback(|| async {
                        (StatusCode::FORBIDDEN, "origin not allowed").into_response()
                    }),
                )
                .await
                .unwrap();
            })
        })
        .await;
        let Err(error) = connect(address).await else {
            panic!("expected the upstream rejection");
        };
        let tokio_tungstenite::tungstenite::Error::Http(response) = *error else {
            panic!("expected the upstream rejection");
        };
        assert_eq!(response.status(), StatusCode::FORBIDDEN);
        assert_eq!(response.body().as_deref(), Some(&b"origin not allowed"[..]));
    }

    #[tokio::test]
    async fn shutdown_closes_open_sockets_after_drain() {
        let (address, stop, server) = gateway(echo).await;
        let mut socket = connect(address).await.unwrap();
        assert_eq!(
            socket.next().await.unwrap().unwrap(),
            Message::text("greeting")
        );
        let started = Instant::now();
        stop.send(()).unwrap();
        tokio::time::timeout(Duration::from_secs(10), server)
            .await
            .unwrap()
            .unwrap();
        assert!(started.elapsed() >= Duration::from_secs(4));
        assert!(!matches!(socket.next().await, Some(Ok(Message::Text(_)))));
    }
}
