use std::{convert::Infallible, io, net::SocketAddr, pin::pin, time::Duration};

use axum::{body::Body, extract::Request};
use hyper::{body::Incoming, server::conn::http1, service::service_fn};
use hyper_util::rt::{TokioIo, TokioTimer};
use tokio::net::TcpListener;

use crate::{Forward, forward::MAX_HEADERS};

/// The client's TCP connection to the public listener.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ClientSocket {
    pub peer: SocketAddr,
    pub local: SocketAddr,
}

/// Serves each public connection with hyper directly, because `axum::serve` cannot configure
/// the connection timeouts uvicorn applied to the same clients.
pub(crate) async fn serve(
    forward: Forward,
    listener: TcpListener,
    shutdown: impl Future<Output = ()>,
    drain: Duration,
) {
    let connections = forward.connections().clone();
    let _close_on_drop = connections.close_on_drop();
    let keepalive = forward.keepalive();
    let builder = builder(keepalive);
    let mut shutdown = pin!(shutdown);
    loop {
        let (stream, peer) = tokio::select! {
            () = &mut shutdown => break,
            accepted = listener.accept() => match accepted {
                Ok(accepted) => accepted,
                Err(error) => {
                    back_off(error).await;
                    continue;
                }
            },
        };
        let local = match stream.local_addr() {
            Ok(local) => local,
            Err(error) => {
                tracing::debug!(%error, "client connection closed before it was served");
                continue;
            }
        };
        if let Err(error) = stream.set_nodelay(true) {
            tracing::warn!(%error, "could not disable Nagle on a client connection");
        }
        let socket = ClientSocket { peer, local };
        let forward = forward.clone();
        let service = service_fn(move |request: Request<Incoming>| {
            let forward = forward.clone();
            async move { Ok::<_, Infallible>(forward.handle(socket, request.map(Body::new)).await) }
        });
        let connection = builder
            .serve_connection(TokioIo::new(stream), service)
            .with_upgrades();
        let draining = connections.clone();
        connections.spawn(async move {
            let mut connection = pin!(connection);
            let ended = tokio::select! {
                ended = connection.as_mut() => ended,
                () = draining.draining() => {
                    connection.as_mut().graceful_shutdown();
                    connection.await
                }
            };
            if let Err(error) = ended {
                tracing::debug!(%error, "client connection ended with an error");
            }
        });
    }
    drop(listener);
    connections.begin_shutdown();
    if tokio::time::timeout(drain, connections.wait())
        .await
        .is_err()
    {
        tracing::warn!(
            ?drain,
            "closing connections still open after the shutdown drain"
        );
    }
    connections.close();
    connections.wait().await;
}

// uvicorn closes a client connection that sits idle for `--keepalive_timeout` between requests.
// Without the same limit, a client that vanishes without closing (NAT expiry, a crashed host)
// would hold its socket for the life of the process until the gateway runs out of descriptors.
// Operators already tune that timeout against their load balancer, so the public side keeps
// Python's value instead of introducing its own. hyper only runs the header timer while a
// connection waits for its next request, so it never cuts a long stream, upload or WebSocket.
fn builder(keepalive: Duration) -> http1::Builder {
    let mut builder = http1::Builder::new();
    builder
        .max_headers(MAX_HEADERS)
        .timer(TokioTimer::new())
        .header_read_timeout(keepalive);
    builder
}

async fn back_off(error: io::Error) {
    if matches!(
        error.kind(),
        io::ErrorKind::ConnectionRefused
            | io::ErrorKind::ConnectionAborted
            | io::ErrorKind::ConnectionReset
    ) {
        return;
    }
    tracing::error!(%error, "could not accept a client connection");
    tokio::time::sleep(Duration::from_secs(1)).await;
}
