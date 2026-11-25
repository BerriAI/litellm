use std::{convert::Infallible, future::ready, io, pin::pin, time::Duration};

use axum::{
    Extension,
    body::Body,
    extract::{ConnectInfo, Request},
    response::Response,
};
use futures_util::{Stream, StreamExt, stream};
use hyper::{body::Incoming, server::conn::http1};
use hyper_util::{
    rt::{TokioIo, TokioTimer},
    service::TowerToHyperService,
};
use tokio::net::{TcpListener, TcpStream};
use tower::{Service, ServiceBuilder};

use crate::{client::ClientSocket, connections::Connections, forward::MAX_HEADERS};

/// Serves each public connection with hyper directly, because `axum::serve` cannot configure
/// the connection timeouts uvicorn applied to the same clients. Every request reaches `app` with
/// its `ClientSocket`, and its peer as axum's `ConnectInfo`, in the extensions. The listener
/// closes once `shutdown` resolves, before the drain starts.
pub(crate) async fn serve<S>(
    app: S,
    listener: TcpListener,
    keepalive: Duration,
    connections: Connections,
    shutdown: impl Future<Output = ()>,
    drain: Duration,
) where
    S: Service<Request, Response = Response, Error = Infallible> + Clone + Send + 'static,
    S::Future: Send + 'static,
{
    let _close_on_drop = connections.close_on_drop();
    let http = builder(keepalive);
    let open = connections.clone();
    clients(listener)
        .take_until(shutdown)
        .for_each(move |(stream, socket)| {
            open.spawn(exchange(
                http.clone(),
                stream,
                socket,
                app.clone(),
                open.clone(),
            ));
            ready(())
        })
        .await;
    connections.shut_down(drain).await;
}

fn clients(listener: TcpListener) -> impl Stream<Item = (TcpStream, ClientSocket)> {
    stream::unfold(listener, |listener| async move {
        let client = accept(&listener).await;
        Some((client, listener))
    })
    .filter_map(ready)
}

async fn accept(listener: &TcpListener) -> Option<(TcpStream, ClientSocket)> {
    let (stream, peer) = match listener.accept().await {
        Ok(accepted) => accepted,
        Err(error) => {
            back_off(error).await;
            return None;
        }
    };
    let local = match stream.local_addr() {
        Ok(local) => local,
        Err(error) => {
            tracing::debug!(%error, "client connection closed before it was served");
            return None;
        }
    };
    if let Err(error) = stream.set_nodelay(true) {
        tracing::warn!(%error, "could not disable Nagle on a client connection");
    }
    Some((stream, ClientSocket { peer, local }))
}

/// One client connection, from its first request until it ends. Once the gateway drains, the
/// connection finishes the exchange it is in and closes.
async fn exchange<S>(
    http: http1::Builder,
    stream: TcpStream,
    socket: ClientSocket,
    app: S,
    connections: Connections,
) where
    S: Service<Request, Response = Response, Error = Infallible> + Clone + Send + 'static,
    S::Future: Send + 'static,
{
    let per_connection = ServiceBuilder::new()
        .map_request(|request: Request<Incoming>| request.map(Body::new))
        .layer(Extension(socket))
        .layer(Extension(ConnectInfo(socket.peer)))
        .service(app);
    let connection = http
        .serve_connection(
            TokioIo::new(stream),
            TowerToHyperService::new(per_connection),
        )
        .with_upgrades();
    let mut connection = pin!(connection);
    let ended = tokio::select! {
        ended = connection.as_mut() => ended,
        () = connections.draining() => {
            connection.as_mut().graceful_shutdown();
            connection.await
        }
    };
    if let Err(error) = ended {
        tracing::debug!(%error, "client connection ended with an error");
    }
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
