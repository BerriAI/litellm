mod admit;
mod client;
mod connections;
mod connector;
mod error;
mod forward;
mod input;
mod rewrite;
mod server;
mod settings;
mod websocket;

use std::{convert::Infallible, fmt::Display, time::Duration};

use axum::{extract::Request, response::Response};
use tokio::net::TcpListener;
use tower::Service;

use connections::Connections;
pub use error::Error;
pub(crate) use error::NotForwarded;
use forward::Forward;
pub use input::input_closed;
pub use rewrite::{ANSWERED_BY_HEADER, PEER_HEADER};
pub use settings::{Settings, Upstream};

/// Binds the public listener and serves `native` in front of Python until `shutdown` resolves,
/// then drains for `settings.shutdown_drain`. `admit` decides, request by request, which of the
/// two answers.
pub async fn run<S, A, Admitted, Why>(
    settings: Settings,
    native: S,
    admit: A,
    shutdown: impl Future<Output = ()> + Send + 'static,
) -> Result<(), Error>
where
    S: Service<Request, Response = Response, Error = Infallible> + Clone + Send + Sync + 'static,
    S::Future: Send + 'static,
    A: Fn(Request) -> Admitted + Clone + Send + Sync + 'static,
    Admitted: Future<Output = Result<Request, (Box<Request>, Why)>> + Send + 'static,
    Why: Display + Into<&'static str> + Send + 'static,
{
    let listener = TcpListener::bind((settings.host.as_str(), settings.port)).await?;
    tracing::info!(address = %listener.local_addr()?, "forwarding gateway listening");
    serve(
        listener,
        settings.upstream,
        settings.upstream_keepalive,
        settings.shutdown_drain,
        native,
        admit,
        shutdown,
    )
    .await;
    Ok(())
}

/// Serves `native` in front of the Python proxy at `upstream` on `listener` until `shutdown`
/// resolves, then lets open requests and WebSockets finish for up to `drain` before closing
/// whatever is left. Python shuts down alongside and ends long streams itself, so `drain` only
/// backstops a Python that never does. Idle client connections are closed after
/// `upstream_keepalive`, as uvicorn did.
///
/// `admit` sees every request first and either admits it into `native` or refuses it, giving it
/// back as the client sent it, with why. Python answers every refused request. `native` and
/// `admit` are composed here and nowhere else, so no layer a caller builds can wrap a request
/// Python answers.
pub async fn serve<S, A, Admitted, Why>(
    listener: TcpListener,
    upstream: Upstream,
    upstream_keepalive: Duration,
    drain: Duration,
    native: S,
    admit: A,
    shutdown: impl Future<Output = ()>,
) where
    S: Service<Request, Response = Response, Error = Infallible> + Clone + Send + Sync + 'static,
    S::Future: Send + 'static,
    A: Fn(Request) -> Admitted + Clone + Send + Sync + 'static,
    Admitted: Future<Output = Result<Request, (Box<Request>, Why)>> + Send + 'static,
    Why: Display + Into<&'static str> + Send + 'static,
{
    let connections = Connections::default();
    let python = Forward::new(&upstream, upstream_keepalive, connections.clone());
    server::serve(
        admit::in_front_of(python, native, admit),
        listener,
        upstream_keepalive,
        connections,
        shutdown,
        drain,
    )
    .await;
}
