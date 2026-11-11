mod connections;
mod connector;
mod error;
mod forward;
mod rewrite;
mod server;
mod settings;
mod websocket;

pub use error::Error;
pub(crate) use error::Rejection;
pub use forward::Forward;
pub use rewrite::PEER_HEADER;
pub use settings::{Settings, Upstream};

/// Binds the public listener and forwards to Python until `shutdown` resolves, then drains for
/// `settings.shutdown_drain`.
pub async fn run(
    settings: Settings,
    shutdown: impl Future<Output = ()> + Send + 'static,
) -> Result<(), Error> {
    run_app(settings, |forward| forward, shutdown).await
}

/// Like `run`, but every request goes to the app `app` builds first, which forwards what it does
/// not serve through the `Forward` it was given.
pub async fn run_app<A>(
    settings: Settings,
    app: impl FnOnce(Forward) -> A,
    shutdown: impl Future<Output = ()> + Send + 'static,
) -> Result<(), Error>
where
    A: tower::Service<axum::extract::Request, Response = axum::response::Response, Error = std::convert::Infallible>
        + Clone
        + Send
        + 'static,
    A::Future: Send,
{
    let forward = Forward::new(&settings.upstream, settings.upstream_keepalive);
    let app = app(forward.clone());
    let listener = tokio::net::TcpListener::bind((settings.host.as_str(), settings.port)).await?;
    tracing::info!(address = %listener.local_addr()?, "forwarding gateway listening");
    forward
        .serve_app(app, listener, shutdown, settings.shutdown_drain)
        .await;
    Ok(())
}
