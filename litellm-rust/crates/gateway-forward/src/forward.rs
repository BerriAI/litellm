use std::{
    net::{Ipv4Addr, SocketAddr},
    time::Duration,
};

use axum::{
    Router,
    body::{Body, HttpBody},
    extract::{ConnectInfo, DefaultBodyLimit, Request, State},
    http::{HeaderValue, StatusCode, Uri, Version, header},
    response::Response,
};
use hyper_util::{
    client::legacy::{Client, connect::HttpConnector},
    rt::{TokioExecutor, TokioTimer},
};
use tokio::net::TcpListener;

use crate::{Error, hop, websocket::Upgrades};

pub const PEER_HEADER: &str = "x-litellm-rust-peer";

#[derive(Clone)]
pub struct Forward {
    upstream: Uri,
    client: Client<HttpConnector, Body>,
    upgrades: Upgrades,
}

impl Forward {
    /// `upstream_keepalive` is how long the upstream keeps an idle connection open. Pooled
    /// connections are dropped well before that, so a request never lands on one it is closing.
    pub fn new(upstream: &str, upstream_keepalive: Duration) -> Result<Self, Error> {
        let upstream: Uri = upstream.parse().map_err(Error::UpstreamUri)?;
        if upstream.scheme_str() != Some("http")
            || !upstream
                .host()
                .and_then(|host| host.parse::<Ipv4Addr>().ok())
                .is_some_and(|ip| ip.is_loopback())
            || upstream
                .authority()
                .is_some_and(|authority| authority.as_str().contains('@'))
            || upstream.path() != "/"
            || upstream.query().is_some()
        {
            return Err(Error::Upstream);
        }
        let client = Client::builder(TokioExecutor::new())
            .pool_timer(TokioTimer::new())
            .pool_idle_timeout(upstream_keepalive / 2)
            .build_http();
        Ok(Self {
            upstream,
            client,
            upgrades: Upgrades::default(),
        })
    }

    pub fn router(self) -> Router {
        Router::new()
            .fallback(forward)
            .layer(DefaultBodyLimit::disable())
            .with_state(self)
    }

    /// Serves until `shutdown` resolves, then gives open WebSockets a bounded drain before closing them.
    pub async fn serve(
        self,
        listener: TcpListener,
        shutdown: impl Future<Output = ()> + Send + 'static,
    ) -> std::io::Result<()> {
        let upgrades = self.upgrades.clone();
        let _close_on_drop = upgrades.close_on_drop();
        let draining = upgrades.clone();
        let result = axum::serve(
            listener,
            self.router()
                .into_make_service_with_connect_info::<SocketAddr>(),
        )
        .with_graceful_shutdown(async move {
            shutdown.await;
            draining.begin_shutdown();
        })
        .await;
        if result.is_err() {
            upgrades.close();
        }
        upgrades.wait().await;
        result
    }
}

async fn forward(
    State(state): State<Forward>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    mut request: Request,
) -> Result<Response, StatusCode> {
    let downstream = state.upgrades.take_handshake(&mut request)?;
    let mut uri = state.upstream.into_parts();
    uri.path_and_query = request.uri().path_and_query().cloned();
    *request.uri_mut() = Uri::from_parts(uri).map_err(|_| StatusCode::BAD_GATEWAY)?;
    *request.version_mut() = Version::HTTP_11;
    let chunked = request.body().size_hint().exact().is_none();
    let headers = request.headers_mut();
    hop::strip(headers);
    if chunked {
        // hyper sends no body for a GET with an unknown length unless told it is chunked.
        headers.insert(
            header::TRANSFER_ENCODING,
            HeaderValue::from_static("chunked"),
        );
    }
    if downstream.is_some() {
        hop::restore_websocket_handshake(headers);
    }
    // Forwarding headers pass through untouched so Python sees the request as if the client
    // connected to it directly. Inserting replaces any client-sent copy: the peer is the one
    // value a client must not choose.
    headers.insert(
        PEER_HEADER,
        HeaderValue::from_str(&peer.to_string()).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?,
    );
    let response = state.client.request(request);
    let mut response = match downstream {
        Some(_) => state.upgrades.until_closed(response).await?,
        None => response.await,
    }
    .map_err(|error| {
        tracing::warn!(%error, "python proxy unreachable");
        StatusCode::BAD_GATEWAY
    })?;
    if response.status() == StatusCode::SWITCHING_PROTOCOLS {
        return state.upgrades.switch(downstream, response).await;
    }
    hop::strip(response.headers_mut());
    Ok(response.map(Body::new))
}
