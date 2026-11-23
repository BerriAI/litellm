use std::{
    convert::Infallible,
    pin::Pin,
    task::{Context, Poll},
    time::Duration,
};

use axum::{
    body::Body,
    extract::Request,
    http::{StatusCode, Uri},
    response::{IntoResponse, Response},
};
use hyper::body::{Frame, Incoming, SizeHint};
use hyper_util::{
    client::legacy::{Client, connect::HttpConnector},
    rt::{TokioExecutor, TokioTimer},
};
use tower::Service;

use crate::{
    NotForwarded, Upstream, client::ClientSocket, connections::Connections, connector::Connector,
    rewrite, websocket,
};

// h11 counts no headers and only caps a head at 16 KiB, so it accepts as many headers as fit in
// that many shortest lines (`a:\r\n`). Python copies every provider header into its response
// too, so hyper's default of 100 would turn ordinary traffic into 431s and 502s.
pub(crate) const MAX_HEADERS: usize = 16 * 1024 / b"a:\r\n".len();

/// The Python proxy, as the service that forwards a request to it.
#[derive(Clone)]
pub(crate) struct Forward {
    upstream: Uri,
    client: Client<Connector, Body>,
    connections: Connections,
}

impl Forward {
    /// `upstream_keepalive` is how long the upstream keeps an idle connection open. Pooled
    /// connections are dropped well before that, so a request never lands on one it is closing.
    /// WebSocket relays run on `connections`, so shutdown drains them with the client connections.
    pub(crate) fn new(
        upstream: &Upstream,
        upstream_keepalive: Duration,
        connections: Connections,
    ) -> Self {
        let mut connector = HttpConnector::new();
        connector.set_nodelay(true);
        let client = Client::builder(TokioExecutor::new())
            .http1_max_headers(MAX_HEADERS)
            .pool_timer(TokioTimer::new())
            .pool_idle_timeout(upstream_keepalive / 2)
            .set_host(false)
            .build(Connector(connector));
        Self {
            upstream: upstream.uri().clone(),
            client,
            connections,
        }
    }

    async fn forward(&self, mut request: Request) -> Result<Response, NotForwarded> {
        let client = ClientSocket::of(&request)?;
        if let Some(downstream) = websocket::take_handshake(&self.connections, &mut request)? {
            return self.forward_websocket(client, request, downstream).await;
        }
        let request = rewrite::upstream_request(request, &self.upstream, client, false)?;
        let response = self
            .client
            .request(request)
            .await
            .map_err(NotForwarded::from_upstream)?;
        http_response(response)
    }

    async fn forward_websocket(
        &self,
        client: ClientSocket,
        request: Request,
        downstream: hyper::upgrade::OnUpgrade,
    ) -> Result<Response, NotForwarded> {
        let request = rewrite::upstream_request(request, &self.upstream, client, true)?;
        let response = self
            .connections
            .until_closed(self.client.request(request))
            .await
            .ok_or(NotForwarded::ShuttingDown)?
            .map_err(NotForwarded::from_upstream)?;
        if response.status() == StatusCode::SWITCHING_PROTOCOLS {
            return websocket::switch(&self.connections, downstream, response).await;
        }
        http_response(response)
    }
}

/// Forwards each request to Python, which sees it as the client sent it. The serving loop puts the
/// client's socket in the request extensions, as axum's `ConnectInfo` does.
impl Service<Request> for Forward {
    type Response = Response;
    type Error = Infallible;
    type Future = Pin<Box<dyn Future<Output = Result<Response, Infallible>> + Send>>;

    fn poll_ready(&mut self, _: &mut Context<'_>) -> Poll<Result<(), Infallible>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, request: Request) -> Self::Future {
        let python = self.clone();
        Box::pin(async move {
            Ok(python
                .forward(request)
                .await
                .unwrap_or_else(IntoResponse::into_response))
        })
    }
}

fn http_response(response: Response<Incoming>) -> Result<Response, NotForwarded> {
    if response.status() == StatusCode::SWITCHING_PROTOCOLS {
        return Err(NotForwarded::UnrequestedUpgrade);
    }
    Ok(
        rewrite::downstream_response(response)
            .map(|body| Body::new(Relayed { body, failed: None })),
    )
}

/// Python's response body on its way to the client.
struct Relayed {
    body: Incoming,
    failed: Option<hyper::Error>,
}

impl hyper::body::Body for Relayed {
    type Data = <Incoming as hyper::body::Body>::Data;
    type Error = hyper::Error;

    // hyper aborts a client connection whose response body fails without flushing the head it
    // buffered, so a head Python sent right before dying would never arrive and SDKs would retry
    // a request Python already ran. Holding the error for one poll lets hyper flush the head.
    fn poll_frame(
        mut self: Pin<&mut Self>,
        context: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        if let Some(error) = self.failed.take() {
            return Poll::Ready(Some(Err(error)));
        }
        match Pin::new(&mut self.body).poll_frame(context) {
            Poll::Ready(Some(Err(error))) => {
                self.failed = Some(error);
                context.waker().wake_by_ref();
                Poll::Pending
            }
            frame => frame,
        }
    }

    fn is_end_stream(&self) -> bool {
        self.failed.is_none() && self.body.is_end_stream()
    }

    fn size_hint(&self) -> SizeHint {
        self.body.size_hint()
    }
}
