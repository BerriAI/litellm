use std::time::Duration;

use axum::{
    body::Body,
    extract::Request,
    http::{HeaderMap, Method, StatusCode, Version, header},
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
pub(crate) struct Upgrades {
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
        crate::hop::strip(response.headers_mut());
        crate::hop::restore_websocket_handshake(response.headers_mut());
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
