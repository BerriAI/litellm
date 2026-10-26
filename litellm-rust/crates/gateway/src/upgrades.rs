use std::time::Duration;

use hyper::upgrade::{OnUpgrade, Upgraded};
use hyper_util::rt::TokioIo;
use tokio::{sync::watch, time::Instant};
use tokio_util::{sync::CancellationToken, task::TaskTracker};

pub(crate) const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(10);
const DRAIN_TIMEOUT: Duration = Duration::from_secs(5);

#[derive(Clone)]
pub(crate) struct Upgrades {
    pub tasks: TaskTracker,
    pub cancel: CancellationToken,
    deadline: watch::Sender<Option<Instant>>,
}

impl Upgrades {
    pub fn new() -> Self {
        Self {
            tasks: TaskTracker::new(),
            cancel: CancellationToken::new(),
            deadline: watch::Sender::new(None),
        }
    }

    pub fn begin_shutdown(&self) {
        self.deadline
            .send_replace(Some(Instant::now() + DRAIN_TIMEOUT));
    }

    pub fn is_draining(&self) -> bool {
        self.deadline.borrow().is_some() || self.cancel.is_cancelled()
    }

    pub async fn cancelled(&self) {
        let mut deadline = self.deadline.subscribe();
        tokio::select! {
            _ = self.cancel.cancelled() => {}
            _ = async {
                let at = *deadline.wait_for(Option::is_some).await.expect("sender is owned by self");
                tokio::time::sleep_until(at.expect("waited for shutdown deadline")).await;
            } => {}
        }
    }

    pub fn relay(&self, downstream: OnUpgrade, upstream: Upgraded) {
        let lifecycle = self.clone();
        self.tasks.spawn(async move {
            tokio::select! {
                _ = lifecycle.cancelled() => {}
                _ = async {
                    let upgraded = tokio::time::timeout(HANDSHAKE_TIMEOUT, downstream).await;
                    let Ok(Ok(downstream)) = upgraded else {
                        tracing::debug!("websocket upgrade did not complete");
                        return;
                    };
                    let (mut downstream_read, mut downstream_write) = tokio::io::split(TokioIo::new(downstream));
                    let (mut upstream_read, mut upstream_write) = tokio::io::split(TokioIo::new(upstream));
                    let result = tokio::select! {
                        result = tokio::io::copy(&mut downstream_read, &mut upstream_write) => result,
                        result = tokio::io::copy(&mut upstream_read, &mut downstream_write) => result,
                    };
                    if result.is_err() {
                        tracing::debug!("websocket relay disconnected");
                    }
                } => {}
            }
        });
    }
}
