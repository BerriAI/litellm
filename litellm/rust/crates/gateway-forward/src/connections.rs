use std::time::Duration;

use tokio_util::{
    sync::{CancellationToken, DropGuard},
    task::TaskTracker,
};

/// Every client connection and WebSocket relay the gateway runs, so shutdown can wait for them
/// and then close whatever outlives the drain.
#[derive(Clone, Default)]
pub(crate) struct Connections {
    tasks: TaskTracker,
    draining: CancellationToken,
    closed: CancellationToken,
}

impl Connections {
    /// Runs `work` until it finishes or the gateway closes every connection.
    pub fn spawn(&self, work: impl Future<Output = ()> + Send + 'static) {
        let closed = self.closed.clone();
        self.tasks.spawn(async move {
            tokio::select! {
                () = closed.cancelled() => {}
                () = work => {}
            }
        });
    }

    pub async fn draining(&self) {
        self.draining.cancelled().await;
    }

    pub fn is_draining(&self) -> bool {
        self.draining.is_cancelled()
    }

    pub async fn until_closed<T>(&self, future: impl Future<Output = T>) -> Option<T> {
        tokio::select! {
            () = self.closed.cancelled() => None,
            output = future => Some(output),
        }
    }

    pub fn close_on_drop(&self) -> DropGuard {
        self.closed.clone().drop_guard()
    }

    /// Lets open requests and WebSockets finish for up to `drain`, then closes whatever is left.
    pub async fn shut_down(&self, drain: Duration) {
        self.draining.cancel();
        if tokio::time::timeout(drain, self.wait()).await.is_err() {
            tracing::warn!(
                ?drain,
                "closing connections still open after the shutdown drain"
            );
        }
        self.closed.cancel();
        self.wait().await;
    }

    async fn wait(&self) {
        self.tasks.close();
        self.tasks.wait().await;
    }
}
