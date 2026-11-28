//! Whether an MCP server the proxy stores forwards a caller's header upstream through its
//! `extra_headers`, which Python's `upstream_credential_headers` then drops from every logged copy
//! of a request's headers. The native app logs those headers as sent, so `admit` leaves s3_v2 to
//! Python while one does. It is read at boot and again on `reload_mcp_servers_job`'s interval, and
//! a read that fails counts as forwarding.

use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

use litellm_repositories::{Database, McpServerRepository};

use crate::periodic::PeriodicJob;

/// What the last read found, shared by every `admit` built over the same state.
#[derive(Clone, Default)]
pub struct McpServers {
    forwards_headers: Arc<AtomicBool>,
    refresh: Option<Arc<PeriodicJob>>,
}

impl McpServers {
    /// Reads `database` now, then every `every`.
    pub(crate) async fn read(database: Database, every: Duration) -> Self {
        let repository = Arc::new(McpServerRepository::new(database));
        let forwards_headers = Arc::new(AtomicBool::new(
            any_server_forwards_headers(&repository).await,
        ));
        let refresh = PeriodicJob::start(every, {
            let forwards = forwards_headers.clone();
            move || {
                let repository = repository.clone();
                let forwards = forwards.clone();
                async move {
                    forwards.store(
                        any_server_forwards_headers(&repository).await,
                        Ordering::Relaxed,
                    );
                }
            }
        });
        Self {
            forwards_headers,
            refresh: Some(Arc::new(refresh)),
        }
    }

    /// As though the last read found `forwards_headers`, with nothing read again.
    #[cfg(test)]
    pub(crate) fn found(forwards_headers: bool) -> Self {
        Self {
            forwards_headers: Arc::new(AtomicBool::new(forwards_headers)),
            refresh: None,
        }
    }

    pub(crate) fn forward_headers(&self) -> bool {
        self.forwards_headers.load(Ordering::Relaxed)
    }

    /// Stops reading the servers again.
    pub(crate) async fn close(&self) {
        if let Some(refresh) = &self.refresh {
            refresh.close().await;
        }
    }
}

async fn any_server_forwards_headers(repository: &McpServerRepository) -> bool {
    repository
        .any_server_forwards_headers()
        .await
        .inspect_err(|error| {
            tracing::warn!(%error, "the stored MCP servers could not be read, so s3_v2 goes to Python");
        })
        .unwrap_or(true)
}
