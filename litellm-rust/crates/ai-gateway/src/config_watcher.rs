//! Config file hot-reload watcher.
//!
//! Watches a YAML config file for changes and triggers a router reload.

use std::path::PathBuf;
use std::sync::Arc;
use notify::{Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use tokio::sync::watch;

use litellm_core::router::Router;

/// Spawn a background task that watches the config file for changes.
///
/// When the file changes, the config is reloaded and the new router is sent
/// through the provided channel. Returns the watcher (must be kept alive).
pub fn spawn_config_watcher(
    config_path: PathBuf,
    router_tx: watch::Sender<Arc<Router>>,
) -> Result<RecommendedWatcher, notify::Error> {
    let path = config_path.clone();
    let mut watcher = RecommendedWatcher::new(
        move |result: Result<Event, notify::Error>| {
            if let Ok(event) = result {
                if matches!(event.kind, EventKind::Modify(_) | EventKind::Create(_)) {
                    tracing::info!(path = ?path, "config file changed, reloading");
                    match crate::config::load_router_from_yaml(path.to_str().unwrap_or("")) {
                        Ok(new_router) => {
                            let router = Arc::new(new_router);
                            let _ = router_tx.send(router);
                            tracing::info!("config reload complete");
                        }
                        Err(e) => {
                            tracing::error!(error = %e, "config reload failed, keeping previous config");
                        }
                    }
                }
            }
        },
        notify::Config::default(),
    )?;

    watcher.watch(config_path.as_ref(), RecursiveMode::NonRecursive)?;
    Ok(watcher)
}
