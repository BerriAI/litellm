use std::sync::{Arc, Mutex, PoisonError};
use std::thread::JoinHandle;
use std::time::Duration;

use litellm_host_python::{release_gil, reserved_for_forking};
use litellm_http::HttpClientConfig;
use litellm_model_catalog_refresh::{RetryPolicy, Source};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use tokio::sync::watch;

use super::fetch::{Outcome, drive_async, integrity_limits};
use super::{CatalogHandle, Fetched};

const HOOKS_MODULE: &str = "litellm.litellm_core_utils.get_model_cost_map";
const ATTEMPTS: u32 = 3;
const TIMEOUT: Duration = Duration::from_secs(5);

/// One background refresh of the bound map per `start` (decisions 2 and 4). Rust owns the loop and installs through
/// `Catalog.install`, so Python only starts it, stops it and restarts it after a fork
#[pyclass(frozen, module = "litellm.rust_bridge._native")]
pub(crate) struct Refresher {
    catalog: Py<CatalogHandle>,
    source: Source,
    /// Only held to swap state, never across a call into Python or a network wait
    state: Mutex<RefresherState>,
}

enum RefresherState {
    Idle,
    Running {
        cancel: watch::Sender<bool>,
        thread: JoinHandle<()>,
        pid: u32,
    },
    Stopped,
}

/// What a running refresh hands back to Python, at most once
struct Run {
    catalog: Py<CatalogHandle>,
    source: Source,
    seen_generation: u64,
}

impl Refresher {
    pub(crate) fn new(catalog: Py<CatalogHandle>, source: Source) -> Self {
        Self {
            catalog,
            source,
            state: Mutex::new(RefresherState::Idle),
        }
    }

    fn swap(&self, next: RefresherState) -> RefresherState {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        std::mem::replace(&mut *state, next)
    }
}

#[pymethods]
impl Refresher {
    /// Starts the refresh thread. A no-op in a process reserved for forking, or when one already runs in this pid
    fn start(&self, py: Python<'_>) -> PyResult<()> {
        self.start_after(py, Duration::ZERO)
    }

    /// Cancels a held fetch or sleep and joins the thread with the GIL released. Idempotent
    fn stop(&self, py: Python<'_>) {
        if let RefresherState::Running {
            cancel,
            thread,
            pid,
        } = self.swap(RefresherState::Stopped)
        {
            let _ = cancel.send(true);
            if pid == std::process::id() {
                release_gil(py, move || {
                    let _ = thread.join();
                });
            }
        }
    }

    /// In a forked child, forgets the parent's thread without joining it, and refreshes again unless the parent
    /// already installed the remote map. A jitter of up to a second spreads the workers' fetches
    fn after_fork(&self, py: Python<'_>) -> PyResult<()> {
        let inherited = {
            let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
            std::mem::replace(&mut *state, RefresherState::Idle)
        };
        let was_running = matches!(inherited, RefresherState::Running { .. });
        std::mem::forget(inherited);
        let remote_installed = self
            .catalog
            .get()
            .provenance()
            .is_some_and(|provenance| provenance.is_remote());
        if was_running && !remote_installed {
            let jitter = Duration::from_millis(u64::from(std::process::id() % 1000));
            self.start_after(py, jitter)?;
        }
        Ok(())
    }

    fn __enter__(slf: Py<Self>, py: Python<'_>) -> PyResult<Py<Self>> {
        slf.get().start(py)?;
        Ok(slf)
    }

    #[pyo3(signature = (*_exc))]
    fn __exit__(&self, py: Python<'_>, _exc: &Bound<'_, pyo3::types::PyTuple>) {
        self.stop(py);
    }
}

impl Refresher {
    fn start_after(&self, py: Python<'_>, delay: Duration) -> PyResult<()> {
        let pid = std::process::id();
        if reserved_for_forking() {
            return Ok(());
        }
        {
            let state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
            if matches!(&*state, RefresherState::Running { pid: running, .. } if *running == pid) {
                return Ok(());
            }
        }
        let policy = RetryPolicy {
            max_attempts: ATTEMPTS,
            timeout: TIMEOUT,
            limits: integrity_limits(py)?,
            ..RetryPolicy::default()
        };
        let config = crate::http::call_config(py, &PyDict::new(py), false)?;
        let run = Run {
            catalog: self.catalog.clone_ref(py),
            source: self.source.clone(),
            seen_generation: self.catalog.get().install_generation(),
        };
        let (cancel, cancelled) = watch::channel(false);
        let source = self.source.clone();
        let thread = std::thread::Builder::new()
            .name("litellm-model-cost-map-refresh".to_owned())
            .spawn(move || refresh(&source, policy, config, delay, cancelled, run))
            .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))?;
        if let RefresherState::Running { cancel, .. } = self.swap(RefresherState::Running {
            cancel,
            thread,
            pid,
        }) {
            let _ = cancel.send(true);
        }
        Ok(())
    }
}

fn refresh(
    source: &Source,
    policy: RetryPolicy,
    config: HttpClientConfig,
    delay: Duration,
    mut cancelled: watch::Receiver<bool>,
    run: Run,
) {
    let Ok(runtime) = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
    else {
        return;
    };
    let outcome = runtime.block_on(async {
        tokio::select! {
            outcome = async {
                tokio::time::sleep(delay).await;
                drive_async(source, policy, config).await
            } => Some(outcome),
            _ = cancelled.wait_for(|stop| *stop) => None,
        }
    });
    runtime.shutdown_timeout(Duration::from_secs(1));
    let Some(outcome) = outcome else {
        return;
    };
    if reserved_for_forking() {
        return;
    }
    let _ = Python::try_attach(|py| hand_over(py, outcome, run));
}

/// Installs a fetched map through Python's install steps, unless an install landed since the refresh started, in
/// which case the newer map stays and this one is dropped
fn hand_over(py: Python<'_>, outcome: Outcome, run: Run) -> PyResult<()> {
    let hooks = py.import(HOOKS_MODULE)?;
    let url = run.source.to_string();
    match outcome {
        Outcome::Fetched {
            parsed,
            revision,
            etag,
        } => {
            if run.catalog.get().install_generation() != run.seen_generation {
                return hooks
                    .call_method1("note_refresh_dropped", (url,))
                    .map(drop);
            }
            let fetched = Fetched {
                parsed: Arc::new(parsed),
                revision,
                etag,
                source: Some(run.source),
                started_at: std::time::SystemTime::now(),
            };
            hooks
                .call_method1("install_refreshed", (fetched, run.catalog, url))
                .map(drop)
        }
        Outcome::Unavailable { reason, .. } => hooks
            .call_method1("note_refresh_failed", (url, reason))
            .map(drop),
    }
}
