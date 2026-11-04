use std::time::Duration;

use litellm_http::ClientVariant;
use litellm_model_catalog_refresh::{
    Attempt, FetchRequest, Pause, Refreshed, Retrying, SourceClients, SourceFetch, attempt,
    refresh, refused, resume,
};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

/// One fetch of the cost map, or a whole refresh
#[pyclass(frozen, module = "litellm.rust_bridge._native")]
pub(crate) enum ModelCostMapFetch {
    /// The document's bytes for Python to decode, with the `git_blob_id` of those bytes
    Adopted {
        body: Py<PyBytes>,
        revision: String,
        etag: Option<String>,
    },
    NotModified {},
    Failed {
        reason: String,
    },
    /// Only from `fetch_model_cost_map_once`, to resume with `refresh_model_cost_map`
    Retryable {
        reason: String,
        retry_after: Option<f64>,
    },
}

fn draw_jitter(rng: &Bound<'_, PyAny>) -> PyResult<f64> {
    rng.call_method1(intern!(rng.py(), "uniform"), (0.0, 1.0))?
        .extract()
}

/// One cost map refresh, built where it is created so the thread that runs it never reads Python settings or
/// imports. `sleep` is `None` to sleep in Rust, `on_retry(attempt, max_attempts, reason, wait_seconds)` runs before
/// each wait
#[pyclass(
    frozen,
    module = "litellm.rust_bridge._native",
    name = "ModelCostMapRefresh"
)]
pub(crate) struct PyRefresh {
    run: Run,
    rng: Py<PyAny>,
    on_retry: Py<PyAny>,
    sleep: Option<Py<PyAny>>,
}

#[pymethods]
impl PyRefresh {
    #[new]
    #[pyo3(signature = (url, *, timeout, max_attempts, rng, on_retry, sleep = None))]
    fn new(
        py: Python<'_>,
        url: &str,
        timeout: f64,
        max_attempts: i64,
        rng: Py<PyAny>,
        on_retry: Py<PyAny>,
        sleep: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        let http = crate::http::host_client(py, ClientVariant::NoRedirect)?;
        Ok(Self::with_client(
            http,
            url,
            timeout,
            max_attempts,
            rng,
            on_retry,
            sleep,
        ))
    }

    /// The refresh `import litellm` starts, for `run_detached`. Its client is built fresh and never pooled, so its
    /// connections end with its runtime and a fork mid-fetch leaves no shared lock or socket in the child
    #[staticmethod]
    #[pyo3(signature = (url, *, timeout, max_attempts, rng, on_retry))]
    fn at_import(
        py: Python<'_>,
        url: &str,
        timeout: f64,
        max_attempts: i64,
        rng: Py<PyAny>,
        on_retry: Py<PyAny>,
    ) -> PyResult<Self> {
        let http = crate::http::unpooled_host_client(py, ClientVariant::NoRedirect)?;
        Ok(Self::with_client(
            http,
            url,
            timeout,
            max_attempts,
            rng,
            on_retry,
            None,
        ))
    }

    /// Attempt 1 alone, which `get_model_cost_map` makes before deciding to retry in the background
    fn attempt(&self, py: Python<'_>) -> PyResult<ModelCostMapFetch> {
        let run = self.run.clone();
        let first = litellm_host_python::run_sync_value(py, async move {
            Ok(match &run.source {
                Ok(fetch) => attempt(fetch, &run.request).await,
                Err(reason) => reason.clone(),
            })
        })?;
        Python::attach(|py| Ok(from_attempt(py, first)))
    }

    /// Every attempt with the GIL released, resuming after `first` when it is a retryable attempt 1
    #[pyo3(signature = (first = None))]
    fn run(
        &self,
        py: Python<'_>,
        first: Option<&Bound<'_, ModelCostMapFetch>>,
    ) -> PyResult<ModelCostMapFetch> {
        let first = first.map(|first| to_attempt(py, first.get()));
        let run = self.run.clone();
        let mut pause = self.pause(py, false);
        let refreshed = litellm_host_python::run_sync_value(py, async move {
            run.refresh(first, &mut pause).await
        })?;
        Python::attach(|py| Ok(from_refreshed(py, refreshed)))
    }

    /// `run` on a runtime that ends with the call, for the refresh `import litellm` starts. The process starts no
    /// runtime thread, so it can still fork workers or reserve itself for forking afterwards
    fn run_detached(&self, py: Python<'_>) -> PyResult<ModelCostMapFetch> {
        let run = self.run.clone();
        let mut pause = self.pause(py, false);
        let refreshed = litellm_host_python::run_detached_value(py, run.refresh(None, &mut pause))?;
        Python::attach(|py| Ok(from_refreshed(py, refreshed)))
    }

    /// `run` as an awaitable, where `sleep` returns an awaitable
    fn arun<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let run = self.run.clone();
        let mut pause = self.pause(py, true);
        litellm_host_python::run_async_value(py, async move {
            let refreshed = run.refresh(None, &mut pause).await?;
            Ok(Python::attach(|py| from_refreshed(py, refreshed)))
        })
    }
}

impl PyRefresh {
    fn with_client(
        http: litellm_http::Client,
        url: &str,
        timeout: f64,
        max_attempts: i64,
        rng: Py<PyAny>,
        on_retry: Py<PyAny>,
        sleep: Option<Py<PyAny>>,
    ) -> Self {
        let env = |name: &str| std::env::var(name).ok();
        let source = SourceFetch::new(url, SourceClients { http, env: &env })
            .map_err(|error| refused(url, &error));
        Self {
            run: Run {
                source,
                request: request(timeout),
                max_attempts,
            },
            rng,
            on_retry,
            sleep,
        }
    }

    fn pause(&self, py: Python<'_>, asynchronous: bool) -> PythonPause {
        PythonPause::new(
            self.rng.clone_ref(py),
            self.on_retry.clone_ref(py),
            self.sleep.as_ref().map(|sleep| sleep.clone_ref(py)),
            asynchronous,
        )
    }
}

#[derive(Clone)]
struct Run {
    source: Result<SourceFetch, Attempt>,
    request: FetchRequest,
    max_attempts: i64,
}

impl Run {
    async fn refresh(self, first: Option<Attempt>, pause: &mut PythonPause) -> PyResult<Refreshed> {
        let fetch = match self.source {
            Ok(fetch) => fetch,
            Err(Attempt::Failed { reason }) => return Ok(Refreshed::Failed { reason }),
            Err(_) => unreachable!("a source that cannot be built is a final failure"),
        };
        match first {
            Some(first) => resume(&fetch, &self.request, 1, self.max_attempts, first, pause).await,
            None => refresh(&fetch, &self.request, self.max_attempts, pause).await,
        }
    }
}

fn request(timeout: f64) -> FetchRequest {
    FetchRequest {
        if_none_match: None,
        timeout: Duration::try_from_secs_f64(timeout).unwrap_or(Duration::MAX),
    }
}

fn from_attempt(py: Python<'_>, attempt: Attempt) -> ModelCostMapFetch {
    match attempt {
        Attempt::Fetched { body, etag } => ModelCostMapFetch::Adopted {
            revision: litellm_model_catalog::revision(&body),
            body: PyBytes::new(py, &body).unbind(),
            etag,
        },
        Attempt::NotModified => ModelCostMapFetch::NotModified {},
        Attempt::Retryable {
            reason,
            retry_after,
        } => ModelCostMapFetch::Retryable {
            reason,
            retry_after,
        },
        Attempt::Failed { reason } => ModelCostMapFetch::Failed { reason },
    }
}

fn to_attempt(py: Python<'_>, fetch: &ModelCostMapFetch) -> Attempt {
    match fetch {
        ModelCostMapFetch::Adopted { body, etag, .. } => Attempt::Fetched {
            body: body.bind(py).as_bytes().to_vec(),
            etag: etag.clone(),
        },
        ModelCostMapFetch::NotModified {} => Attempt::NotModified,
        ModelCostMapFetch::Failed { reason } => Attempt::Failed {
            reason: reason.clone(),
        },
        ModelCostMapFetch::Retryable {
            reason,
            retry_after,
        } => Attempt::Retryable {
            reason: reason.clone(),
            retry_after: *retry_after,
        },
    }
}

fn from_refreshed(py: Python<'_>, refreshed: Refreshed) -> ModelCostMapFetch {
    match refreshed {
        Refreshed::Adopted {
            body,
            revision,
            etag,
        } => ModelCostMapFetch::Adopted {
            body: PyBytes::new(py, &body).unbind(),
            revision,
            etag,
        },
        Refreshed::NotModified => ModelCostMapFetch::NotModified {},
        Refreshed::Failed { reason } => ModelCostMapFetch::Failed { reason },
    }
}

/// Python's `rng`, `sleep` and retry report. A failure in any of them ends the refresh at the next sleep with that
/// exception, as main's loop raised it
struct PythonPause {
    rng: Py<PyAny>,
    on_retry: Py<PyAny>,
    sleep: Option<Py<PyAny>>,
    asynchronous: bool,
    failed: Option<PyErr>,
}

impl PythonPause {
    fn new(
        rng: Py<PyAny>,
        on_retry: Py<PyAny>,
        sleep: Option<Py<PyAny>>,
        asynchronous: bool,
    ) -> Self {
        Self {
            rng,
            on_retry,
            sleep,
            asynchronous,
            failed: None,
        }
    }
}

impl Pause for PythonPause {
    type Stop = PyErr;

    fn jitter(&mut self) -> f64 {
        Python::attach(|py| draw_jitter(self.rng.bind(py))).unwrap_or_else(|error| {
            self.failed.get_or_insert(error);
            0.0
        })
    }

    fn retrying(&mut self, retrying: &Retrying<'_>) {
        let reported = Python::attach(|py| {
            self.on_retry
                .call1(
                    py,
                    (
                        retrying.attempt,
                        retrying.max_attempts,
                        retrying.reason,
                        retrying.wait_seconds,
                    ),
                )
                .map(drop)
        });
        if let Err(error) = reported {
            self.failed.get_or_insert(error);
        }
    }

    async fn sleep(&mut self, seconds: f64) -> PyResult<()> {
        if let Some(error) = self.failed.take() {
            return Err(error);
        }
        let Some(sleep) = &self.sleep else {
            tokio::time::sleep(Duration::try_from_secs_f64(seconds).unwrap_or_default()).await;
            return Ok(());
        };
        if !self.asynchronous {
            return Python::attach(|py| sleep.call1(py, (seconds,)).map(drop));
        }
        let slept = Python::attach(|py| {
            pyo3_async_runtimes::tokio::into_future(sleep.bind(py).call1((seconds,))?)
        })?;
        slept.await.map(drop)
    }
}
