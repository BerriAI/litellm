use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use litellm_host_python::release_gil;
use litellm_http::media::PublicDnsResolver;
use litellm_http::{ClientVariant, HttpClientConfig, HttpClientPool};
use litellm_model_catalog::IntegrityLimits;
use litellm_model_catalog_refresh::{
    Installed, Origin, RetryPolicy, Source, Step, Transports, fetch_with_retry,
};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::{BUNDLED, BUNDLED_REVISION, Fetched, Parsed, Unavailable, bundled, remote};

#[pyclass(frozen, module = "litellm.rust_bridge._native")]
pub(crate) struct Remote {
    #[pyo3(get)]
    url: String,
}

#[pyclass(frozen, module = "litellm.rust_bridge._native")]
pub(crate) struct File {
    #[pyo3(get)]
    path: String,
}

#[pyclass(frozen, module = "litellm.rust_bridge._native")]
pub(crate) struct Bundled;

/// One snapshot of where the installed map came from, replacing main's field-by-field `ModelCostMapSourceInfo`
#[pyclass(frozen, skip_from_py_object, module = "litellm.rust_bridge._native")]
#[derive(Clone)]
pub(crate) struct Provenance {
    /// `"local"` or `"remote"`, main's values
    #[pyo3(get)]
    source: &'static str,
    #[pyo3(get)]
    url: Option<String>,
    #[pyo3(get)]
    is_env_forced: bool,
    #[pyo3(get)]
    fallback_reason: Option<String>,
    loaded_at: SystemTime,
    #[pyo3(get)]
    source_revision: String,
    #[pyo3(get)]
    etag: Option<String>,
}

#[pymethods]
impl Remote {
    #[new]
    fn new(url: String) -> Self {
        Self { url }
    }
}

#[pymethods]
impl File {
    #[new]
    fn new(path: PathBuf) -> Self {
        Self {
            path: path.to_string_lossy().into_owned(),
        }
    }
}

#[pymethods]
impl Bundled {
    #[new]
    fn new() -> Self {
        Self
    }
}

#[pymethods]
impl Provenance {
    /// When the load that installed this map started, as an aware UTC datetime
    #[getter]
    fn loaded_at<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let seconds = self
            .loaded_at
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();
        let datetime = py.import("datetime")?;
        let utc = datetime.getattr("timezone")?.getattr("utc")?;
        datetime
            .getattr("datetime")?
            .call_method1("fromtimestamp", (seconds, utc))
    }
}

impl Provenance {
    pub(crate) fn of(
        fetched: &Fetched,
        env_forced: bool,
        fallback_reason: Option<String>,
        url: Option<String>,
    ) -> Self {
        Self {
            source: if fetched.source.is_some() {
                "remote"
            } else {
                "local"
            },
            url: url.or_else(|| fetched.source.as_ref().map(ToString::to_string)),
            is_env_forced: env_forced,
            fallback_reason,
            loaded_at: fetched.started_at,
            source_revision: fetched.revision.clone(),
            etag: fetched.etag.clone(),
        }
    }
}

/// What a Python `Source` names
pub(crate) enum Requested {
    Bundled,
    Source(Source),
    Invalid(String),
}

pub(crate) fn requested(source: &Bound<'_, PyAny>) -> PyResult<Requested> {
    if source.is_instance_of::<Bundled>() {
        return Ok(Requested::Bundled);
    }
    if let Ok(remote) = source.cast::<Remote>() {
        let url = &remote.get().url;
        return Ok(Source::parse(url).map_or_else(
            |error| Requested::Invalid(error.to_string()),
            Requested::Source,
        ));
    }
    if let Ok(file) = source.cast::<File>() {
        return Ok(Requested::Source(Source::File(PathBuf::from(
            &file.get().path,
        ))));
    }
    Err(PyTypeError::new_err(
        "source must be Remote, File or Bundled",
    ))
}

enum Outcome {
    Fetched {
        parsed: Parsed,
        revision: String,
        etag: Option<String>,
    },
    Unavailable {
        reason: String,
        retry_after: Option<Duration>,
    },
}

/// Fetches with the GIL released, retrying under main's policy, and validates the body before returning it
pub(crate) fn fetch(
    py: Python<'_>,
    requested: Requested,
    timeout: f64,
    attempts: u32,
) -> PyResult<Py<PyAny>> {
    let source = match requested {
        Requested::Bundled => return Ok(bundled().into_pyobject(py)?.into_any().unbind()),
        Requested::Invalid(reason) => return unavailable(py, reason, None),
        Requested::Source(source) => source,
    };
    let timeout = Duration::try_from_secs_f64(timeout)
        .map_err(|_| PyValueError::new_err("timeout must be a non-negative number of seconds"))?;
    let policy = RetryPolicy {
        max_attempts: attempts.max(1),
        timeout,
        limits: integrity_limits(py)?,
        ..RetryPolicy::default()
    };
    let config = crate::http::call_config(py, &PyDict::new(py), false)?;
    let started_at = SystemTime::now();
    let fetched_from = source.clone();
    match release_gil(py, move || drive(&source, policy, config)) {
        Outcome::Fetched {
            parsed,
            revision,
            etag,
        } => Ok(Fetched {
            parsed: Arc::new(parsed),
            revision,
            etag,
            source: Some(fetched_from),
            started_at,
        }
        .into_pyobject(py)?
        .into_any()
        .unbind()),
        Outcome::Unavailable {
            reason,
            retry_after,
        } => unavailable(py, reason, retry_after),
    }
}

fn unavailable(
    py: Python<'_>,
    reason: String,
    retry_after: Option<Duration>,
) -> PyResult<Py<PyAny>> {
    Ok(Unavailable {
        reason,
        retry_after: retry_after.map(|wait| wait.as_secs_f64()),
    }
    .into_pyobject(py)?
    .into_any()
    .unbind())
}

/// Main's `MODEL_COST_MAP_MIN_MODEL_COUNT` and `MODEL_COST_MAP_MAX_SHRINK_RATIO`, which are env-overridable
fn integrity_limits(py: Python<'_>) -> PyResult<IntegrityLimits> {
    let constants = py.import("litellm.constants")?;
    let minimum: usize = constants
        .getattr("MODEL_COST_MAP_MIN_MODEL_COUNT")?
        .extract()?;
    let ratio: f64 = constants
        .getattr("MODEL_COST_MAP_MAX_SHRINK_RATIO")?
        .extract()?;
    IntegrityLimits::new(minimum, ratio).ok_or_else(|| {
        PyValueError::new_err("MODEL_COST_MAP_MAX_SHRINK_RATIO must be between 0 and 1")
    })
}

/// A current-thread runtime of its own, so a fetch at import never starts the shared native runtime, and a client
/// built inside it, so no pooled connection outlives the runtime that spawned it
fn drive(source: &Source, policy: RetryPolicy, config: HttpClientConfig) -> Outcome {
    if tokio::runtime::Handle::try_current().is_ok() {
        return Outcome::Unavailable {
            reason: "the model cost map fetch cannot run from a Tokio context".to_owned(),
            retry_after: None,
        };
    }
    let runtime = match tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(error) => {
            return Outcome::Unavailable {
                reason: format!("the model cost map fetch runtime could not start: {error}"),
                retry_after: None,
            };
        }
    };
    let step = runtime.block_on(async {
        let pool = HttpClientPool::new(Arc::new(PublicDnsResolver));
        let client = match pool.client(&config, ClientVariant::Provider) {
            Ok(client) => client,
            Err(error) => {
                return Step::Done {
                    reason: format!("the HTTP client could not be built: {error}"),
                    retry_after: None,
                };
            }
        };
        let installed = Installed {
            origin: Origin::Bundled,
            etag: None,
            revision: BUNDLED_REVISION.clone(),
            model_count: BUNDLED.raw.model_count(),
        };
        fetch_with_retry(
            source,
            &installed,
            policy,
            None,
            &Transports {
                http: &client,
                s3: None,
            },
            tokio::time::sleep,
        )
        .await
    });
    runtime.shutdown_timeout(Duration::from_secs(1));
    match step {
        Step::Adopt {
            catalog,
            installed,
            errors,
        } => Outcome::Fetched {
            parsed: remote(*catalog, &errors),
            revision: installed.revision,
            etag: installed.etag,
        },
        Step::Done {
            reason,
            retry_after,
        } => Outcome::Unavailable {
            reason,
            retry_after,
        },
        Step::Fetch(_) | Step::Sleep(_) => Outcome::Unavailable {
            reason: "the model cost map fetch stopped before it finished".to_owned(),
            retry_after: None,
        },
    }
}
