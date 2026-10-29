use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use litellm_host_python::release_gil;
use litellm_http::media::PublicDnsResolver;
use litellm_http::{ClientVariant, HttpClientConfig, HttpClientPool};
use litellm_model_catalog::IntegrityLimits;
use litellm_model_catalog_refresh::{
    Installed, Origin, RetryPolicy, Source, Step, Transports, fetch_with_retry, local_map_forced,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::{BUNDLED, BUNDLED_REVISION, Contents, MapOrigin, Parsed, Unavailable, bundled, remote};

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
    pub(crate) fn of(parsed: &Parsed) -> Self {
        let (source, url, is_env_forced, fallback_reason) = match &parsed.origin {
            MapOrigin::Bundled => ("local", None, false, None),
            MapOrigin::Forced => ("local", None, true, None),
            MapOrigin::Fallback { url, reason } => {
                ("local", Some(url.clone()), false, Some(reason.clone()))
            }
            MapOrigin::Remote(from) => ("remote", Some(from.to_string()), false, None),
        };
        Self {
            source,
            url,
            is_env_forced,
            fallback_reason,
            loaded_at: parsed.started_at,
            source_revision: parsed.revision.clone(),
            etag: parsed.etag.clone(),
        }
    }
}

#[derive(Debug, thiserror::Error)]
enum LoadError {
    #[error("the model cost map load cannot run from a Tokio context")]
    InsideTokio,
    #[error("the model cost map load runtime could not start")]
    Runtime(#[source] std::io::Error),
    #[error("the HTTP client for the model cost map could not be built")]
    Client(#[source] litellm_http::Error),
    #[error(transparent)]
    Refresh(litellm_model_catalog_refresh::Error),
    #[error("the model cost map fetch stopped before it finished")]
    Unfinished,
}

enum Outcome {
    Fetched {
        contents: Contents,
        revision: String,
        etag: Option<String>,
    },
    Unavailable {
        error: LoadError,
        retry_after: Option<Duration>,
    },
}

/// The map the operator configured for `url`. The bundled map when `LITELLM_LOCAL_MODEL_COST_MAP` or
/// `LITELLM_OFFLINE_MODE` forces it, otherwise a fetch with the GIL released, retried under main's policy and
/// validated before it is returned
pub(crate) fn load(
    py: Python<'_>,
    url: String,
    timeout: f64,
    attempts: u32,
) -> PyResult<Py<PyAny>> {
    let started_at = SystemTime::now();
    if local_map_forced(|name| std::env::var(name).ok()) {
        return Ok(Parsed {
            origin: MapOrigin::Forced,
            ..bundled()
        }
        .into_pyobject(py)?
        .into_any()
        .unbind());
    }
    let source = match Source::parse(&url) {
        Ok(source) => source,
        Err(error) => {
            return unavailable(py, LoadError::Refresh(error), None, url, started_at);
        }
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
    let fetched_from = source.clone();
    match release_gil(py, move || drive(&source, policy, config)) {
        Outcome::Fetched {
            contents,
            revision,
            etag,
        } => Ok(Parsed {
            contents: Arc::new(contents),
            revision,
            etag,
            origin: MapOrigin::Remote(fetched_from),
            started_at,
        }
        .into_pyobject(py)?
        .into_any()
        .unbind()),
        Outcome::Unavailable { error, retry_after } => {
            unavailable(py, error, retry_after, url, started_at)
        }
    }
}

fn unavailable(
    py: Python<'_>,
    error: LoadError,
    retry_after: Option<Duration>,
    url: String,
    started_at: SystemTime,
) -> PyResult<Py<PyAny>> {
    Ok(Unavailable {
        reason: litellm_error::Report::new(&error).to_string(),
        retry_after: retry_after.map(|wait| wait.as_secs_f64()),
        url: Some(url),
        started_at,
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
            error: LoadError::InsideTokio,
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
                error: LoadError::Runtime(error),
                retry_after: None,
            };
        }
    };
    let step = runtime.block_on(async {
        let pool = HttpClientPool::new(Arc::new(PublicDnsResolver));
        let client = pool
            .client(&config, ClientVariant::Provider)
            .map_err(LoadError::Client)?;
        let installed = Installed {
            origin: Origin::Bundled,
            etag: None,
            revision: BUNDLED_REVISION.clone(),
            model_count: BUNDLED.raw.model_count(),
        };
        Ok(fetch_with_retry(
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
        .await)
    });
    runtime.shutdown_timeout(Duration::from_secs(1));
    match step {
        Ok(Step::Adopt {
            catalog,
            installed,
            errors,
        }) => Outcome::Fetched {
            contents: remote(*catalog, &errors),
            revision: installed.revision,
            etag: installed.etag,
        },
        Ok(Step::Done { error, retry_after }) => Outcome::Unavailable {
            error: LoadError::Refresh(error),
            retry_after,
        },
        Ok(Step::Fetch(_) | Step::Sleep(_)) => Outcome::Unavailable {
            error: LoadError::Unfinished,
            retry_after: None,
        },
        Err(error) => Outcome::Unavailable {
            error,
            retry_after: None,
        },
    }
}
