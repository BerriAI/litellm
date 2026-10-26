use std::{
    sync::{
        Arc, Weak,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

use async_trait::async_trait;
use litellm_host_python::release_gil;
use litellm_model_catalog::{
    FetchRequest, Installed, Origin, Reply, RetryPolicy, Revalidation, Step,
};
use litellm_model_catalog_refresh::{Adopted, Checked, Fetch, Install, Refresher, Source, Stale};
use pyo3::{
    PyTraverseError, PyVisit,
    exceptions::{PyRuntimeError, PyTypeError, PyValueError},
    prelude::*,
    types::PyDict,
};

use super::{BUNDLED, ModelRegistry, ParsedRegistry, RowError, row_error};

#[pyclass(frozen, module = "litellm.rust_bridge._native")]
pub(crate) struct FetchedRegistry {
    #[pyo3(get)]
    registry: ModelRegistry,
    #[pyo3(get)]
    revision: String,
    #[pyo3(get)]
    validator: Option<String>,
}

#[pyclass(frozen, module = "litellm.rust_bridge._native")]
pub(crate) struct RegistryNotModified;

#[pyclass(frozen, module = "litellm.rust_bridge._native")]
pub(crate) struct RegistryUnavailable {
    #[pyo3(get)]
    reason: String,
    #[pyo3(get)]
    retryable: bool,
    #[pyo3(get)]
    retry_after: Option<f64>,
}

#[derive(IntoPyObject)]
enum FetchResult {
    Fetched(FetchedRegistry),
    NotModified(RegistryNotModified),
    Unavailable(RegistryUnavailable),
}

fn installed(revision: Option<String>, validator: Option<String>, source: &Source) -> Installed {
    Installed {
        catalog: Arc::clone(&BUNDLED.catalog),
        origin: if revision.is_some() {
            Origin::Remote {
                source: source.id(),
            }
        } else {
            Origin::Bundled
        },
        revision: revision.unwrap_or_default(),
        validator,
    }
}

fn unavailable(reason: String, retryable: bool, after: Option<Duration>) -> FetchResult {
    FetchResult::Unavailable(RegistryUnavailable {
        reason,
        retryable,
        retry_after: after.map(|delay| delay.as_secs_f64()),
    })
}

#[pyfunction]
#[pyo3(signature = (source, *, if_none_match, timeout_seconds))]
pub(crate) fn fetch_once(
    py: Python<'_>,
    source: &str,
    if_none_match: Option<String>,
    timeout_seconds: f64,
) -> PyResult<Py<PyAny>> {
    fetch_with_retry(py, source, if_none_match, timeout_seconds, 1, 30.0)
}

#[pyfunction]
#[pyo3(signature = (source, *, if_none_match, timeout_seconds, max_attempts, max_wait_seconds = 30.0))]
pub(crate) fn fetch_with_retry(
    py: Python<'_>,
    source: &str,
    if_none_match: Option<String>,
    timeout_seconds: f64,
    max_attempts: usize,
    max_wait_seconds: f64,
) -> PyResult<Py<PyAny>> {
    let source = Source::parse(source).map_err(|error| PyValueError::new_err(error.to_string()))?;
    let timeout = duration(timeout_seconds, "timeout_seconds")?;
    let max_wait = duration(max_wait_seconds, "max_wait_seconds")?;
    if max_attempts == 0 {
        return Err(PyValueError::new_err("max_attempts must be positive"));
    }
    if tokio::runtime::Handle::try_current().is_ok() {
        return Err(PyRuntimeError::new_err(
            "model catalog fetch cannot run from a Tokio context",
        ));
    }
    let http = crate::http::call_config(py, &PyDict::new(py), false)?;
    let result = release_gil(py, move || {
        let fetch = match source.fetcher(&http) {
            Ok(fetch) => fetch,
            Err(_) => {
                return unavailable(
                    "model catalog transport configuration failed".into(),
                    false,
                    None,
                );
            }
        };
        let runtime = match tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
        {
            Ok(runtime) => runtime,
            Err(_) => return unavailable("model catalog runtime unavailable".into(), false, None),
        };
        let (state, step) = Revalidation::new(
            source.id(),
            &installed(None, if_none_match, &source),
            RetryPolicy {
                max_attempts,
                max_wait,
                timeout,
                ..RetryPolicy::default()
            },
            None,
        );
        let result = runtime.block_on(drive_fetch(fetch.as_ref(), state, step));
        runtime.shutdown_background();
        result
    });
    Ok(result.into_pyobject(py)?.unbind())
}

fn duration(seconds: f64, name: &str) -> PyResult<Duration> {
    Duration::try_from_secs_f64(seconds)
        .ok()
        .filter(|duration| std::time::Instant::now().checked_add(*duration).is_some())
        .ok_or_else(|| {
            PyValueError::new_err(format!(
                "{name} must be finite, non-negative and within the supported duration range"
            ))
        })
}

async fn drive_fetch(fetch: &dyn Fetch, mut state: Revalidation, mut step: Step) -> FetchResult {
    let mut retry = None;
    loop {
        step = match step {
            Step::Fetch(request) => {
                let reply = fetch.fetch(request).await;
                retry = match &reply {
                    Reply::Retry { after, .. } => Some(*after),
                    _ => None,
                };
                state.on_reply(reply)
            }
            Step::Sleep(delay) => {
                tokio::time::sleep(delay).await;
                state.on_slept()
            }
            Step::Adopt { installed, errors } => {
                return FetchResult::Fetched(FetchedRegistry {
                    registry: ModelRegistry {
                        parsed: Arc::new(ParsedRegistry {
                            catalog: installed.catalog,
                            errors: Arc::from(errors),
                        }),
                        source_revision: Some(installed.revision.clone()),
                        source_validator: installed.validator.clone(),
                    },
                    revision: installed.revision,
                    validator: installed.validator,
                });
            }
            Step::Done { reason: None } => return FetchResult::NotModified(RegistryNotModified),
            Step::Done {
                reason: Some(reason),
            } => {
                return unavailable(reason, retry.is_some(), retry.flatten());
            }
        };
    }
}

struct PythonInstall {
    install: Py<PyAny>,
    checked: Py<PyAny>,
    generation: Py<PyAny>,
    stopped: Arc<AtomicBool>,
}

struct CallbackFetch {
    inner: Arc<dyn Fetch>,
    stopped: Arc<AtomicBool>,
}

#[async_trait]
impl Fetch for CallbackFetch {
    async fn fetch(&self, request: FetchRequest) -> Reply {
        if self.stopped.load(Ordering::Acquire) {
            return Reply::Failed {
                reason: "model catalog callbacks stopped".into(),
            };
        }
        self.inner.fetch(request).await
    }
}

impl PythonInstall {
    fn call<T>(&self, call: impl FnOnce(Python<'_>) -> PyResult<T>) -> Option<T> {
        Python::try_attach(|py| {
            if self.stopped.load(Ordering::Acquire) {
                return None;
            }
            match call(py) {
                Ok(value) => Some(value),
                Err(error) => {
                    self.stopped.store(true, Ordering::Release);
                    error.write_unraisable(py, None);
                    None
                }
            }
        })
        .flatten()
    }
}

impl Install for PythonInstall {
    fn generation(&self) -> u64 {
        self.call(|py| self.generation.call0(py)?.extract(py))
            .unwrap_or_default()
    }

    fn install(&self, seen: u64, adopted: Adopted) -> Result<u64, Stale> {
        let registry = ModelRegistry {
            parsed: Arc::new(ParsedRegistry {
                catalog: adopted.installed.catalog,
                errors: Arc::from([]),
            }),
            source_revision: Some(adopted.installed.revision),
            source_validator: adopted.installed.validator,
        };
        let left_out: Vec<RowError> = adopted.left_out.iter().map(row_error).collect();
        self.call(|py| {
            self.install
                .call1(py, (seen, registry, left_out))?
                .extract::<Option<u64>>(py)
        })
        .flatten()
        .ok_or(Stale)
    }

    fn checked(&self, outcome: Checked) {
        let reason = match outcome {
            Checked::NotModified => None,
            Checked::Failed { reason } => Some(reason),
        };
        self.call(|py| self.checked.call1(py, (reason,)).map(|_| ()));
    }
}

#[pyclass(module = "litellm.rust_bridge._native")]
pub(crate) struct CatalogRefresher {
    inner: Option<Refresher>,
    callbacks: Weak<PythonInstall>,
}

#[pymethods]
impl CatalogRefresher {
    #[new]
    #[pyo3(signature = (source, *, installed_revision, installed_validator, install, checked, generation))]
    fn new(
        py: Python<'_>,
        source: &str,
        installed_revision: Option<String>,
        installed_validator: Option<String>,
        install: Py<PyAny>,
        checked: Py<PyAny>,
        generation: Py<PyAny>,
    ) -> PyResult<Self> {
        let source =
            Source::parse(source).map_err(|error| PyValueError::new_err(error.to_string()))?;
        if !install.bind(py).is_callable()
            || !checked.bind(py).is_callable()
            || !generation.bind(py).is_callable()
        {
            return Err(PyTypeError::new_err(
                "install, checked and generation must be callable",
            ));
        }
        let http = crate::http::call_config(py, &PyDict::new(py), false)?;
        let logger = crate::logger::capture(py);
        let callbacks = Arc::new(PythonInstall {
            install,
            checked,
            generation,
            stopped: Arc::new(AtomicBool::new(false)),
        });
        let weak = Arc::downgrade(&callbacks);
        let inner = release_gil(py, move || {
            let fetch = Arc::new(CallbackFetch {
                inner: source.fetcher(&http).map_err(crate::http::client_error)?,
                stopped: Arc::clone(&callbacks.stopped),
            });
            let installed = installed(installed_revision, installed_validator, &source);
            Ok::<_, PyErr>(
                logger.scope(|| Refresher::start(source.id(), fetch, &installed, callbacks)),
            )
        })?;
        Ok(Self {
            inner: Some(inner),
            callbacks: weak,
        })
    }

    fn stop(&self) {
        if let Some(callbacks) = self.callbacks.upgrade() {
            callbacks.stopped.store(true, Ordering::Release);
        }
        if let Some(inner) = &self.inner {
            inner.stop();
        }
    }

    fn after_fork(&self, py: Python<'_>) {
        if self
            .callbacks
            .upgrade()
            .is_some_and(|callbacks| callbacks.stopped.load(Ordering::Acquire))
        {
            return;
        }
        if let Some(inner) = &self.inner {
            release_gil(py, || inner.after_fork());
        }
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        if let Some(callbacks) = self.callbacks.upgrade() {
            // A running worker owns these edges independently of this Python object.
            if Arc::strong_count(&callbacks) == 2 {
                visit.call(&callbacks.install)?;
                visit.call(&callbacks.checked)?;
                visit.call(&callbacks.generation)?;
            }
        }
        Ok(())
    }

    fn __clear__(&mut self) {
        self.stop();
        self.inner.take();
    }
}

impl Drop for CatalogRefresher {
    fn drop(&mut self) {
        self.stop();
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::CString;

    use pyo3::types::PyList;
    use rstest::{fixture, rstest};

    use super::*;

    #[fixture]
    fn python() {
        Python::initialize();
    }

    fn callbacks(py: Python<'_>, install_result: &str) -> (PythonInstall, Py<PyList>) {
        let locals = PyDict::new(py);
        py.run(&CString::new(format!(
            "events = []\ndef install(seen, registry, errors):\n events.append((seen, registry, errors))\n return {install_result}\ndef checked(reason):\n events.append(reason)\ndef generation():\n return 7"
        )).unwrap(), Some(&locals), None).unwrap();
        let callback = |name| locals.get_item(name).unwrap().unwrap().unbind();
        let result = PythonInstall {
            install: callback("install"),
            checked: callback("checked"),
            generation: callback("generation"),
            stopped: Arc::new(AtomicBool::new(false)),
        };
        (
            result,
            locals
                .get_item("events")
                .unwrap()
                .unwrap()
                .cast_into::<PyList>()
                .unwrap()
                .unbind(),
        )
    }

    #[rstest]
    #[case::adopted("8", Ok(8))]
    #[case::stale("None", Err(Stale))]
    fn install_return_maps_to_generation_or_stale(
        #[from(python)] _python: (),
        #[case] answer: &str,
        #[case] expected: Result<u64, Stale>,
    ) {
        Python::attach(|py| {
            let (callbacks, events) = callbacks(py, answer);
            let source = Source::parse("https://example.test/catalog").unwrap();
            assert_eq!(callbacks.generation(), 7);
            let adopted = Adopted {
                installed: installed(Some("revision".into()), Some("validator".into()), &source),
                left_out: vec![],
            };
            assert_eq!(callbacks.install(7, adopted), expected);
            let event = events.bind(py).get_item(0).unwrap();
            assert_eq!(event.get_item(0).unwrap().extract::<u64>().unwrap(), 7);
            let registry = event
                .get_item(1)
                .unwrap()
                .extract::<PyRef<'_, ModelRegistry>>()
                .unwrap();
            assert!(Arc::ptr_eq(&registry.parsed.catalog, &BUNDLED.catalog));
            assert_eq!(registry.source_revision.as_deref(), Some("revision"));
            assert_eq!(registry.source_validator.as_deref(), Some("validator"));
            assert!(
                event
                    .get_item(2)
                    .unwrap()
                    .cast::<PyList>()
                    .unwrap()
                    .is_empty()
            );
        });
    }

    #[rstest]
    #[case::not_modified(Checked::NotModified, None)]
    #[case::failed(Checked::Failed { reason: "offline".into() }, Some("offline"))]
    fn checked_maps_reason(
        #[from(python)] _python: (),
        #[case] outcome: Checked,
        #[case] expected: Option<&str>,
    ) {
        Python::attach(|py| {
            let (callbacks, events) = callbacks(py, "8");
            callbacks.checked(outcome);
            assert_eq!(
                events
                    .bind(py)
                    .get_item(0)
                    .unwrap()
                    .extract::<Option<String>>()
                    .unwrap()
                    .as_deref(),
                expected
            );
        });
    }

    #[rstest]
    fn stopped_callbacks_are_not_invoked(#[from(python)] _python: ()) {
        Python::attach(|py| {
            let (callbacks, events) = callbacks(py, "8");
            callbacks.stopped.store(true, Ordering::Release);
            assert_eq!(callbacks.generation(), 0);
            callbacks.checked(Checked::NotModified);
            assert!(events.bind(py).is_empty());
        });
    }

    #[rstest]
    #[case::unsupported_scheme("ftp://example.test")]
    #[case::relative("catalog.json")]
    fn invalid_source_precedes_python_config(#[from(python)] _python: (), #[case] source: &str) {
        Python::attach(|py| {
            let error = fetch_once(py, source, None, 1.0).unwrap_err();
            assert!(error.is_instance_of::<PyValueError>(py));
            let error = fetch_with_retry(py, source, None, 1.0, 3, 30.0).unwrap_err();
            assert!(error.is_instance_of::<PyValueError>(py));
        });
    }

    #[rstest]
    #[case::negative(-1.0)]
    #[case::nan(f64::NAN)]
    #[case::infinite(f64::INFINITY)]
    fn invalid_timeout_precedes_python_config(#[from(python)] _python: (), #[case] timeout: f64) {
        Python::attach(|py| {
            let error = fetch_once(py, "https://example.test", None, timeout).unwrap_err();
            assert!(error.is_instance_of::<PyValueError>(py));
        });
    }

    #[rstest]
    #[case::negative_timeout(-1.0, 3, 30.0)]
    #[case::nan_timeout(f64::NAN, 3, 30.0)]
    #[case::infinite_timeout(f64::INFINITY, 3, 30.0)]
    #[case::overflowing_timeout(1e19, 3, 30.0)]
    #[case::zero_attempts(1.0, 0, 30.0)]
    #[case::negative_wait(1.0, 3, -1.0)]
    #[case::nan_wait(1.0, 3, f64::NAN)]
    #[case::infinite_wait(1.0, 3, f64::INFINITY)]
    #[case::overflowing_wait(1.0, 3, 1e19)]
    fn invalid_retry_policy_precedes_python_config(
        #[from(python)] _python: (),
        #[case] timeout: f64,
        #[case] attempts: usize,
        #[case] wait: f64,
    ) {
        Python::attach(|py| {
            let error = fetch_with_retry(py, "https://example.test", None, timeout, attempts, wait)
                .unwrap_err();
            assert!(error.is_instance_of::<PyValueError>(py));
        });
    }

    #[rstest]
    fn nested_runtime_is_rejected(#[from(python)] _python: ()) {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        let _entered = runtime.enter();
        Python::attach(|py| {
            let error = fetch_once(py, "https://example.test", None, 1.0).unwrap_err();
            assert!(error.is_instance_of::<PyRuntimeError>(py));
            let error =
                fetch_with_retry(py, "https://example.test", None, 1.0, 3, 30.0).unwrap_err();
            assert!(error.is_instance_of::<PyRuntimeError>(py));
        });
    }
}
