use std::time::Duration;

use litellm_http::{ClientVariant, HttpClientConfig};
use litellm_model_catalog_refresh::{Failed, Fetched, GcsAccess, RetryPolicy, S3Access, Start};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

/// A cost map URL and how to connect to it. An `s3://` URL reaches S3 as an AWS SDK finds it in the environment,
/// with LiteLLM's `AWS_REGION_NAME` winning over the region it finds. A `gs://` URL reaches Cloud Storage with Google
/// application default credentials
#[pyclass(
    frozen,
    name = "NativeCostMapSource",
    module = "litellm.rust_bridge._native"
)]
pub(crate) struct NativeCostMapSource {
    url: String,
    http: HttpClientConfig,
}

#[pymethods]
impl NativeCostMapSource {
    #[new]
    #[pyo3(signature = (url, *, use_http_settings))]
    fn new(py: Python<'_>, url: String, use_http_settings: bool) -> PyResult<Self> {
        Ok(Self {
            url,
            http: crate::http::settings_config(py, use_http_settings)?,
        })
    }

    /// Fetches on the shared runtime with a pooled client
    fn fetch<'py>(
        &self,
        py: Python<'py>,
        policy: PyRef<'_, NativeFetchPolicy>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let client = crate::http::pool()
            .client(&self.http, ClientVariant::Provider)
            .map_err(crate::http::client_error)?;
        let url = self.url.clone();
        let policy = policy.0;
        crate::execution::run_async_value(py, async move {
            Ok(Outcome(
                litellm_model_catalog_refresh::fetch(
                    &client,
                    &s3(),
                    &GcsAccess::Environment,
                    &url,
                    policy,
                    Start::Fresh,
                )
                .await,
            ))
        })
    }

    /// Fetches on a throwaway runtime with a client of its own. `resume_from` continues a fetch whose first attempt
    /// failed retryably
    #[pyo3(signature = (policy, resume_from = None))]
    fn fetch_blocking(
        &self,
        py: Python<'_>,
        policy: PyRef<'_, NativeFetchPolicy>,
        resume_from: Option<PyRef<'_, NativeFetchFailed>>,
    ) -> PyResult<Py<PyAny>> {
        let start = resume_from.map_or(Start::Fresh, |failed| Start::After(failed.0.clone()));
        let (config, url, policy) = (self.http.clone(), self.url.clone(), policy.0);
        let logger = crate::logger::capture(py);
        let fetched = litellm_host_python::release_gil(py, move || {
            logger.scope(|| {
                let client = crate::http::pool()
                    .unpooled(&config, ClientVariant::Provider)
                    .map_err(crate::http::client_error)?;
                litellm_host_python::run_isolated(litellm_model_catalog_refresh::fetch(
                    &client,
                    &s3(),
                    &GcsAccess::Environment,
                    &url,
                    policy,
                    start,
                ))
                .map_err(PyErr::from)
            })
        })?;
        Outcome(fetched).into_pyobject(py).map(Bound::unbind)
    }
}

fn s3() -> S3Access {
    S3Access::Environment {
        region: std::env::var("AWS_REGION_NAME").ok(),
    }
}

#[pyclass(
    frozen,
    name = "NativeFetchPolicy",
    module = "litellm.rust_bridge._native"
)]
pub(crate) struct NativeFetchPolicy(RetryPolicy);

#[pymethods]
impl NativeFetchPolicy {
    #[new]
    /// Retries wait an exponential backoff with jitter from `min_delay_seconds` up to `max_wait_seconds`, or the
    /// server's `Retry-After` up to that cap
    #[pyo3(signature = (*, max_attempts, max_wait_seconds, timeout_seconds, min_delay_seconds))]
    fn new(
        max_attempts: u32,
        max_wait_seconds: f64,
        timeout_seconds: f64,
        min_delay_seconds: f64,
    ) -> PyResult<Self> {
        if max_attempts == 0 {
            return Err(PyValueError::new_err("max_attempts must be at least 1"));
        }
        let seconds = |name: &str, value: f64| {
            Duration::try_from_secs_f64(value).map_err(|_| {
                PyValueError::new_err(format!("{name} must be a non-negative number of seconds"))
            })
        };
        let min_delay = seconds("min_delay_seconds", min_delay_seconds)?;
        let max_delay = seconds("max_wait_seconds", max_wait_seconds)?;
        if min_delay > max_delay {
            return Err(PyValueError::new_err(
                "min_delay_seconds must not exceed max_wait_seconds",
            ));
        }
        Ok(Self(RetryPolicy {
            max_attempts,
            timeout: seconds("timeout_seconds", timeout_seconds)?,
            min_delay,
            max_delay,
        }))
    }
}

#[pyclass(frozen, name = "NativeFetched", module = "litellm.rust_bridge._native")]
pub(crate) struct NativeFetched {
    #[pyo3(get)]
    body: Py<PyBytes>,
    #[pyo3(get)]
    etag: Option<String>,
}

#[pyclass(
    frozen,
    skip_from_py_object,
    name = "NativeFetchFailed",
    module = "litellm.rust_bridge._native"
)]
pub(crate) struct NativeFetchFailed(Failed);

#[pymethods]
impl NativeFetchFailed {
    #[getter]
    fn reason(&self) -> &str {
        &self.0.reason
    }

    /// Another attempt could succeed. Only the attempt limit ended the fetch
    #[getter]
    fn retryable(&self) -> bool {
        self.0.retryable
    }

    #[getter]
    fn retry_after(&self) -> Option<f64> {
        self.0.retry_after.map(|wait| wait.as_secs_f64())
    }
}

struct Outcome(Result<Fetched, Failed>);

impl<'py> IntoPyObject<'py> for Outcome {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        match self.0 {
            Ok(fetched) => Bound::new(
                py,
                NativeFetched {
                    body: PyBytes::new(py, &fetched.body).unbind(),
                    etag: fetched.etag,
                },
            )
            .map(Bound::into_any),
            Err(failed) => Bound::new(py, NativeFetchFailed(failed)).map(Bound::into_any),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::io::{BufRead, BufReader, Write};
    use std::net::TcpListener;

    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::equal_bounds(2.0, 2.0, true)]
    #[case::min_below_max(2.0, 30.0, true)]
    #[case::min_above_max(31.0, 30.0, false)]
    fn a_policy_accepts_a_min_delay_only_up_to_the_max_wait(
        #[case] min_delay_seconds: f64,
        #[case] max_wait_seconds: f64,
        #[case] accepted: bool,
    ) {
        Python::initialize();
        let policy = NativeFetchPolicy::new(3, max_wait_seconds, 5.0, min_delay_seconds);
        assert_eq!(policy.is_ok(), accepted);
    }

    #[rstest]
    fn a_blocking_fetch_hands_python_the_document_without_starting_the_shared_runtime() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/cost.json", listener.local_addr().unwrap());
        std::thread::spawn(move || {
            let mut stream = listener.incoming().next().unwrap().unwrap();
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut line = String::new();
            while reader.read_line(&mut line).unwrap() > 2 {
                line.clear();
            }
            stream
                .write_all(b"HTTP/1.1 200 OK\r\nETag: \"v1\"\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}")
                .unwrap();
        });
        Python::initialize();
        Python::attach(|py| {
            let source = NativeCostMapSource::new(py, url, false).unwrap();
            let policy =
                Bound::new(py, NativeFetchPolicy::new(3, 30.0, 5.0, 2.0).unwrap()).unwrap();
            let outcome = source.fetch_blocking(py, policy.borrow(), None).unwrap();
            let fetched = outcome.bind(py).cast::<NativeFetched>().unwrap().get();
            assert_eq!(fetched.body.bind(py).as_bytes(), b"{}");
            assert_eq!(fetched.etag.as_deref(), Some("\"v1\""));
        });
        assert!(!litellm_host_python::runtime_started());
    }
}
