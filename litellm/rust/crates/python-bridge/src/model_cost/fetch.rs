use std::time::Duration;

use litellm_http::{ClientVariant, HttpClientConfig};
use litellm_model_catalog_refresh::{
    Access, Failed, Fetched, GcsAccess, Remote, RemoteSource, RetryPolicy, S3Access, Source,
    redacted,
};
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
pub(crate) struct NativeCostMapSource(pub(super) Connection);

/// What a `RemoteSource` is built from. Plain data, so a caller can clone it and build the source off the GIL
#[derive(Clone)]
pub(super) struct Connection {
    url: String,
    http: HttpClientConfig,
}

impl Connection {
    pub(super) fn url(&self) -> &str {
        &self.url
    }

    /// On the shared runtime's pooled client, for workers after the fork
    pub(super) fn pooled(&self, policy: RetryPolicy) -> PyResult<RemoteSource> {
        let client = crate::http::pool()
            .client(&self.http, ClientVariant::Provider)
            .map_err(crate::http::client_error)?;
        Ok(RemoteSource::new(
            self.url.clone(),
            client,
            access(),
            policy,
        ))
    }

    /// With a client of its own, so an importing process can still fork
    pub(super) fn unpooled(&self, policy: RetryPolicy) -> PyResult<RemoteSource> {
        let client = crate::http::pool()
            .unpooled(&self.http, ClientVariant::Provider)
            .map_err(crate::http::client_error)?;
        Ok(RemoteSource::new(
            self.url.clone(),
            client,
            access(),
            policy,
        ))
    }
}

#[pymethods]
impl NativeCostMapSource {
    #[new]
    #[pyo3(signature = (url, *, use_http_settings))]
    fn new(py: Python<'_>, url: String, use_http_settings: bool) -> PyResult<Self> {
        Ok(Self(Connection {
            url,
            http: crate::http::settings_config(py, use_http_settings)?,
        }))
    }

    /// The source at `url`, or `None` when the local override or offline mode turns the remote cost map off
    #[staticmethod]
    #[pyo3(signature = (url, *, local_only, offline, use_http_settings))]
    fn remote(
        py: Python<'_>,
        url: String,
        local_only: bool,
        offline: bool,
        use_http_settings: bool,
    ) -> PyResult<Option<Self>> {
        match Remote::new(url.into(), local_only, offline) {
            Remote::Off => Ok(None),
            Remote::At(url) => Self::new(py, url.expose().to_owned(), use_http_settings).map(Some),
        }
    }

    /// The URL as source info and logs show it, without the credentials a presigned URL carries
    #[getter]
    fn shown_url(&self) -> String {
        redacted(&self.0.url).into_owned()
    }

    /// Fetches on a throwaway runtime with a client of its own
    fn fetch_blocking(
        &self,
        py: Python<'_>,
        policy: PyRef<'_, NativeFetchPolicy>,
    ) -> PyResult<Py<PyAny>> {
        let (remote, policy) = (self.0.clone(), policy.0);
        let logger = crate::logger::capture(py);
        let fetched = litellm_host_python::release_gil(py, move || {
            logger.scope(|| {
                let source = remote.unpooled(policy)?;
                litellm_host_python::run_isolated(source.fetch()).map_err(PyErr::from)
            })
        })?;
        Outcome(fetched).into_pyobject(py).map(Bound::unbind)
    }
}

fn access() -> Access {
    Access {
        s3: S3Access::Environment {
            region: std::env::var("AWS_REGION_NAME").ok(),
        },
        gcs: GcsAccess::Environment,
    }
}

#[pyclass(
    frozen,
    name = "NativeFetchPolicy",
    module = "litellm.rust_bridge._native"
)]
pub(crate) struct NativeFetchPolicy(pub(super) RetryPolicy);

#[pymethods]
impl NativeFetchPolicy {
    #[new]
    /// Retries wait an exponential backoff with jitter from `min_delay_seconds` up to `max_wait_seconds`, or the
    /// server's `Retry-After` up to that cap. An argument left out takes its value from `RetryPolicy::DEFAULT`
    #[pyo3(signature = (*, max_attempts = None, max_wait_seconds = None, timeout_seconds = None, min_delay_seconds = None))]
    fn new(
        max_attempts: Option<u32>,
        max_wait_seconds: Option<f64>,
        timeout_seconds: Option<f64>,
        min_delay_seconds: Option<f64>,
    ) -> PyResult<Self> {
        let default = RetryPolicy::DEFAULT;
        let max_attempts = max_attempts.unwrap_or(default.max_attempts);
        if max_attempts == 0 {
            return Err(PyValueError::new_err("max_attempts must be at least 1"));
        }
        let seconds = |name: &str, value: Option<f64>, default: Duration| match value {
            None => Ok(default),
            Some(value) => Duration::try_from_secs_f64(value).map_err(|_| {
                PyValueError::new_err(format!("{name} must be a non-negative number of seconds"))
            }),
        };
        let min_delay = seconds("min_delay_seconds", min_delay_seconds, default.min_delay)?;
        let max_delay = seconds("max_wait_seconds", max_wait_seconds, default.max_delay)?;
        if min_delay > max_delay {
            return Err(PyValueError::new_err(
                "min_delay_seconds must not exceed max_wait_seconds",
            ));
        }
        Ok(Self(RetryPolicy {
            max_attempts,
            timeout: seconds("timeout_seconds", timeout_seconds, default.timeout)?,
            min_delay,
            max_delay,
        }))
    }

    /// The attempt limit, the default's when the caller left it out
    #[getter]
    fn max_attempts(&self) -> u32 {
        self.0.max_attempts
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
pub(crate) struct NativeFetchFailed(pub(super) Failed);

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
        let policy = NativeFetchPolicy::new(
            Some(3),
            Some(max_wait_seconds),
            Some(5.0),
            Some(min_delay_seconds),
        );
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
                Bound::new(py, NativeFetchPolicy::new(None, None, None, None).unwrap()).unwrap();
            let outcome = source.fetch_blocking(py, policy.borrow()).unwrap();
            let fetched = outcome.bind(py).cast::<NativeFetched>().unwrap().get();
            assert_eq!(fetched.body.bind(py).as_bytes(), b"{}");
            assert_eq!(fetched.etag.as_deref(), Some("\"v1\""));
        });
        assert!(!litellm_host_python::runtime_started());
    }
}
