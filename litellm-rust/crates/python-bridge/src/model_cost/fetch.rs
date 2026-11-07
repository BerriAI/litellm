use std::sync::LazyLock;
use std::time::Duration;

use litellm_auth_aws::{AwsAuthService, AwsCredentialsProvider};
use litellm_http::{ClientVariant, HttpClientConfig};
use litellm_model_catalog_refresh::{Failed, Fetched, RetryPolicy, Start, Target};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

static AWS: LazyLock<AwsAuthService> = LazyLock::new(AwsAuthService::default);

/// The default chain main's `boto3.client("s3")` read its credentials from
fn default_chain(auth: AwsAuthService, target: &Target) -> AwsCredentialsProvider {
    AwsCredentialsProvider::default_chain(auth, target.aws_region.clone())
}

/// Where to fetch the cost map from and how to connect, built by Python from env and `litellm.*`
#[pyclass(
    frozen,
    name = "NativeCostMapSource",
    module = "litellm.rust_bridge._native"
)]
pub(crate) struct NativeCostMapSource {
    target: Target,
    http: HttpClientConfig,
}

#[pymethods]
impl NativeCostMapSource {
    /// `http_settings` is `litellm.rust_bridge.host.settings.http_settings()`, or `None` for the environment alone
    #[new]
    #[pyo3(signature = (url, *, http_settings, aws_region, aws_endpoint_url = None))]
    fn new(
        py: Python<'_>,
        url: String,
        http_settings: Option<Bound<'_, PyAny>>,
        aws_region: String,
        aws_endpoint_url: Option<String>,
    ) -> PyResult<Self> {
        Ok(Self {
            target: Target {
                url,
                aws_region,
                aws_endpoint_url,
            },
            http: crate::http::settings_config(py, http_settings.as_ref())?,
        })
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
    #[pyo3(signature = (
        *,
        max_attempts,
        max_wait_seconds,
        timeout_seconds,
        backoff_base_seconds = 2.0,
        jitter_min_seconds = 0.0,
        jitter_max_seconds = 1.0,
    ))]
    fn new(
        max_attempts: u32,
        max_wait_seconds: f64,
        timeout_seconds: f64,
        backoff_base_seconds: f64,
        jitter_min_seconds: f64,
        jitter_max_seconds: f64,
    ) -> PyResult<Self> {
        if max_attempts == 0 {
            return Err(PyValueError::new_err("max_attempts must be at least 1"));
        }
        let seconds = |name: &str, value: f64| {
            Duration::try_from_secs_f64(value).map_err(|_| {
                PyValueError::new_err(format!("{name} must be a non-negative number of seconds"))
            })
        };
        Ok(Self(RetryPolicy {
            max_attempts,
            max_wait: seconds("max_wait_seconds", max_wait_seconds)?,
            timeout: seconds("timeout_seconds", timeout_seconds)?,
            backoff_base_seconds,
            jitter_min_seconds,
            jitter_max_seconds,
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
        self.0.retry_after
    }
}

struct Outcome(Result<Fetched, Failed>);

/// Fetches on the shared runtime with a pooled client. For callers inside a worker, after any fork
pub(crate) fn fetch<'py>(
    py: Python<'py>,
    source: &NativeCostMapSource,
    policy: &NativeFetchPolicy,
) -> PyResult<Bound<'py, PyAny>> {
    let client = crate::http::pool()
        .client(&source.http, ClientVariant::NoRedirect)
        .map_err(crate::http::client_error)?;
    let target = source.target.clone();
    let credentials = default_chain(AWS.clone(), &target);
    let policy = policy.0;
    crate::execution::run_async_value(py, async move {
        let fetched = litellm_model_catalog_refresh::fetch(
            &client,
            &credentials,
            &target,
            policy,
            Start::Fresh,
        )
        .await;
        Ok(Outcome(fetched))
    })
}

/// Fetches on a throwaway runtime with a client and AWS credential cache of its own, so a process that fetches here
/// can still fork. `resume_from` continues a fetch whose first attempt failed retryably
pub(crate) fn fetch_blocking(
    py: Python<'_>,
    source: &NativeCostMapSource,
    policy: &NativeFetchPolicy,
    resume_from: Option<&NativeFetchFailed>,
) -> PyResult<Py<PyAny>> {
    let start = resume_from.map_or(Start::Fresh, |failed| Start::After(failed.0.clone()));
    let config = source.http.clone();
    let target = source.target.clone();
    let policy = policy.0;
    let logger = crate::logger::capture(py);
    let fetched = litellm_host_python::release_gil(py, move || {
        logger.scope(|| {
            let client = crate::http::pool()
                .unpooled(&config, ClientVariant::NoRedirect)
                .map_err(crate::http::client_error)?;
            litellm_host_python::run_isolated(async move {
                let credentials = default_chain(AwsAuthService::default(), &target);
                litellm_model_catalog_refresh::fetch(&client, &credentials, &target, policy, start)
                    .await
            })
            .map_err(PyErr::from)
        })
    })?;
    Outcome(fetched).into_pyobject(py).map(Bound::unbind)
}

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
            let source = NativeCostMapSource::new(py, url, None, "us-east-1".into(), None).unwrap();
            let policy = NativeFetchPolicy::new(3, 30.0, 5.0, 2.0, 0.0, 1.0).unwrap();
            let outcome = fetch_blocking(py, &source, &policy, None).unwrap();
            let fetched = outcome.bind(py).cast::<NativeFetched>().unwrap().get();
            assert_eq!(fetched.body.bind(py).as_bytes(), b"{}");
            assert_eq!(fetched.etag.as_deref(), Some("\"v1\""));
        });
        assert!(!litellm_host_python::runtime_started());
    }
}
