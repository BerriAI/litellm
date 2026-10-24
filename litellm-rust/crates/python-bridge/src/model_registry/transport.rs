use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use bytes::Bytes;
use litellm_host_python::release_gil;
use litellm_http::{ClientVariant, HttpClientConfig, HttpClientPool, media::PublicDnsResolver};
use litellm_model_catalog::{
    BUNDLED_REGISTRY, Catalog, FetchRequest, Installed, Origin, Reply, RetryPolicy, Revalidation,
    Step, git_blob_id,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedBytes;
use pyo3::types::{PyAny, PyDict};

use super::{ModelRegistry, ParsedRegistry, RowError, row_error};

#[pyclass(frozen, module = "litellm.rust_bridge._native")]
pub(crate) struct FetchedRegistry {
    registry: ModelRegistry,
    #[pyo3(get)]
    revision: String,
    #[pyo3(get)]
    etag: Option<String>,
}

#[pymethods]
impl FetchedRegistry {
    #[getter]
    fn registry(&self) -> ModelRegistry {
        self.registry.clone()
    }
}

#[pyclass(frozen, module = "litellm.rust_bridge._native")]
pub(crate) struct RegistryNotModified {}

#[pyclass(frozen, module = "litellm.rust_bridge._native")]
pub(crate) struct RegistryUnavailable {
    #[pyo3(get)]
    reason: String,
    #[pyo3(get)]
    retry_after: Option<f64>,
}

#[pyclass(frozen, module = "litellm.rust_bridge._native")]
pub(crate) struct FetchStep {
    request: FetchRequest,
}

#[pymethods]
impl FetchStep {
    #[getter]
    fn url(&self) -> &str {
        &self.request.url
    }

    #[getter]
    fn if_none_match(&self) -> Option<&str> {
        self.request.if_none_match.as_deref()
    }

    #[getter]
    fn timeout_seconds(&self) -> f64 {
        self.request.timeout.as_secs_f64()
    }
}

#[pyclass(frozen, module = "litellm.rust_bridge._native")]
pub(crate) struct SleepStep {
    #[pyo3(get)]
    seconds: f64,
}

#[pyclass(frozen, module = "litellm.rust_bridge._native")]
pub(crate) struct AdoptStep {
    registry: ModelRegistry,
    #[pyo3(get)]
    revision: String,
    #[pyo3(get)]
    etag: Option<String>,
    #[pyo3(get)]
    errors: Vec<RowError>,
}

#[pymethods]
impl AdoptStep {
    #[getter]
    fn registry(&self) -> ModelRegistry {
        self.registry.clone()
    }

    #[getter]
    fn catalog(&self) -> ModelRegistry {
        self.registry.clone()
    }
}

#[pyclass(frozen, module = "litellm.rust_bridge._native")]
pub(crate) struct DoneStep {
    #[pyo3(get)]
    reason: Option<String>,
}

#[pyclass(frozen, module = "litellm.rust_bridge._native")]
pub(crate) struct RevalidationState {
    state: Mutex<RevalidationSession>,
    config: Option<HttpClientConfig>,
}

struct RevalidationSession {
    state: Revalidation,
    status: SessionStatus,
}

enum SessionStatus {
    Fetch(FetchRequest),
    InFlight,
    Sleep,
    Terminal,
}

impl SessionStatus {
    fn from_step(step: &Step) -> Self {
        match step {
            Step::Fetch(request) => Self::Fetch(request.clone()),
            Step::Sleep(_) => Self::Sleep,
            Step::Adopt { .. } | Step::Done { .. } => Self::Terminal,
        }
    }
}

#[pymethods]
impl RevalidationState {
    fn perform(&self, py: Python<'_>, step: PyRef<'_, FetchStep>) -> PyResult<Py<PyAny>> {
        let request = step.request.clone();
        {
            let mut session = self.state.lock().unwrap_or_else(PoisonError::into_inner);
            if !matches!(&session.status, SessionStatus::Fetch(expected) if expected == &request) {
                return step_to_py(
                    py,
                    Step::Done {
                        reason: Some("fetch step is stale or unrelated".to_owned()),
                    },
                );
            }
            session.status = SessionStatus::InFlight;
        }
        let config = self.config.clone();
        let (reply, _) = release_gil(py, move || fetch_reply(config, request));
        let next = {
            let mut session = self.state.lock().unwrap_or_else(PoisonError::into_inner);
            if !matches!(&session.status, SessionStatus::InFlight) {
                return step_to_py(
                    py,
                    Step::Done {
                        reason: Some("fetch step is no longer active".to_owned()),
                    },
                );
            }
            let next = session.state.on_reply(reply);
            session.status = SessionStatus::from_step(&next);
            next
        };
        step_to_py(py, next)
    }

    fn slept(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let next = {
            let mut session = self.state.lock().unwrap_or_else(PoisonError::into_inner);
            if !matches!(&session.status, SessionStatus::Sleep) {
                return step_to_py(
                    py,
                    Step::Done {
                        reason: Some("revalidation is not awaiting a sleep".to_owned()),
                    },
                );
            }
            session.status = SessionStatus::InFlight;
            let next = session.state.on_slept();
            session.status = SessionStatus::from_step(&next);
            next
        };
        step_to_py(py, next)
    }

    fn reply(
        &self,
        py: Python<'_>,
        status: u16,
        etag: Option<String>,
        retry_after: Option<String>,
        body: PyBackedBytes,
    ) -> PyResult<Py<PyAny>> {
        let next = {
            let mut session = self.state.lock().unwrap_or_else(PoisonError::into_inner);
            if !matches!(&session.status, SessionStatus::Fetch(_)) {
                return step_to_py(
                    py,
                    Step::Done {
                        reason: Some("revalidation is not awaiting a reply".to_owned()),
                    },
                );
            }
            session.status = SessionStatus::InFlight;
            let next = session.state.on_reply(Reply::Response {
                status,
                etag,
                retry_after,
                body: Bytes::copy_from_slice(&body),
            });
            session.status = SessionStatus::from_step(&next);
            next
        };
        step_to_py(py, next)
    }

    fn transport_error(&self, py: Python<'_>, reason: String) -> PyResult<Py<PyAny>> {
        let next = {
            let mut session = self.state.lock().unwrap_or_else(PoisonError::into_inner);
            if !matches!(&session.status, SessionStatus::Fetch(_)) {
                return step_to_py(
                    py,
                    Step::Done {
                        reason: Some("revalidation is not awaiting a reply".to_owned()),
                    },
                );
            }
            session.status = SessionStatus::InFlight;
            let next = session.state.on_reply(Reply::Transport { reason });
            session.status = SessionStatus::from_step(&next);
            next
        };
        step_to_py(py, next)
    }
}

#[pyfunction(name = "Revalidation")]
#[pyo3(signature = (url, installed_revision=None, installed_etag=None, seed=None, max_attempts=None, timeout_seconds=5.0, installed_url=None))]
#[expect(
    clippy::too_many_arguments,
    reason = "keep existing positional arguments compatible while adding installed_url"
)]
pub(crate) fn revalidation(
    py: Python<'_>,
    url: String,
    installed_revision: Option<String>,
    installed_etag: Option<String>,
    seed: Option<u64>,
    max_attempts: Option<usize>,
    timeout_seconds: f64,
    installed_url: Option<String>,
) -> PyResult<(RevalidationState, Py<PyAny>)> {
    let bundled = Catalog::parse(BUNDLED_REGISTRY)
        .expect("bundled model registry is valid")
        .value;
    let origin = installed_url.map_or(Origin::Bundled, |url| Origin::Remote { url });
    let installed = Installed {
        catalog: Arc::new(bundled),
        revision: installed_revision.unwrap_or_else(|| git_blob_id(BUNDLED_REGISTRY)),
        etag: installed_etag,
        origin,
    };
    let timeout = Duration::try_from_secs_f64(timeout_seconds)
        .map_err(|_| PyValueError::new_err("timeout_seconds must be positive"))?;
    let policy = RetryPolicy {
        max_attempts: max_attempts.unwrap_or_else(|| RetryPolicy::default().max_attempts),
        timeout,
        ..RetryPolicy::default()
    };
    let (state, initial_step) = Revalidation::new(url, &installed, policy, seed);
    let config = if matches!(&initial_step, Step::Fetch(_)) {
        crate::http::call_config(py, &PyDict::new(py), false).ok()
    } else {
        None
    };
    let status = SessionStatus::from_step(&initial_step);
    Ok((
        RevalidationState {
            state: Mutex::new(RevalidationSession { state, status }),
            config,
        },
        step_to_py(py, initial_step)?,
    ))
}

#[pyfunction]
#[pyo3(signature = (source=None, *, file=None, if_none_match=None, timeout_seconds=5.0))]
pub(crate) fn fetch_model_registry(
    py: Python<'_>,
    source: Option<String>,
    file: Option<String>,
    if_none_match: Option<String>,
    timeout_seconds: f64,
) -> PyResult<Py<PyAny>> {
    let result = fetch(py, source, file, if_none_match, timeout_seconds)?;
    fetch_result_to_py(py, result)
}

#[pyfunction]
#[pyo3(signature = (source=None, *, file=None, if_none_match=None, timeout_seconds=5.0))]
pub(crate) fn fetch_model_registry_async<'py>(
    py: Python<'py>,
    source: Option<String>,
    file: Option<String>,
    if_none_match: Option<String>,
    timeout_seconds: f64,
) -> PyResult<Bound<'py, PyAny>> {
    let config = if requires_http_configuration(source.as_deref(), file.as_deref()) {
        crate::http::call_config(py, &PyDict::new(py), false).ok()
    } else {
        None
    };
    if source.is_some() && file.is_some() {
        return crate::execution::run_async_value(py, async {
            Python::attach(|py| {
                fetch_result_to_py(
                    py,
                    FetchResult::Unavailable(
                        "source and file cannot both be specified".to_owned(),
                        None,
                    ),
                )
            })
        });
    }
    crate::execution::run_async_value(py, async move {
        let result = tokio::task::spawn_blocking(move || {
            fetch_without_python(source, file, if_none_match, timeout_seconds, config)
        })
        .await
        .unwrap_or_else(|_| {
            FetchResult::Unavailable("model cost map transport failed".to_owned(), None)
        });
        Python::attach(|py| fetch_result_to_py(py, result))
    })
}

fn fetch(
    py: Python<'_>,
    source: Option<String>,
    file: Option<String>,
    if_none_match: Option<String>,
    timeout_seconds: f64,
) -> PyResult<FetchResult> {
    if source.is_some() && file.is_some() {
        return Ok(FetchResult::Unavailable(
            "source and file cannot both be specified".to_owned(),
            None,
        ));
    }
    let config = if requires_http_configuration(source.as_deref(), file.as_deref()) {
        crate::http::call_config(py, &PyDict::new(py), false).ok()
    } else {
        None
    };
    Ok(release_gil(py, move || {
        fetch_without_python(source, file, if_none_match, timeout_seconds, config)
    }))
}

enum FetchResult {
    Fetched(ModelRegistry, String, Option<String>),
    NotModified,
    Unavailable(String, Option<f64>),
}

fn requires_http_configuration(source: Option<&str>, file: Option<&str>) -> bool {
    file.is_none() && source.is_some_and(|source| !source.starts_with("s3://"))
}

fn fetch_without_python(
    source: Option<String>,
    file: Option<String>,
    if_none_match: Option<String>,
    timeout_seconds: f64,
    config: Option<HttpClientConfig>,
) -> FetchResult {
    if source.is_some() && file.is_some() {
        return FetchResult::Unavailable(
            "source and file cannot both be specified".to_owned(),
            None,
        );
    }
    if let Some(path) = file {
        return match std::fs::read(&path) {
            Ok(body) => parse_fetched_registry(&body, None),
            Err(_) => {
                FetchResult::Unavailable(format!("reading model cost map file {path} failed"), None)
            }
        };
    }
    let Some(url) = source else {
        return parse_fetched_registry(BUNDLED_REGISTRY, None);
    };
    if url.starts_with("s3://") {
        return FetchResult::Unavailable("s3 is fetched by the host".to_owned(), None);
    }
    if !timeout_seconds.is_finite() || timeout_seconds <= 0.0 {
        return FetchResult::Unavailable("timeout_seconds must be positive".to_owned(), None);
    }
    let Some(config) = config else {
        return FetchResult::Unavailable(
            "HTTP client configuration is unavailable".to_owned(),
            None,
        );
    };
    let installed = match Catalog::parse(BUNDLED_REGISTRY) {
        Ok(loaded) => Installed {
            catalog: Arc::new(loaded.value),
            revision: git_blob_id(BUNDLED_REGISTRY),
            etag: if_none_match,
            origin: Origin::Remote { url: url.clone() },
        },
        Err(error) => {
            return FetchResult::Unavailable(error.to_string(), None);
        }
    };
    let policy = RetryPolicy {
        max_attempts: RetryPolicy::default().max_attempts,
        timeout: Duration::try_from_secs_f64(timeout_seconds).unwrap_or_default(),
        ..RetryPolicy::default()
    };
    let (state, initial_step) = Revalidation::new(url, &installed, policy, Some(0));
    let Step::Fetch(request) = initial_step else {
        return FetchResult::Unavailable(
            "model cost map fetch did not request transport".to_owned(),
            None,
        );
    };
    let (reply, failure) = fetch_reply(Some(config), request);
    let mut state = state;
    let step = state.on_reply(reply);
    match step {
        Step::Adopt { installed, errors } => FetchResult::Fetched(
            ModelRegistry {
                parsed: Arc::new(ParsedRegistry {
                    catalog: Arc::clone(&installed.catalog),
                    errors: Arc::from(errors),
                }),
            },
            installed.revision,
            installed.etag,
        ),
        Step::Done { reason: None } => FetchResult::NotModified,
        Step::Done {
            reason: Some(reason),
        } => FetchResult::Unavailable(reason, None),
        Step::Sleep(delay) => FetchResult::Unavailable(
            failure.unwrap_or_else(|| "model cost map fetch failed".to_owned()),
            Some(delay.as_secs_f64()),
        ),
        Step::Fetch(_) => FetchResult::Unavailable(
            "model cost map fetch requires another reply".to_owned(),
            None,
        ),
    }
}

fn parse_fetched_registry(body: &[u8], etag: Option<String>) -> FetchResult {
    match Catalog::parse(body) {
        Ok(loaded) => FetchResult::Fetched(
            ModelRegistry {
                parsed: Arc::new(ParsedRegistry {
                    catalog: Arc::new(loaded.value),
                    errors: Arc::from(loaded.errors),
                }),
            },
            git_blob_id(body),
            etag,
        ),
        Err(error) => FetchResult::Unavailable(error.to_string(), None),
    }
}

fn fetch_reply(config: Option<HttpClientConfig>, request: FetchRequest) -> (Reply, Option<String>) {
    if tokio::runtime::Handle::try_current().is_ok() {
        return (
            Reply::Transport {
                reason: "HTTP fetch cannot run inside a Tokio runtime".to_owned(),
            },
            None,
        );
    }
    let runtime = match tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(_) => {
            return (
                Reply::Transport {
                    reason: "HTTP fetch runtime could not be created".to_owned(),
                },
                None,
            );
        }
    };
    let result = runtime.block_on(async {
        let Some(config) = config else {
            return (
                Reply::Transport {
                    reason: "HTTP client configuration is unavailable".to_owned(),
                },
                None,
            );
        };
        let pool = HttpClientPool::new(Arc::new(PublicDnsResolver));
        let client = match pool.client(&config, ClientVariant::Provider) {
            Ok(client) => client,
            Err(_) => {
                return (
                    Reply::Transport {
                        reason: "HTTP client could not be created".to_owned(),
                    },
                    None,
                );
            }
        };
        let mut request_builder = client.get(&request.url).timeout(request.timeout);
        if let Some(etag) = &request.if_none_match {
            request_builder = request_builder.header(reqwest::header::IF_NONE_MATCH, etag);
        }
        let result = match request_builder.send().await {
            Ok(response) => {
                let status = response.status().as_u16();
                let etag = response
                    .headers()
                    .get(reqwest::header::ETAG)
                    .and_then(|value| value.to_str().ok())
                    .map(str::to_owned);
                let retry_after = response
                    .headers()
                    .get(reqwest::header::RETRY_AFTER)
                    .and_then(|value| value.to_str().ok())
                    .map(str::to_owned);
                let body = if status == 200 {
                    match response.bytes().await {
                        Ok(body) => body,
                        Err(_) => {
                            return (
                                Reply::Transport {
                                    reason: "reading model cost map response failed".to_owned(),
                                },
                                None,
                            );
                        }
                    }
                } else {
                    Bytes::new()
                };
                (
                    Reply::Response {
                        status,
                        etag,
                        retry_after,
                        body,
                    },
                    None,
                )
            }
            Err(_) => (
                Reply::Transport {
                    reason: "model cost map request failed".to_owned(),
                },
                None,
            ),
        };
        drop(client);
        drop(pool);
        result
    });
    runtime.shutdown_timeout(Duration::from_secs(1));
    result
}

fn fetch_result_to_py(py: Python<'_>, result: FetchResult) -> PyResult<Py<PyAny>> {
    match result {
        FetchResult::Fetched(registry, revision, etag) => Ok(Py::new(
            py,
            FetchedRegistry {
                registry,
                revision,
                etag,
            },
        )?
        .into_any()),
        FetchResult::NotModified => Ok(Py::new(py, RegistryNotModified {})?.into_any()),
        FetchResult::Unavailable(reason, retry_after) => Ok(Py::new(
            py,
            RegistryUnavailable {
                reason,
                retry_after,
            },
        )?
        .into_any()),
    }
}

fn step_to_py(py: Python<'_>, step: Step) -> PyResult<Py<PyAny>> {
    match step {
        Step::Fetch(request) => Ok(Py::new(py, FetchStep { request })?.into_any()),
        Step::Sleep(delay) => Ok(Py::new(
            py,
            SleepStep {
                seconds: delay.as_secs_f64(),
            },
        )?
        .into_any()),
        Step::Adopt { installed, errors } => {
            let row_errors = errors.iter().map(row_error).collect();
            Ok(Py::new(
                py,
                AdoptStep {
                    registry: ModelRegistry {
                        parsed: Arc::new(ParsedRegistry {
                            catalog: Arc::clone(&installed.catalog),
                            errors: Arc::from(errors),
                        }),
                    },
                    revision: installed.revision,
                    etag: installed.etag,
                    errors: row_errors,
                },
            )?
            .into_any())
        }
        Step::Done { reason } => Ok(Py::new(py, DoneStep { reason })?.into_any()),
    }
}

#[cfg(test)]
mod tests {
    use std::{
        io::{Read, Write},
        net::TcpListener,
        process::Command,
        thread,
    };

    use litellm_http::{HttpSettings, Resolution};
    use litellm_model_catalog::{Error as CatalogError, Located};
    use rstest::rstest;

    use super::*;

    fn serve(status: &str, headers: &str, body: &[u8]) -> (String, thread::JoinHandle<String>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind local server");
        let address = listener.local_addr().expect("local address");
        let response = format!(
            "HTTP/1.1 {status}\r\n{headers}Content-Length: {}\r\nConnection: close\r\n\r\n",
            body.len()
        )
        .into_bytes()
        .into_iter()
        .chain(body.iter().copied())
        .collect::<Vec<_>>();
        let handle = thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept local request");
            let mut request = Vec::new();
            let mut byte = [0];
            loop {
                if stream.read_exact(&mut byte).is_err() {
                    break;
                }
                request.push(byte[0]);
                if request.ends_with(b"\r\n\r\n") {
                    break;
                }
            }
            stream.write_all(&response).expect("write local response");
            String::from_utf8(request).expect("HTTP request should be UTF-8")
        });
        (format!("http://{address}/catalog?token=secret"), handle)
    }

    fn serve_self_signed_tls() -> (String, thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind local TLS server");
        let address = listener.local_addr().expect("local TLS address");
        let identity = rcgen::generate_simple_self_signed(vec!["localhost".to_owned()])
            .expect("self-signed identity");
        let key = rustls::pki_types::PrivateKeyDer::Pkcs8(
            rustls::pki_types::PrivatePkcs8KeyDer::from(identity.signing_key.serialize_der()),
        );
        let server_config = rustls::ServerConfig::builder_with_provider(Arc::new(
            rustls::crypto::ring::default_provider(),
        ))
        .with_safe_default_protocol_versions()
        .expect("TLS protocol versions")
        .with_no_client_auth()
        .with_single_cert(vec![identity.cert.der().clone()], key)
        .expect("TLS certificate");
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            BUNDLED_REGISTRY.len()
        )
        .into_bytes()
        .into_iter()
        .chain(BUNDLED_REGISTRY.iter().copied())
        .collect::<Vec<_>>();
        let handle = thread::spawn(move || {
            let (stream, _) = listener.accept().expect("accept TLS request");
            let connection =
                rustls::ServerConnection::new(Arc::new(server_config)).expect("TLS connection");
            let mut stream = rustls::StreamOwned::new(connection, stream);
            let mut request = [0; 2048];
            if stream.read(&mut request).is_ok() {
                let _ = stream.write_all(&response);
            }
        });
        (
            format!("https://localhost:{}/catalog", address.port()),
            handle,
        )
    }

    fn http_config() -> HttpClientConfig {
        Resolution::from(&HttpSettings::default()).config
    }

    fn fetch_local(url: String) -> FetchResult {
        fetch_without_python(Some(url), None, None, 2.0, Some(http_config()))
    }

    fn thread_count() -> usize {
        std::fs::read_dir("/proc/self/task")
            .expect("process thread directory")
            .count()
    }

    fn install_test_python_modules(py: Python<'_>, locals: &Bound<'_, PyDict>) {
        py.run(
            pyo3::ffi::c_str!(
                r#"
import sys
import types

litellm = types.ModuleType("litellm")
litellm.__path__ = []
litellm.ssl_verify = True
bridge = types.ModuleType("litellm.rust_bridge")
bridge.__path__ = []
host = types.ModuleType("litellm.rust_bridge.host")
host.__path__ = []
settings = types.ModuleType("litellm.rust_bridge.host.settings")
def http_settings():
    return types.SimpleNamespace(
        ssl_verify=litellm.ssl_verify,
        ssl_certificate=None,
        ssl_security_level=None,
        ssl_ecdh_curve=None,
        force_ipv4=False,
        http2=False,
        aiohttp_trust_env=False,
        disable_aiohttp_trust_env=False,
        disable_aiohttp_transport=False,
        user_agent="test",
    )
settings.http_settings = http_settings
logger = types.ModuleType("litellm.rust_bridge.logger")
logger.context = lambda: ("test", "test")
logger.enabled = lambda level: False
logger.emit = lambda *args: None
litellm.rust_bridge = bridge
bridge.host = host
host.settings = settings
sys.modules["litellm"] = litellm
sys.modules["litellm.rust_bridge"] = bridge
sys.modules["litellm.rust_bridge.host"] = host
sys.modules["litellm.rust_bridge.host.settings"] = settings
sys.modules["litellm.rust_bridge.logger"] = logger
"#
            ),
            Some(locals),
            Some(locals),
        )
        .expect("install synthetic LiteLLM modules");
    }

    fn run_isolated(test_name: &str, child_marker: &str) -> bool {
        if std::env::var_os(child_marker).is_some() {
            return true;
        }
        let output = Command::new(std::env::current_exe().expect("current test executable"))
            .args(["--exact", test_name, "--nocapture", "--test-threads=1"])
            .env(child_marker, "1")
            .output()
            .expect("run isolated native test");
        assert!(
            output.status.success(),
            "isolated test failed\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        false
    }

    #[rstest]
    fn fetches_local_registry_and_retains_etag() {
        let (url, server) = serve("200 OK", "ETag: \"catalog-v1\"\r\n", BUNDLED_REGISTRY);
        let result = fetch_without_python(
            Some(url),
            None,
            Some("\"catalog-v1\"".to_owned()),
            2.0,
            Some(http_config()),
        );
        let request = server.join().expect("server thread");
        assert!(
            request
                .to_ascii_lowercase()
                .contains("if-none-match: \"catalog-v1\""),
            "request headers were {request}"
        );
        assert!(matches!(
            result,
            FetchResult::Fetched(_, _, Some(etag)) if etag == "\"catalog-v1\""
        ));
    }

    #[rstest]
    fn returns_not_modified_for_local_304() {
        let (url, server) = serve("304 Not Modified", "ETag: \"catalog-v1\"\r\n", b"secret");
        let result = fetch_local(url);
        server.join().expect("server thread");
        assert!(matches!(result, FetchResult::NotModified));
    }

    #[rstest]
    fn preserves_retry_after_from_native_policy() {
        let (url, server) = serve("429 Too Many Requests", "Retry-After: 9\r\n", b"secret");
        let result = fetch_local(url);
        server.join().expect("server thread");
        assert!(matches!(
            result,
            FetchResult::Unavailable(_, Some(seconds)) if seconds == 9.0
        ));
    }

    #[rstest]
    #[case::unauthorized("401 Unauthorized")]
    #[case::not_found("404 Not Found")]
    fn terminal_status_does_not_expose_body_or_url(#[case] status: &str) {
        let (url, server) = serve(status, "", b"secret-response-body");
        let result = fetch_local(url);
        server.join().expect("server thread");
        let FetchResult::Unavailable(reason, retry_after) = result else {
            panic!("terminal status must be unavailable");
        };
        assert!(reason.contains(status.split_whitespace().next().expect("status code")));
        assert!(!reason.contains("secret"));
        assert!(!reason.contains("token"));
        assert_eq!(retry_after, None);
    }

    #[rstest]
    fn bundled_file_and_source_validation_do_not_need_http_configuration() {
        assert!(matches!(
            fetch_without_python(None, None, None, 5.0, None),
            FetchResult::Fetched(_, _, _)
        ));
        let path = std::env::temp_dir().join(format!(
            "litellm-model-registry-{}.json",
            std::process::id()
        ));
        std::fs::write(&path, BUNDLED_REGISTRY).expect("write catalog");
        assert!(matches!(
            fetch_without_python(None, Some(path.display().to_string()), None, 5.0, None),
            FetchResult::Fetched(_, _, _)
        ));
        std::fs::remove_file(path).expect("remove temporary catalog");
        assert!(matches!(
            fetch_without_python(
                Some("https://example.invalid".to_owned()),
                Some("catalog.json".to_owned()),
                None,
                5.0,
                None
            ),
            FetchResult::Unavailable(_, None)
        ));
        assert!(matches!(
            fetch_without_python(Some("s3://bucket/catalog".to_owned()), None, None, 5.0, None),
            FetchResult::Unavailable(reason, None) if reason == "s3 is fetched by the host"
        ));
    }

    #[rstest]
    fn transport_failure_does_not_include_url_credentials() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind local listener");
        let address = listener.local_addr().expect("local address");
        drop(listener);
        let result = fetch_local(format!("http://user:password@{address}/?token=secret"));
        let FetchResult::Unavailable(reason, _) = result else {
            panic!("closed local port must fail");
        };
        assert!(!reason.contains("password"));
        assert!(!reason.contains("secret"));
        assert!(!reason.contains("127.0.0.1"));
    }

    #[rstest]
    fn self_signed_tls_obeys_verify_configuration() {
        if !run_isolated(
            "model_registry::transport::tests::self_signed_tls_obeys_verify_configuration",
            "LITELLM_MODEL_REGISTRY_TLS_CHILD",
        ) {
            return;
        }
        Python::initialize();
        let (disabled_url, disabled_server) = serve_self_signed_tls();
        let (enabled_url, enabled_server) = serve_self_signed_tls();
        Python::attach(|py| {
            let module = crate::native_module(py);
            let locals = PyDict::new(py);
            locals
                .set_item("native", module)
                .expect("bind native module");
            locals
                .set_item("disabled_url", disabled_url)
                .expect("bind disabled-verification URL");
            locals
                .set_item("enabled_url", enabled_url)
                .expect("bind enabled-verification URL");
            install_test_python_modules(py, &locals);
            py.run(
                pyo3::ffi::c_str!(
                    r#"
import litellm

litellm.ssl_verify = False
disabled = native.fetch_model_registry(source=disabled_url, timeout_seconds=3.0)
assert isinstance(disabled, native.FetchedRegistry), type(disabled)

litellm.ssl_verify = True
enabled = native.fetch_model_registry(source=enabled_url, timeout_seconds=3.0)
assert isinstance(enabled, native.RegistryUnavailable), type(enabled)
"#
                ),
                Some(&locals),
                Some(&locals),
            )
            .expect("fetch through Python's ssl_verify setting");
        });
        disabled_server.join().expect("disabled TLS server thread");
        enabled_server.join().expect("enabled TLS server thread");
    }

    #[rstest]
    fn async_fetch_returns_the_native_result_from_a_blocking_worker() {
        if !run_isolated(
            "model_registry::transport::tests::async_fetch_returns_the_native_result_from_a_blocking_worker",
            "LITELLM_MODEL_REGISTRY_ASYNC_CHILD",
        ) {
            return;
        }
        Python::initialize();
        Python::attach(|py| {
            let module = crate::native_module(py);
            let locals = PyDict::new(py);
            locals
                .set_item("native", module)
                .expect("bind native module");
            install_test_python_modules(py, &locals);
            py.run(
                pyo3::ffi::c_str!(
                    r#"
import asyncio
async def exercise():
    result = await native.fetch_model_registry_async()
    assert result.revision
    assert result.registry.served()

asyncio.run(exercise())
"#
                ),
                Some(&locals),
                Some(&locals),
            )
            .expect("async native fetch should complete");
        });
    }

    #[rstest]
    fn nested_tokio_fetch_returns_unavailable_without_panicking() {
        let request = FetchRequest {
            url: "http://127.0.0.1:1/?token=secret".to_owned(),
            if_none_match: None,
            timeout: Duration::from_secs(1),
        };
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("test runtime");
        let (reply, _) = runtime.block_on(async { fetch_reply(Some(http_config()), request) });
        assert!(matches!(
            reply,
            Reply::Transport { reason } if reason == "HTTP fetch cannot run inside a Tokio runtime"
        ));
    }

    #[rstest]
    #[case::json(
        CatalogError::Json(
            <serde_json::Error as serde::de::Error>::custom("invalid test payload")
        ),
        "json",
        false
    )]
    #[case::empty(CatalogError::Empty, "empty", false)]
    #[case::below_minimum(CatalogError::BelowMinimum { actual: 1, minimum: 2 }, "below_minimum", false)]
    #[case::shrunk(CatalogError::Shrunk { actual: 1, reference: 2, ratio: 0.5 }, "shrunk", false)]
    #[case::not_an_object(CatalogError::NotAnObject, "not_an_object", true)]
    #[case::invalid(
        CatalogError::Invalid(
            <serde_json::Error as serde::de::Error>::custom("invalid test payload")
        ),
        "invalid",
        true
    )]
    #[case::unknown_block_key(CatalogError::UnknownBlockKey, "unknown_block_key", true)]
    #[case::missing_provider(CatalogError::MissingProvider, "missing_provider", false)]
    #[case::unknown_timezone(CatalogError::UnknownTimezone { zone: "Invalid/Zone".to_owned() }, "unknown_timezone", true)]
    #[case::alias_shadows_model(CatalogError::AliasShadowsModel { alias: "alias".to_owned() }, "alias_shadows_model", true)]
    #[case::alias_taken(CatalogError::AliasTaken { alias: "alias".to_owned(), taken_by: "model".to_owned() }, "alias_taken", true)]
    #[case::duplicate_price(CatalogError::DuplicatePrice { first: "first".to_owned(), second: "second".to_owned() }, "duplicate_price", true)]
    #[case::suffix_inside_block(CatalogError::SuffixInsideBlock { block: "block" }, "suffix_inside_block", true)]
    #[case::tier_range(CatalogError::TierRange, "tier_range", true)]
    #[case::tier_reversed(CatalogError::TierReversed, "tier_reversed", true)]
    #[case::tier_gap(CatalogError::TierGap { start: 2, expected: 1 }, "tier_gap", true)]
    #[case::mixed_tier_basis(CatalogError::MixedTierBasis, "mixed_tier_basis", true)]
    #[case::not_priced_by_results(CatalogError::NotPricedByResults, "not_priced_by_results", true)]
    #[case::bands_from_two_sources(
        CatalogError::BandsFromTwoSources,
        "bands_from_two_sources",
        true
    )]
    #[case::not_an_off_peak_rate(CatalogError::NotAnOffPeakRate, "not_an_off_peak_rate", true)]
    #[case::no_off_peak_prices(CatalogError::NoOffPeakPrices, "no_off_peak_prices", true)]
    #[case::no_off_peak_hours(CatalogError::NoOffPeakHours, "no_off_peak_hours", true)]
    #[case::empty_list(CatalogError::EmptyList, "empty_list", true)]
    #[case::invalid_date(CatalogError::InvalidDate, "invalid_date", true)]
    fn row_error_maps_every_catalog_error_to_rust_policy(
        #[case] error: CatalogError,
        #[case] expected_kind: &str,
        #[case] expected_rejects: bool,
    ) {
        let located = Located::new("/model/field", error);
        let row = super::row_error(&located);
        assert_eq!(row.key, "model");
        assert_eq!(row.field.as_deref(), Some("field"));
        assert_eq!(row.kind, expected_kind);
        assert_eq!(row.rejects, expected_rejects);
        assert_eq!(
            row.rejects,
            litellm_model_catalog::rejects_row(&located.error)
        );
    }

    #[rstest]
    fn sync_fetch_does_not_start_shared_runtime() {
        if !run_isolated(
            "model_registry::transport::tests::sync_fetch_does_not_start_shared_runtime",
            "LITELLM_MODEL_REGISTRY_SYNC_CHILD",
        ) {
            return;
        }
        let threads_before = thread_count();
        let (url, server) = serve("304 Not Modified", "", b"");
        assert!(!litellm_host_python::runtime_started());
        let result = fetch_local(url);
        server.join().expect("server thread");
        assert!(matches!(result, FetchResult::NotModified));
        assert!(!litellm_host_python::runtime_started());
        assert_eq!(thread_count(), threads_before);
    }
}
