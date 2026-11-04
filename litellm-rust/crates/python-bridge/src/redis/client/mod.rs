mod credentials;

use std::{
    collections::BTreeMap,
    sync::{Arc, LazyLock, Mutex, PoisonError},
    time::Duration,
};

use litellm_circuit_breaker::Config as BreakerConfig;
use litellm_core_utils::settings::ProcessEnvironment;
use litellm_host_python::PythonContext;
use litellm_redis_client::{RedisClient, RedisEvents, WeakRedisClient};
use litellm_redis_connect::{CredentialSource, HostServices, SharedEnvironment};
use litellm_redis_resolve::Kwargs;
use litellm_redis_types::{ProviderHandle, Resolution};
use pyo3::{
    prelude::*,
    sync::MutexExt,
    types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple},
};

use self::credentials::PythonCredentialSource;
use crate::redis::events::{ServiceLoggingEvents, state_label};

/// The kwargs `_merge_redis_kwargs` returns for a `RedisCache`, which the resolver reads the way
/// redis-py's constructors do. A key whose value is a Python object, such as `retry`, reaches the
/// resolver by name only, so it can name the setting it cannot honor. A `credential_provider` object
/// is also kept, so the connector can ask it for credentials, and reaches the resolver as a handle.
pub(crate) struct RedisSpec {
    kwargs: Kwargs,
    credential_provider: Option<Py<PyAny>>,
}

impl<'a, 'py> FromPyObject<'a, 'py> for RedisSpec {
    type Error = PyErr;

    fn extract(kwargs: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let py = kwargs.py();
        let plain = PyDict::new(py);
        let mut host_objects = Vec::new();
        for (name, value) in kwargs.cast::<PyDict>()?.iter() {
            match value.is_none() || is_plain(&value) {
                true => plain.set_item(&name, value)?,
                false => host_objects.push(name.extract::<String>()?),
            }
        }
        let credential_provider = kwargs
            .cast::<PyDict>()?
            .get_item("credential_provider")?
            .filter(|provider| !provider.is_none() && !is_plain(provider))
            .map(Bound::unbind);
        let fields: BTreeMap<String, serde_json::Value> =
            litellm_host_python::from_py_argument(plain.as_any())?;
        let objects = host_objects
            .into_iter()
            .map(|name| (name, serde_json::Value::Bool(true)));
        Ok(Self {
            kwargs: litellm_redis_resolve::kwargs(fields.into_iter().chain(objects)),
            credential_provider,
        })
    }
}

/// `litellm.constants`' `REDIS_CIRCUIT_BREAKER_*`, which Python reads from the environment once.
pub(crate) struct RedisBreaker(BreakerConfig);

impl<'a, 'py> FromPyObject<'a, 'py> for RedisBreaker {
    type Error = PyErr;

    fn extract(breaker: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let (failure_threshold, recovery_timeout, timeout_min_duration): (u32, f64, f64) =
            breaker.extract()?;
        let seconds = |seconds: f64| {
            Duration::try_from_secs_f64(seconds)
                .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))
        };
        Ok(Self(BreakerConfig {
            failure_threshold,
            recovery_timeout: seconds(recovery_timeout)?,
            timeout_min_duration: seconds(timeout_min_duration)?,
        }))
    }
}

fn is_plain(value: &Bound<'_, PyAny>) -> bool {
    value.is_instance_of::<PyString>()
        || value.is_instance_of::<PyBool>()
        || value.is_instance_of::<PyInt>()
        || value.is_instance_of::<PyFloat>()
        || value.is_instance_of::<PyList>()
        || value.is_instance_of::<PyTuple>()
        || value.is_instance_of::<PyDict>()
}

/// A client the process built, held weakly so a discarded `Router` drops its client and provider.
struct Shared {
    resolved: Resolution,
    client: WeakRedisClient,
}

static CLIENTS: LazyLock<Mutex<Vec<Shared>>> = LazyLock::new(|| Mutex::new(Vec::new()));

fn live_clients(py: Python<'_>) -> std::sync::MutexGuard<'static, Vec<Shared>> {
    let mut clients = CLIENTS
        .lock_py_attached(py)
        .unwrap_or_else(PoisonError::into_inner);
    clients.retain(|shared| shared.client.upgrade().is_some());
    clients
}

/// The one client for a resolved Redis config and credential provider, so every family in this process shares its pool
/// and its breaker. Like redis-py, which never connects at construction, a config the resolver or
/// the connector cannot honor leaves the family on memory with a warning rather than failing the
/// `Router`.
fn shared_client(
    py: Python<'_>,
    spec: RedisSpec,
    namespace: Option<String>,
    breaker: Option<RedisBreaker>,
) -> PyResult<Option<RedisClient>> {
    let environment: SharedEnvironment = Arc::new(ProcessEnvironment);
    let RedisSpec {
        kwargs,
        credential_provider,
    } = spec;
    // The registry entry's client keeps the provider alive, so no live entry shares its address
    let handle = credential_provider
        .as_ref()
        .map(|provider| ProviderHandle::new(provider.as_ptr() as usize as u64));
    let logger = crate::logger::capture(py);
    let resolved = match logger.scope(|| litellm_redis_resolve::interpret(&kwargs, handle)) {
        Ok(resolution) => {
            logger.scope(|| {
                for ignored in &resolution.ignored {
                    litellm_tracing::warn!(
                        "Redis setting {} is not supported by the Rust client and is ignored: {}",
                        ignored.setting,
                        ignored.consequence
                    );
                }
            });
            Resolution {
                namespace,
                ..resolution
            }
        }
        Err(error) => return Ok(memory_only(py, &error)),
    };
    let runtime = litellm_host_python::runtime_handle()?;
    let events: Arc<dyn RedisEvents> = Arc::new(ServiceLoggingEvents::new(py, runtime.clone())?);
    let host = HostServices {
        events: Some(events),
        http: crate::http::host_client(py, litellm_http::ClientVariant::Provider)?,
        environment,
        credential_provider: credential_provider
            .as_ref()
            .map(|provider| -> PyResult<Arc<dyn CredentialSource>> {
                Ok(Arc::new(PythonCredentialSource::new(
                    provider.clone_ref(py),
                    PythonContext::capture(py)?,
                )))
            })
            .transpose()?,
    };
    let mut clients = live_clients(py);
    if let Some(client) = clients
        .iter()
        .filter(|shared| shared.resolved == resolved)
        .find_map(|shared| shared.client.upgrade())
    {
        return Ok(Some(client));
    }
    let connected = {
        let _runtime = runtime.enter();
        litellm_redis_connect::connect(
            resolved.clone(),
            host,
            breaker.map(|breaker| breaker.0),
        )
    };
    let client = match connected {
        Ok(client) => client,
        Err(error) => {
            drop(clients);
            return Ok(memory_only(py, &error));
        }
    };
    clients.push(Shared {
        resolved,
        client: client.downgrade(),
    });
    Ok(Some(client))
}

/// The Rust client for one `RedisCache`, built once and handed to every native store, so a store
/// never sees the redis-py kwargs or the namespace.
#[pyclass(frozen, name = "NativeRedisClient")]
pub(crate) struct NativeRedisClient(pub(crate) RedisClient);

#[pymethods]
impl NativeRedisClient {
    /// `None` when Redis is not usable, which leaves every store given it on memory.
    #[staticmethod]
    #[pyo3(signature = (redis_kwargs, *, namespace = None, breaker = None))]
    fn connect(
        py: Python<'_>,
        redis_kwargs: RedisSpec,
        namespace: Option<String>,
        breaker: Option<RedisBreaker>,
    ) -> PyResult<Option<Self>> {
        Ok(shared_client(py, redis_kwargs, namespace, breaker)?.map(Self))
    }
}

/// The breaker state of every native Redis client this process built, so the Prometheus state
/// gauge counts them next to the Python `RedisCache` breakers.
#[pyfunction]
pub(crate) fn native_redis_breaker_states(py: Python<'_>) -> Vec<&'static str> {
    live_clients(py)
        .iter()
        .filter_map(|shared| shared.client.upgrade())
        .filter_map(|client| client.breaker_state())
        .map(|state| state_label(&state))
        .collect()
}

fn memory_only(py: Python<'_>, error: &(dyn std::error::Error + 'static)) -> Option<RedisClient> {
    let report = litellm_error::Report::new(error).to_string();
    crate::logger::capture(py).scope(|| {
        litellm_tracing::warn!(
            "Redis is not usable from the Rust stores, their state stays in memory: {report}"
        )
    });
    None
}
