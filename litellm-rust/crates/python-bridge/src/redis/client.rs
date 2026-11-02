use std::{
    collections::BTreeSet,
    sync::{Arc, LazyLock, Mutex, PoisonError},
};

use litellm_config::{RedisSettings, ResolvedRedis, SecretSnapshot};
use litellm_core_utils::settings::ProcessEnvironment;
use litellm_host_python::PythonContext;
use litellm_redis_client::{RedisClient, RedisEvents};
use litellm_redis_connect::{CredentialSource, HostServices, SharedEnvironment};
use pyo3::{
    prelude::*,
    types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple},
};

use super::{
    credentials::PythonCredentialSource,
    events::{ServiceLoggingEvents, state_label},
};

/// The `redis_kwargs` dict `RedisCache` hands redis-py, read the way `litellm/_redis.py` reads it.
/// A value that is not plain data reaches the resolver by its type name, so the resolver or the
/// connector names the setting it cannot honor. A `credential_provider` object is also kept, so the
/// connector can ask it for credentials.
pub(crate) struct PythonRedisSettings {
    settings: RedisSettings,
    credential_provider: Option<Py<PyAny>>,
}

impl<'a, 'py> FromPyObject<'a, 'py> for PythonRedisSettings {
    type Error = PyErr;

    fn extract(kwargs: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let py = kwargs.py();
        let kwargs = kwargs.cast::<PyDict>()?;
        let plain = PyDict::new(py);
        for (name, value) in kwargs.iter() {
            if value.is_none() {
                continue;
            }
            if is_plain(&value) {
                plain.set_item(&name, value)?;
            } else {
                plain.set_item(&name, value.get_type().name()?)?;
            }
        }
        let credential_provider = kwargs
            .get_item("credential_provider")?
            .filter(|provider| !provider.is_none() && !is_plain(provider))
            .map(Bound::unbind);
        Ok(Self {
            settings: litellm_host_python::from_py_argument(plain.as_any())?,
            credential_provider,
        })
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

struct Shared {
    resolved: ResolvedRedis,
    credential_provider: Option<Py<PyAny>>,
    client: RedisClient,
}

impl Shared {
    fn serves(&self, resolved: &ResolvedRedis, credential_provider: Option<&Py<PyAny>>) -> bool {
        let same_provider = match (&self.credential_provider, credential_provider) {
            (None, None) => true,
            (Some(shared), Some(provider)) => shared.is(provider),
            _ => false,
        };
        same_provider && self.resolved == *resolved
    }
}

static CLIENTS: LazyLock<Mutex<Vec<Shared>>> = LazyLock::new(|| Mutex::new(Vec::new()));

/// The one client for a resolved Redis config and credential provider, so every family in this process shares its pool
/// and its breaker. Like redis-py, which never connects at construction, a config the resolver or
/// the connector cannot honor leaves the family on memory with a warning rather than failing the
/// `Router`.
pub(crate) fn shared_client(
    py: Python<'_>,
    kwargs: PythonRedisSettings,
    namespace: Option<String>,
) -> PyResult<Option<RedisClient>> {
    let environment: SharedEnvironment = Arc::new(ProcessEnvironment);
    let PythonRedisSettings {
        settings,
        credential_provider,
    } = kwargs;
    let secrets = match secret_snapshot(py, settings.secret_names())? {
        Ok(secrets) => secrets,
        Err(error) => return Ok(memory_only(py, &error)),
    };
    let logger = crate::logger::capture(py);
    let resolved = match logger.scope(|| settings.resolve(&secrets, &*environment)) {
        Ok(resolved) => ResolvedRedis {
            namespace,
            ..resolved
        },
        Err(error) => return Ok(memory_only(py, &error)),
    };
    let mut clients = CLIENTS.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some(shared) = clients
        .iter()
        .find(|shared| shared.serves(&resolved, credential_provider.as_ref()))
    {
        return Ok(Some(shared.client.clone()));
    }
    let events: Arc<dyn RedisEvents> = Arc::new(ServiceLoggingEvents::new(py)?);
    let breaker = match logger.scope(|| litellm_redis_connect::breaker_config(&*environment)) {
        Ok(breaker) => breaker,
        Err(error) => return Ok(memory_only(py, &error)),
    };
    let host = HostServices {
        events: Some(events),
        breaker,
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
    let target = resolved.clone();
    let client = match crate::execution::run_sync_value(py, async move {
        Ok(litellm_redis_connect::connect(target, host))
    })? {
        Ok(client) => client,
        Err(error) => return Ok(memory_only(py, &error)),
    };
    clients.push(Shared {
        resolved,
        credential_provider,
        client: client.clone(),
    });
    Ok(Some(client))
}

/// The breaker state of every native Redis client this process built, so the Prometheus state
/// gauge counts them next to the Python `RedisCache` breakers.
#[pyfunction]
pub(crate) fn native_redis_breaker_states() -> Vec<&'static str> {
    let clients = CLIENTS.lock().unwrap_or_else(PoisonError::into_inner);
    clients
        .iter()
        .filter_map(|shared| shared.client.breaker_state())
        .map(|state| state_label(&state))
        .collect()
}

/// Every name the resolver may read, fetched once the way Python's `get_secret` fetches it: the
/// configured secret manager first, then the process environment.
fn secret_snapshot(
    py: Python<'_>,
    names: BTreeSet<String>,
) -> PyResult<Result<SecretSnapshot, litellm_secrets::Error>> {
    let source = crate::secrets::source(py)?;
    let owned: Vec<String> = names.iter().cloned().collect();
    let lookup = crate::execution::run_sync_value(py, async move {
        let borrowed: Vec<&str> = owned.iter().map(String::as_str).collect();
        Ok(source.resolve(&borrowed).await)
    })?;
    Ok(lookup.map(|lookup| SecretSnapshot::read(names, &*lookup)))
}

fn memory_only(py: Python<'_>, error: &(dyn std::error::Error + 'static)) -> Option<RedisClient> {
    let report = litellm_error::Report::new(error).to_string();
    crate::logger::capture(py).scope(|| {
        litellm_tracing::warn!(
            "Redis is not usable from the Rust router stores, cooldowns and usage stay in memory: {report}"
        )
    });
    None
}
