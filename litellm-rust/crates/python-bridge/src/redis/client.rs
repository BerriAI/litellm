use std::{
    collections::BTreeSet,
    sync::{Arc, LazyLock, Mutex, PoisonError},
};

use litellm_config::{RedisSettings, ResolvedRedis, SecretSnapshot};
use litellm_core_utils::settings::ProcessEnvironment;
use litellm_redis_client::{RedisClient, RedisEvents};
use litellm_redis_connect::{HostServices, SharedEnvironment};
use pyo3::{
    prelude::*,
    types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple},
};

use super::events::{ServiceLoggingEvents, state_label};

/// The `redis_kwargs` dict `RedisCache` hands redis-py, read the way `litellm/_redis.py` reads it.
/// A value that is not plain data, such as a `credential_provider` object, reaches the resolver by
/// its type name, so the resolver or the connector names the setting it cannot honor.
pub(crate) struct PythonRedisSettings(RedisSettings);

impl<'a, 'py> FromPyObject<'a, 'py> for PythonRedisSettings {
    type Error = PyErr;

    fn extract(kwargs: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let py = kwargs.py();
        let plain = PyDict::new(py);
        for (name, value) in kwargs.cast::<PyDict>()?.iter() {
            if value.is_none() {
                continue;
            }
            if is_plain(&value) {
                plain.set_item(&name, value)?;
            } else {
                plain.set_item(&name, value.get_type().name()?)?;
            }
        }
        Ok(Self(litellm_host_python::from_py_argument(plain.as_any())?))
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
    client: RedisClient,
}

static CLIENTS: LazyLock<Mutex<Vec<Shared>>> = LazyLock::new(|| Mutex::new(Vec::new()));

/// The one client for a resolved Redis config, so every family in this process shares its pool
/// and its breaker. Like redis-py, which never connects at construction, a config the resolver or
/// the connector cannot honor leaves the family on memory with a warning rather than failing the
/// `Router`.
pub(crate) fn shared_client(
    py: Python<'_>,
    kwargs: PythonRedisSettings,
    namespace: Option<String>,
) -> PyResult<Option<RedisClient>> {
    let environment: SharedEnvironment = Arc::new(ProcessEnvironment);
    let secrets = match secret_snapshot(py, kwargs.0.secret_names())? {
        Ok(secrets) => secrets,
        Err(error) => return Ok(memory_only(py, &error)),
    };
    let logger = crate::logger::capture(py);
    let resolved = match logger.scope(|| kwargs.0.resolve(&secrets, &*environment)) {
        Ok(resolved) => ResolvedRedis {
            namespace,
            ..resolved
        },
        Err(error) => return Ok(memory_only(py, &error)),
    };
    let mut clients = CLIENTS.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some(shared) = clients.iter().find(|shared| shared.resolved == resolved) {
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
