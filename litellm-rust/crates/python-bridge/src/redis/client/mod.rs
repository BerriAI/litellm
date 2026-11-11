mod credentials;

use std::{collections::BTreeMap, sync::Arc, time::Duration};

use litellm_circuit_breaker::Config as BreakerConfig;
use litellm_core_utils::settings::ProcessEnvironment;
use litellm_host_python::PythonContext;
use litellm_python_compat::number::{self, SecondsError};
use litellm_redis_client::{RedisClient, RedisEvents};
use litellm_redis_connect::{CredentialSource, HostServices, SharedEnvironment};
use litellm_redis_resolve::{Kwargs, PythonException};
use litellm_redis_types::{Endpoint, ProviderHandle, Resolution};
use pyo3::{
    exceptions::{PyOverflowError, PyTypeError, PyValueError},
    prelude::*,
    types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyMapping, PyString, PyTuple},
};

use self::credentials::PythonCredentialSource;
use crate::redis::{breaker::BreakerSettings, events::ServiceLoggingEvents, hosts};

pyo3::import_exception!(redis.exceptions, RedisClusterException);

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
        let source = PyDict::new(py);
        let mapping = kwargs.cast::<PyMapping>()?;
        source.update(&mapping)?;
        let plain = PyDict::new(py);
        let mut host_objects = Vec::new();
        for (name, value) in source.iter() {
            match value.is_none() || is_plain(&value) {
                true => plain.set_item(&name, value)?,
                false => host_objects.push(name.extract::<String>()?),
            }
        }
        let credential_provider = source
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

/// `litellm.redis.store_settings.RedisStoreSettings`, which a `RedisCache` captures when it is built.
#[derive(FromPyObject)]
pub(crate) struct StoreSettings {
    redis_kwargs: RedisSpec,
    namespace: Option<String>,
    default_ttl: f64,
    breaker: Option<BreakerSettings>,
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

/// The one client for a resolved Redis config and credential provider, so every family in this process shares its pool
/// and its breaker. A config redis-py refuses building its client raises the same class here. One it only fails on
/// once a command connects, where `RedisCache` logs every failed call, leaves the family on memory with an error.
fn shared_client(
    py: Python<'_>,
    spec: RedisSpec,
    namespace: Option<String>,
    breaker: Option<BreakerConfig>,
) -> PyResult<Option<RedisClient>> {
    let environment: SharedEnvironment = Arc::new(ProcessEnvironment);
    let logger = crate::logger::capture(py);
    let resolved = match logger.scope(|| resolve_spec(py, &spec))? {
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
        Err(error) => return unresolved(py, &error),
    };
    let runtime = litellm_host_python::runtime_handle()?;
    let events: Arc<dyn RedisEvents> = Arc::new(ServiceLoggingEvents::new(py, runtime.clone())?);
    let host = HostServices {
        events: Some(events),
        http: crate::http::host_client(py, litellm_http::ClientVariant::Provider)?,
        environment,
        credential_provider: spec
            .credential_provider
            .as_ref()
            .map(|provider| -> PyResult<Arc<dyn CredentialSource>> {
                Ok(Arc::new(PythonCredentialSource::new(
                    provider.clone_ref(py),
                    PythonContext::capture(py)?,
                )))
            })
            .transpose()?,
    };
    let connected = py.detach(|| {
        let _runtime = runtime.enter();
        hosts().client(resolved, host, breaker)
    });
    match connected {
        Ok(client) => Ok(Some(client)),
        Err(error) => Ok(unconnected(py, &error)),
    }
}

/// The Rust client for one `RedisCache`, built once and handed to every native store, so a store
/// never sees the redis-py kwargs or the namespace.
#[pyclass(frozen, name = "NativeRedisClient")]
pub(crate) struct NativeRedisClient {
    pub(crate) client: RedisClient,
    /// `RedisCache.default_ttl`, the TTL a store's write takes when its caller gives none.
    pub(crate) default_ttl: Duration,
}

#[pymethods]
impl NativeRedisClient {
    /// `None` when redis-py would fail only once a command connects, which leaves every store given it on memory.
    #[staticmethod]
    fn connect(py: Python<'_>, settings: StoreSettings) -> PyResult<Option<Self>> {
        let StoreSettings {
            redis_kwargs,
            namespace,
            default_ttl,
            breaker,
        } = settings;
        let default_ttl = number::seconds(default_ttl).map_err(seconds_error)?;
        let breaker = breaker.as_ref().map(BreakerSettings::config).transpose()?;
        Ok(
            shared_client(py, redis_kwargs, namespace, breaker)?.map(|client| Self {
                client,
                default_ttl,
            }),
        )
    }

    #[getter]
    fn default_ttl_seconds(&self) -> f64 {
        self.default_ttl.as_secs_f64()
    }
}

/// The Redis server `spec` names, whatever its db, credentials or namespace, or none when the Rust client cannot reach it.
pub(crate) fn resolved_endpoint(py: Python<'_>, spec: &RedisSpec) -> PyResult<Option<Endpoint>> {
    Ok(resolve_spec(py, spec)?
        .ok()
        .and_then(|resolution| resolution.target.endpoint()))
}

// A live registry entry's client keeps its provider alive, so no live entry shares its address
fn resolve_spec(
    py: Python<'_>,
    spec: &RedisSpec,
) -> PyResult<Result<Resolution, litellm_redis_resolve::Error>> {
    let handle = spec
        .credential_provider
        .as_ref()
        .map(|provider| ProviderHandle::new(provider.as_ptr() as usize as u64));
    resolve(&spec.kwargs, handle, &py.import("os")?.getattr("environ")?)
}

/// What redis-py makes of the kwargs, with `REDIS_USERNAME` read from `environ` the way Python's Azure AD auth
/// reads `os.environ`.
fn resolve(
    kwargs: &Kwargs,
    provider: Option<ProviderHandle>,
    environ: &Bound<'_, PyAny>,
) -> PyResult<Result<Resolution, litellm_redis_resolve::Error>> {
    let redis_username: Option<String> = environ
        .call_method1("get", ("REDIS_USERNAME",))?
        .extract()?;
    Ok(litellm_redis_resolve::interpret(
        kwargs,
        provider,
        redis_username.as_deref(),
    ))
}

fn unresolved(
    py: Python<'_>,
    error: &litellm_redis_resolve::Error,
) -> PyResult<Option<RedisClient>> {
    let message = || litellm_error::Report::new(error).to_string();
    match error.raised_building_the_python_client() {
        Some(PythonException::ValueError) => Err(PyValueError::new_err(message())),
        Some(PythonException::TypeError) => Err(PyTypeError::new_err(message())),
        Some(PythonException::RedisClusterException) => {
            Err(RedisClusterException::new_err(message()))
        }
        None => Ok(memory_only(py, error)),
    }
}

// redis-py reads TLS files and fetches credentials only once a command connects, so none of these fails building
// its client
fn unconnected(py: Python<'_>, error: &litellm_redis_connect::Error) -> Option<RedisClient> {
    match error {
        litellm_redis_connect::Error::Connect(_)
        | litellm_redis_connect::Error::NoHostCredentialProvider
        | litellm_redis_connect::Error::ProcessEnvironmentRequired { .. }
        | litellm_redis_connect::Error::Credential { .. }
        | litellm_redis_connect::Error::CredentialTimeout { .. }
        | litellm_redis_connect::Error::ExpiredCredential
        | litellm_redis_connect::Error::MissingCredential { .. } => memory_only(py, error),
    }
}

fn memory_only(py: Python<'_>, error: &(dyn std::error::Error + 'static)) -> Option<RedisClient> {
    let report = litellm_error::Report::new(error).to_string();
    crate::logger::capture(py).scope(|| {
        litellm_tracing::error!(
            "Redis is not usable from the Rust stores, their state stays in memory: {report}"
        )
    });
    None
}

fn seconds_error(error: SecondsError) -> PyErr {
    match error {
        SecondsError::Overflow => PyOverflowError::new_err(error.to_string()),
        SecondsError::NotANumber => PyValueError::new_err(error.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::CStr;

    use litellm_redis_types::Auth;
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    fn kwargs(fields: serde_json::Value) -> Kwargs {
        let fields: BTreeMap<String, serde_json::Value> = serde_json::from_value(fields).unwrap();
        litellm_redis_resolve::kwargs(fields)
    }

    #[rstest]
    #[case::set(c"environ = {'REDIS_USERNAME': 'azure-user'}", Some("azure-user"))]
    #[case::empty_is_unset(c"environ = {'REDIS_USERNAME': ''}", None)]
    fn azure_auth_takes_its_username_from_the_environ_like_python(
        #[case] source: &CStr,
        #[case] expected: Option<&str>,
    ) {
        Python::initialize();
        Python::attach(|py| {
            let locals = PyDict::new(py);
            py.run(source, Some(&locals), Some(&locals)).unwrap();
            let environ = locals.get_item("environ").unwrap().unwrap();
            let kwargs = kwargs(json!({"host": "redis", "azure_redis_ad_token": "true"}));

            let resolved = resolve(&kwargs, None, &environ).unwrap().unwrap();

            assert!(
                matches!(&resolved.auth, Auth::AzureAd(azure) if azure.username.as_deref() == expected),
                "{:?}",
                resolved.auth
            );
        });
    }

    #[rstest]
    #[case::no_target(json!({"port": 6379}), Some("ValueError"))]
    #[case::unenforceable(json!({"host": "redis", "ssl": true, "ssl_validate_ocsp": true}), Some("TypeError"))]
    #[case::fails_only_once_a_command_connects(
        json!({"host": "redis", "ssl": true, "ssl_cert_reqs": "None"}),
        None
    )]
    fn a_config_raises_where_redis_py_raises_building_its_client(
        #[case] fields: serde_json::Value,
        #[case] raised: Option<&str>,
    ) {
        Python::initialize();
        Python::attach(|py| {
            let error = litellm_redis_resolve::interpret(&kwargs(fields), None, None).unwrap_err();

            let outcome = unresolved(py, &error)
                .map(|client| client.is_some())
                .map_err(|error| error.get_type(py).name().unwrap().to_string());

            assert_eq!(
                outcome,
                raised.map_or(Ok(false), |class| Err(class.to_owned()))
            );
        });
    }

    #[rstest]
    #[case::overflow(f64::INFINITY, "OverflowError")]
    #[case::not_a_number(f64::NAN, "ValueError")]
    fn a_default_ttl_python_cannot_make_an_int_raises_like_int(
        #[case] seconds: f64,
        #[case] raised: &str,
    ) {
        Python::initialize();
        Python::attach(|py| {
            let error = number::seconds(seconds).map_err(seconds_error).unwrap_err();

            assert_eq!(error.get_type(py).name().unwrap().to_string(), raised);
        });
    }
}
