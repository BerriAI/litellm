use std::collections::BTreeMap;

use litellm_config::{RedisSettings, RouterRedis, Value};
use litellm_env::Lookup;
use litellm_redis_types::{ProviderHandle, Resolution};
use litellm_secrets::SecretResolver;
use veil::Redact;

use crate::{Error, Expected, Kwargs, keys::Kwarg, layers, secrets::SecretSnapshot};

const CACHE_ONLY: &[&str] = &[
    "type",
    "mode",
    "ttl",
    "default_in_memory_ttl",
    "default_in_redis_ttl",
    "supported_call_types",
    "similarity_threshold",
    "max_messages",
    "_backend",
];

const STARTUP_NODES_ALIAS: &str = "redis_startup_nodes";

/// Where a gateway's Redis settings are written.
#[derive(Clone, Copy, Debug)]
pub enum Surface<'a> {
    /// `general_settings.coordination_redis`, which the proxy validates as a pydantic model first.
    Coordination(&'a RedisSettings),
    /// `litellm_settings.cache_params` with `type: redis`, as `litellm.Cache` hands it to `RedisCache`.
    CacheParams(&'a RedisSettings),
    /// `router_settings`' `redis_*` and `cache_kwargs`, as the router hands them to `RedisCache`.
    Router(&'a RouterRedis),
    /// No configured settings, so only `REDIS_*` and the other names the merge reads apply.
    Environment,
}

/// The Redis one surface configures, merged the way `_get_redis_client_logic` merges it. Every name
/// Python reads with `get_secret` is read through `secrets`, and the one it reads straight from
/// `os.environ`, Azure's `REDIS_USERNAME`, from the environment under `secrets`. `None` is
/// `Surface::Environment` with no name that says where Redis is, or `Surface::Router` naming none.
pub async fn resolve(
    surface: Surface<'_>,
    secrets: &SecretResolver,
) -> Result<Option<Resolution>, Error> {
    let configured = match surface {
        Surface::Coordination(settings) => ConfiguredRedis::coordination(settings)?,
        Surface::CacheParams(settings) => ConfiguredRedis::cache_params(settings),
        Surface::Router(settings) => match ConfiguredRedis::router(settings) {
            Some(configured) => configured,
            None => return Ok(None),
        },
        Surface::Environment => ConfiguredRedis::environment_only(),
    };
    let snapshot = SecretSnapshot::read(configured.secret_names(), secrets).await?;
    if matches!(surface, Surface::Environment) && !layers::secrets_name_a_target(&snapshot) {
        return Ok(None);
    }
    configured
        .resolve(&snapshot, secrets.environment().as_ref())
        .map(Some)
}

/// The redis-py kwargs one surface hands `_get_redis_client_logic`, before `REDIS_*` and secrets.
#[derive(Clone, Redact)]
pub(crate) struct ConfiguredRedis {
    #[redact(with = "[REDACTED]")]
    pub(crate) fields: BTreeMap<String, Value>,
    pub(crate) nulls: Nulls,
    pub(crate) credential_provider: Option<ProviderHandle>,
}

/// Whether a null setting reaches redis-py as an explicit `None`, which outranks its `REDIS_*` value.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Nulls {
    /// `coordination_redis` reaches redis-py through `model_dump(exclude_none=True)`.
    Dropped,
    /// `Cache` drops a null only for the parameters it names and passes any other key on.
    KeptUnlessNamedByCache,
    /// `RedisCache.redis_kwargs` already holds only the nulls redis-py receives.
    #[cfg(test)]
    Kept,
}

impl ConfiguredRedis {
    /// An `os.environ/` port is only known after merging, where it is read like any other port.
    pub(crate) fn coordination(settings: &RedisSettings) -> Result<Self, Error> {
        let key = Kwarg::Port.name();
        if let Some(port) = settings.fields().get(key).and_then(text)
            && !port.starts_with("os.environ/")
            && port.parse::<u16>().is_err()
        {
            return Err(Error::Invalid {
                setting: key.to_owned(),
                expected: Expected::Integer,
            });
        }
        Ok(Self {
            fields: settings.fields().clone(),
            nulls: Nulls::Dropped,
            credential_provider: None,
        })
    }

    pub(crate) fn cache_params(settings: &RedisSettings) -> Self {
        let fields = settings.fields();
        let startup_nodes = Kwarg::StartupNodes.name();
        let alias = fields
            .get(STARTUP_NODES_ALIAS)
            .filter(|_| !fields.contains_key(startup_nodes))
            .map(|nodes| (startup_nodes.to_owned(), nodes.clone()));
        let fields = fields
            .iter()
            .filter(|(key, _)| {
                key.as_str() != STARTUP_NODES_ALIAS && !CACHE_ONLY.contains(&key.as_str())
            })
            .map(|(key, value)| (key.clone(), value.clone()))
            .chain(alias)
            .collect();
        Self {
            fields,
            nulls: Nulls::KeptUnlessNamedByCache,
            credential_provider: None,
        }
    }

    /// `redis_cache_for` in `litellm/router_python_host.py`: a `redis_url`, or a `redis_host` with a
    /// `redis_port`, names the router's Redis, and `cache_kwargs` are further `RedisCache` keywords
    /// that replace the named ones. `RedisCache` drops a null only for a parameter it names, each of
    /// which `Cache` names too, and passes any other null on.
    pub(crate) fn router(settings: &RouterRedis) -> Option<Self> {
        let RouterRedis {
            redis_url,
            redis_host,
            redis_port,
            redis_password,
            redis_db,
            cache_kwargs,
        } = settings;
        if redis_url.is_none() && (redis_host.is_none() || redis_port.is_none()) {
            return None;
        }
        let named = [
            (
                Kwarg::Url,
                redis_url.as_ref().map(|url| url.expose().to_owned()),
            ),
            (Kwarg::Host, redis_host.clone()),
            (Kwarg::Port, redis_port.as_ref().map(ToString::to_string)),
            (
                Kwarg::Password,
                redis_password
                    .as_ref()
                    .map(|password| password.expose().to_owned()),
            ),
            (Kwarg::Db, redis_db.as_ref().map(ToString::to_string)),
        ]
        .into_iter()
        .filter_map(|(kwarg, value)| Some((kwarg.name().to_owned(), Value::String(value?))));
        let keywords = cache_kwargs.iter().flatten().filter_map(|(kwarg, value)| {
            Some((kwarg.clone(), serde_json::from_value(value.clone()).ok()?))
        });
        Some(Self {
            fields: named.chain(keywords).collect(),
            nulls: Nulls::KeptUnlessNamedByCache,
            credential_provider: None,
        })
    }

    /// `RedisCache.redis_kwargs`, the dict every Python surface ends in.
    #[cfg(test)]
    pub(crate) fn redis_cache_kwargs(fields: BTreeMap<String, Value>) -> Self {
        Self {
            fields,
            nulls: Nulls::Kept,
            credential_provider: None,
        }
    }

    /// The handle the host holds for the `credential_provider` object in these settings.
    #[cfg(test)]
    pub(crate) fn with_credential_provider(self, handle: ProviderHandle) -> Self {
        Self {
            credential_provider: Some(handle),
            ..self
        }
    }

    pub(crate) fn environment_only() -> Self {
        Self {
            fields: BTreeMap::new(),
            nulls: Nulls::Dropped,
            credential_provider: None,
        }
    }

    /// `_merge_redis_kwargs` in `litellm/_redis.py`: the kwargs over their `REDIS_*` values with every
    /// `get_secret` fallback filled.
    pub(crate) fn merged(&self, secrets: &SecretSnapshot) -> Result<Kwargs, Error> {
        layers::merged(self, secrets)
    }

    /// The environment is only for what Python reads straight from `os.environ`, `REDIS_USERNAME` for Azure.
    pub(crate) fn resolve(
        &self,
        secrets: &SecretSnapshot,
        environment: &dyn Lookup,
    ) -> Result<Resolution, Error> {
        let username = Kwarg::Username
            .environment_name()
            .and_then(|name| environment.get(&name));
        crate::interpret(
            &self.merged(secrets)?,
            self.credential_provider,
            username.as_deref(),
        )
    }
}

fn text(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => Some(text.clone()),
        Value::Number(number) => Some(number.to_string()),
        Value::Bool(flag) => Some(flag.to_string()),
        _ => None,
    }
}
