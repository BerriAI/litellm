use std::collections::BTreeMap;

use litellm_config::{RedisSettings, Value};
use litellm_core_utils::settings::Lookup;
use litellm_redis_resolve::{Expected, Kwargs};
use litellm_redis_types::{ProviderHandle, Resolution};
use veil::Redact;

use super::{layers, secrets::SecretSnapshot};
use crate::Error;

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

/// The redis-py kwargs one surface hands `_get_redis_client_logic`, before `REDIS_*` and secrets.
#[derive(Clone, Redact)]
pub(crate) struct ConfiguredRedis {
    #[redact(with = "[REDACTED]")]
    pub(super) fields: BTreeMap<String, Value>,
    pub(super) nulls: Nulls,
    pub(super) credential_provider: Option<ProviderHandle>,
}

/// Whether a null setting reaches redis-py as an explicit `None`, which outranks its `REDIS_*` value.
#[derive(Clone, Copy, Debug)]
pub(super) enum Nulls {
    /// `coordination_redis` reaches redis-py through `model_dump(exclude_none=True)`.
    Dropped,
    /// `Cache` drops a null only for the parameters it names and passes any other key on.
    KeptUnlessNamedByCache,
    /// `RedisCache.redis_kwargs` already holds only the nulls redis-py receives.
    #[cfg(test)]
    Kept,
}

impl ConfiguredRedis {
    /// `general_settings.coordination_redis`, which the proxy validates as a pydantic model first.
    /// An `os.environ/` port is only known after merging, where it is read like any other port.
    pub(crate) fn coordination(
        settings: &RedisSettings,
    ) -> Result<Self, litellm_redis_resolve::Error> {
        if let Some(port) = settings.fields().get("port").and_then(text)
            && !port.starts_with("os.environ/")
            && port.parse::<u16>().is_err()
        {
            return Err(litellm_redis_resolve::Error::Invalid {
                setting: "port".to_owned(),
                expected: Expected::Integer,
            });
        }
        Ok(Self {
            fields: settings.fields().clone(),
            nulls: Nulls::Dropped,
            credential_provider: None,
        })
    }

    /// `litellm_settings.cache_params` with `type: redis`, as `litellm.Cache` hands it to `RedisCache`.
    pub(crate) fn cache_params(settings: &RedisSettings) -> Self {
        let fields = settings.fields();
        let alias = fields
            .get("redis_startup_nodes")
            .filter(|_| !fields.contains_key("startup_nodes"))
            .map(|nodes| ("startup_nodes".to_owned(), nodes.clone()));
        let fields = fields
            .iter()
            .filter(|(key, _)| {
                key.as_str() != "redis_startup_nodes" && !CACHE_ONLY.contains(&key.as_str())
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

    /// No configured settings, so only `REDIS_*` and the other environment fallbacks apply.
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
        Ok(litellm_redis_resolve::interpret(
            &self.merged(secrets)?,
            self.credential_provider,
            environment.get("REDIS_USERNAME").as_deref(),
        )?)
    }
}

pub(super) fn text(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => Some(text.clone()),
        Value::Number(number) => Some(number.to_string()),
        Value::Bool(flag) => Some(flag.to_string()),
        _ => None,
    }
}
