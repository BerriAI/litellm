use std::{
    collections::{BTreeMap, BTreeSet},
    time::Duration,
};

use litellm_core_utils::settings::Lookup;
use litellm_redis_types::{Auth, HealthCheck, ProviderHandle, Resolution, StaticAuth, Target};
use serde::{Deserialize, Deserializer, de::Error as _};
use veil::Redact;

use crate::{Object, Value};

mod auth;
mod breaker;
mod endpoint;
mod error;
mod keys;
mod layers;
mod options;
mod secrets;
mod tls;

pub use error::{Expected, RedisError};
pub use secrets::SecretSnapshot;

use options::Options;

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

/// A Redis section as `config.yaml` spells it. Each surface reads it its own way, so it resolves
/// only through [`RedisSettings::coordination`] or [`RedisSettings::cache_params`].
#[derive(Clone, Default, Deserialize, Redact)]
#[serde(transparent)]
pub struct RedisSettings {
    #[redact(with = "[REDACTED]")]
    fields: BTreeMap<String, Value>,
}

#[derive(Clone, Debug)]
pub enum CacheSettings {
    Redis(Box<RedisSettings>),
    Other(Object),
}

impl<'de> Deserialize<'de> for CacheSettings {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let fields = BTreeMap::<String, Value>::deserialize(deserializer)?;
        let is_redis = matches!(fields.get("type"), Some(Value::String(kind)) if kind == "redis");
        if !is_redis {
            return Ok(Self::Other(Object::from_fields(fields)));
        }
        let mapping = fields
            .into_iter()
            .map(|(key, value)| (Value::String(key), value))
            .collect();
        serde_yaml_ng::from_value(Value::Mapping(mapping))
            .map(|settings| Self::Redis(Box::new(settings)))
            .map_err(D::Error::custom)
    }
}

/// The redis-py kwargs one surface hands `_get_redis_client_logic`, before `REDIS_*` and secrets.
#[derive(Clone, Redact)]
pub struct ConfiguredRedis {
    #[redact(with = "[REDACTED]")]
    fields: BTreeMap<String, Value>,
    nulls: Nulls,
    credential_provider: Option<ProviderHandle>,
    host_objects: BTreeSet<String>,
}

/// Whether a null setting reaches redis-py as an explicit `None`, which outranks its `REDIS_*` value.
#[derive(Clone, Copy, Debug)]
enum Nulls {
    /// `coordination_redis` reaches redis-py through `model_dump(exclude_none=True)`.
    Dropped,
    /// `Cache` drops a null only for the parameters it names and passes any other key on.
    KeptUnlessNamedByCache,
    /// `RedisCache.redis_kwargs` already holds only the nulls redis-py receives.
    Kept,
}

impl RedisSettings {
    /// `general_settings.coordination_redis`, which the proxy validates as a pydantic model first.
    /// An `os.environ/` port is only known after resolution, where it is read like any other port.
    pub fn coordination(&self) -> Result<ConfiguredRedis, RedisError> {
        if let Some(port) = self.fields.get("port").and_then(options::as_text)
            && !port.starts_with("os.environ/")
            && port.parse::<u16>().is_err()
        {
            return Err(RedisError::Invalid {
                setting: "port".to_owned(),
                expected: Expected::Integer,
            });
        }
        Ok(ConfiguredRedis {
            fields: self.fields.clone(),
            nulls: Nulls::Dropped,
            credential_provider: None,
            host_objects: BTreeSet::new(),
        })
    }

    /// `litellm_settings.cache_params` with `type: redis`, as `litellm.Cache` hands it to `RedisCache`.
    pub fn cache_params(&self) -> ConfiguredRedis {
        let alias = self
            .fields
            .get("redis_startup_nodes")
            .filter(|_| !self.fields.contains_key("startup_nodes"))
            .map(|nodes| ("startup_nodes".to_owned(), nodes.clone()));
        let fields = self
            .fields
            .iter()
            .filter(|(key, _)| {
                key.as_str() != "redis_startup_nodes" && !CACHE_ONLY.contains(&key.as_str())
            })
            .map(|(key, value)| (key.clone(), value.clone()))
            .chain(alias)
            .collect();
        ConfiguredRedis {
            fields,
            nulls: Nulls::KeptUnlessNamedByCache,
            credential_provider: None,
            host_objects: BTreeSet::new(),
        }
    }
}

impl ConfiguredRedis {
    /// `RedisCache.redis_kwargs`, the dict every Python surface ends in and the one python-bridge
    /// hands over.
    pub fn redis_cache_kwargs(fields: BTreeMap<String, Value>) -> Self {
        Self {
            fields,
            nulls: Nulls::Kept,
            credential_provider: None,
            host_objects: BTreeSet::new(),
        }
    }

    /// Keys whose value is a host object, which the host keeps instead of handing over. Each one hides
    /// its `REDIS_*` value like any configured key.
    pub fn with_host_objects(self, names: BTreeSet<String>) -> Self {
        Self {
            host_objects: names,
            ..self
        }
    }

    /// The handle the host holds for the `credential_provider` object in these settings.
    pub fn with_credential_provider(self, handle: ProviderHandle) -> Self {
        Self {
            credential_provider: Some(handle),
            ..self
        }
    }

    /// No configured settings, so only `REDIS_*` and the other environment fallbacks apply.
    pub fn environment_only() -> Self {
        Self {
            fields: BTreeMap::new(),
            nulls: Nulls::Dropped,
            credential_provider: None,
            host_objects: BTreeSet::new(),
        }
    }

    pub fn resolve(
        &self,
        secrets: &SecretSnapshot,
        environment: &dyn Lookup,
    ) -> Result<Resolution, RedisError> {
        let kwargs = layers::merged(self, secrets);
        let (options, unmodeled) = Options::read(kwargs, &self.host_objects, secrets)?;
        let auth = auth::redis_auth(&options, self.credential_provider, environment)?;
        let (target, url_ignored) = endpoint::target(&options, &auth)?;
        let auth = match (&target, auth) {
            (Target::Url(_), Auth::Static(StaticAuth { username, .. })) => {
                Auth::Static(StaticAuth {
                    username,
                    password: None,
                })
            }
            (_, auth) => auth,
        };
        Ok(Resolution {
            target,
            auth,
            namespace: options::text(options.namespace.as_ref()),
            timeout: options::seconds(options.socket_timeout.as_ref())?,
            connect_timeout: options::seconds(options.socket_connect_timeout.as_ref())?,
            health_check: health_check(options.health_check_interval.as_ref())?,
            breaker: breaker::breaker(environment),
            passthrough: unmodeled.passthrough,
            host_objects: unmodeled.host_objects,
            ignored: unmodeled.ignored.into_iter().chain(url_ignored).collect(),
        })
    }
}

// redis-py checks health only when the interval is truthy, so zero turns it off and a negative interval checks
// before every command
fn health_check(entry: Option<&layers::Entry>) -> Result<HealthCheck, RedisError> {
    Ok(match options::integer::<i64>(entry)? {
        None => HealthCheck::Unset,
        Some(0) => HealthCheck::Disabled,
        Some(seconds) => HealthCheck::Every(Duration::from_secs(seconds.try_into().unwrap_or(0))),
    })
}

pub fn environment_has_target(lookup: &dyn Lookup) -> bool {
    [
        "REDIS_HOST",
        "REDIS_URL",
        "REDIS_CLUSTER_NODES",
        "REDIS_SENTINEL_NODES",
    ]
    .iter()
    .any(|name| lookup.get(name).is_some())
}

#[cfg(test)]
mod tests {
    use super::RedisSettings;
    use rstest::rstest;

    #[rstest]
    #[case::scalar(serde_json::json!("credential-value"))]
    #[case::object(serde_json::json!({"secret": "credential-value"}))]
    #[case::array(serde_json::json!(["credential-value"]))]
    fn credentials_are_redacted(
        #[case] password: serde_json::Value,
        #[values(false, true)] configured: bool,
    ) {
        let settings: RedisSettings = serde_json::from_value(serde_json::json!({
            "host": "redis", "password": password,
        }))
        .unwrap();
        let debug = match configured {
            false => format!("{settings:?}"),
            true => format!("{:?}", settings.cache_params()),
        };
        assert!(!debug.contains("credential-value"));
        assert!(debug.contains("[REDACTED]"));
    }
}
