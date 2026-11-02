use std::{collections::BTreeMap, time::Duration};

use litellm_core_utils::settings::Lookup;
use litellm_python_compat::{BigInt, number};
use litellm_redis_types::{
    ClusterEndpoint, ClusterNode as RedisNode, Endpoint, HostEndpoint, SentinelEndpoint,
};
use litellm_secret_value::SecretValue;
use serde::{Deserialize, Deserializer, de::Error as _};
use veil::Redact;

use crate::{Object, Value};

mod auth;
mod error;
mod keys;
mod secrets;
mod sentinel;
mod tls;

use keys::Handling;

pub use auth::{AwsIamAuth, AzureAdAuth, RedisAuth};
pub use error::{Expected, RedisError};
pub use secrets::SecretSnapshot;

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

#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedRedis {
    pub endpoint: Endpoint,
    pub auth: RedisAuth,
    pub namespace: Option<String>,
    pub timeout: Option<Duration>,
    pub connect_timeout: Option<Duration>,
}

#[derive(Clone)]
struct Entry {
    value: Value,
    source: String,
}

type Kwargs = BTreeMap<String, Entry>;

impl RedisSettings {
    pub fn resolve(
        &self,
        secrets: &SecretSnapshot,
        environment: &dyn Lookup,
    ) -> Result<ResolvedRedis, RedisError> {
        self.resolve_with(secrets, environment, false)
    }

    pub fn resolve_cache(
        &self,
        secrets: &SecretSnapshot,
        environment: &dyn Lookup,
    ) -> Result<ResolvedRedis, RedisError> {
        self.resolve_with(secrets, environment, true)
    }

    fn resolve_with(
        &self,
        secrets: &SecretSnapshot,
        environment: &dyn Lookup,
        cache: bool,
    ) -> Result<ResolvedRedis, RedisError> {
        let lookup: &dyn Lookup = secrets;
        let block = self.kwargs(cache);
        let block: Kwargs = block
            .into_iter()
            .filter_map(|(key, entry)| {
                let value = environment_reference(&entry.value, lookup);
                let value = match (&value, cache, key.as_str()) {
                    (Value::String(text), false, "ssl") => {
                        coordination_bool(text).map(Value::Bool).unwrap_or(value)
                    }
                    _ => value,
                };
                (!value.is_null() || keeps_explicit_none(cache, &key))
                    .then(|| (key, Entry { value, ..entry }))
            })
            .collect();
        let environment_kwargs = environment_kwargs(lookup);
        let names_a_target = ["host", "startup_nodes", "sentinel_nodes"]
            .iter()
            .any(|key| is_set(&block, key));
        let drop_environment_url = names_a_target && !is_set(&block, "url");
        let merged: Kwargs = environment_kwargs
            .into_iter()
            .filter(|(key, _)| !(key == "url" && drop_environment_url))
            .chain(block)
            .collect();
        reject_unsupported(&merged)?;
        warn_ignored(&merged);

        let startup_nodes = match configured(&merged, "startup_nodes") {
            Some(entry) => Some(
                cluster_nodes(&entry.value)
                    .ok_or_else(|| invalid(&entry.source, Expected::StartupNodes))?,
            ),
            None => keys::fallback("startup_nodes", lookup)
                .map(|(source, nodes)| {
                    cluster_nodes(&Value::String(nodes))
                        .ok_or_else(|| invalid(source, Expected::StartupNodes))
                })
                .transpose()?,
        }
        .filter(|nodes| !nodes.is_empty());
        let url = text_value(&merged, "url");
        let ssl = bool_value(&merged, "ssl");
        let uses_tls = match (&startup_nodes, &url) {
            (Some(_), _) | (None, None) => ssl,
            (None, Some(url)) => url
                .split_once("://")
                .is_some_and(|(scheme, _)| scheme.eq_ignore_ascii_case("rediss")),
        };
        let auth = auth::redis_auth(&merged, lookup, environment, uses_tls)?;
        let gcp_ca_certs = match &auth {
            RedisAuth::GcpIam { .. } if ssl => {
                text_or_fallback(&merged, "gcp_ssl_ca_certs", lookup)
                    .filter(|path| !path.is_empty())
            }
            _ => None,
        };
        let tls = || {
            ssl.then(|| tls::tls_options(&merged, gcp_ca_certs.clone()))
                .transpose()
        };
        let db = || integer(&merged, "db").map(|db| db.unwrap_or(0));
        let endpoint = match (startup_nodes, url) {
            (Some(nodes), _) => {
                reject_cluster_db(&merged)?;
                Endpoint::Cluster(ClusterEndpoint { nodes, tls: tls()? })
            }
            (None, Some(url)) => Endpoint::Url {
                tls: tls::url_tls(&url)?,
                url: SecretValue::new(match &auth {
                    RedisAuth::Static { .. } => url,
                    _ => without_userinfo(&url),
                }),
            },
            (None, None) => match sentinel::sentinel_target(&merged, lookup)? {
                Some(target) => {
                    reject_sentinel_host(&merged)?;
                    Endpoint::Sentinel(SentinelEndpoint {
                        sentinels: target.sentinels,
                        service_name: target.service_name,
                        sentinel_password: target.sentinel_password,
                        tls: tls()?,
                        db: db()?,
                    })
                }
                None => Endpoint::Host(HostEndpoint {
                    host: text_value(&merged, "host").ok_or(RedisError::NoTarget)?,
                    port: integer(&merged, "port")?.unwrap_or(6379),
                    tls: tls()?,
                    db: db()?,
                }),
            },
        };
        let auth = match (&endpoint, auth) {
            (Endpoint::Url { .. }, RedisAuth::Static { username, .. }) => RedisAuth::Static {
                username,
                password: None,
            },
            (_, auth) => auth,
        };
        Ok(ResolvedRedis {
            endpoint,
            auth,
            namespace: text_value(&merged, "namespace"),
            timeout: seconds(&merged, "socket_timeout")?,
            connect_timeout: seconds(&merged, "socket_connect_timeout")?,
        })
    }

    pub fn validate_coordination(&self) -> Result<(), RedisError> {
        let text = |key: &str| self.fields.get(key).and_then(as_text);
        if text("port").is_some_and(|port| port.parse::<u16>().is_err()) {
            return Err(invalid("port", Expected::Integer));
        }
        if text("ssl").is_some_and(|ssl| coordination_bool(&ssl).is_none()) {
            return Err(invalid("ssl", Expected::Boolean));
        }
        Ok(())
    }

    fn kwargs(&self, cache: bool) -> Kwargs {
        let entry = |key: &str, value: &Value| {
            (
                key.to_owned(),
                Entry {
                    value: value.clone(),
                    source: key.to_owned(),
                },
            )
        };
        let aliased = self
            .fields
            .get("redis_startup_nodes")
            .filter(|_| cache)
            .map(|value| entry("startup_nodes", value));
        let direct = self
            .fields
            .iter()
            .filter(|(key, _)| {
                !(cache
                    && (key.as_str() == "redis_startup_nodes"
                        || CACHE_ONLY.contains(&key.as_str())))
            })
            .map(|(key, value)| entry(key, value));
        aliased.into_iter().chain(direct).collect()
    }
}

// _get_redis_url_kwargs keeps only plain Connection kwargs, so redis-py drops every ssl_* setting for a url
// and a credential provider replaces the url's userinfo, which _url_without_userinfo strips
fn without_userinfo(url: &str) -> String {
    let Some((scheme, rest)) = url.split_once("://") else {
        return url.to_owned();
    };
    let (authority, tail) = rest.split_at(rest.find(['/', '?', '#']).unwrap_or(rest.len()));
    let host = authority
        .rsplit_once('@')
        .map_or(authority, |(_, host)| host);
    format!("{scheme}://{host}{tail}")
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

fn reject_unsupported(merged: &Kwargs) -> Result<(), RedisError> {
    match merged.iter().find(|(key, _)| is_unsupported(key)) {
        Some((_, entry)) => Err(RedisError::Unsupported {
            setting: entry.source.clone(),
        }),
        None => Ok(()),
    }
}

fn invalid(setting: &str, expected: Expected) -> RedisError {
    RedisError::Invalid {
        setting: setting.to_owned(),
        expected,
    }
}

fn parsed<T>(
    kwargs: &Kwargs,
    key: &str,
    expected: Expected,
    parse: impl Fn(&str) -> Option<T>,
) -> Result<Option<T>, RedisError> {
    kwargs
        .get(key)
        .and_then(|entry| Some((as_text(&entry.value)?, &entry.source)))
        .map(|(text, source)| parse(&text).ok_or_else(|| invalid(source, expected)))
        .transpose()
}

fn seconds(merged: &Kwargs, key: &str) -> Result<Option<Duration>, RedisError> {
    coerced(merged, key, Expected::Seconds, number::float, |seconds| {
        Duration::try_from_secs_f64(seconds).ok()
    })
}

fn coerced<N, T>(
    kwargs: &Kwargs,
    key: &str,
    expected: Expected,
    parse: impl Fn(&str) -> Option<N>,
    convert: impl Fn(N) -> Option<T>,
) -> Result<Option<T>, RedisError> {
    let Some((entry, text)) = kwargs
        .get(key)
        .and_then(|entry| Some((entry, as_text(&entry.value)?)))
    else {
        return Ok(None);
    };
    match parse(&text) {
        Some(number) => convert(number)
            .map(Some)
            .ok_or_else(|| invalid(&entry.source, expected)),
        None => {
            tracing::warn!(setting = %entry.source, "Redis setting is not a number, so the default is used like redis-py");
            Ok(None)
        }
    }
}

fn warn_ignored(merged: &Kwargs) {
    for key in merged.keys() {
        if keys::setting(key).is_some_and(|setting| setting.handling == Handling::Warn) {
            tracing::warn!(setting = %key, "Redis setting is accepted but ignored");
        }
    }
}

fn is_unsupported(key: &str) -> bool {
    keys::setting(key).is_some_and(|setting| setting.handling == Handling::Reject)
}

fn environment_kwargs(lookup: &dyn Lookup) -> Kwargs {
    keys::SETTINGS
        .iter()
        .filter_map(|setting| {
            let source = setting.environment_name()?;
            let value = lookup.get(&source)?;
            Some((
                setting.kwarg.to_owned(),
                Entry {
                    value: Value::String(value),
                    source,
                },
            ))
        })
        .collect()
}

// load_config resolves os.environ/ to None when unset, so a missing reference is a null like any other
fn environment_reference(value: &Value, lookup: &dyn Lookup) -> Value {
    match value {
        Value::String(text) => match text.strip_prefix("os.environ/") {
            Some(name) => lookup.get(name).map_or(Value::Null, Value::String),
            None => value.clone(),
        },
        _ => value.clone(),
    }
}

// coordination_redis reaches redis-py through model_dump(exclude_none=True), so every null is dropped. Cache
// and RedisCache drop a null only for the parameters they name, and pass any other key on as an explicit None
// that overrides the REDIS_* environment value
fn keeps_explicit_none(cache: bool, key: &str) -> bool {
    const NAMED_BY_CACHE: &[&str] = &[
        "host",
        "port",
        "password",
        "socket_timeout",
        "startup_nodes",
        "namespace",
        "gcp_service_account",
        "gcp_ssl_ca_certs",
    ];
    cache && !NAMED_BY_CACHE.contains(&key)
}

// redis-py's RedisCluster raises unless db is 0 or None
fn reject_cluster_db(merged: &Kwargs) -> Result<(), RedisError> {
    let Some(entry) = merged.get("db").filter(|entry| !entry.value.is_null()) else {
        return Ok(());
    };
    match &entry.value {
        Value::String(_) if matches!(integer::<i64>(merged, "db"), Ok(None | Some(0))) => Ok(()),
        value if !truthy(value) => Ok(()),
        _ => Err(RedisError::DbInCluster {
            setting: entry.source.clone(),
        }),
    }
}

// redis.asyncio.Sentinel passes the connection kwargs to each monitor's Redis(host=..., port=...), so a host or
// port setting raises TypeError when litellm builds the client
fn reject_sentinel_host(merged: &Kwargs) -> Result<(), RedisError> {
    match ["host", "port"]
        .iter()
        .find_map(|key| merged.get(*key).filter(|entry| !entry.value.is_null()))
    {
        Some(entry) => Err(RedisError::SentinelWithHost {
            setting: entry.source.clone(),
        }),
        None => Ok(()),
    }
}

fn is_set(kwargs: &Kwargs, key: &str) -> bool {
    kwargs.get(key).is_some_and(|entry| !entry.value.is_null())
}

// Python chains these settings with `or`, so a falsy configured value ("", [], 0, False) falls through too
fn configured<'a>(kwargs: &'a Kwargs, key: &str) -> Option<&'a Entry> {
    kwargs.get(key).filter(|entry| truthy(&entry.value))
}

fn truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(flag) => *flag,
        Value::Number(number) => number.as_f64().is_some_and(|number| number != 0.0),
        Value::String(text) => !text.is_empty(),
        Value::Sequence(items) => !items.is_empty(),
        Value::Mapping(entries) => !entries.is_empty(),
        Value::Tagged(_) => true,
    }
}

fn text_or_fallback(kwargs: &Kwargs, key: &str, lookup: &dyn Lookup) -> Option<String> {
    configured(kwargs, key)
        .and_then(|entry| as_text(&entry.value))
        .or_else(|| keys::fallback_value(key, lookup))
}

fn as_text(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => Some(text.clone()),
        Value::Number(number) => Some(number.to_string()),
        Value::Bool(flag) => Some(flag.to_string()),
        _ => None,
    }
}

fn text_value(kwargs: &Kwargs, key: &str) -> Option<String> {
    kwargs.get(key).and_then(|entry| as_text(&entry.value))
}

fn secret_value(kwargs: &Kwargs, key: &str) -> Option<SecretValue> {
    text_value(kwargs, key).map(SecretValue::new)
}

fn bool_value(kwargs: &Kwargs, key: &str) -> bool {
    match kwargs.get(key).map(|entry| &entry.value) {
        Some(Value::Bool(flag)) => *flag,
        Some(value) => as_text(value)
            .is_some_and(|text| matches!(text.to_ascii_lowercase().as_str(), "true" | "1" | "yes")),
        None => false,
    }
}

fn integer<T: TryFrom<BigInt>>(kwargs: &Kwargs, key: &str) -> Result<Option<T>, RedisError> {
    coerced(kwargs, key, Expected::Integer, number::int, |number| {
        T::try_from(number).ok()
    })
}

fn cluster_nodes(value: &Value) -> Option<Vec<RedisNode>> {
    nodes(value, |node| {
        node_address(
            node.get("host").and_then(as_text),
            node.get("port").and_then(as_text),
        )
    })
}

fn nodes(value: &Value, parse: impl Fn(&Value) -> Option<RedisNode>) -> Option<Vec<RedisNode>> {
    match value {
        Value::String(json) => serde_json::from_str::<Vec<Value>>(json)
            .ok()?
            .iter()
            .map(parse)
            .collect(),
        Value::Sequence(nodes) => nodes.iter().map(parse).collect(),
        _ => None,
    }
}

fn node_address(host: Option<String>, port: Option<String>) -> Option<RedisNode> {
    Some(RedisNode {
        host: host?,
        port: port?.parse().ok()?,
    })
}

fn coordination_bool(text: &str) -> Option<bool> {
    match text.to_ascii_lowercase().as_str() {
        "1" | "on" | "t" | "true" | "y" | "yes" => Some(true),
        "0" | "off" | "f" | "false" | "n" | "no" => Some(false),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::{RedisSettings, SecretSnapshot};
    use litellm_redis_types::Endpoint;
    use rstest::rstest;

    #[rstest]
    #[case::on("on", true)]
    #[case::t("t", true)]
    #[case::y("y", true)]
    #[case::upper_on("ON", true)]
    #[case::off("off", false)]
    #[case::f("f", false)]
    #[case::n("n", false)]
    fn coordination_ssl_aliases_are_normalized_without_changing_raw_cache_kwargs(
        #[case] value: &str,
        #[case] tls: bool,
        #[values(false, true)] cache: bool,
    ) {
        let settings: RedisSettings = serde_json::from_value(serde_json::json!({
            "host": "redis", "ssl": value,
        }))
        .unwrap();
        settings.validate_coordination().unwrap();
        let environment = |_: &str| None;
        let secrets = SecretSnapshot::read(settings.secret_names(), &environment);
        let resolved = if cache {
            settings.resolve_cache(&secrets, &environment)
        } else {
            settings.resolve(&secrets, &environment)
        }
        .unwrap();
        let Endpoint::Host(host) = resolved.endpoint else {
            panic!("expected host endpoint");
        };
        assert_eq!(host.tls.is_some(), tls && !cache);
    }

    #[rstest]
    #[case::scalar(serde_json::json!("credential-value"))]
    #[case::object(serde_json::json!({"secret": "credential-value"}))]
    #[case::array(serde_json::json!(["credential-value"]))]
    fn credentials_are_redacted(#[case] password: serde_json::Value) {
        let settings: RedisSettings = serde_json::from_value(serde_json::json!({
            "host": "redis", "password": password,
        }))
        .unwrap();
        let debug = format!("{settings:?}");
        assert!(!debug.contains("credential-value"));
        assert!(debug.contains("[REDACTED]"));
    }
}
