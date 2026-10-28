use std::{collections::BTreeMap, fmt, time::Duration};

use litellm_auth_types::SecretValue;
use litellm_router_types::Spelled;
use serde::{Deserialize, Deserializer, de::Error as _};

use crate::{AdditionalFields, Error, Object, Value};

pub type Lookup<'a> = &'a dyn Fn(&str) -> Option<String>;

const SUPPORTED: &[&str] = &[
    "url",
    "host",
    "port",
    "username",
    "password",
    "ssl",
    "db",
    "socket_timeout",
    "startup_nodes",
    "namespace",
];

const IGNORED: &[&str] = &[
    "max_connections",
    "single_connection_client",
    "health_check_interval",
    "socket_keepalive",
    "socket_keepalive_options",
    "decode_responses",
    "encoding",
    "encoding_errors",
    "charset",
    "errors",
    "client_name",
    "lib_name",
    "lib_version",
    "sentinel_password",
    "gcp_ssl_ca_certs",
    "azure_client_id",
    "azure_tenant_id",
    "azure_client_secret",
    "aws_iam_user_name",
    "aws_iam_cache_name",
    "aws_iam_region",
    "aws_iam_serverless",
];

const WARN_IGNORED: &[&str] = &[
    "max_connections",
    "health_check_interval",
    "socket_keepalive",
    "socket_keepalive_options",
];

const REDIS_PY_KWARGS: &[&str] = &[
    "aws_iam_auth",
    "aws_iam_cache_name",
    "aws_iam_region",
    "aws_iam_serverless",
    "aws_iam_user_name",
    "azure_client_id",
    "azure_client_secret",
    "azure_redis_ad_token",
    "azure_tenant_id",
    "cache",
    "cache_config",
    "charset",
    "client_name",
    "db",
    "decode_responses",
    "encoding",
    "encoding_errors",
    "errors",
    "event_dispatcher",
    "gcp_service_account",
    "gcp_ssl_ca_certs",
    "health_check_interval",
    "host",
    "lib_name",
    "lib_version",
    "max_connections",
    "password",
    "port",
    "protocol",
    "redis_connect_func",
    "retry_on_error",
    "retry_on_timeout",
    "single_connection_client",
    "socket_connect_timeout",
    "socket_keepalive",
    "socket_keepalive_options",
    "socket_timeout",
    "ssl",
    "ssl_ca_certs",
    "ssl_ca_data",
    "ssl_ca_path",
    "ssl_cert_reqs",
    "ssl_certfile",
    "ssl_check_hostname",
    "ssl_ciphers",
    "ssl_keyfile",
    "ssl_min_version",
    "ssl_ocsp_context",
    "ssl_ocsp_expected_cert",
    "ssl_password",
    "ssl_validate_ocsp",
    "ssl_validate_ocsp_stapled",
    "unix_socket_path",
    "url",
    "username",
];

const UNSUPPORTED_EXTRA: &[&str] = &[
    "credential_provider",
    "sentinel_nodes",
    "retry",
    "connection_pool",
];

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

#[derive(Clone, Default)]
pub struct RedisSettings {
    pub url: Option<SecretValue>,
    pub host: Option<Spelled<String>>,
    pub port: Option<Spelled<u16>>,
    pub username: Option<SecretValue>,
    pub password: Option<SecretValue>,
    pub ssl: Option<Spelled<bool>>,
    pub db: Option<Spelled<i64>>,
    pub socket_timeout: Option<Spelled<f64>>,
    pub startup_nodes: Option<Value>,
    pub namespace: Option<Spelled<String>>,
    pub additional_fields: AdditionalFields,
    unparsed: DeferredInputs,
}

impl<'de> Deserialize<'de> for RedisSettings {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let fields = BTreeMap::<String, Value>::deserialize(deserializer)?;
        let scalar = |key: &str| fields.get(key).and_then(as_text);
        Ok(Self {
            url: scalar("url").map(SecretValue::new),
            host: scalar("host").map(Spelled::Value),
            port: spelled(scalar("port")),
            username: scalar("username").map(SecretValue::new),
            password: scalar("password").map(SecretValue::new),
            ssl: spelled(scalar("ssl")),
            db: spelled(scalar("db")),
            socket_timeout: spelled(scalar("socket_timeout")),
            startup_nodes: fields
                .get("startup_nodes")
                .filter(|value| !value.is_null())
                .cloned(),
            namespace: scalar("namespace").map(Spelled::Value),
            additional_fields: cloned(&fields, |key, _| !SUPPORTED.contains(&key)),
            unparsed: deferred_inputs(&fields),
        })
    }
}

impl fmt::Debug for RedisSettings {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RedisSettings")
            .field("url", &self.url)
            .field("host", &self.host)
            .field("port", &self.port)
            .field("username", &self.username)
            .field("password", &self.password)
            .field("ssl", &self.ssl)
            .field("db", &self.db)
            .field("socket_timeout", &self.socket_timeout)
            .field(
                "startup_nodes",
                &self.startup_nodes.as_ref().map(|_| "<configured>"),
            )
            .field("namespace", &self.namespace)
            .field("additional_fields", &self.additional_fields.keys())
            .field("unparsed", &self.unparsed.keys())
            .finish()
    }
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

#[derive(Clone, PartialEq)]
pub struct ResolvedRedis {
    pub endpoint: RedisEndpoint,
    pub namespace: Option<String>,
    pub timeout: Option<Duration>,
}

impl fmt::Debug for ResolvedRedis {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ResolvedRedis")
            .field("endpoint", &self.endpoint)
            .field("namespace", &self.namespace)
            .field("timeout", &self.timeout)
            .finish()
    }
}

#[derive(Clone, PartialEq)]
pub enum RedisEndpoint {
    Url(SecretValue),
    Host(RedisHost),
    Cluster(RedisCluster),
}

impl fmt::Debug for RedisEndpoint {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Url(_) => formatter.debug_tuple("Url").field(&"<redacted>").finish(),
            Self::Host(host) => host.fmt(formatter),
            Self::Cluster(cluster) => cluster.fmt(formatter),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RedisHost {
    pub host: String,
    pub port: u16,
    pub username: Option<SecretValue>,
    pub password: Option<SecretValue>,
    pub tls: bool,
    pub db: i64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RedisCluster {
    pub nodes: Vec<RedisNode>,
    pub username: Option<SecretValue>,
    pub password: Option<SecretValue>,
    pub tls: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RedisNode {
    pub host: String,
    pub port: u16,
}

#[derive(Clone)]
struct Entry {
    value: Value,
    source: String,
}

type Kwargs = BTreeMap<String, Entry>;

impl RedisSettings {
    pub fn resolve(&self, lookup: Lookup<'_>) -> Result<ResolvedRedis, Error> {
        self.resolve_with(lookup, false)
    }

    pub fn resolve_cache(&self, lookup: Lookup<'_>) -> Result<ResolvedRedis, Error> {
        self.resolve_with(lookup, true)
    }

    fn resolve_with(&self, lookup: Lookup<'_>, cache: bool) -> Result<ResolvedRedis, Error> {
        let block = self.kwargs(cache);
        let block: Kwargs = block
            .into_iter()
            .filter_map(|(key, entry)| {
                let value = environment_reference(&entry.value, lookup)?;
                Some((key, Entry { value, ..entry }))
            })
            .collect();
        let environment = environment_kwargs(lookup);
        let names_a_target = ["host", "startup_nodes", "sentinel_nodes"]
            .iter()
            .any(|key| block.contains_key(*key));
        let drop_environment_url = names_a_target && !block.contains_key("url");
        let merged: Kwargs = environment
            .into_iter()
            .filter(|(key, _)| !(key == "url" && drop_environment_url))
            .chain(block)
            .collect();
        reject_unsupported(&merged, lookup)?;
        warn_ignored(&merged);

        let startup_nodes = match merged.get("startup_nodes") {
            Some(entry) => Some(cluster_nodes(&entry.value)?),
            None => lookup("REDIS_CLUSTER_NODES")
                .map(|nodes| cluster_nodes(&Value::String(nodes)))
                .transpose()?,
        };
        let username = secret_value(&merged, "username");
        let password = secret_value(&merged, "password");
        let tls = bool_value(&merged, "ssl");
        let endpoint = match (startup_nodes, text_value(&merged, "url")) {
            (Some(nodes), _) if !nodes.is_empty() => RedisEndpoint::Cluster(RedisCluster {
                nodes,
                username,
                password,
                tls,
            }),
            (_, Some(url)) => RedisEndpoint::Url(SecretValue::new(url)),
            _ => RedisEndpoint::Host(RedisHost {
                host: text_value(&merged, "host").ok_or(Error::RedisInvalid {
                    setting: "host",
                    reason: "either host or url must be specified for redis",
                })?,
                port: integer(&merged, "port")?.unwrap_or(6379),
                username,
                password,
                tls,
                db: integer(&merged, "db")?.unwrap_or(0),
            }),
        };
        let timeout = merged
            .get("socket_timeout")
            .and_then(|entry| as_text(&entry.value))
            .map(|seconds| {
                seconds
                    .parse::<f64>()
                    .ok()
                    .and_then(|seconds| Duration::try_from_secs_f64(seconds).ok())
                    .ok_or(Error::RedisInvalid {
                        setting: "socket_timeout",
                        reason: "must be a non-negative number of seconds",
                    })
            })
            .transpose()?;
        Ok(ResolvedRedis {
            endpoint,
            namespace: text_value(&merged, "namespace"),
            timeout,
        })
    }

    fn kwargs(&self, cache: bool) -> Kwargs {
        let known = [
            (
                "url",
                self.url.as_ref().map(|value| value.expose().to_owned()),
            ),
            ("host", spelled_text(self.host.as_ref())),
            ("port", spelled_text(self.port.as_ref())),
            (
                "username",
                self.username
                    .as_ref()
                    .map(|value| value.expose().to_owned()),
            ),
            (
                "password",
                self.password
                    .as_ref()
                    .map(|value| value.expose().to_owned()),
            ),
            ("ssl", spelled_text(self.ssl.as_ref())),
            ("db", spelled_text(self.db.as_ref())),
            ("socket_timeout", spelled_text(self.socket_timeout.as_ref())),
            ("namespace", spelled_text(self.namespace.as_ref())),
        ];
        let known = known.into_iter().filter_map(|(key, value)| {
            value.map(|value| {
                (
                    key.to_owned(),
                    Entry {
                        value: Value::String(value),
                        source: key.to_owned(),
                    },
                )
            })
        });
        let startup_nodes = self.startup_nodes.clone().map(|value| {
            (
                "startup_nodes".to_owned(),
                Entry {
                    value,
                    source: "startup_nodes".to_owned(),
                },
            )
        });
        let extras = self.additional_fields.iter().filter_map(|(key, value)| {
            if cache && CACHE_ONLY.contains(&key.as_str()) {
                return None;
            }
            let key = match (cache, key.as_str()) {
                (true, "redis_startup_nodes") => "startup_nodes".to_owned(),
                _ => key.clone(),
            };
            Some((
                key.clone(),
                Entry {
                    value: value.clone(),
                    source: key,
                },
            ))
        });
        let unparsed = self.unparsed.iter().map(|(key, value)| {
            (
                key.clone(),
                Entry {
                    value: value.0.clone(),
                    source: key.clone(),
                },
            )
        });
        extras
            .chain(unparsed)
            .chain(known)
            .chain(startup_nodes)
            .collect()
    }
}

pub fn environment_has_target(lookup: Lookup<'_>) -> bool {
    [
        "REDIS_HOST",
        "REDIS_URL",
        "REDIS_CLUSTER_NODES",
        "REDIS_SENTINEL_NODES",
    ]
    .iter()
    .any(|name| lookup(name).is_some())
}

fn reject_unsupported(merged: &Kwargs, lookup: Lookup<'_>) -> Result<(), Error> {
    let active_auth = [
        merged
            .get("gcp_service_account")
            .map(|entry| entry.source.as_str())
            .or_else(|| lookup("REDIS_GCP_SERVICE_ACCOUNT").map(|_| "REDIS_GCP_SERVICE_ACCOUNT")),
        text_value(merged, "azure_redis_ad_token")
            .or_else(|| lookup("REDIS_AZURE_AD_TOKEN"))
            .filter(|value| value.eq_ignore_ascii_case("true"))
            .map(|_| {
                merged
                    .get("azure_redis_ad_token")
                    .map_or("REDIS_AZURE_AD_TOKEN", |entry| entry.source.as_str())
            }),
        bool_value(merged, "aws_iam_auth").then(|| merged["aws_iam_auth"].source.as_str()),
        merged
            .get("sentinel_nodes")
            .map(|entry| entry.source.as_str())
            .or_else(|| lookup("REDIS_SENTINEL_NODES").map(|_| "REDIS_SENTINEL_NODES")),
    ];
    if let Some(setting) = active_auth.into_iter().flatten().next() {
        return Err(Error::RedisSetting(setting.to_owned()));
    }
    match merged
        .iter()
        .find(|(key, _)| is_unsupported(key))
        .map(|(_, entry)| entry.source.clone())
    {
        Some(setting) => Err(Error::RedisSetting(setting)),
        None => Ok(()),
    }
}

fn warn_ignored(merged: &Kwargs) {
    for setting in WARN_IGNORED {
        if merged.contains_key(*setting) {
            tracing::warn!(setting, "Redis setting is accepted but ignored");
        }
    }
}

fn is_unsupported(key: &str) -> bool {
    let known = REDIS_PY_KWARGS.contains(&key) || UNSUPPORTED_EXTRA.contains(&key);
    let handled = SUPPORTED.contains(&key)
        || IGNORED.contains(&key)
        || matches!(
            key,
            "gcp_service_account" | "azure_redis_ad_token" | "aws_iam_auth"
        );
    known && !handled
}

fn environment_kwargs(lookup: Lookup<'_>) -> Kwargs {
    REDIS_PY_KWARGS
        .iter()
        .filter_map(|key| {
            let source = format!("REDIS_{}", key.to_ascii_uppercase());
            let value = lookup(&source)?;
            Some((
                (*key).to_owned(),
                Entry {
                    value: Value::String(value),
                    source,
                },
            ))
        })
        .collect()
}

fn environment_reference(value: &Value, lookup: Lookup<'_>) -> Option<Value> {
    match value {
        Value::String(text) => match text.strip_prefix("os.environ/") {
            Some(name) => lookup(name).map(Value::String),
            None => Some(value.clone()),
        },
        Value::Null => None,
        _ => Some(value.clone()),
    }
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

fn integer<T: std::str::FromStr>(kwargs: &Kwargs, key: &'static str) -> Result<Option<T>, Error> {
    kwargs
        .get(key)
        .and_then(|entry| as_text(&entry.value))
        .map(|text| {
            text.parse().map_err(|_| Error::RedisInvalid {
                setting: key,
                reason: "must be an integer",
            })
        })
        .transpose()
}

const INVALID_NODES: Error = Error::RedisInvalid {
    setting: "startup_nodes",
    reason: "must be a list of {host, port} objects",
};

fn cluster_nodes(value: &Value) -> Result<Vec<RedisNode>, Error> {
    match value {
        Value::String(json) => {
            let nodes: Vec<serde_json::Map<String, serde_json::Value>> =
                serde_json::from_str(json).map_err(|_| INVALID_NODES)?;
            nodes
                .iter()
                .map(|node| {
                    cluster_node(
                        node.get("host").and_then(json_text),
                        node.get("port").and_then(json_text),
                    )
                })
                .collect()
        }
        Value::Sequence(nodes) => nodes
            .iter()
            .map(|node| {
                cluster_node(
                    node.get("host").and_then(as_text),
                    node.get("port").and_then(as_text),
                )
            })
            .collect(),
        _ => Err(INVALID_NODES),
    }
}

fn cluster_node(host: Option<String>, port: Option<String>) -> Result<RedisNode, Error> {
    match (host, port.and_then(|port| port.parse().ok())) {
        (Some(host), Some(port)) => Ok(RedisNode { host, port }),
        _ => Err(INVALID_NODES),
    }
}

fn json_text(value: &serde_json::Value) -> Option<String> {
    match value {
        serde_json::Value::String(text) => Some(text.clone()),
        serde_json::Value::Number(number) => Some(number.to_string()),
        _ => None,
    }
}

fn spelled_text<T: ToString>(value: Option<&Spelled<T>>) -> Option<String> {
    value.map(|value| match value {
        Spelled::Value(value) => value.to_string(),
        Spelled::Text(value) => value.clone(),
    })
}

fn spelled<T: std::str::FromStr>(text: Option<String>) -> Option<Spelled<T>> {
    text.map(|text| match text.parse() {
        Ok(value) => Spelled::Value(value),
        Err(_) => Spelled::Text(text),
    })
}

fn cloned(
    fields: &BTreeMap<String, Value>,
    keep: impl Fn(&str, &Value) -> bool,
) -> AdditionalFields {
    fields
        .iter()
        .filter(|(key, value)| keep(key, value))
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect()
}

#[derive(Clone)]
struct DeferredInput(Value);

type DeferredInputs = BTreeMap<String, DeferredInput>;

fn deferred_inputs(fields: &BTreeMap<String, Value>) -> DeferredInputs {
    fields
        .iter()
        .filter(|(key, value)| is_unparsed(key, value))
        .map(|(key, value)| (key.clone(), DeferredInput(value.clone())))
        .collect()
}

fn is_unparsed(key: &str, value: &Value) -> bool {
    SUPPORTED.contains(&key)
        && match key {
            "startup_nodes" => value.is_null(),
            _ => as_text(value).is_none(),
        }
}
