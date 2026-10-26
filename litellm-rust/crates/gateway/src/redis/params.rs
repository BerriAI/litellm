use std::{collections::BTreeMap, time::Duration};

use litellm_config::{Object, Value};
use litellm_redis_client::{
    ClusterEndpoint, ClusterNode, DEFAULT_TIMEOUT, Endpoint, HostEndpoint, Namespace,
};

use crate::Error;

pub(crate) type Lookup<'a> = &'a dyn Fn(&str) -> Option<String>;

type Kwargs = BTreeMap<String, Value>;

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
    "redis_flush_size",
    "service_name",
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

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Params {
    pub endpoint: Endpoint,
    pub namespace: Namespace,
    pub timeout: Duration,
}

pub(crate) fn environment_has_target(lookup: Lookup<'_>) -> bool {
    [
        "REDIS_HOST",
        "REDIS_URL",
        "REDIS_CLUSTER_NODES",
        "REDIS_SENTINEL_NODES",
    ]
    .iter()
    .any(|name| lookup(name).is_some())
}

pub(crate) fn resolve(block: &Kwargs, lookup: Lookup<'_>) -> Result<Params, Error> {
    let block: Kwargs = block
        .iter()
        .filter_map(|(key, value)| Some((key.clone(), environment_reference(value, lookup)?)))
        .collect();
    let environment = environment_kwargs(lookup);
    let names_a_target = ["host", "startup_nodes", "sentinel_nodes"]
        .iter()
        .any(|key| block.contains_key(*key));
    let drop_environment_url = names_a_target && !block.contains_key("url");
    let environment = environment
        .into_iter()
        .filter(|(key, _)| !(key == "url" && drop_environment_url));
    let merged: Kwargs = environment.chain(block).collect();
    reject_unsupported(&merged, lookup)?;

    let startup_nodes = match merged.get("startup_nodes") {
        Some(nodes) => Some(cluster_nodes(nodes)?),
        None => lookup("REDIS_CLUSTER_NODES")
            .map(|nodes| cluster_nodes(&Value::String(nodes)))
            .transpose()?,
    };
    let username = text(&merged, "username");
    let password = text(&merged, "password");
    let tls = flag(&merged, "ssl");
    let endpoint = match (startup_nodes, text(&merged, "url")) {
        (Some(nodes), _) if !nodes.is_empty() => Endpoint::Cluster(ClusterEndpoint {
            nodes,
            username,
            password,
            tls,
        }),
        (_, Some(url)) => Endpoint::Url(url),
        _ => Endpoint::Host(HostEndpoint {
            host: text(&merged, "host").ok_or(Error::RedisInvalid {
                setting: "host",
                reason: "either host or url must be specified for redis",
            })?,
            port: number(&merged, "port")?.unwrap_or(6379),
            username,
            password,
            tls,
            db: number(&merged, "db")?.unwrap_or(0),
        }),
    };
    let timeout = match merged.get("socket_timeout").and_then(as_text) {
        Some(seconds) => seconds
            .parse::<f64>()
            .ok()
            .and_then(|seconds| Duration::try_from_secs_f64(seconds).ok())
            .ok_or(Error::RedisInvalid {
                setting: "socket_timeout",
                reason: "must be a non-negative number of seconds",
            })?,
        None => DEFAULT_TIMEOUT,
    };
    Ok(Params {
        endpoint,
        namespace: text(&merged, "namespace")
            .map(Namespace::new)
            .unwrap_or_default(),
        timeout,
    })
}

pub(crate) fn cache_block(cache_params: &Object) -> Kwargs {
    cache_params
        .iter()
        .filter(|(key, _)| !CACHE_ONLY.contains(&key.as_str()))
        .map(|(key, value)| match key.as_str() {
            "redis_startup_nodes" => ("startup_nodes".to_owned(), value.clone()),
            _ => (key.clone(), value.clone()),
        })
        .collect()
}

const CACHE_ONLY: &[&str] = &[
    "type",
    "mode",
    "namespace",
    "ttl",
    "default_in_memory_ttl",
    "default_in_redis_ttl",
    "supported_call_types",
    "similarity_threshold",
    "max_messages",
    "_backend",
];

fn reject_unsupported(merged: &Kwargs, lookup: Lookup<'_>) -> Result<(), Error> {
    let active_auth = [
        (
            "gcp_service_account",
            merged.contains_key("gcp_service_account")
                || lookup("REDIS_GCP_SERVICE_ACCOUNT").is_some(),
        ),
        (
            "azure_redis_ad_token",
            text(merged, "azure_redis_ad_token")
                .or_else(|| lookup("REDIS_AZURE_AD_TOKEN"))
                .is_some_and(|value| value.eq_ignore_ascii_case("true")),
        ),
        ("aws_iam_auth", flag(merged, "aws_iam_auth")),
        (
            "sentinel_nodes",
            merged.contains_key("sentinel_nodes") || lookup("REDIS_SENTINEL_NODES").is_some(),
        ),
    ];
    if let Some((setting, _)) = active_auth.iter().find(|(_, active)| *active) {
        return Err(Error::RedisSetting((*setting).to_owned()));
    }
    match merged.keys().find(|key| is_unsupported(key)) {
        Some(key) => Err(Error::RedisSetting(key.clone())),
        None => Ok(()),
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
            let value = lookup(&format!("REDIS_{}", key.to_ascii_uppercase()))?;
            Some(((*key).to_owned(), Value::String(value)))
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

fn text(kwargs: &Kwargs, key: &str) -> Option<String> {
    kwargs.get(key).and_then(as_text)
}

fn flag(kwargs: &Kwargs, key: &str) -> bool {
    match kwargs.get(key) {
        Some(Value::Bool(flag)) => *flag,
        Some(value) => as_text(value)
            .is_some_and(|text| matches!(text.to_ascii_lowercase().as_str(), "true" | "1" | "yes")),
        None => false,
    }
}

fn number<T: std::str::FromStr>(kwargs: &Kwargs, key: &'static str) -> Result<Option<T>, Error> {
    kwargs
        .get(key)
        .and_then(as_text)
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

fn cluster_nodes(value: &Value) -> Result<Vec<ClusterNode>, Error> {
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

fn cluster_node(host: Option<String>, port: Option<String>) -> Result<ClusterNode, Error> {
    match (host, port.and_then(|port| port.parse().ok())) {
        (Some(host), Some(port)) => Ok(ClusterNode { host, port }),
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
