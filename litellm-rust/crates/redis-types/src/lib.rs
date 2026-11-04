use std::{
    collections::{BTreeMap, BTreeSet},
    path::PathBuf,
    time::Duration,
};

use litellm_secret_value::SecretValue;
use veil::Redact;

/// One Redis connection with every setting, environment value and secret already applied. Each
/// client library builds its client from this one value, so they cannot read a setting differently.
#[derive(Clone, Debug, PartialEq)]
pub struct Resolution {
    pub target: Target,
    pub auth: Auth,
    pub namespace: Option<String>,
    pub timeout: Option<Duration>,
    pub connect_timeout: Option<Duration>,
    pub health_check: HealthCheck,
    /// Plain settings the resolver does not model, kept verbatim for the redis-py projection.
    pub passthrough: Passthrough,
    /// Settings whose value is a host object, like redis-py's `retry` or `connection_class`, which
    /// only the host can reattach.
    pub host_objects: BTreeSet<String>,
    pub ignored: Vec<Ignored>,
}

/// redis-py's `health_check_interval`, which `_coerce_redis_kwargs_types` reads as an `int` of
/// seconds. Unset and zero differ, because a client may give an unset interval its own default.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum HealthCheck {
    Unset,
    Disabled,
    Every(Duration),
}

/// A setting the Rust client accepts without honoring, so the host can log what ignoring it means.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Ignored {
    /// The key, the environment variable or the url query parameter that set it.
    pub setting: String,
    pub consequence: &'static str,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Target {
    Standalone(Standalone),
    /// Like redis-py's `UnixDomainSocketConnection`, which replaces host, port and TLS.
    Unix(UnixSocket),
    Url(RedisUrl),
    Cluster(Cluster),
    Sentinel(Sentinel),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Standalone {
    pub host: String,
    pub port: u16,
    pub tls: Option<TlsOptions>,
    pub db: i64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct UnixSocket {
    pub path: PathBuf,
    pub db: i64,
}

/// A url kept verbatim, plus what redis-py's `parse_url` reads from its query that the url's own
/// grammar does not carry.
#[derive(Clone, Debug, PartialEq)]
pub struct RedisUrl {
    pub url: SecretValue,
    /// Set only for a `rediss://` url, from its query parameters.
    pub tls: Option<TlsOptions>,
    /// `?db=`, which wins over the path.
    pub db: Option<i64>,
    /// `?username=` and `?password=`, which the url's userinfo wins over.
    pub username: Option<SecretValue>,
    pub password: Option<SecretValue>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Cluster {
    pub nodes: Vec<Node>,
    pub tls: Option<TlsOptions>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Node {
    pub host: String,
    pub port: u16,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Sentinel {
    pub monitors: Vec<Node>,
    pub service_name: String,
    /// How to authenticate to the monitors, which are separate servers from the data node.
    pub monitor_auth: Option<StaticAuth>,
    pub tls: Option<TlsOptions>,
    pub db: i64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TlsOptions {
    pub ca_certs: Option<PathBuf>,
    /// PEM text trusted alongside `ca_certs`, redis-py's `ssl_ca_data`
    pub ca_data: Option<String>,
    /// A directory whose files are PEM certificates to trust, redis-py's `ssl_ca_path`
    pub ca_path: Option<PathBuf>,
    pub certfile: Option<PathBuf>,
    pub keyfile: Option<PathBuf>,
    /// Unlocks an encrypted `keyfile`, redis-py's `ssl_password`
    pub key_password: Option<SecretValue>,
    pub verify_certificate: bool,
    pub check_hostname: bool,
}

impl TlsOptions {
    pub fn verified() -> Self {
        Self {
            ca_certs: None,
            ca_data: None,
            ca_path: None,
            certfile: None,
            keyfile: None,
            key_password: None,
            verify_certificate: true,
            check_hostname: false,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Auth {
    Static(StaticAuth),
    /// Credentials come from a host object, like a redis-py `credential_provider`.
    Provider(ProviderHandle),
    GcpIam {
        service_account: String,
    },
    AzureAd(AzureAdAuth),
    AwsIam(AwsIamAuth),
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct StaticAuth {
    pub username: Option<SecretValue>,
    pub password: Option<SecretValue>,
}

/// Names one credential provider the host holds. Two resolutions with the same handle use the
/// same provider.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ProviderHandle(u64);

impl ProviderHandle {
    pub const fn new(id: u64) -> Self {
        Self(id)
    }

    pub const fn id(self) -> u64 {
        self.0
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct AzureAdAuth {
    pub client_id: Option<String>,
    pub tenant_id: Option<String>,
    pub client_secret: Option<SecretValue>,
    pub username: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AwsIamAuth {
    pub user_name: String,
    pub cache_name: String,
    pub region: String,
    pub serverless: bool,
}

/// Settings passed through untouched. They can hold credentials, so they never print.
#[derive(Clone, Default, PartialEq, Redact)]
pub struct Passthrough(#[redact(with = "[REDACTED]")] BTreeMap<String, serde_json::Value>);

impl Passthrough {
    pub fn new(settings: BTreeMap<String, serde_json::Value>) -> Self {
        Self(settings)
    }

    pub fn get(&self, key: &str) -> Option<&serde_json::Value> {
        self.0.get(key)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, &serde_json::Value)> {
        self.0.iter().map(|(key, value)| (key.as_str(), value))
    }
}
