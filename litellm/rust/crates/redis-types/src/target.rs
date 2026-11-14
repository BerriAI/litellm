use std::path::PathBuf;

use litellm_secret_value::SecretValue;

use crate::{StaticAuth, TlsOptions};

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
