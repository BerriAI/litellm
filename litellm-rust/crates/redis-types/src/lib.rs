use std::path::PathBuf;

use litellm_secret_value::SecretValue;

#[derive(Clone, Debug, PartialEq)]
pub enum Endpoint {
    Url { url: SecretValue, tls: TlsOptions },
    Host(HostEndpoint),
    Cluster(ClusterEndpoint),
    Sentinel(SentinelEndpoint),
}

#[derive(Clone, Debug, PartialEq)]
pub struct HostEndpoint {
    pub host: String,
    pub port: u16,
    pub tls: Option<TlsOptions>,
    pub db: i64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ClusterEndpoint {
    pub nodes: Vec<ClusterNode>,
    pub tls: Option<TlsOptions>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ClusterNode {
    pub host: String,
    pub port: u16,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SentinelEndpoint {
    pub sentinels: Vec<ClusterNode>,
    pub service_name: String,
    pub sentinel_password: Option<SecretValue>,
    pub tls: Option<TlsOptions>,
    pub db: i64,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct TlsOptions {
    pub ca_certs: Option<PathBuf>,
    pub certfile: Option<PathBuf>,
    pub keyfile: Option<PathBuf>,
    pub verify_certificate: bool,
    pub check_hostname: bool,
}

impl TlsOptions {
    pub fn verified() -> Self {
        Self {
            verify_certificate: true,
            ..Self::default()
        }
    }
}
