use std::{path::PathBuf, time::Duration};

use litellm_secret_value::SecretValue;

/// One Redis connection with every setting, environment value and secret already applied, the
/// value a host hands to the connector.
#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedRedis {
    pub endpoint: Endpoint,
    pub auth: RedisAuth,
    pub namespace: Option<String>,
    pub timeout: Option<Duration>,
    pub connect_timeout: Option<Duration>,
}

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

#[derive(Clone, Debug, PartialEq)]
pub enum RedisAuth {
    Static {
        username: Option<SecretValue>,
        password: Option<SecretValue>,
    },
    /// The host supplies credentials, like a redis-py `credential_provider` object.
    Host,
    GcpIam {
        service_account: String,
    },
    AzureAd(AzureAdAuth),
    AwsIam(AwsIamAuth),
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
