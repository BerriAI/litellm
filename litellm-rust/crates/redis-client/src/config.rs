use std::time::Duration;

use redis::{ConnectionAddr, ConnectionInfo, IntoConnectionInfo, RedisConnectionInfo, RedisResult};

use litellm_circuit_breaker::Breaker;
use litellm_secret_value::SecretValue;

use crate::Namespace;

pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(5);

pub struct RedisConfig {
    pub endpoint: Endpoint,
    pub namespace: Namespace,
    pub timeout: Duration,
    pub breaker: Option<Breaker>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Endpoint {
    Url(SecretValue),
    Host(HostEndpoint),
    Cluster(ClusterEndpoint),
}

#[derive(Clone, Debug, PartialEq)]
pub struct HostEndpoint {
    pub host: String,
    pub port: u16,
    pub username: Option<SecretValue>,
    pub password: Option<SecretValue>,
    pub tls: bool,
    pub db: i64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ClusterEndpoint {
    pub nodes: Vec<ClusterNode>,
    pub username: Option<SecretValue>,
    pub password: Option<SecretValue>,
    pub tls: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ClusterNode {
    pub host: String,
    pub port: u16,
}

impl ClusterEndpoint {
    pub(crate) fn addrs(&self) -> impl Iterator<Item = ConnectionAddr> + '_ {
        self.nodes
            .iter()
            .map(|node| addr(&node.host, node.port, self.tls))
    }
}

impl HostEndpoint {
    pub(crate) fn connection_info(&self) -> RedisResult<ConnectionInfo> {
        let settings = RedisConnectionInfo::default().set_db(self.db);
        let settings = match &self.username {
            Some(username) => settings.set_username(username.expose()),
            None => settings,
        };
        let settings = match &self.password {
            Some(password) => settings.set_password(password.expose()),
            None => settings,
        };
        Ok(addr(&self.host, self.port, self.tls)
            .into_connection_info()?
            .set_redis_settings(settings))
    }
}

fn addr(host: &str, port: u16, tls: bool) -> ConnectionAddr {
    match tls {
        true => ConnectionAddr::TcpTls {
            host: host.to_owned(),
            port,
            insecure: false,
            tls_params: None,
        },
        false => ConnectionAddr::Tcp(host.to_owned(), port),
    }
}

#[cfg(test)]
mod tests {
    use litellm_secret_value::SecretValue;
    use rstest::rstest;

    use super::{ClusterEndpoint, Endpoint, HostEndpoint};

    #[rstest]
    #[case::url(Endpoint::Url(SecretValue::new("redis://credential-value@cache:6379/0")))]
    #[case::host(
        Endpoint::Host(HostEndpoint {
            host: "cache".to_owned(),
            port: 6379,
            username: Some(SecretValue::new("credential-value")),
            password: Some(SecretValue::new("credential-value")),
            tls: false,
            db: 0,
        })
    )]
    #[case::cluster(Endpoint::Cluster(ClusterEndpoint {
        nodes: Vec::new(),
        username: Some(SecretValue::new("credential-value")),
        password: Some(SecretValue::new("credential-value")),
        tls: false,
    }))]
    fn debug_output_never_shows_the_secret(#[case] endpoint: Endpoint) {
        let debug = format!("{endpoint:?}");

        assert!(!debug.contains("credential-value"), "{debug}");
        assert!(debug.contains("[REDACTED]"), "{debug}");
    }
}
