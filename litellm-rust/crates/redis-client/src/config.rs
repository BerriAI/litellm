use std::{fmt, time::Duration};

use redis::{ConnectionAddr, ConnectionInfo, IntoConnectionInfo, RedisConnectionInfo, RedisResult};

use litellm_circuit_breaker::Breaker;

use crate::Namespace;

pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(5);

pub struct RedisConfig {
    pub endpoint: Endpoint,
    pub namespace: Namespace,
    pub timeout: Duration,
    pub breaker: Option<Breaker>,
}

#[derive(Clone, PartialEq)]
pub enum Endpoint {
    Url(String),
    Host(HostEndpoint),
    Cluster(ClusterEndpoint),
}

#[derive(Clone, PartialEq)]
pub struct HostEndpoint {
    pub host: String,
    pub port: u16,
    pub username: Option<String>,
    pub password: Option<String>,
    pub tls: bool,
    pub db: i64,
}

#[derive(Clone, PartialEq)]
pub struct ClusterEndpoint {
    pub nodes: Vec<ClusterNode>,
    pub username: Option<String>,
    pub password: Option<String>,
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
            Some(username) => settings.set_username(username),
            None => settings,
        };
        let settings = match &self.password {
            Some(password) => settings.set_password(password),
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

impl fmt::Debug for Endpoint {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Url(url) => formatter
                .debug_tuple("Url")
                .field(&redacted_url(url))
                .finish(),
            Self::Host(host) => formatter
                .debug_struct("Host")
                .field("host", &host.host)
                .field("port", &host.port)
                .field("username", &host.username)
                .field("password", &host.password.as_ref().map(|_| "<redacted>"))
                .field("tls", &host.tls)
                .field("db", &host.db)
                .finish(),
            Self::Cluster(cluster) => formatter
                .debug_struct("Cluster")
                .field("nodes", &cluster.nodes)
                .field("username", &cluster.username)
                .field("password", &cluster.password.as_ref().map(|_| "<redacted>"))
                .field("tls", &cluster.tls)
                .finish(),
        }
    }
}

fn redacted_url(url: &str) -> String {
    match url
        .split_once("://")
        .and_then(|(scheme, rest)| Some((scheme, rest.rsplit_once('@')?.1)))
    {
        Some((scheme, host)) => format!("{scheme}://<redacted>@{host}"),
        None => url.to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::{Endpoint, HostEndpoint};

    #[rstest]
    #[case::url_with_password(Endpoint::Url("redis://user:secret@cache:6379/0".to_owned()), "redis://<redacted>@cache:6379/0")]
    #[case::url_with_at_in_password(Endpoint::Url("rediss://:p@ss@cache:6380".to_owned()), "rediss://<redacted>@cache:6380")]
    #[case::host_password(
        Endpoint::Host(HostEndpoint {
            host: "cache".to_owned(),
            port: 6379,
            username: None,
            password: Some("secret".to_owned()),
            tls: false,
            db: 0,
        }),
        "<redacted>"
    )]
    fn debug_output_never_shows_the_password(#[case] endpoint: Endpoint, #[case] shown: &str) {
        let debug = format!("{endpoint:?}");

        assert!(
            !debug.contains("secret") && !debug.contains("p@ss"),
            "{debug}"
        );
        assert!(debug.contains(shown), "{debug}");
    }
}
