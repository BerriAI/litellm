use std::time::Duration;

use redis::{ConnectionAddr, ConnectionInfo, IntoConnectionInfo, RedisConnectionInfo, RedisResult};

use crate::Namespace;

pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(5);

pub struct RedisConfig {
    pub endpoint: Endpoint,
    pub namespace: Namespace,
    pub timeout: Duration,
}

pub enum Endpoint {
    Url(String),
    Host(HostEndpoint),
    Cluster(ClusterEndpoint),
}

pub struct HostEndpoint {
    pub host: String,
    pub port: u16,
    pub username: Option<String>,
    pub password: Option<String>,
    pub tls: bool,
    pub db: i64,
}

pub struct ClusterEndpoint {
    pub nodes: Vec<ClusterNode>,
    pub username: Option<String>,
    pub password: Option<String>,
    pub tls: bool,
}

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
