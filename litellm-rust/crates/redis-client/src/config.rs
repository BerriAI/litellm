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
}

pub struct HostEndpoint {
    pub host: String,
    pub port: u16,
    pub username: Option<String>,
    pub password: Option<String>,
    pub tls: bool,
    pub db: i64,
}

impl Endpoint {
    pub(crate) fn connection_info(&self) -> RedisResult<ConnectionInfo> {
        match self {
            Self::Url(url) => url.as_str().into_connection_info(),
            Self::Host(host) => host.connection_info(),
        }
    }
}

impl HostEndpoint {
    fn connection_info(&self) -> RedisResult<ConnectionInfo> {
        let addr = match self.tls {
            true => ConnectionAddr::TcpTls {
                host: self.host.clone(),
                port: self.port,
                insecure: false,
                tls_params: None,
            },
            false => ConnectionAddr::Tcp(self.host.clone(), self.port),
        };
        let settings = RedisConnectionInfo::default().set_db(self.db);
        let settings = match &self.username {
            Some(username) => settings.set_username(username),
            None => settings,
        };
        let settings = match &self.password {
            Some(password) => settings.set_password(password),
            None => settings,
        };
        Ok(addr.into_connection_info()?.set_redis_settings(settings))
    }
}
