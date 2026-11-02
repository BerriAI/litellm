use std::{fmt, path::PathBuf, pin::Pin, sync::Arc, time::Duration};

use futures_util::Stream;

use redis::{
    BasicAuth, ClientTlsConfig, ConnectionAddr, ConnectionInfo, IntoConnectionInfo,
    RedisConnectionInfo, RedisResult, StreamingCredentialsProvider, TlsCertificates, TlsMode,
    io::tcp::TcpSettings,
};

use litellm_circuit_breaker::Breaker;
use litellm_secret_value::SecretValue;

use crate::{ConnectError, Namespace, RedisEvents};
use litellm_redis_types::{ClusterEndpoint, Endpoint, TlsOptions};

// litellm/constants.py REDIS_SOCKET_TIMEOUT, which _coerce_redis_kwargs_types sets when socket_timeout is unset
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(5);

pub struct RedisConfig {
    pub endpoint: Endpoint,
    pub credentials: Credentials,
    pub namespace: Namespace,
    pub timeouts: Timeouts,
    pub breaker: Option<Breaker>,
    pub events: Option<Arc<dyn RedisEvents>>,
    /// Ping every node before the next command once the client sat idle this long, like
    /// redis-py's `health_check_interval`, so a connection the server dropped silently is
    /// replaced before a real command pays for it.
    pub health_check: Option<Duration>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Timeouts {
    pub response: Duration,
    pub connect: Duration,
}

impl Timeouts {
    pub fn new(socket_timeout: Duration, connect_timeout: Option<Duration>) -> Self {
        Self {
            response: socket_timeout,
            connect: connect_timeout.unwrap_or(socket_timeout),
        }
    }
}

impl Default for Timeouts {
    fn default() -> Self {
        Self::new(DEFAULT_TIMEOUT, None)
    }
}

#[derive(Clone, Debug)]
pub enum Credentials {
    Static {
        username: Option<SecretValue>,
        password: Option<SecretValue>,
    },
    Provider(CredentialsProvider),
}

#[derive(Clone)]
pub struct CredentialsProvider(pub Arc<dyn StreamingCredentialsProvider>);

impl fmt::Debug for CredentialsProvider {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CredentialsProvider")
            .finish_non_exhaustive()
    }
}

impl StreamingCredentialsProvider for CredentialsProvider {
    fn subscribe(&self) -> Pin<Box<dyn Stream<Item = RedisResult<BasicAuth>> + Send + 'static>> {
        self.0.subscribe()
    }
}

impl Credentials {
    pub const NONE: Self = Self::Static {
        username: None,
        password: None,
    };
}

pub(crate) fn tls_mode(tls: &TlsOptions) -> TlsMode {
    match tls.verify_certificate {
        true => TlsMode::Secure,
        false => TlsMode::Insecure,
    }
}

pub(crate) fn tls_certificates(tls: &TlsOptions) -> Result<TlsCertificates, ConnectError> {
    let client_tls = match (&tls.certfile, &tls.keyfile) {
        (Some(certfile), keyfile) => Some(ClientTlsConfig {
            client_cert: read(certfile)?,
            client_key: read(keyfile.as_ref().unwrap_or(certfile))?,
        }),
        (None, _) => None,
    };
    let from_file: Option<Vec<u8>> = tls.ca_certs.as_ref().map(read).transpose()?;
    let from_directory: Vec<Vec<u8>> = tls
        .ca_path
        .as_ref()
        .map(read_directory)
        .transpose()?
        .unwrap_or_default();
    let configured: Vec<Vec<u8>> = from_file
        .into_iter()
        .chain(tls.ca_data.iter().map(|pem| pem.clone().into_bytes()))
        .chain(from_directory)
        .collect();
    Ok(TlsCertificates {
        client_tls,
        root_cert: (!configured.is_empty()).then(|| with_system_roots(configured.join(&b'\n'))),
    })
}

/// Every regular file in `path`, in name order, as redis-py hands `ssl_ca_path` to
/// `load_verify_locations(capath=...)`
fn read_directory(path: &PathBuf) -> Result<Vec<Vec<u8>>, ConnectError> {
    let tls_file = |source| ConnectError::TlsFile {
        path: path.display().to_string(),
        source,
    };
    let mut files: Vec<PathBuf> = std::fs::read_dir(path)
        .map_err(tls_file)?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<Result<_, _>>()
        .map_err(tls_file)?;
    files.sort();
    files
        .iter()
        .filter(|file| file.is_file())
        .map(read)
        .collect()
}

pub(crate) fn apply_tls(
    tls: &TlsOptions,
    info: ConnectionInfo,
) -> Result<ConnectionInfo, ConnectError> {
    let tls_info = redis::Client::build_with_tls(info, tls_certificates(tls)?)?
        .get_connection_info()
        .clone();
    let mut addr = match tls_info.addr().clone() {
        ConnectionAddr::TcpTls {
            host,
            port,
            tls_params,
            ..
        } => ConnectionAddr::TcpTls {
            host,
            port,
            insecure: !tls.verify_certificate,
            tls_params,
        },
        addr => addr,
    };
    addr.set_danger_accept_invalid_hostnames(!tls.check_hostname);
    Ok(tls_info.set_addr(addr))
}

fn with_system_roots(configured: Vec<u8>) -> Vec<u8> {
    let system = rustls_native_certs::load_native_certs();
    let system_pem = system
        .certs
        .iter()
        .map(|cert| pem::encode(&pem::Pem::new("CERTIFICATE", cert.to_vec())));
    configured
        .into_iter()
        .chain(b"\n".iter().copied())
        .chain(system_pem.flat_map(String::into_bytes))
        .collect()
}

fn read(path: &PathBuf) -> Result<Vec<u8>, ConnectError> {
    std::fs::read(path).map_err(|source| ConnectError::TlsFile {
        path: path.display().to_string(),
        source,
    })
}

pub(crate) fn addr(host: &str, port: u16, tls: Option<&TlsOptions>) -> ConnectionAddr {
    match tls {
        Some(tls) => ConnectionAddr::TcpTls {
            host: host.to_owned(),
            port,
            insecure: !tls.verify_certificate,
            tls_params: None,
        },
        None => ConnectionAddr::Tcp(host.to_owned(), port),
    }
}

pub(crate) fn cluster_addrs(
    endpoint: &ClusterEndpoint,
) -> impl Iterator<Item = ConnectionAddr> + '_ {
    endpoint
        .nodes
        .iter()
        .map(|node| addr(&node.host, node.port, endpoint.tls.as_ref()))
}

pub(crate) fn connection_info(
    endpoint: &Endpoint,
    credentials: &Credentials,
) -> Result<ConnectionInfo, ConnectError> {
    let (info, tls) = match endpoint {
        Endpoint::Url { url, tls } => (url.expose().into_connection_info()?, Some(tls)),
        Endpoint::Host(host) => {
            let (addr, tls) = match &host.unix_socket_path {
                Some(path) => (ConnectionAddr::Unix(path.clone()), None),
                None => (
                    addr(&host.host, host.port, host.tls.as_ref()),
                    host.tls.as_ref(),
                ),
            };
            let info = addr.into_connection_info()?;
            let settings = info.redis_settings().clone().set_db(host.db);
            (info.set_redis_settings(settings), tls)
        }
        Endpoint::Cluster(_) | Endpoint::Sentinel(_) => {
            unreachable!("cluster and sentinel endpoints build their own connection info")
        }
    };
    let info = with_static_credentials(info, credentials)
        .set_tcp_settings(TcpSettings::default().set_nodelay(true));
    match (tls, info.addr()) {
        (Some(tls), ConnectionAddr::TcpTls { .. }) => apply_tls(tls, info),
        _ => Ok(info),
    }
}

fn with_static_credentials(info: ConnectionInfo, credentials: &Credentials) -> ConnectionInfo {
    let Credentials::Static { username, password } = credentials else {
        return info;
    };
    let settings = info.redis_settings().clone();
    let settings = match (username, settings.username()) {
        (Some(username), None) => settings.set_username(username.expose()),
        _ => settings,
    };
    let settings: RedisConnectionInfo = match (password, settings.password()) {
        (Some(password), None) => settings.set_password(password.expose()),
        _ => settings,
    };
    info.set_redis_settings(settings)
}

#[cfg(test)]
mod tests {
    use litellm_redis_types::HostEndpoint;

    use super::*;

    #[test]
    fn a_unix_socket_path_replaces_host_port_and_tls_like_redis_py() {
        let endpoint = Endpoint::Host(HostEndpoint {
            host: "ignored".to_owned(),
            port: 1,
            tls: Some(TlsOptions::verified()),
            db: 3,
            unix_socket_path: Some("/tmp/redis.sock".into()),
        });

        let info = connection_info(&endpoint, &Credentials::NONE).unwrap();

        assert_eq!(*info.addr(), ConnectionAddr::Unix("/tmp/redis.sock".into()));
        assert_eq!(info.redis_settings().db(), 3);
    }
}
