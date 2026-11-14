use std::{fmt, path::PathBuf, pin::Pin, sync::Arc, time::Duration};

use futures_util::{Stream, StreamExt};

use redis::{
    BasicAuth, ClientTlsConfig, ConnectionAddr, ConnectionInfo, ErrorKind, IntoConnectionInfo,
    ProtocolVersion, RedisConnectionInfo, RedisError, RedisResult, StreamingCredentialsProvider,
    TlsCertificates, TlsMode, io::tcp::TcpSettings,
};

use litellm_circuit_breaker::Breaker;
use litellm_secret_value::SecretValue;

use pkcs8::EncryptedPrivateKeyInfo;

use crate::{ConnectError, KeyError, Namespace, RedisEvents};
use litellm_redis_types::{Cluster, RedisUrl, Target, TlsOptions};

// litellm/constants.py REDIS_SOCKET_TIMEOUT, which _coerce_redis_kwargs_types sets when socket_timeout is unset
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(5);
// below the idle timeouts of common cloud load balancers (AWS NLB 350s, Azure Load Balancer 240s), which the
// OS default of two hours never probes within
const KEEPALIVE_IDLE: Duration = Duration::from_secs(60);

pub struct RedisConfig {
    pub target: Target,
    pub credentials: Credentials,
    pub namespace: Namespace,
    pub timeouts: Timeouts,
    pub breaker: Option<Breaker>,
    pub events: Option<Arc<dyn RedisEvents>>,
    /// Ping every node before the next command once the client sat idle this long, like
    /// redis-py's `health_check_interval`, so a connection the server dropped silently is
    /// replaced before a real command pays for it.
    pub health_check: Option<Duration>,
    /// Probe a connection with TCP keepalive once it sat idle, like redis-py's `socket_keepalive`.
    pub keepalive: bool,
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
            client_key: client_key(
                keyfile.as_ref().unwrap_or(certfile),
                tls.key_password.as_ref(),
            )?,
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

/// The key as unencrypted PEM, decrypting a PKCS#8 `ENCRYPTED PRIVATE KEY` with `ssl_password` the way
/// `load_cert_chain(password=...)` does, since rustls only reads unencrypted keys. Like OpenSSL, a password next to
/// an unencrypted key is unused
fn client_key(path: &PathBuf, password: Option<&SecretValue>) -> Result<Vec<u8>, ConnectError> {
    let contents = read(path)?;
    let key_error = |reason| ConnectError::TlsKey {
        path: path.display().to_string(),
        reason,
    };
    let Ok(sections) = pem::parse_many(&contents) else {
        return Ok(contents);
    };
    let encrypted = sections
        .iter()
        .find(|section| section.tag() == "ENCRYPTED PRIVATE KEY");
    match (encrypted, password) {
        (Some(_), None) => Err(key_error(KeyError::MissingPassword)),
        (Some(section), Some(password)) => {
            let decrypted = EncryptedPrivateKeyInfo::try_from(section.contents())
                .and_then(|info| info.decrypt(password.expose()))
                .map_err(|source| key_error(KeyError::Decrypt(source)))?;
            Ok(pem::encode(&pem::Pem::new("PRIVATE KEY", decrypted.as_bytes())).into_bytes())
        }
        (None, _) if sections.iter().any(is_legacy_encrypted) => {
            Err(key_error(KeyError::LegacyEncryption))
        }
        (None, _) => Ok(contents),
    }
}

fn is_legacy_encrypted(section: &pem::Pem) -> bool {
    section
        .headers()
        .get("Proc-Type")
        .is_some_and(|proc_type| proc_type.contains("ENCRYPTED"))
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

pub(crate) fn cluster_addrs(cluster: &Cluster) -> impl Iterator<Item = ConnectionAddr> + '_ {
    cluster
        .nodes
        .iter()
        .map(|node| addr(&node.host, node.port, cluster.tls.as_ref()))
}

pub(crate) fn tcp_settings(keepalive: bool) -> TcpSettings {
    let settings = TcpSettings::default().set_nodelay(true);
    match keepalive {
        true => settings.set_keepalive(socket2::TcpKeepalive::new().with_time(KEEPALIVE_IDLE)),
        false => settings,
    }
}

pub(crate) fn connection_info(
    target: &Target,
    credentials: &Credentials,
    keepalive: bool,
) -> Result<ConnectionInfo, ConnectError> {
    let (info, tls) = match target {
        Target::Url(url) => (url_info(url, credentials)?, url.tls.as_ref()),
        Target::Standalone(host) => {
            let info = addr(&host.host, host.port, host.tls.as_ref()).into_connection_info()?;
            let settings = info.redis_settings().clone().set_db(host.db);
            (info.set_redis_settings(settings), host.tls.as_ref())
        }
        Target::Unix(socket) => {
            let info = ConnectionAddr::Unix(socket.path.clone()).into_connection_info()?;
            let settings = info.redis_settings().clone().set_db(socket.db);
            (info.set_redis_settings(settings), None)
        }
        Target::Cluster(_) | Target::Sentinel(_) => {
            unreachable!("cluster and sentinel endpoints build their own connection info")
        }
    };
    let info = with_static_credentials(info, credentials).set_tcp_settings(tcp_settings(keepalive));
    match (tls, info.addr()) {
        (Some(tls), ConnectionAddr::TcpTls { .. }) => apply_tls(tls, info),
        _ => Ok(info),
    }
}

// parse_url reads ?db= over the path, and the url's userinfo over ?username= and ?password=. A credential
// provider supersedes both, as it does in redis-py. ?protocol=3 would switch redis-rs to RESP3, which no
// reply decode here expects, so the protocol is ignored as the resolver warns
fn url_info(url: &RedisUrl, credentials: &Credentials) -> Result<ConnectionInfo, ConnectError> {
    let info = url.url.expose().into_connection_info()?;
    let settings = info
        .redis_settings()
        .clone()
        .set_protocol(ProtocolVersion::RESP2);
    let settings = match url.db {
        Some(db) => settings.set_db(db),
        None => settings,
    };
    let info = info.set_redis_settings(settings);
    Ok(match credentials {
        Credentials::Static { .. } => with_static_credentials(
            info,
            &Credentials::Static {
                username: url.username.clone(),
                password: url.password.clone(),
            },
        ),
        Credentials::Provider(_) => info,
    })
}

fn with_static_credentials(info: ConnectionInfo, credentials: &Credentials) -> ConnectionInfo {
    let settings = static_credentials(info.redis_settings().clone(), credentials);
    info.set_redis_settings(settings)
}

fn static_credentials(
    settings: RedisConnectionInfo,
    credentials: &Credentials,
) -> RedisConnectionInfo {
    let Credentials::Static { username, password } = credentials else {
        return settings;
    };
    let settings = match (username, settings.username()) {
        (Some(username), None) => settings.set_username(username.expose()),
        _ => settings,
    };
    match (password, settings.password()) {
        (Some(password), None) => settings.set_password(password.expose()),
        _ => settings,
    }
}

/// `settings` with the credentials a connection opened now sends. A provider is asked once for its
/// current pair, which is when redis-py asks its `credential_provider`.
pub(crate) async fn authenticated(
    settings: RedisConnectionInfo,
    credentials: &Credentials,
) -> RedisResult<RedisConnectionInfo> {
    let Credentials::Provider(provider) = credentials else {
        return Ok(static_credentials(settings, credentials));
    };
    let current = provider.subscribe().next().await.ok_or_else(|| {
        RedisError::from((
            ErrorKind::AuthenticationFailed,
            "the Redis credential provider ended before giving credentials",
        ))
    })??;
    Ok(settings
        .set_username(current.username())
        .set_password(current.password()))
}

#[cfg(test)]
mod tests {
    use litellm_redis_types::{RedisUrl, UnixSocket};
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::host(Target::Standalone(crate::Standalone { host: "redis".into(), port: 6379, tls: None, db: 0 }))]
    #[case::url(Target::Url(RedisUrl {
        url: SecretValue::new("redis://cache:6379".to_owned()),
        db: None,
        username: None,
        password: None,
        tls: None,
    }))]
    fn keepalive_probes_an_idle_socket_only_when_enabled(#[case] target: Target) {
        let probed = |keepalive| {
            connection_info(&target, &Credentials::NONE, keepalive)
                .unwrap()
                .tcp_settings()
                .keepalive()
                .is_some()
        };

        assert!(probed(true));
        assert!(!probed(false));
    }

    #[test]
    fn a_unix_socket_replaces_host_port_and_tls_like_redis_py() {
        let target = Target::Unix(UnixSocket {
            path: "/tmp/redis.sock".into(),
            db: 3,
        });

        let info = connection_info(&target, &Credentials::NONE, true).unwrap();

        assert_eq!(*info.addr(), ConnectionAddr::Unix("/tmp/redis.sock".into()));
        assert_eq!(info.redis_settings().db(), 3);
    }

    #[rstest]
    #[case::the_query_db_wins_over_the_path("redis://cache:6379/1", Some(3), None, 3, None)]
    #[case::the_path_db_without_a_query_db("redis://cache:6379/1", None, None, 1, None)]
    #[case::the_query_password_fills_a_missing_userinfo(
        "redis://cache:6379",
        None,
        Some("query"),
        0,
        Some("query")
    )]
    #[case::the_userinfo_wins_over_the_query_password(
        "redis://:userinfo@cache:6379",
        None,
        Some("query"),
        0,
        Some("userinfo")
    )]
    fn a_url_reads_its_query_like_parse_url(
        #[case] url: &str,
        #[case] db: Option<i64>,
        #[case] password: Option<&str>,
        #[case] expected_db: i64,
        #[case] expected_password: Option<&str>,
    ) {
        let target = Target::Url(RedisUrl {
            url: SecretValue::new(url),
            tls: None,
            db,
            username: None,
            password: password.map(SecretValue::new),
        });
        let static_password = Credentials::Static {
            username: None,
            password: Some(SecretValue::new("static")),
        };

        let info = connection_info(&target, &static_password, true).unwrap();

        assert_eq!(info.redis_settings().db(), expected_db);
        assert_eq!(
            info.redis_settings().password(),
            expected_password.or(Some("static"))
        );
    }

    #[rstest]
    #[case::by_number("redis://cache:6379/?protocol=3")]
    #[case::by_name("redis://cache:6379/?protocol=resp3")]
    fn a_url_never_switches_the_protocol_from_resp2(#[case] url: &str) {
        let target = Target::Url(RedisUrl {
            url: SecretValue::new(url),
            tls: None,
            db: None,
            username: None,
            password: None,
        });

        let info = connection_info(&target, &Credentials::NONE, true).unwrap();

        assert_eq!(info.redis_settings().protocol(), ProtocolVersion::RESP2);
    }
}
