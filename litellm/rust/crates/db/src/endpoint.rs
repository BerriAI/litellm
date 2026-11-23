use std::{
    path::PathBuf,
    time::{Duration, SystemTime},
};

use litellm_secret_value::SecretValue;
use serde::Deserialize;
use sqlx::{
    PgPool,
    postgres::{PgConnectOptions, PgPoolOptions, PgSslMode},
};

use crate::Error;

const WAIT_FOREVER: Duration = Duration::from_secs(100 * 365 * 24 * 60 * 60);

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct Endpoint {
    pub host: String,
    pub port: u16,
    pub socket: Option<PathBuf>,
    pub username: String,
    pub password: Option<SecretValue>,
    pub database: String,
    pub ssl_mode: SslMode,
    pub ssl_root_cert: Option<PathBuf>,
    pub ssl_client: Option<ClientIdentity>,
    pub application_name: Option<String>,
    pub options: Vec<(String, String)>,
    pub statement_cache_capacity: usize,
    pub pool: PoolLimits,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum SslMode {
    Disable,
    Prefer,
    Require,
    VerifyCa,
    VerifyFull,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct ClientIdentity {
    pub cert_pem: String,
    pub key_pem: SecretValue,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq)]
pub struct PoolLimits {
    pub max_connections: u32,
    pub acquire_timeout_seconds: Option<f64>,
    pub idle_timeout_seconds: Option<f64>,
    pub socket_timeout_seconds: Option<f64>,
}

impl From<SslMode> for PgSslMode {
    fn from(mode: SslMode) -> Self {
        match mode {
            SslMode::Disable => Self::Disable,
            SslMode::Prefer => Self::Prefer,
            SslMode::Require => Self::Require,
            SslMode::VerifyCa => Self::VerifyCa,
            SslMode::VerifyFull => Self::VerifyFull,
        }
    }
}

impl Endpoint {
    pub fn with_password(&self, password: SecretValue) -> Self {
        Self {
            password: Some(password),
            ..self.clone()
        }
    }

    pub fn connect_options(&self) -> PgConnectOptions {
        let options = PgConnectOptions::new_without_pgpass()
            .host(&self.host)
            .port(self.port)
            .username(&self.username)
            .database(&self.database)
            .ssl_mode(self.ssl_mode.into())
            .options(self.options.iter().map(|(name, value)| (name, value)))
            .statement_cache_capacity(self.statement_cache_capacity);
        let options = match &self.socket {
            Some(directory) => options.socket(directory),
            None => options,
        };
        let options = match &self.password {
            Some(password) => options.password(password.expose()),
            None => options,
        };
        let options = match &self.application_name {
            Some(name) => options.application_name(name),
            None => options,
        };
        let options = match &self.ssl_client {
            Some(identity) => options
                .ssl_client_cert_from_pem(&identity.cert_pem)
                .ssl_client_key_from_pem(identity.key_pem.expose()),
            None => pgsslkey_as_pkcs8(options),
        };
        let options = match &self.ssl_root_cert {
            Some(cert) => options.ssl_root_cert(cert),
            None => options,
        };
        match self.ssl_mode {
            SslMode::Prefer | SslMode::Require => unchecked_root_cert(options),
            SslMode::VerifyCa | SslMode::VerifyFull if self.root_cert_holds_no_pem() => {
                unchecked_root_cert(options)
            }
            SslMode::Disable | SslMode::VerifyCa | SslMode::VerifyFull => options,
        }
    }

    /// The root certificate sqlx reads: the one the URL names, otherwise `PGSSLROOTCERT`.
    #[cfg(feature = "native-tls")]
    fn root_cert(&self) -> Option<Input> {
        match &self.ssl_root_cert {
            Some(path) => Some(Input::File(path.clone())),
            None => Input::from_env("PGSSLROOTCERT"),
        }
    }

    /// Whether the root certificate sqlx reads holds no PEM certificate, as a DER file does. rustls adds the PEM
    /// certificates it finds to the system's roots, so with none it checks the server against the system's roots
    /// alone, while native-tls fails every connection on it. A file that cannot be read fails both, so it counts as
    /// holding one.
    #[cfg(feature = "native-tls")]
    fn root_cert_holds_no_pem(&self) -> bool {
        const PEM_CERTIFICATE: &[u8] = b"-----BEGIN CERTIFICATE-----";
        self.root_cert()
            .and_then(|input| input.data())
            .is_some_and(|data| {
                !data
                    .windows(PEM_CERTIFICATE.len())
                    .any(|window| window == PEM_CERTIFICATE)
            })
    }

    #[cfg(not(feature = "native-tls"))]
    fn root_cert_holds_no_pem(&self) -> bool {
        false
    }

    pub fn pool_options(&self) -> Result<PgPoolOptions, Error> {
        let limits = self.pool;
        Ok(PgPoolOptions::new()
            .min_connections(0)
            .max_connections(limits.max_connections)
            .acquire_timeout(seconds(limits.acquire_timeout_seconds)?.unwrap_or(WAIT_FOREVER))
            .idle_timeout(seconds(limits.idle_timeout_seconds)?))
    }

    pub fn pool(&self) -> Result<PgPool, Error> {
        Ok(self
            .pool_options()?
            .connect_lazy_with(self.connect_options()))
    }

    pub fn names_statements(&self) -> bool {
        self.statement_cache_capacity > 0
    }

    pub fn socket_timeout(&self) -> Result<Option<Duration>, Error> {
        seconds(self.pool.socket_timeout_seconds)
    }
}

/// A certificate or key setting, read the way sqlx reads it: the value itself when it looks like PEM, otherwise the
/// path of a file.
#[cfg(feature = "native-tls")]
enum Input {
    Inline(Vec<u8>),
    File(PathBuf),
}

#[cfg(feature = "native-tls")]
impl Input {
    fn from_env(name: &str) -> Option<Self> {
        let value = std::env::var(name).ok()?;
        let trimmed = value.trim();
        Some(
            if trimmed.starts_with("-----BEGIN") && trimmed.ends_with("-----") {
                Self::Inline(value.into_bytes())
            } else {
                Self::File(PathBuf::from(value))
            },
        )
    }

    fn data(&self) -> Option<Vec<u8>> {
        match self {
            Self::Inline(data) => Some(data.clone()),
            Self::File(path) => std::fs::read(path).ok(),
        }
    }

    fn file(&self) -> Option<&std::path::Path> {
        match self {
            Self::Inline(_) => None,
            Self::File(path) => Some(path),
        }
    }
}

/// The `PGSSLKEY` key sqlx reads, handed to it as PKCS#8 PEM when native-tls cannot load it as it is: native-tls
/// only loads a key that starts with a PKCS#8 PEM block, while libpq and rustls also load PKCS#1 and SEC1 keys. The
/// `PGSSLCERT` certificate is read at the same moment, so the two always match; the pool builds its options again
/// when either file changes. A key native-tls loads, one OpenSSL cannot read, and one that needs a password are left
/// for sqlx to read on each connection: OpenSSL gets an empty password instead of asking for it on the terminal.
#[cfg(feature = "native-tls")]
fn pgsslkey_as_pkcs8(options: PgConnectOptions) -> PgConnectOptions {
    let read = |name: &str| Input::from_env(name).and_then(|input| input.data());
    match read("PGSSLKEY").and_then(|pem| pkcs8_pem(&pem)) {
        Some(key) => match read("PGSSLCERT") {
            Some(cert) => options
                .ssl_client_cert_from_pem(cert)
                .ssl_client_key_from_pem(key),
            None => options.ssl_client_key_from_pem(key),
        },
        None => options,
    }
}

#[cfg(not(feature = "native-tls"))]
fn pgsslkey_as_pkcs8(options: PgConnectOptions) -> PgConnectOptions {
    options
}

/// The `PGSSLCERT`, `PGSSLKEY` and root certificate files as they are now, or that they are missing, so a pool can
/// tell when a file its options were built from changed. Without native-tls the options never depend on the files'
/// contents, since sqlx reads them on each connection, so there is nothing to follow.
#[derive(PartialEq)]
pub(crate) struct ClientFiles([Option<Stamp>; 3]);

impl ClientFiles {
    #[cfg(feature = "native-tls")]
    pub(crate) fn of(endpoint: &Endpoint) -> Self {
        let stamp = |input: Option<Input>| Some(stamp(&std::fs::metadata(input?.file()?).ok()?));
        Self([
            stamp(Input::from_env("PGSSLCERT")),
            stamp(Input::from_env("PGSSLKEY")),
            stamp(endpoint.root_cert()),
        ])
    }

    #[cfg(not(feature = "native-tls"))]
    pub(crate) fn of(_: &Endpoint) -> Self {
        Self([None, None, None])
    }
}

/// What tells that a file changed. On Unix that is its device and inode, which change when a new file is renamed
/// over it, and its status change time, which every write and every change to its timestamps moves, so a
/// replacement that keeps the old modified time and size is still seen.
#[cfg(unix)]
type Stamp = (u64, u64, i64, i64, Option<SystemTime>, u64);

#[cfg(all(unix, feature = "native-tls"))]
fn stamp(metadata: &std::fs::Metadata) -> Stamp {
    use std::os::unix::fs::MetadataExt;
    (
        metadata.dev(),
        metadata.ino(),
        metadata.ctime(),
        metadata.ctime_nsec(),
        metadata.modified().ok(),
        metadata.len(),
    )
}

#[cfg(not(unix))]
type Stamp = (Option<SystemTime>, u64);

#[cfg(all(not(unix), feature = "native-tls"))]
fn stamp(metadata: &std::fs::Metadata) -> Stamp {
    (metadata.modified().ok(), metadata.len())
}

#[cfg(feature = "native-tls")]
fn pkcs8_pem(pem: &[u8]) -> Option<Vec<u8>> {
    if pem.starts_with(b"-----BEGIN PRIVATE KEY-----") {
        return None;
    }
    openssl::pkey::PKey::private_key_from_pem_passphrase(pem, b"")
        .and_then(|key| key.private_key_to_pem_pkcs8())
        .ok()
}

/// sqlx's native-tls handshake loads the root certificate even in the modes that do not check the server's
/// certificate, so a `PGSSLROOTCERT` file that is missing or not PEM fails every connection, where rustls never reads
/// it. In the modes that check it, a file with no PEM certificate fails every connection too, where rustls checks
/// against the system's roots alone. sqlx cannot drop the file it read from the environment, so in those cases it
/// gets a throwaway self-signed certificate in its place, which nothing chains to.
#[cfg(feature = "native-tls")]
fn unchecked_root_cert(options: PgConnectOptions) -> PgConnectOptions {
    static CERTIFICATE: std::sync::OnceLock<Option<Vec<u8>>> = std::sync::OnceLock::new();
    match CERTIFICATE.get_or_init(|| self_signed_pem().ok()) {
        Some(pem) => options.ssl_root_cert_from_pem(pem.clone()),
        None => options,
    }
}

#[cfg(not(feature = "native-tls"))]
fn unchecked_root_cert(options: PgConnectOptions) -> PgConnectOptions {
    options
}

#[cfg(feature = "native-tls")]
fn self_signed_pem() -> Result<Vec<u8>, openssl::error::ErrorStack> {
    use openssl::{
        asn1::Asn1Time,
        ec::{EcGroup, EcKey},
        hash::MessageDigest,
        nid::Nid,
        pkey::PKey,
        x509::X509,
    };

    let p256 = EcGroup::from_curve_name(Nid::X9_62_PRIME256V1)?;
    let key = PKey::from_ec_key(EcKey::generate(&p256)?)?;
    let mut certificate = X509::builder()?;
    certificate.set_pubkey(&key)?;
    certificate.set_not_before(&*Asn1Time::days_from_now(0)?)?;
    certificate.set_not_after(&*Asn1Time::days_from_now(1)?)?;
    certificate.sign(&key, MessageDigest::sha256())?;
    certificate.build().to_pem()
}

fn seconds(value: Option<f64>) -> Result<Option<Duration>, Error> {
    value
        .map(Duration::try_from_secs_f64)
        .transpose()
        .map_err(Error::from)
}

#[cfg(test)]
mod tests {
    use std::{error::Error as StdError, path::PathBuf, time::Duration};

    use litellm_secret_value::SecretValue;
    use rstest::rstest;
    use sqlx::postgres::PgSslMode;

    use super::{Endpoint, PoolLimits, SslMode, WAIT_FOREVER};
    use crate::{Error, test_support::database_endpoint};

    const LIMITS: PoolLimits = PoolLimits {
        max_connections: 7,
        acquire_timeout_seconds: Some(2.5),
        idle_timeout_seconds: Some(60.0),
        socket_timeout_seconds: None,
    };

    fn endpoint() -> Endpoint {
        Endpoint {
            host: "db.internal".to_owned(),
            port: 6543,
            socket: None,
            username: "litellm".to_owned(),
            password: Some(SecretValue::new("first")),
            database: "proxy".to_owned(),
            ssl_mode: SslMode::VerifyFull,
            ssl_root_cert: None,
            ssl_client: None,
            application_name: Some("litellm-native".to_owned()),
            options: vec![("statement_timeout".to_owned(), "30000".to_owned())],
            statement_cache_capacity: 100,
            pool: LIMITS,
        }
    }

    #[test]
    fn maps_endpoint_to_connect_options() {
        let options = endpoint().connect_options();

        assert_eq!(options.get_host(), "db.internal");
        assert_eq!(options.get_port(), 6543);
        assert_eq!(options.get_socket(), None);
        assert_eq!(options.get_username(), "litellm");
        assert_eq!(options.get_database(), Some("proxy"));
        assert_eq!(options.get_application_name(), Some("litellm-native"));
        assert!(matches!(options.get_ssl_mode(), PgSslMode::VerifyFull));
        assert_eq!(options.get_options(), Some("-c statement_timeout=30000"));
    }

    #[test]
    fn connects_through_a_unix_socket_directory() {
        let options = Endpoint {
            socket: Some(PathBuf::from("/cloudsql/project:region:instance")),
            ..endpoint()
        }
        .connect_options();

        assert_eq!(
            options.get_socket(),
            Some(&PathBuf::from("/cloudsql/project:region:instance"))
        );
    }

    #[tokio::test]
    async fn applies_pool_limits() -> Result<(), Error> {
        let pool = endpoint().pool()?;

        assert_eq!(pool.options().get_max_connections(), 7);
        assert_eq!(pool.options().get_min_connections(), 0);
        assert_eq!(
            pool.options().get_acquire_timeout(),
            Duration::from_millis(2500)
        );
        assert_eq!(
            pool.options().get_idle_timeout(),
            Some(Duration::from_secs(60))
        );
        assert_eq!(pool.size(), 0);
        Ok(())
    }

    #[tokio::test]
    async fn missing_acquire_timeout_waits_forever() -> Result<(), Error> {
        let pool = Endpoint {
            pool: PoolLimits {
                acquire_timeout_seconds: None,
                ..LIMITS
            },
            ..endpoint()
        }
        .pool()?;

        assert_eq!(pool.options().get_acquire_timeout(), WAIT_FOREVER);
        Ok(())
    }

    #[rstest]
    #[case::negative(-1.0)]
    #[case::not_a_number(f64::NAN)]
    #[tokio::test]
    async fn rejects_invalid_durations(#[case] seconds: f64) {
        let error = Endpoint {
            pool: PoolLimits {
                acquire_timeout_seconds: Some(seconds),
                ..LIMITS
            },
            ..endpoint()
        }
        .pool()
        .expect_err("an invalid duration must not build a pool");

        assert!(matches!(error, Error::InvalidDuration(_)));
    }

    #[test]
    fn redacts_password() {
        assert!(!format!("{:?}", endpoint()).contains("first"));
    }

    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn waits_past_an_exhausted_pool_without_a_timeout() -> Result<(), Box<dyn StdError>> {
        let endpoint = database_endpoint()?;
        let pool = Endpoint {
            pool: PoolLimits {
                max_connections: 1,
                acquire_timeout_seconds: None,
                ..endpoint.pool
            },
            ..endpoint
        }
        .pool()?;
        let held = pool.acquire().await?;
        let waiter = tokio::spawn({
            let pool = pool.clone();
            async move { pool.acquire().await.map(drop) }
        });

        tokio::time::sleep(Duration::from_millis(300)).await;
        assert!(!waiter.is_finished());
        drop(held);

        waiter.await??;
        Ok(())
    }

    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn applies_session_options() -> Result<(), Box<dyn StdError>> {
        let pool = Endpoint {
            options: vec![
                ("statement_timeout".to_owned(), "1234".to_owned()),
                (
                    "search_path".to_owned(),
                    "\"Other Schema\", public".to_owned(),
                ),
            ],
            application_name: Some("litellm-native".to_owned()),
            ..database_endpoint()?
        }
        .pool()?;

        #[expect(
            clippy::disallowed_methods,
            reason = "Session settings are not part of the compile-checked schema"
        )]
        let (statement_timeout, search_path, application_name): (String, String, String) =
            sqlx::query_as(
                "SELECT current_setting('statement_timeout'), current_setting('search_path'), \
                 current_setting('application_name')",
            )
            .fetch_one(&pool)
            .await?;

        assert_eq!(statement_timeout, "1234ms");
        assert_eq!(search_path, "\"Other Schema\", public");
        assert_eq!(application_name, "litellm-native");
        Ok(())
    }

    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn rotated_password_reaches_new_connections() -> Result<(), Box<dyn StdError>> {
        let endpoint = database_endpoint()?;
        let pool = endpoint
            .with_password(SecretValue::new("not-the-password"))
            .pool()?;
        #[expect(clippy::disallowed_methods, reason = "A probe query needs no schema")]
        let probe = || sqlx::query("SELECT 1");
        let stale = probe().execute(&pool).await;

        pool.set_connect_options(endpoint.connect_options());
        probe().execute(&pool).await?;

        assert!(stale.is_err());
        Ok(())
    }

    #[rstest]
    #[case::prefer(SslMode::Prefer, false)]
    #[case::require(SslMode::Require, false)]
    #[case::verify_full(SslMode::VerifyFull, true)]
    #[tokio::test]
    async fn reads_the_root_cert_only_when_the_server_certificate_is_checked(
        #[case] ssl_mode: SslMode,
        #[case] checked: bool,
    ) -> Result<(), Box<dyn StdError>> {
        let server = std::net::TcpListener::bind("127.0.0.1:0")?;
        let port = server.local_addr()?.port();
        let accepts_tls_then_hangs_up = std::thread::spawn(move || -> std::io::Result<()> {
            let (mut socket, _) = server.accept()?;
            std::io::Read::read_exact(&mut socket, &mut [0; 8])?;
            std::io::Write::write_all(&mut socket, b"S")
        });
        let options = Endpoint {
            host: "127.0.0.1".to_owned(),
            port,
            ssl_mode,
            ssl_root_cert: Some(PathBuf::from("/nonexistent/root.crt")),
            ..endpoint()
        }
        .connect_options();

        let error = <sqlx::PgConnection as sqlx::Connection>::connect_with(&options)
            .await
            .expect_err("the server hangs up before the handshake");
        accepts_tls_then_hangs_up
            .join()
            .map_err(|_| "the server thread panicked")??;

        let root_cert_missing = matches!(
            &error,
            sqlx::Error::Io(io) if io.kind() == std::io::ErrorKind::NotFound
        );
        assert_eq!(root_cert_missing, checked, "{error}");
        Ok(())
    }

    #[cfg(feature = "native-tls")]
    mod native_tls_client_key {
        use std::error::Error as StdError;

        use openssl::{
            ec::{EcGroup, EcKey},
            error::ErrorStack,
            nid::Nid,
            pkey::PKey,
            rsa::Rsa,
            symm::Cipher,
        };
        use rstest::rstest;

        use super::super::pkcs8_pem;

        fn pkcs1_rsa() -> Result<Vec<u8>, ErrorStack> {
            Rsa::generate(2048)?.private_key_to_pem()
        }

        fn sec1_ec() -> Result<Vec<u8>, ErrorStack> {
            let p256 = EcGroup::from_curve_name(Nid::X9_62_PRIME256V1)?;
            EcKey::generate(&p256)?.private_key_to_pem()
        }

        fn pkcs12_export() -> Result<Vec<u8>, ErrorStack> {
            let pkcs8 = PKey::from_rsa(Rsa::generate(2048)?)?.private_key_to_pem_pkcs8()?;
            Ok([b"Bag Attributes\n    localKeyID: 01\n".as_slice(), &pkcs8].concat())
        }

        #[rstest]
        #[case::pkcs1_rsa(pkcs1_rsa)]
        #[case::sec1_ec(sec1_ec)]
        #[case::pkcs12_export_with_bag_attributes(pkcs12_export)]
        fn hands_the_same_key_to_native_tls_as_pkcs8(
            #[case] pem: fn() -> Result<Vec<u8>, ErrorStack>,
        ) -> Result<(), Box<dyn StdError>> {
            let original = pem()?;

            let pkcs8 = pkcs8_pem(&original).ok_or("OpenSSL reads the key")?;

            assert!(pkcs8.starts_with(b"-----BEGIN PRIVATE KEY-----\n"));
            assert_eq!(
                PKey::private_key_from_pem(&pkcs8)?.public_key_to_der()?,
                PKey::private_key_from_pem(&original)?.public_key_to_der()?
            );
            Ok(())
        }

        #[rstest]
        fn leaves_a_key_native_tls_loads_to_sqlx() -> Result<(), Box<dyn StdError>> {
            let pkcs8 = PKey::from_rsa(Rsa::generate(2048)?)?.private_key_to_pem_pkcs8()?;

            assert_eq!(pkcs8_pem(&pkcs8), None);
            Ok(())
        }

        #[rstest]
        fn leaves_a_file_openssl_cannot_read_to_sqlx() {
            assert_eq!(pkcs8_pem(b"not a key"), None);
        }

        #[rstest]
        fn leaves_a_key_that_needs_a_password_to_sqlx() -> Result<(), Box<dyn StdError>> {
            let encrypted = PKey::from_rsa(Rsa::generate(2048)?)?
                .private_key_to_pem_pkcs8_passphrase(Cipher::aes_256_cbc(), b"secret")?;

            assert_eq!(pkcs8_pem(&encrypted), None);
            Ok(())
        }
    }
}
