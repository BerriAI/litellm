use std::{path::PathBuf, time::Duration};

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
            None => options,
        };
        match &self.ssl_root_cert {
            Some(cert) => options.ssl_root_cert(cert),
            None => options,
        }
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
}
