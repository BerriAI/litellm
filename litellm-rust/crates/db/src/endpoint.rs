use std::{path::PathBuf, time::Duration};

use litellm_secret_value::SecretValue;
use serde::Deserialize;
use sqlx::{
    PgPool,
    postgres::{PgConnectOptions, PgPoolOptions, PgSslMode},
};

use crate::Error;

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct Endpoint {
    pub host: String,
    pub port: u16,
    pub username: String,
    pub password: Option<SecretValue>,
    pub database: String,
    pub ssl_mode: SslMode,
    pub ssl_root_cert: Option<PathBuf>,
    pub options: Vec<(String, String)>,
    pub statement_cache: bool,
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

#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq)]
pub struct PoolLimits {
    pub max_connections: Option<u32>,
    pub acquire_timeout_seconds: Option<f64>,
    pub idle_timeout_seconds: Option<f64>,
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
            .options(self.options.iter().map(|(name, value)| (name, value)));
        let options = if self.statement_cache {
            options
        } else {
            options.statement_cache_capacity(0)
        };
        let options = match &self.password {
            Some(password) => options.password(password.expose()),
            None => options,
        };
        match &self.ssl_root_cert {
            Some(cert) => options.ssl_root_cert(cert),
            None => options,
        }
    }

    pub fn pool(&self) -> Result<PgPool, Error> {
        let limits = self.pool;
        let options = PgPoolOptions::new()
            .min_connections(0)
            .idle_timeout(seconds(limits.idle_timeout_seconds)?);
        let options = match limits.max_connections {
            Some(max) => options.max_connections(max),
            None => options,
        };
        let options = match seconds(limits.acquire_timeout_seconds)? {
            Some(timeout) => options.acquire_timeout(timeout),
            None => options,
        };
        Ok(options.connect_lazy_with(self.connect_options()))
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
    use std::{error::Error as StdError, str::FromStr, time::Duration};

    use litellm_secret_value::SecretValue;
    use rstest::rstest;
    use sqlx::postgres::{PgConnectOptions, PgSslMode};

    use super::{Endpoint, PoolLimits, SslMode};
    use crate::Error;

    fn endpoint() -> Endpoint {
        Endpoint {
            host: "db.internal".to_owned(),
            port: 6543,
            username: "litellm".to_owned(),
            password: Some(SecretValue::new("first")),
            database: "proxy".to_owned(),
            ssl_mode: SslMode::VerifyFull,
            ssl_root_cert: None,
            options: vec![("statement_timeout".to_owned(), "30000".to_owned())],
            statement_cache: true,
            pool: PoolLimits::default(),
        }
    }

    fn database_endpoint() -> Result<Endpoint, Box<dyn StdError>> {
        let url = PgConnectOptions::from_str(&std::env::var("DATABASE_URL")?)?;
        let password = url::Url::parse(&std::env::var("DATABASE_URL")?)?
            .password()
            .map(SecretValue::new);
        Ok(Endpoint {
            host: url.get_host().to_owned(),
            port: url.get_port(),
            username: url.get_username().to_owned(),
            password,
            database: url.get_database().unwrap_or("postgres").to_owned(),
            ssl_mode: SslMode::Prefer,
            ssl_root_cert: None,
            options: Vec::new(),
            statement_cache: true,
            pool: PoolLimits::default(),
        })
    }

    #[test]
    fn maps_endpoint_to_connect_options() {
        let options = endpoint().connect_options();

        assert_eq!(options.get_host(), "db.internal");
        assert_eq!(options.get_port(), 6543);
        assert_eq!(options.get_username(), "litellm");
        assert_eq!(options.get_database(), Some("proxy"));
        assert!(matches!(options.get_ssl_mode(), PgSslMode::VerifyFull));
        assert_eq!(options.get_options(), Some("-c statement_timeout=30000"));
    }

    #[tokio::test]
    async fn applies_pool_limits() -> Result<(), Error> {
        let pool = Endpoint {
            pool: PoolLimits {
                max_connections: Some(7),
                acquire_timeout_seconds: Some(2.5),
                idle_timeout_seconds: Some(60.0),
            },
            ..endpoint()
        }
        .pool()?;

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

    #[rstest]
    #[case::negative(-1.0)]
    #[case::not_a_number(f64::NAN)]
    #[tokio::test]
    async fn rejects_invalid_durations(#[case] seconds: f64) {
        let error = Endpoint {
            pool: PoolLimits {
                acquire_timeout_seconds: Some(seconds),
                ..PoolLimits::default()
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
    async fn applies_session_options() -> Result<(), Box<dyn StdError>> {
        let pool = Endpoint {
            options: vec![
                ("statement_timeout".to_owned(), "1234".to_owned()),
                (
                    "search_path".to_owned(),
                    "\"Other Schema\", public".to_owned(),
                ),
            ],
            ..database_endpoint()?
        }
        .pool()?;

        #[expect(
            clippy::disallowed_methods,
            reason = "Session settings are not part of the compile-checked schema"
        )]
        let (statement_timeout, search_path): (String, String) = sqlx::query_as(
            "SELECT current_setting('statement_timeout'), current_setting('search_path')",
        )
        .fetch_one(&pool)
        .await?;

        assert_eq!(statement_timeout, "1234ms");
        assert_eq!(search_path, "\"Other Schema\", public");
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
