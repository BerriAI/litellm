use std::{str::FromStr, time::Duration};

use sqlx::{
    PgPool,
    postgres::{PgConnectOptions, PgPoolOptions},
};

use crate::Error;

/// Pool knobs named after the proxy's `general_settings.database_*` settings.
#[derive(Clone, Debug)]
pub struct PoolSettings {
    pub max_connections: u32,
    pub acquire_timeout: Duration,
    pub idle_timeout: Option<Duration>,
}

impl Default for PoolSettings {
    fn default() -> Self {
        Self {
            max_connections: 10,
            acquire_timeout: Duration::from_secs(60),
            idle_timeout: Some(Duration::from_secs(60)),
        }
    }
}

/// Builds a pool that connects on first use, so a host starts while the database is down.
pub fn connect_lazy(url: &str, settings: &PoolSettings) -> Result<PgPool, Error> {
    let options = PgConnectOptions::from_str(url).map_err(Error::Url)?;
    Ok(PgPoolOptions::new()
        .max_connections(settings.max_connections.max(1))
        .acquire_timeout(settings.acquire_timeout)
        .idle_timeout(settings.idle_timeout)
        .connect_lazy_with(options))
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::postgres("postgres://user:pass@localhost:5432/litellm")]
    #[case::postgresql("postgresql://user:pass@localhost/litellm")]
    #[tokio::test]
    async fn accepts_the_url_forms_the_proxy_uses(#[case] url: &str) {
        assert!(connect_lazy(url, &PoolSettings::default()).is_ok());
    }

    #[rstest]
    #[tokio::test]
    async fn rejects_a_malformed_url() {
        assert!(matches!(
            connect_lazy(
                "postgres://localhost:port/litellm",
                &PoolSettings::default()
            ),
            Err(Error::Url(_))
        ));
    }
}
