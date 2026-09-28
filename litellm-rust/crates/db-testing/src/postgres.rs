use std::path::Path;

use sqlx::{PgPool, migrate::Migrator};
use testcontainers_modules::{
    postgres::Postgres,
    testcontainers::{ContainerAsync, ImageExt, runners::AsyncRunner},
};

use crate::{Error, PRISMA_MIGRATIONS_DIR, prisma_migrations};

const POSTGRES_TAG: &str =
    "16@sha256:e17e86066e5ef83e0952a9347f5c792b7ece00972e2aa787a6986f471b3dd3d5";

pub struct MigratedPostgres {
    _container: ContainerAsync<Postgres>,
    url: String,
    pool: PgPool,
}

impl MigratedPostgres {
    pub async fn start() -> Result<Self, Error> {
        let container = Postgres::default().with_tag(POSTGRES_TAG).start().await?;
        let url = connection_url(
            &container.get_host().await?.to_string(),
            container.get_host_port_ipv4(5432).await?,
            "postgres",
            "postgres",
            "postgres",
        )?;
        let pool = PgPool::connect(&url).await?;
        Migrator::with_migrations(prisma_migrations(Path::new(PRISMA_MIGRATIONS_DIR))?)
            .run(&pool)
            .await?;
        Ok(Self {
            _container: container,
            url,
            pool,
        })
    }

    pub fn url(&self) -> &str {
        &self.url
    }

    pub fn pool(&self) -> &PgPool {
        &self.pool
    }
}

fn connection_url(
    host: &str,
    port: u16,
    username: &str,
    password: &str,
    database: &str,
) -> Result<String, Error> {
    let mut url = url::Url::parse("postgres://localhost").expect("static URL");
    match host.parse::<std::net::IpAddr>() {
        Ok(host) => url
            .set_ip_host(host)
            .map_err(|()| url::ParseError::InvalidIpv6Address)?,
        Err(_) => url.set_host(Some(host))?,
    }
    url.set_port(Some(port))
        .map_err(|()| url::ParseError::InvalidPort)?;
    url.set_username(username)
        .map_err(|()| url::ParseError::EmptyHost)?;
    url.set_password(Some(password))
        .map_err(|()| url::ParseError::EmptyHost)?;
    url.path_segments_mut()
        .map_err(|()| url::ParseError::RelativeUrlWithoutBase)?
        .clear()
        .push(database);
    Ok(url.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[rstest::rstest]
    #[case::ipv4("127.0.0.1", "127.0.0.1")]
    #[case::ipv6("::1", "[::1]")]
    #[case::dns("postgres.test", "postgres.test")]
    fn connection_components_are_encoded(#[case] host: &str, #[case] expected_host: &str) {
        let value = connection_url(host, 15432, "user@tenant", "p/a?#%", "db/name").unwrap();
        let parsed = url::Url::parse(&value).unwrap();
        assert_eq!(parsed.host_str(), Some(expected_host));
        assert_eq!(parsed.port(), Some(15432));
        assert_eq!(parsed.username(), "user%40tenant");
        assert_eq!(parsed.password(), Some("p%2Fa%3F%23%"));
        assert_eq!(parsed.path(), "/db%2Fname");
        assert!(parsed.query().is_none());
    }
}
