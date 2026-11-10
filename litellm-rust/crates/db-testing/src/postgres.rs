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
        let url = format!(
            "postgres://postgres:postgres@{}:{}/postgres",
            container.get_host().await?,
            container.get_host_port_ipv4(5432).await?,
        );
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
