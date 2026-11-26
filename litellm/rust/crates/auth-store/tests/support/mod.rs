//! What the database tests share: the repositories' handle on a migrated Postgres, fixture
//! statements, and a record as the bytes the auth cache holds for it.

// Shared across test targets; each target uses a different subset.
#![allow(dead_code)]

use std::{str::FromStr, sync::Arc};

use litellm_db::{Endpoint, Observer, PoolLimits, Pools, QueryEvent, SslMode};
use litellm_db_testing::MigratedPostgres;
use litellm_redis_store::IntoValue;
use litellm_repositories::Database;
use litellm_secret_value::SecretValue;
use sqlx::{PgPool, postgres::PgConnectOptions};

struct NoEvents;

impl Observer for NoEvents {
    fn record(&self, _event: QueryEvent) {}
}

pub fn database(postgres: &MigratedPostgres) -> Database {
    let options = PgConnectOptions::from_str(postgres.url()).unwrap();
    let endpoint = Endpoint {
        host: options.get_host().to_owned(),
        port: options.get_port(),
        socket: None,
        username: "postgres".to_owned(),
        password: Some(SecretValue::new("postgres")),
        database: "postgres".to_owned(),
        ssl_mode: SslMode::Disable,
        ssl_root_cert: None,
        ssl_client: None,
        application_name: None,
        options: Vec::new(),
        statement_cache_capacity: 100,
        pool: PoolLimits {
            max_connections: 2,
            acquire_timeout_seconds: Some(10.0),
            idle_timeout_seconds: None,
            socket_timeout_seconds: None,
        },
    };
    Database::new(
        Pools::new(&endpoint, None).unwrap(),
        Arc::new(NoEvents),
        None,
        endpoint.names_statements(),
    )
}

pub async fn execute(pool: &PgPool, sql: &'static str) {
    #[expect(
        clippy::disallowed_methods,
        reason = "Fixture statements are outside the compile-checked query set"
    )]
    let statement = sqlx::query(sql);
    statement.execute(pool).await.unwrap();
}

pub fn dumped(value: impl IntoValue) -> String {
    litellm_python_compat::json::dumps(&value.into_value()).unwrap()
}
