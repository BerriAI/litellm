//! The proxy's database as a gateway test holds it: a migrated Postgres with the keys the test
//! puts in it, read through the repositories the gateway reads through.

// Shared across test targets; each target uses a different subset.
#![allow(dead_code)]

use std::{str::FromStr, sync::Arc};

use litellm_db::{Call, Endpoint, Observer, PoolLimits, Pools, QueryEvent, SslMode};
use litellm_db_testing::MigratedPostgres;
use litellm_gateway_auth_core::Token;
use litellm_redis_store::auth_cache::{AuthCache, Settings};
use litellm_repositories::Database;
use litellm_secret_value::SecretValue;
use parking_lot::Mutex;
use serde_json::{Value, json};
use sqlx::postgres::PgConnectOptions;

/// The table columns of a key generated with no parameters, as `/key/generate` stores it.
const PLAIN: &str = include_str!("plain.json");

/// Every query that ran to its end, as its repository names it.
#[derive(Default)]
struct Queries(Mutex<Vec<Call>>);

impl Observer for Queries {
    fn record(&self, event: QueryEvent) {
        self.0.lock().push(event.call);
    }
}

pub struct Held {
    postgres: MigratedPostgres,
    queries: Arc<Queries>,
}

impl Held {
    /// A database that holds no key.
    pub async fn start() -> Self {
        Self {
            postgres: MigratedPostgres::start().await.unwrap(),
            queries: Arc::default(),
        }
    }

    /// A database that holds the key of `token` alone, a plain one with `columns` set on it.
    pub async fn holding(token: &str, columns: Value) -> Self {
        let held = Self::start().await;
        held.key(token, columns).await;
        held
    }

    /// Stores the key of `token`: a plain row under the token's hash, with `columns` set on it.
    pub async fn key(&self, token: &str, columns: Value) {
        let mut row: Value = serde_json::from_str(PLAIN).unwrap();
        let fields = row.as_object_mut().unwrap();
        fields.extend(
            json!({"token": Token::new(token).hash().as_str()})
                .as_object()
                .unwrap()
                .clone(),
        );
        fields.extend(columns.as_object().unwrap().clone());
        self.insert("LiteLLM_VerificationToken", &row).await;
    }

    /// Stores `row` in `table`, each field in the column of its name.
    pub async fn insert(&self, table: &str, row: &Value) {
        #[expect(
            clippy::disallowed_methods,
            reason = "Fixture statements are outside the compile-checked query set"
        )]
        let statement = sqlx::query(sqlx::AssertSqlSafe(format!(
            r#"INSERT INTO "{table}" SELECT * FROM jsonb_populate_record(NULL::"{table}", $1::jsonb)"#
        )));
        statement
            .bind(sqlx::types::Json(row))
            .execute(self.postgres.pool())
            .await
            .unwrap();
    }

    /// Stores `row` in `table`, each field in the column of its name and every other column at its
    /// default.
    pub async fn insert_named(&self, table: &str, row: &Value) {
        let columns = row
            .as_object()
            .unwrap()
            .keys()
            .map(|column| format!(r#""{column}""#))
            .collect::<Vec<_>>()
            .join(", ");
        #[expect(
            clippy::disallowed_methods,
            reason = "Fixture statements are outside the compile-checked query set"
        )]
        let statement = sqlx::query(sqlx::AssertSqlSafe(format!(
            r#"INSERT INTO "{table}" ({columns}) SELECT {columns} FROM jsonb_populate_record(NULL::"{table}", $1::jsonb)"#
        )));
        statement
            .bind(sqlx::types::Json(row))
            .execute(self.postgres.pool())
            .await
            .unwrap();
    }

    /// Runs the fixture statements of `sql`.
    pub async fn execute(&self, sql: &'static str) {
        #[expect(
            clippy::disallowed_methods,
            reason = "Fixture statements are outside the compile-checked query set"
        )]
        let statements = sqlx::raw_sql(sql);
        statements.execute(self.postgres.pool()).await.unwrap();
    }

    /// A transaction that holds `table` locked, so every read of it waits until it ends.
    pub async fn locking(&self, table: &str) -> sqlx::Transaction<'static, sqlx::Postgres> {
        let mut transaction = self.postgres.pool().begin().await.unwrap();
        #[expect(
            clippy::disallowed_methods,
            reason = "Fixture statements are outside the compile-checked query set"
        )]
        let statement = sqlx::query(sqlx::AssertSqlSafe(format!(
            r#"LOCK TABLE "{table}" IN ACCESS EXCLUSIVE MODE"#
        )));
        statement.execute(&mut *transaction).await.unwrap();
        transaction
    }

    /// Where a config names this database, as `general_settings.database_url`.
    pub fn url(&self) -> &str {
        self.postgres.url()
    }

    /// What the gateway's auth store and its repositories run their queries on this database with.
    pub fn database(&self) -> Database {
        connected(self.postgres.url(), self.queries.clone())
    }

    /// How many times the key view was read through [`Self::database`].
    pub fn key_reads(&self) -> usize {
        self.count(|call| call.table == Some("combined_view"))
    }

    /// How many queries a repository names `call_type` ran to their end through
    /// [`Self::database`].
    pub fn queries(&self, call_type: &str) -> usize {
        self.count(|call| call.call_type == call_type)
    }

    fn count(&self, counted: impl Fn(&Call) -> bool) -> usize {
        let queries = self.queries.0.lock();
        queries.iter().filter(|call| counted(call)).count()
    }
}

/// The auth cache of a gateway with no coordination Redis: this process's memory alone.
pub fn memory_cache() -> Arc<AuthCache> {
    Arc::new(AuthCache::new(None, Settings::default()))
}

/// A database nothing listens on, whose every read fails.
pub fn unreachable() -> Database {
    connected(
        "postgres://postgres:postgres@127.0.0.1:1/postgres",
        Arc::<Queries>::default(),
    )
}

fn connected(url: &str, events: Arc<Queries>) -> Database {
    let options = PgConnectOptions::from_str(url).unwrap();
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
            acquire_timeout_seconds: Some(2.0),
            idle_timeout_seconds: None,
            socket_timeout_seconds: None,
        },
    };
    Database::new(
        Pools::new(&endpoint, None).unwrap(),
        events,
        None,
        endpoint.names_statements(),
    )
}
