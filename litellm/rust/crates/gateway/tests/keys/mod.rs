//! The proxy's database as a gateway test holds it: a migrated Postgres with the keys the test
//! puts in it, read through the repositories the gateway reads through.

// Shared across test targets; each target uses a different subset.
#![allow(dead_code)]

use std::{
    str::FromStr,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use litellm_auth_store::{KeyDatabase, TeamDatabase};
use litellm_db::{Endpoint, Observer, PoolLimits, Pools, QueryEvent, SslMode};
use litellm_db_testing::MigratedPostgres;
use litellm_gateway_auth_core::Token;
use litellm_redis_store::auth_cache::{AuthCache, Settings};
use litellm_repositories::{Database, KeyRepository, ObjectPermissionRepository, TeamRepository};
use litellm_secret_value::SecretValue;
use serde_json::{Value, json};
use sqlx::postgres::PgConnectOptions;

/// The table columns of a key generated with no parameters, as `/key/generate` stores it.
const PLAIN: &str = include_str!("plain.json");

/// The table columns of a team created with no parameters, as `/team/new` stores it.
const PLAIN_TEAM: &str = include_str!("team.json");

/// How many times the key view was read.
#[derive(Default)]
struct KeyReads(AtomicUsize);

impl Observer for KeyReads {
    fn record(&self, event: QueryEvent) {
        if event.call.table == Some("combined_view") {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }
}

pub struct Held {
    postgres: MigratedPostgres,
    reads: Arc<KeyReads>,
}

impl Held {
    /// A database that holds no key.
    pub async fn start() -> Self {
        Self {
            postgres: MigratedPostgres::start().await.unwrap(),
            reads: Arc::default(),
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

    /// Stores the team `team_id`: a plain row with `columns` set on it.
    pub async fn team(&self, team_id: &str, columns: Value) {
        let mut row: Value = serde_json::from_str(PLAIN_TEAM).unwrap();
        let fields = row.as_object_mut().unwrap();
        fields.extend(json!({"team_id": team_id}).as_object().unwrap().clone());
        fields.extend(columns.as_object().unwrap().clone());
        self.insert("LiteLLM_TeamTable", &row).await;
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

    /// Runs the fixture statements of `sql`.
    pub async fn execute(&self, sql: &'static str) {
        #[expect(
            clippy::disallowed_methods,
            reason = "Fixture statements are outside the compile-checked query set"
        )]
        let statements = sqlx::raw_sql(sql);
        statements.execute(self.postgres.pool()).await.unwrap();
    }

    /// What a repository runs its queries on this database with.
    pub fn repositories(&self) -> Database {
        connected(self.postgres.url(), self.reads.clone())
    }

    /// The key read the gateway makes behind its auth cache.
    pub fn database(&self) -> KeyDatabase {
        key_database(self.repositories())
    }

    /// The team read the gateway makes behind its auth cache.
    pub fn teams(&self) -> TeamDatabase {
        TeamDatabase::new(Arc::new(TeamRepository::new(self.repositories())))
    }

    /// How many times the key view was read through [`Self::repositories`].
    pub fn key_reads(&self) -> usize {
        self.reads.0.load(Ordering::SeqCst)
    }
}

/// The auth cache of a gateway with no coordination Redis: this process's memory alone.
pub fn memory_cache() -> Arc<AuthCache> {
    Arc::new(AuthCache::new(None, Settings::default()))
}

/// A database nothing listens on, whose every read fails.
pub fn unreachable() -> KeyDatabase {
    key_database(nowhere())
}

/// The team read of a database nothing listens on.
pub fn unreachable_teams() -> TeamDatabase {
    TeamDatabase::new(Arc::new(TeamRepository::new(nowhere())))
}

fn nowhere() -> Database {
    connected(
        "postgres://postgres:postgres@127.0.0.1:1/postgres",
        Arc::<KeyReads>::default(),
    )
}

fn key_database(database: Database) -> KeyDatabase {
    KeyDatabase::new(
        Arc::new(KeyRepository::new(database.clone())),
        Arc::new(ObjectPermissionRepository::new(database)),
    )
}

fn connected(url: &str, events: Arc<KeyReads>) -> Database {
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
