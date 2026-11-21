//! Replays `tests/goldens/stored.json`, recorded from the Python writer by
//! `scripts/record_goldens.py`, against a Prisma-migrated database: the rows and deliveries a
//! flush replayed once stores.

#![recursion_limit = "256"]

use std::{error::Error as StdError, str::FromStr, sync::Arc, time::SystemTime};

use litellm_db::{Endpoint, Observer, PoolLimits, Pools, QueryEvent, SslMode};
use litellm_repositories::SpendLogRepository;
use litellm_secret_value::SecretValue;
use litellm_spend_log_writer::{
    PgLedgerStore, QueuedSpendLog, SpendLogFlusher, SpendLogWriter, WriterSettings,
};
use rstest::{fixture, rstest};
use serde::Deserialize;
use serde_json::{Value, json};
use sqlx::{PgPool, postgres::PgConnectOptions};

struct Ignored;

impl Observer for Ignored {
    fn record(&self, _event: QueryEvent) {}
}

fn endpoint() -> Result<Endpoint, Box<dyn StdError>> {
    let database_url = std::env::var("DATABASE_URL")?;
    let options = PgConnectOptions::from_str(&database_url)?;
    Ok(Endpoint {
        host: options.get_host().to_owned(),
        port: options.get_port(),
        socket: None,
        username: options.get_username().to_owned(),
        password: url::Url::parse(&database_url)?
            .password()
            .map(SecretValue::new),
        database: options.get_database().unwrap_or("postgres").to_owned(),
        ssl_mode: SslMode::Prefer,
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
    })
}

struct Database {
    writer: SpendLogWriter<PgLedgerStore>,
    pool: PgPool,
    prefix: String,
}

#[fixture]
fn database() -> Result<Database, Box<dyn StdError>> {
    let endpoint = endpoint()?;
    let pools = Pools::new(&endpoint, None)?;
    let pool = pools.writer().clone();
    let settings = WriterSettings::default();
    let store = PgLedgerStore::new(
        SpendLogRepository::new(litellm_repositories::Database::new(
            pools,
            Arc::new(Ignored),
            None,
            true,
        )),
        settings.transaction_timeout,
    );
    let prefix = format!(
        "spend-log-writer-{}",
        SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)?
            .as_nanos()
    );
    Ok(Database {
        writer: SpendLogWriter::new(SpendLogFlusher::new(store, settings), || {}),
        pool,
        prefix,
    })
}

/// What the Python writer stored, under this run's request id prefix.
#[derive(Deserialize)]
struct Stored {
    rows: Vec<String>,
    logs: Vec<Value>,
    deliveries: Vec<Value>,
}

impl Database {
    fn id(&self, name: &str) -> String {
        format!("{}-{name}", self.prefix)
    }

    fn golden(&self) -> Result<Stored, Box<dyn StdError>> {
        let text = include_str!("goldens/stored.json").replace("golden-", &self.id(""));
        Ok(serde_json::from_str(&text)?)
    }

    fn queued(&self, texts: &[String]) -> Result<Vec<QueuedSpendLog>, Box<dyn StdError>> {
        texts
            .iter()
            .map(|text| QueuedSpendLog::from_python(text).map_err(|rejected| rejected.error.into()))
            .collect()
    }

    fn row(&self, name: &str, messages: &str) -> String {
        let request_id = self.id(name);
        json!({
            "request_id": request_id,
            "call_type": "acompletion",
            "api_key": "hashed",
            "spend": 0.25,
            "total_tokens": 3,
            "prompt_tokens": 1,
            "completion_tokens": 2,
            "startTime": "2026-11-02T12:00:00+00:00",
            "endTime": "2026-11-02T12:00:01+00:00",
            "completionStartTime": "2026-11-02T12:00:00.500000+00:00",
            "model": "gpt",
            "model_id": "deployment",
            "model_group": "gpt",
            "mcp_namespaced_tool_name": null,
            "agent_id": null,
            "billing_agent_id": null,
            "api_base": "https://example.test",
            "user": "user",
            "metadata": "{\"status\": \"success\"}",
            "cache_hit": "False",
            "cache_key": "Cache OFF",
            "request_tags": "[]",
            "team_id": "team",
            "organization_id": "",
            "end_user": "",
            "requester_ip_address": null,
            "custom_llm_provider": "openai",
            "messages": messages,
            "response": "{}",
            "proxy_server_request": "{}",
            "session_id": null,
            "request_duration_ms": 1000,
            "status": "success",
            "litellm_call_id": "call",
            "accounting_delivery_id": format!("{request_id}-delivery"),
            "accounting_payload": {"request_id": request_id, "spend": 0.25},
            "accounting_log_disabled": false,
        })
        .to_string()
    }

    async fn stored(&self, query: &'static str) -> Result<Vec<Value>, Box<dyn StdError>> {
        #[expect(
            clippy::disallowed_methods,
            reason = "Test assertions are outside the compile-checked query set"
        )]
        let rows = sqlx::query_scalar::<_, Value>(sqlx::AssertSqlSafe(query))
            .bind(format!("{}%", self.prefix))
            .fetch_all(&self.pool)
            .await?;
        Ok(rows)
    }

    async fn clean(&self) -> Result<(), Box<dyn StdError>> {
        for query in [
            r#"DELETE FROM "LiteLLM_SpendLogs" WHERE request_id LIKE $1"#,
            r#"DELETE FROM "LiteLLM_SpendLogDelivery" WHERE request_id LIKE $1"#,
        ] {
            #[expect(
                clippy::disallowed_methods,
                reason = "Test cleanup is outside the compile-checked query set"
            )]
            sqlx::query(sqlx::AssertSqlSafe(query))
                .bind(format!("{}%", self.prefix))
                .execute(&self.pool)
                .await?;
        }
        Ok(())
    }
}

const LOGS: &str = r#"SELECT to_jsonb(s) - 'created_at' - 'updated_at' FROM "LiteLLM_SpendLogs" s WHERE request_id LIKE $1 ORDER BY request_id"#;
const DELIVERIES: &str = r#"SELECT to_jsonb(d) - 'created_at' - 'next_attempt_at' FROM "LiteLLM_SpendLogDelivery" d WHERE request_id LIKE $1 ORDER BY delivery_id"#;

#[rstest]
#[tokio::test]
#[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
async fn a_replayed_flush_stores_what_python_stores(
    database: Result<Database, Box<dyn StdError>>,
) -> Result<(), Box<dyn StdError>> {
    let database = database?;
    let golden = database.golden()?;
    for _ in 0..2 {
        database
            .writer
            .enqueue(database.queued(&golden.rows)?, false);
        database.writer.flush(usize::MAX, None).await?;
    }

    let logs = database.stored(LOGS).await?;
    let deliveries = database.stored(DELIVERIES).await?;
    database.clean().await?;
    assert_eq!(logs, golden.logs);
    assert_eq!(deliveries, golden.deliveries);
    Ok(())
}

#[rstest]
#[tokio::test]
#[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
async fn a_poison_row_is_dropped_but_keeps_its_charge(
    database: Result<Database, Box<dyn StdError>>,
) -> Result<(), Box<dyn StdError>> {
    let database = database?;
    let rows = [
        database.row("good-1", "[]"),
        database.row("poison", "not json"),
        database.row("good-2", "[]"),
    ];
    database.writer.enqueue(database.queued(&rows)?, false);
    assert_eq!(database.writer.flush(usize::MAX, None).await?.len(), 3);

    let logs = database.stored(LOGS).await?;
    let deliveries = database.stored(DELIVERIES).await?;
    database.clean().await?;
    let logged: Vec<&Value> = logs.iter().map(|row| &row["request_id"]).collect();
    assert_eq!(
        logged,
        [&json!(database.id("good-1")), &json!(database.id("good-2"))]
    );
    assert_eq!(deliveries.len(), 3);
    Ok(())
}
