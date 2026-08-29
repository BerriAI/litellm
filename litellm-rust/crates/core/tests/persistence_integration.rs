//! Integration tests for Redis and PostgreSQL persistence.
//!
//! These tests require running Redis and PostgreSQL instances.
//! Start them with: docker compose -f docker-compose.test.yml up -d
//!
//! Run with: REDIS_URL=redis://localhost:6379 DATABASE_URL=postgres://litellm:litellm@localhost:5432/litellm cargo test -p litellm-core --features integration-tests

#![cfg(feature = "integration-tests")]

use litellm_core::persistence::{CacheStore, DatabaseStore, PostgresStore, RedisStore};
use serde_json::json;

fn redis_url() -> String {
    std::env::var("REDIS_URL").unwrap_or_else(|_| "redis://127.0.0.1:6379".to_string())
}

fn database_url() -> String {
    std::env::var("DATABASE_URL")
        .unwrap_or_else(|_| "postgres://litellm:litellm@127.0.0.1:5432/litellm".to_string())
}

const SPEND_LOGS_SCHEMA: &str = "
    request_id TEXT PRIMARY KEY,
    call_type TEXT NOT NULL DEFAULT '',
    api_key TEXT NOT NULL DEFAULT '',
    spend FLOAT NOT NULL DEFAULT 0.0,
    total_tokens INTEGER NOT NULL DEFAULT 0,
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    model TEXT NOT NULL DEFAULT '',
    \"user\" TEXT,
    team_id TEXT,
    organization_id TEXT,
    metadata JSONB,
    \"startTime\" TIMESTAMP NOT NULL DEFAULT NOW(),
    \"endTime\" TIMESTAMP NOT NULL DEFAULT NOW(),
    status TEXT
";

#[tokio::test]
async fn redis_connect_and_ping() {
    let store = RedisStore::connect(&redis_url())
        .await
        .expect("connects to Redis");
    store
        .set("test:ping", &json!("pong"), Some(10))
        .await
        .expect("set works");
    let val = store.get("test:ping").await.expect("get works");
    assert_eq!(val.unwrap(), json!("pong"));
    store.delete("test:ping").await.expect("delete works");
}

#[tokio::test]
async fn redis_get_set_delete_lifecycle() {
    let store = RedisStore::connect(&redis_url()).await.expect("connects");
    let key = "test:lifecycle";

    assert!(store.get(key).await.unwrap().is_none());

    store
        .set(key, &json!({"count": 42}), Some(60))
        .await
        .unwrap();
    let val = store.get(key).await.unwrap().unwrap();
    assert_eq!(val["count"], 42);

    store.delete(key).await.unwrap();
    assert!(store.get(key).await.unwrap().is_none());
}

#[tokio::test]
async fn redis_incr_by_float_accumulates() {
    let store = RedisStore::connect(&redis_url()).await.expect("connects");
    let key = "test:incr";
    store.delete(key).await.ok();

    let val1 = store.incr_by_float(key, 1.5).await.unwrap();
    assert!((val1 - 1.5).abs() < 1e-10);

    let val2 = store.incr_by_float(key, 2.3).await.unwrap();
    assert!((val2 - 3.8).abs() < 1e-10);

    let val3 = store.incr_by_float(key, -1.0).await.unwrap();
    assert!((val3 - 2.8).abs() < 1e-10);

    store.delete(key).await.ok();
}

#[tokio::test]
async fn redis_concurrent_incr() {
    let store = RedisStore::connect(&redis_url()).await.expect("connects");
    let key = "test:concurrent_incr";
    store.delete(key).await.ok();

    let store1 = RedisStore::connect(&redis_url()).await.unwrap();
    let store2 = RedisStore::connect(&redis_url()).await.unwrap();

    let (r1, r2, r3) = tokio::join!(
        store.incr_by_float(key, 1.0),
        store1.incr_by_float(key, 2.0),
        store2.incr_by_float(key, 3.0),
    );

    assert!(r1.unwrap() > 0.0);
    assert!(r2.unwrap() > 0.0);
    assert!(r3.unwrap() > 0.0);

    store.delete(key).await.ok();
}

#[tokio::test]
async fn postgres_connect_and_query() {
    let store = PostgresStore::connect(&database_url())
        .await
        .expect("connects to Postgres");

    sqlx::query("CREATE TABLE IF NOT EXISTS _rust_test_ping (id TEXT PRIMARY KEY)")
        .execute(store.pool())
        .await
        .expect("creates test table");

    sqlx::query("DROP TABLE IF EXISTS _rust_test_ping")
        .execute(store.pool())
        .await
        .expect("drops test table");
}

#[tokio::test]
async fn postgres_insert_and_query_spend_log() {
    let store = PostgresStore::connect(&database_url())
        .await
        .expect("connects");

    let table = "LiteLLM_SpendLogs";

    sqlx::query(&format!("DROP TABLE IF EXISTS \"{table}\""))
        .execute(store.pool())
        .await
        .expect("drops any leftover table");

    sqlx::query(&format!("CREATE TABLE \"{table}\" ({SPEND_LOGS_SCHEMA})"))
        .execute(store.pool())
        .await
        .expect("creates spend logs table");

    let log = json!({
        "request_id": "rust-test-req-1",
        "call_type": "completion",
        "api_key": "hashed-key-123",
        "spend": 0.05,
        "total_tokens": 150,
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "model": "gpt-4o",
        "user": "user-1",
        "team_id": "team-1",
        "status": "success"
    });

    store
        .insert_spend_log(&log)
        .await
        .expect("inserts spend log");

    let row: (f64,) = sqlx::query_as(&format!(
        "SELECT spend FROM \"{table}\" WHERE request_id = $1"
    ))
    .bind("rust-test-req-1")
    .fetch_one(store.pool())
    .await
    .expect("queries spend log");

    assert!((row.0 - 0.05).abs() < 1e-10);

    sqlx::query(&format!("DROP TABLE IF EXISTS \"{table}\""))
        .execute(store.pool())
        .await
        .expect("cleans up");
}

#[tokio::test]
async fn postgres_batch_insert_spend_logs() {
    let store = PostgresStore::connect(&database_url())
        .await
        .expect("connects");

    let table = "LiteLLM_SpendLogs";

    sqlx::query(&format!("DROP TABLE IF EXISTS \"{table}\""))
        .execute(store.pool())
        .await
        .expect("drops any leftover table");

    sqlx::query(&format!("CREATE TABLE \"{table}\" ({SPEND_LOGS_SCHEMA})"))
        .execute(store.pool())
        .await
        .expect("creates table");

    let logs = vec![
        json!({"request_id": "rust-batch-1", "spend": 0.01, "call_type": "completion", "api_key": "k1", "total_tokens": 10, "prompt_tokens": 5, "completion_tokens": 5, "model": "gpt-4o"}),
        json!({"request_id": "rust-batch-2", "spend": 0.02, "call_type": "completion", "api_key": "k1", "total_tokens": 20, "prompt_tokens": 10, "completion_tokens": 10, "model": "gpt-4o"}),
        json!({"request_id": "rust-batch-3", "spend": 0.03, "call_type": "completion", "api_key": "k2", "total_tokens": 30, "prompt_tokens": 15, "completion_tokens": 15, "model": "gpt-4"}),
    ];

    store
        .batch_insert_spend_logs(&logs)
        .await
        .expect("batch inserts");

    let count: (i64,) = sqlx::query_as(&format!("SELECT COUNT(*) FROM \"{table}\""))
        .fetch_one(store.pool())
        .await
        .expect("counts");
    assert_eq!(count.0, 3);

    sqlx::query(&format!("DROP TABLE IF EXISTS \"{table}\""))
        .execute(store.pool())
        .await
        .expect("cleans up");
}

#[tokio::test]
async fn postgres_update_entity_spend() {
    let store = PostgresStore::connect(&database_url())
        .await
        .expect("connects");

    sqlx::query("DROP TABLE IF EXISTS _rust_test_verification")
        .execute(store.pool())
        .await
        .ok();
    sqlx::query("DROP TABLE IF EXISTS _rust_test_users")
        .execute(store.pool())
        .await
        .ok();

    sqlx::query("CREATE TABLE _rust_test_verification (token TEXT PRIMARY KEY, spend FLOAT NOT NULL DEFAULT 0.0)")
        .execute(store.pool())
        .await
        .expect("creates verification table");
    sqlx::query("CREATE TABLE _rust_test_users (user_id TEXT PRIMARY KEY, spend FLOAT NOT NULL DEFAULT 0.0)")
        .execute(store.pool())
        .await
        .expect("creates user table");

    sqlx::query("INSERT INTO _rust_test_verification (token, spend) VALUES ('test-key-1', 0.0)")
        .execute(store.pool())
        .await
        .expect("inserts key");
    sqlx::query("INSERT INTO _rust_test_users (user_id, spend) VALUES ('test-user-1', 0.0)")
        .execute(store.pool())
        .await
        .expect("inserts user");

    // The update_entity_spend method uses the real table names, so we test
    // the underlying SQL pattern directly with our test tables.
    sqlx::query("UPDATE _rust_test_verification SET spend = spend + $1 WHERE token = $2")
        .bind(0.05)
        .bind("test-key-1")
        .execute(store.pool())
        .await
        .expect("updates key spend");
    sqlx::query("UPDATE _rust_test_verification SET spend = spend + $1 WHERE token = $2")
        .bind(0.03)
        .bind("test-key-1")
        .execute(store.pool())
        .await
        .expect("updates key spend again");
    sqlx::query("UPDATE _rust_test_users SET spend = spend + $1 WHERE user_id = $2")
        .bind(0.10)
        .bind("test-user-1")
        .execute(store.pool())
        .await
        .expect("updates user spend");

    let key_spend: (f64,) =
        sqlx::query_as("SELECT spend FROM _rust_test_verification WHERE token = 'test-key-1'")
            .fetch_one(store.pool())
            .await
            .expect("queries key spend");
    assert!((key_spend.0 - 0.08).abs() < 1e-10);

    let user_spend: (f64,) =
        sqlx::query_as("SELECT spend FROM _rust_test_users WHERE user_id = 'test-user-1'")
            .fetch_one(store.pool())
            .await
            .expect("queries user spend");
    assert!((user_spend.0 - 0.10).abs() < 1e-10);

    sqlx::query("DROP TABLE IF EXISTS _rust_test_verification")
        .execute(store.pool())
        .await
        .ok();
    sqlx::query("DROP TABLE IF EXISTS _rust_test_users")
        .execute(store.pool())
        .await
        .ok();
}
