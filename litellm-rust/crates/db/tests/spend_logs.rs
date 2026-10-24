use std::error::Error as StdError;

use litellm_db::{SpendLog, insert_spend_logs};
use rstest::{fixture, rstest};
use serde_json::{Value, json};
use sqlx::{Acquire, PgExecutor, PgPool, Row, types::chrono::NaiveDateTime};

mod common;
use common::database_pool;

#[fixture]
fn logs() -> [SpendLog; 2] {
    [
        SpendLog {
            request_id: "spend-logs-a".into(),
            call_type: "acompletion".into(),
            api_key: "key-a".into(),
            spend: 0.125,
            total_tokens: 17,
            prompt_tokens: 11,
            completion_tokens: 6,
            start_time: "2026-01-02 03:04:05.123".into(),
            end_time: "2026-01-02 03:04:06.789".into(),
            completion_start_time: Some("2026-01-02 03:04:05.456".into()),
            request_duration_ms: Some(1666),
            model: "model-a".into(),
            model_id: "deployment-a".into(),
            model_group: "group-a".into(),
            custom_llm_provider: "provider-a".into(),
            api_base: "https://a.example.test".into(),
            user: "user-a".into(),
            metadata: json!({"tenant": "a", "nested": {"tokens": 11}, "flag": false}),
            cache_hit: "False".into(),
            cache_key: "cache-a".into(),
            team_id: "team-a".into(),
            organization_id: "org-a".into(),
            end_user: "end-user-a".into(),
            requester_ip_address: Some("192.0.2.1".into()),
            session_id: Some("session-a".into()),
            status: "success".into(),
            litellm_call_id: Some("call-a".into()),
        },
        SpendLog {
            request_id: "spend-logs-b".into(),
            call_type: "aembedding".into(),
            api_key: "key-b".into(),
            spend: 2.75,
            total_tokens: 43,
            prompt_tokens: 29,
            completion_tokens: 14,
            start_time: "2026-02-03 14:15:16.234".into(),
            end_time: "2026-02-03 14:15:18.567".into(),
            completion_start_time: None,
            request_duration_ms: None,
            model: "model-b".into(),
            model_id: "deployment-b".into(),
            model_group: "group-b".into(),
            custom_llm_provider: "provider-b".into(),
            api_base: "https://b.example.test".into(),
            user: "user-b".into(),
            metadata: json!({"tenant": "b", "tags": ["distinct", 7], "nullable": null}),
            cache_hit: "True".into(),
            cache_key: "cache-b".into(),
            team_id: "team-b".into(),
            organization_id: "org-b".into(),
            end_user: "end-user-b".into(),
            requester_ip_address: None,
            session_id: None,
            status: "failure".into(),
            litellm_call_id: None,
        },
    ]
}

#[derive(Debug, PartialEq)]
struct StoredLog {
    log: SpendLog,
    request_tags: Value,
    messages: Value,
    response: Value,
    proxy_server_request: Value,
}

fn timestamp(value: NaiveDateTime) -> String {
    value.format("%Y-%m-%d %H:%M:%S%.3f").to_string()
}

async fn read_log(
    executor: impl PgExecutor<'_>,
    request_id: &str,
) -> Result<StoredLog, sqlx::Error> {
    #[expect(
        clippy::disallowed_methods,
        reason = "Test assertions are outside the compile-checked query set"
    )]
    let row = sqlx::query(r#"SELECT * FROM "LiteLLM_SpendLogs" WHERE request_id = $1"#)
        .bind(request_id)
        .fetch_one(executor)
        .await?;
    Ok(StoredLog {
        log: SpendLog {
            request_id: row.try_get("request_id")?,
            call_type: row.try_get("call_type")?,
            api_key: row.try_get("api_key")?,
            spend: row.try_get("spend")?,
            total_tokens: row.try_get("total_tokens")?,
            prompt_tokens: row.try_get("prompt_tokens")?,
            completion_tokens: row.try_get("completion_tokens")?,
            start_time: timestamp(row.try_get("startTime")?),
            end_time: timestamp(row.try_get("endTime")?),
            completion_start_time: row
                .try_get::<Option<NaiveDateTime>, _>("completionStartTime")?
                .map(timestamp),
            request_duration_ms: row.try_get("request_duration_ms")?,
            model: row.try_get("model")?,
            model_id: row.try_get("model_id")?,
            model_group: row.try_get("model_group")?,
            custom_llm_provider: row.try_get("custom_llm_provider")?,
            api_base: row.try_get("api_base")?,
            user: row.try_get("user")?,
            metadata: row.try_get("metadata")?,
            cache_hit: row.try_get("cache_hit")?,
            cache_key: row.try_get("cache_key")?,
            team_id: row.try_get("team_id")?,
            organization_id: row.try_get("organization_id")?,
            end_user: row.try_get("end_user")?,
            requester_ip_address: row.try_get("requester_ip_address")?,
            session_id: row.try_get("session_id")?,
            status: row.try_get("status")?,
            litellm_call_id: row.try_get("litellm_call_id")?,
        },
        request_tags: row.try_get("request_tags")?,
        messages: row.try_get("messages")?,
        response: row.try_get("response")?,
        proxy_server_request: row.try_get("proxy_server_request")?,
    })
}

#[rstest]
#[case::all_columns_and_duplicate()]
#[tokio::test]
async fn inserts_distinct_rows_without_replacing_duplicates(
    logs: [SpendLog; 2],
    #[future(awt)] database_pool: Result<Option<PgPool>, sqlx::Error>,
) -> Result<(), Box<dyn StdError>> {
    let Some(pool) = database_pool? else {
        return Ok(());
    };
    let mut tx = pool.begin().await?;
    insert_spend_logs(&mut *tx, &logs).await?;
    let replacement = SpendLog {
        request_id: logs[0].request_id.clone(),
        ..logs[1].clone()
    };
    insert_spend_logs(&mut *tx, &[replacement]).await?;
    assert_eq!(
        read_log(&mut *tx, &logs[0].request_id).await?,
        StoredLog {
            log: logs[0].clone(),
            request_tags: json!([]),
            messages: json!({}),
            response: json!({}),
            proxy_server_request: json!({}),
        }
    );
    assert_eq!(
        read_log(&mut *tx, &logs[1].request_id).await?,
        StoredLog {
            log: logs[1].clone(),
            request_tags: json!([]),
            messages: json!({}),
            response: json!({}),
            proxy_server_request: json!({}),
        }
    );
    tx.rollback().await?;
    Ok(())
}

#[rstest]
#[case::empty_batch()]
#[tokio::test]
async fn empty_input_leaves_rows_unchanged(
    #[future(awt)] database_pool: Result<Option<PgPool>, sqlx::Error>,
) -> Result<(), Box<dyn StdError>> {
    let Some(pool) = database_pool? else {
        return Ok(());
    };
    let mut tx = pool.begin().await?;
    #[expect(
        clippy::disallowed_methods,
        reason = "Test assertions are outside the compile-checked query set"
    )]
    let before: (i64,) = sqlx::query_as(r#"SELECT count(*) FROM "LiteLLM_SpendLogs""#)
        .fetch_one(&mut *tx)
        .await?;
    insert_spend_logs(&mut *tx, &[]).await?;
    #[expect(
        clippy::disallowed_methods,
        reason = "Test assertions are outside the compile-checked query set"
    )]
    let after: (i64,) = sqlx::query_as(r#"SELECT count(*) FROM "LiteLLM_SpendLogs""#)
        .fetch_one(&mut *tx)
        .await?;
    assert_eq!(before, after);
    tx.rollback().await?;
    Ok(())
}

#[rstest]
#[case::invalid_timestamp("not-a-timestamp")]
#[tokio::test]
async fn invalid_timestamp_allows_caller_rollback(
    #[case] invalid_timestamp: &str,
    logs: [SpendLog; 2],
    #[future(awt)] database_pool: Result<Option<PgPool>, sqlx::Error>,
) -> Result<(), Box<dyn StdError>> {
    let Some(pool) = database_pool? else {
        return Ok(());
    };
    let mut outer = pool.begin().await?;
    let mut tx = outer.begin().await?;
    let valid = SpendLog {
        request_id: "spend-logs-rollback-valid".into(),
        ..logs[0].clone()
    };
    let invalid = SpendLog {
        request_id: "spend-logs-rollback-invalid".into(),
        start_time: invalid_timestamp.into(),
        ..logs[1].clone()
    };
    insert_spend_logs(&mut *tx, std::slice::from_ref(&valid)).await?;
    let result = insert_spend_logs(&mut *tx, &[invalid]).await;
    assert!(matches!(
        result,
        Err(litellm_db::Error::Query(sqlx::Error::Database(ref error)))
            if error.code().as_deref() == Some("22007")
    ));
    tx.rollback().await?;
    #[expect(
        clippy::disallowed_methods,
        reason = "Test assertions are outside the compile-checked query set"
    )]
    let remaining: (i64,) =
        sqlx::query_as(r#"SELECT count(*) FROM "LiteLLM_SpendLogs" WHERE request_id = ANY($1)"#)
            .bind([valid.request_id.as_str(), "spend-logs-rollback-invalid"].as_slice())
            .fetch_one(&mut *outer)
            .await?;
    assert_eq!(remaining.0, 0);
    outer.rollback().await?;
    Ok(())
}
