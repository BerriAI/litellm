use chrono::{DateTime, NaiveDateTime};
use sqlx::PgExecutor;

use crate::Error;

#[macro_rules_attribute::apply(crate::db_input)]
#[derive(Default)]
pub struct SpendLog {
    pub request_id: String,
    pub call_type: String,
    pub api_key: String,
    pub spend: f64,
    pub total_tokens: i32,
    pub prompt_tokens: i32,
    pub completion_tokens: i32,
    pub start_time: String,
    pub end_time: String,
    pub completion_start_time: Option<String>,
    pub request_duration_ms: Option<i32>,
    pub model: String,
    pub model_id: String,
    pub model_group: String,
    pub custom_llm_provider: String,
    pub api_base: String,
    pub user: String,
    pub metadata: serde_json::Value,
    pub cache_hit: String,
    pub cache_key: String,
    pub team_id: String,
    pub organization_id: String,
    pub end_user: String,
    pub requester_ip_address: Option<String>,
    pub session_id: Option<String>,
    pub status: String,
    pub litellm_call_id: Option<String>,
}

#[macro_rules_attribute::apply(crate::db_row)]
pub struct SpendLogRow {
    pub request_id: String,
    pub call_type: String,
    pub api_key: String,
    pub spend: f64,
    pub total_tokens: i32,
    pub prompt_tokens: i32,
    pub completion_tokens: i32,
    #[sqlx(rename = "startTime")]
    pub start_time: NaiveDateTime,
    #[sqlx(rename = "endTime")]
    pub end_time: NaiveDateTime,
    #[sqlx(rename = "completionStartTime")]
    pub completion_start_time: Option<NaiveDateTime>,
    pub request_duration_ms: Option<i32>,
    pub model: String,
    pub model_id: Option<String>,
    pub model_group: Option<String>,
    pub custom_llm_provider: Option<String>,
    pub api_base: Option<String>,
    pub user: Option<String>,
    pub metadata: Option<serde_json::Value>,
    pub cache_hit: Option<String>,
    pub cache_key: Option<String>,
    pub request_tags: Option<serde_json::Value>,
    pub team_id: Option<String>,
    pub organization_id: Option<String>,
    pub end_user: Option<String>,
    pub requester_ip_address: Option<String>,
    pub messages: Option<serde_json::Value>,
    pub response: Option<serde_json::Value>,
    pub proxy_server_request: Option<serde_json::Value>,
    pub session_id: Option<String>,
    pub status: Option<String>,
    pub litellm_call_id: Option<String>,
}

fn parse_timestamp(value: &str) -> Result<NaiveDateTime, chrono::ParseError> {
    DateTime::parse_from_rfc3339(value)
        .map(|timestamp| timestamp.naive_utc())
        .or_else(|_| NaiveDateTime::parse_from_str(value, "%Y-%m-%dT%H:%M:%S%.f"))
        .or_else(|_| NaiveDateTime::parse_from_str(value, "%Y-%m-%d %H:%M:%S%.f"))
}

pub async fn insert_spend_logs(
    executor: impl PgExecutor<'_>,
    logs: &[SpendLog],
) -> Result<(), Error> {
    if logs.is_empty() {
        return Ok(());
    }
    let start_times = logs
        .iter()
        .map(|log| parse_timestamp(&log.start_time))
        .collect::<Result<Vec<_>, _>>()?;
    let end_times = logs
        .iter()
        .map(|log| parse_timestamp(&log.end_time))
        .collect::<Result<Vec<_>, _>>()?;
    let completion_start_times = logs
        .iter()
        .map(|log| {
            log.completion_start_time
                .as_deref()
                .map(parse_timestamp)
                .transpose()
        })
        .collect::<Result<Vec<_>, _>>()?;
    macro_rules! column {
        ($field:ident) => {
            logs.iter()
                .map(|log| log.$field.clone())
                .collect::<Vec<_>>()
        };
    }
    sqlx::query_file!(
        "queries/spend/insert_spend_logs.sql",
        &column!(request_id),
        &column!(call_type),
        &column!(api_key),
        &column!(spend),
        &column!(total_tokens),
        &column!(prompt_tokens),
        &column!(completion_tokens),
        &start_times,
        &end_times,
        &completion_start_times as &[Option<NaiveDateTime>],
        &column!(request_duration_ms) as &[Option<i32>],
        &column!(model),
        &column!(model_id),
        &column!(model_group),
        &column!(custom_llm_provider),
        &column!(api_base),
        &column!(user),
        &column!(metadata),
        &column!(cache_hit),
        &column!(cache_key),
        &column!(team_id),
        &column!(organization_id),
        &column!(end_user),
        &column!(requester_ip_address) as &[Option<String>],
        &column!(session_id) as &[Option<String>],
        &column!(status),
        &column!(litellm_call_id) as &[Option<String>],
    )
    .execute(executor)
    .await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::error::Error as StdError;

    use super::{SpendLog, SpendLogRow, insert_spend_logs};
    use crate::test_support::database_pool;
    use rstest::{fixture, rstest};
    use serde_json::json;
    use sqlx::{Acquire, Connection, PgExecutor, PgPool, types::chrono::NaiveDateTime};

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

    async fn read_log(
        executor: impl PgExecutor<'_>,
        request_id: &str,
    ) -> Result<SpendLogRow, sqlx::Error> {
        #[expect(
            clippy::disallowed_methods,
            reason = "Test assertions are outside the compile-checked query set"
        )]
        sqlx::query_as::<_, SpendLogRow>(
            r#"SELECT * FROM "LiteLLM_SpendLogs" WHERE request_id = $1"#,
        )
        .bind(request_id)
        .fetch_one(executor)
        .await
    }

    fn expected_log(log: &SpendLog) -> Result<SpendLogRow, chrono::ParseError> {
        Ok(SpendLogRow {
            request_id: log.request_id.clone(),
            call_type: log.call_type.clone(),
            api_key: log.api_key.clone(),
            spend: log.spend,
            total_tokens: log.total_tokens,
            prompt_tokens: log.prompt_tokens,
            completion_tokens: log.completion_tokens,
            start_time: log.start_time.replace(' ', "T").parse()?,
            end_time: log.end_time.replace(' ', "T").parse()?,
            completion_start_time: log
                .completion_start_time
                .as_ref()
                .map(|value| value.replace(' ', "T").parse())
                .transpose()?,
            request_duration_ms: log.request_duration_ms,
            model: log.model.clone(),
            model_id: Some(log.model_id.clone()),
            model_group: Some(log.model_group.clone()),
            custom_llm_provider: Some(log.custom_llm_provider.clone()),
            api_base: Some(log.api_base.clone()),
            user: Some(log.user.clone()),
            metadata: Some(log.metadata.clone()),
            cache_hit: Some(log.cache_hit.clone()),
            cache_key: Some(log.cache_key.clone()),
            request_tags: Some(json!([])),
            team_id: Some(log.team_id.clone()),
            organization_id: Some(log.organization_id.clone()),
            end_user: Some(log.end_user.clone()),
            requester_ip_address: log.requester_ip_address.clone(),
            messages: Some(json!({})),
            response: Some(json!({})),
            proxy_server_request: Some(json!({})),
            session_id: log.session_id.clone(),
            status: Some(log.status.clone()),
            litellm_call_id: log.litellm_call_id.clone(),
        })
    }

    #[rstest]
    #[case::all_columns_and_duplicate()]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn inserts_distinct_rows_without_replacing_duplicates(
        logs: [SpendLog; 2],
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut tx = pool.begin().await?;
        insert_spend_logs(&mut *tx, &logs).await?;
        let replacement = SpendLog {
            request_id: logs[0].request_id.clone(),
            ..logs[1].clone()
        };
        insert_spend_logs(&mut *tx, &[replacement]).await?;
        assert_eq!(
            read_log(&mut *tx, &logs[0].request_id).await?,
            expected_log(&logs[0])?
        );
        assert_eq!(
            read_log(&mut *tx, &logs[1].request_id).await?,
            expected_log(&logs[1])?
        );
        tx.rollback().await?;
        Ok(())
    }

    #[rstest]
    #[case::partitioned_duplicate()]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn inserts_into_partitioned_logs_without_replacing_duplicates(
        logs: [SpendLog; 2],
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut tx = pool.begin().await?;
        #[expect(
            clippy::disallowed_methods,
            reason = "The test creates a temporary partitioned table at runtime"
        )]
        sqlx::raw_sql(
            r#"CREATE TEMP TABLE "LiteLLM_SpendLogs" (
                LIKE public."LiteLLM_SpendLogs" INCLUDING DEFAULTS INCLUDING GENERATED,
                PRIMARY KEY (request_id, "startTime")
            ) PARTITION BY RANGE ("startTime");
            CREATE TEMP TABLE spend_logs_default
                PARTITION OF pg_temp."LiteLLM_SpendLogs" DEFAULT;
            SET LOCAL search_path TO pg_temp, public;"#,
        )
        .execute(&mut *tx)
        .await?;
        tx.clear_cached_statements().await?;
        insert_spend_logs(&mut *tx, &logs).await?;
        let replacement = SpendLog {
            spend: 99.0,
            ..logs[0].clone()
        };
        insert_spend_logs(&mut *tx, &[replacement]).await?;
        #[expect(
            clippy::disallowed_methods,
            reason = "The partitioned test fixture is created at runtime"
        )]
        let count: (i64,) = sqlx::query_as(r#"SELECT count(*) FROM pg_temp."LiteLLM_SpendLogs""#)
            .fetch_one(&mut *tx)
            .await?;
        assert_eq!(count.0, 2);
        assert_eq!(
            read_log(&mut *tx, &logs[0].request_id).await?,
            expected_log(&logs[0])?
        );
        assert_eq!(
            read_log(&mut *tx, &logs[1].request_id).await?,
            expected_log(&logs[1])?
        );
        let distinct_partition_key = SpendLog {
            request_id: logs[0].request_id.clone(),
            ..logs[1].clone()
        };
        insert_spend_logs(&mut *tx, &[distinct_partition_key]).await?;
        #[expect(
            clippy::disallowed_methods,
            reason = "The partitioned test fixture is created at runtime"
        )]
        let count: (i64,) = sqlx::query_as(r#"SELECT count(*) FROM pg_temp."LiteLLM_SpendLogs""#)
            .fetch_one(&mut *tx)
            .await?;
        assert_eq!(count.0, 3);
        tx.rollback().await?;
        Ok(())
    }

    #[rstest]
    #[case::positive_offset_across_midnight(
        ["2026-01-02T00:04:05.1236+02:00", "2026-01-02T00:04:06.7894+02:00", "2026-01-02T00:04:05.4567+02:00"],
        ["2026-01-01T22:04:05.124", "2026-01-01T22:04:06.789", "2026-01-01T22:04:05.457"]
    )]
    #[case::negative_offset_across_midnight(
        ["2026-01-01T23:04:05-02:00", "2026-01-01T23:04:06-02:00", "2026-01-01T23:04:05.456-02:00"],
        ["2026-01-02T01:04:05", "2026-01-02T01:04:06", "2026-01-02T01:04:05.456"]
    )]
    #[case::utc(
        ["2026-01-02T03:04:05Z", "2026-01-02T03:04:06Z", "2026-01-02T03:04:05.456Z"],
        ["2026-01-02T03:04:05", "2026-01-02T03:04:06", "2026-01-02T03:04:05.456"]
    )]
    #[case::naive_iso(
        ["2026-01-02T03:04:05", "2026-01-02T03:04:06", "2026-01-02T03:04:05.456"],
        ["2026-01-02T03:04:05", "2026-01-02T03:04:06", "2026-01-02T03:04:05.456"]
    )]
    #[case::naive_space(
        ["2026-01-02 03:04:05", "2026-01-02 03:04:06", "2026-01-02 03:04:05.456"],
        ["2026-01-02T03:04:05", "2026-01-02T03:04:06", "2026-01-02T03:04:05.456"]
    )]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn stores_utc_timestamps(
        #[case] input: [&str; 3],
        #[case] expected: [&str; 3],
        logs: [SpendLog; 2],
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut tx = pool.begin().await?;
        #[expect(
            clippy::disallowed_methods,
            reason = "The test sets a transaction-local timezone to verify timestamp handling"
        )]
        sqlx::raw_sql("SET LOCAL TIME ZONE 'America/Los_Angeles'")
            .execute(&mut *tx)
            .await?;
        let log = SpendLog {
            start_time: input[0].into(),
            end_time: input[1].into(),
            completion_start_time: Some(input[2].into()),
            ..logs[0].clone()
        };
        insert_spend_logs(&mut *tx, std::slice::from_ref(&log)).await?;
        let row = read_log(&mut *tx, &log.request_id).await?;
        assert_eq!(row.start_time, expected[0].parse::<NaiveDateTime>()?);
        assert_eq!(row.end_time, expected[1].parse::<NaiveDateTime>()?);
        assert_eq!(
            row.completion_start_time,
            Some(expected[2].parse::<NaiveDateTime>()?)
        );
        tx.rollback().await?;
        Ok(())
    }

    #[rstest]
    #[case::empty_batch()]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn empty_input_leaves_rows_unchanged(
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
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
    #[case::invalid_timestamp("not-a-timestamp", "2026-01-02T03:04:06", None)]
    #[case::invalid_end("2026-01-02T03:04:05", "not-a-timestamp", None)]
    #[case::invalid_completion(
        "2026-01-02T03:04:05",
        "2026-01-02T03:04:06",
        Some("not-a-timestamp")
    )]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn invalid_timestamp_allows_caller_rollback(
        #[case] start_time: &str,
        #[case] end_time: &str,
        #[case] completion_start_time: Option<&str>,
        logs: [SpendLog; 2],
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut outer = pool.begin().await?;
        let mut tx = outer.begin().await?;
        let valid = SpendLog {
            request_id: "spend-logs-rollback-valid".into(),
            ..logs[0].clone()
        };
        let invalid = SpendLog {
            request_id: "spend-logs-rollback-invalid".into(),
            start_time: start_time.into(),
            end_time: end_time.into(),
            completion_start_time: completion_start_time.map(str::to_owned),
            ..logs[1].clone()
        };
        insert_spend_logs(&mut *tx, std::slice::from_ref(&valid)).await?;
        let result = insert_spend_logs(&mut *tx, &[logs[0].clone(), invalid]).await;
        assert!(matches!(result, Err(crate::Error::InvalidTimestamp(_))));
        #[expect(
            clippy::disallowed_methods,
            reason = "Test assertions are outside the compile-checked query set"
        )]
        let partial_batch: (i64,) = sqlx::query_as(
            r#"SELECT count(*) FROM "LiteLLM_SpendLogs" WHERE request_id = ANY($1)"#,
        )
        .bind([logs[0].request_id.as_str(), "spend-logs-rollback-invalid"].as_slice())
        .fetch_one(&mut *tx)
        .await?;
        assert_eq!(partial_batch.0, 0);
        tx.rollback().await?;
        #[expect(
            clippy::disallowed_methods,
            reason = "Test assertions are outside the compile-checked query set"
        )]
        let remaining: (i64,) = sqlx::query_as(
            r#"SELECT count(*) FROM "LiteLLM_SpendLogs" WHERE request_id = ANY($1)"#,
        )
        .bind([valid.request_id.as_str(), "spend-logs-rollback-invalid"].as_slice())
        .fetch_one(&mut *outer)
        .await?;
        assert_eq!(remaining.0, 0);
        outer.rollback().await?;
        Ok(())
    }
}
