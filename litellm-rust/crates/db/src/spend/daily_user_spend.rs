use sqlx::PgExecutor;

use crate::Error;

#[macro_rules_attribute::apply(crate::db_input)]
#[derive(Default)]
pub struct DailySpend {
    pub entity_id: String,
    pub date: String,
    pub api_key: String,
    pub model: String,
    pub custom_llm_provider: String,
    pub mcp_namespaced_tool_name: String,
    pub endpoint: String,
    pub model_group: Option<String>,
    pub prompt_tokens: i64,
    pub completion_tokens: i64,
    pub api_requests: i64,
    pub successful_requests: i64,
    pub failed_requests: i64,
    pub cache_read_input_tokens: i64,
    pub cache_creation_input_tokens: i64,
    pub total_response_time_ms: i64,
    pub timed_requests: i64,
    pub spend: f64,
}

pub async fn upsert_daily_user_spend(
    executor: impl PgExecutor<'_>,
    rows: &[DailySpend],
) -> Result<(), Error> {
    if rows.is_empty() {
        return Ok(());
    }
    macro_rules! column {
        ($field:ident) => {
            rows.iter()
                .map(|row| row.$field.clone())
                .collect::<Vec<_>>()
        };
    }
    sqlx::query_file!(
        "queries/spend/upsert_daily_user_spend.sql",
        &column!(entity_id),
        &column!(date),
        &column!(api_key),
        &column!(model),
        &column!(custom_llm_provider),
        &column!(endpoint),
        &column!(model_group) as &[Option<String>],
        &column!(prompt_tokens),
        &column!(completion_tokens),
        &column!(api_requests),
        &column!(successful_requests),
        &column!(failed_requests),
        &column!(cache_read_input_tokens),
        &column!(cache_creation_input_tokens),
        &column!(total_response_time_ms),
        &column!(timed_requests),
        &column!(spend),
        &column!(mcp_namespaced_tool_name),
    )
    .execute(executor)
    .await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::error::Error as StdError;

    use super::{DailySpend, upsert_daily_user_spend};
    use crate::{DailyUserRow, test_support::database_pool};
    use rstest::rstest;
    use sqlx::{PgPool, types::chrono::NaiveDateTime};

    #[rstest]
    #[case::distinct_tools("server/search", "server/write")]
    #[case::default_empty_tool("", "server/search")]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn daily_user_spend_separates_tools_and_accumulates(
        #[case] first_tool: &str,
        #[case] second_tool: &str,
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut tx = pool.begin().await?;
        let first = DailySpend {
            entity_id: format!("daily-user-tools-{first_tool}"),
            date: "2026-01-15".to_owned(),
            api_key: "tool-key".to_owned(),
            model: "tool-model".to_owned(),
            custom_llm_provider: "tool-provider".to_owned(),
            endpoint: "/mcp".to_owned(),
            model_group: Some("tool-group".to_owned()),
            mcp_namespaced_tool_name: first_tool.to_owned(),
            prompt_tokens: 3,
            api_requests: 1,
            spend: 0.25,
            ..Default::default()
        };
        let second = DailySpend {
            mcp_namespaced_tool_name: second_tool.to_owned(),
            prompt_tokens: 7,
            api_requests: 2,
            spend: 0.5,
            ..first.clone()
        };
        upsert_daily_user_spend(&mut *tx, &[second.clone(), first.clone()]).await?;
        upsert_daily_user_spend(&mut *tx, &[first.clone(), second]).await?;

        #[expect(
            clippy::disallowed_methods,
            reason = "The integration test reads stored rows to verify tool attribution"
        )]
        let rows: Vec<DailyUserRow> = sqlx::query_as(
            r#"SELECT * FROM "LiteLLM_DailyUserSpend"
               WHERE user_id = $1
               ORDER BY mcp_namespaced_tool_name, endpoint"#,
        )
        .bind(&first.entity_id)
        .fetch_all(&mut *tx)
        .await?;

        assert_eq!(rows.len(), 2);
        assert_eq!(
            rows[0].mcp_namespaced_tool_name.as_deref(),
            Some(first_tool)
        );
        assert_eq!(
            rows[1].mcp_namespaced_tool_name.as_deref(),
            Some(second_tool)
        );
        assert_eq!(
            (rows[0].prompt_tokens, rows[0].api_requests, rows[0].spend),
            (6, 2, 0.5)
        );
        assert_eq!(
            (rows[1].prompt_tokens, rows[1].api_requests, rows[1].spend),
            (14, 4, 1.0)
        );
        Ok(())
    }

    #[rstest]
    #[case::full_keys(
        "daily-user-base",
        "daily-key-base",
        "daily-model",
        "daily-provider",
        "/chat/completions",
        Some("first-group")
    )]
    #[case::initial_model_group_none(
        "daily-user-base",
        "daily-key-base",
        "daily-model",
        "daily-provider",
        "/chat/completions",
        None
    )]
    #[case::empty_optional_keys("", "", "", "", "", Some(""))]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn daily_user_spend_upsert_preserves_keys_and_accumulates(
        #[case] user_id: &'static str,
        #[case] api_key: &'static str,
        #[case] model: &'static str,
        #[case] custom_llm_provider: &'static str,
        #[case] endpoint: &'static str,
        #[case] first_model_group: Option<&'static str>,
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut tx = pool.begin().await?;

        #[expect(
            clippy::disallowed_methods,
            reason = "The integration test exercises UTC timestamp behavior under a non-UTC session"
        )]
        sqlx::query("SET LOCAL TIME ZONE 'America/Los_Angeles'")
            .execute(&mut *tx)
            .await?;

        let first_batch = [
            DailySpend {
                entity_id: user_id.to_owned(),
                date: "2026-01-15".to_owned(),
                api_key: api_key.to_owned(),
                model: model.to_owned(),
                custom_llm_provider: custom_llm_provider.to_owned(),
                mcp_namespaced_tool_name: String::new(),
                endpoint: endpoint.to_owned(),
                model_group: first_model_group.map(str::to_owned),
                prompt_tokens: 1,
                completion_tokens: 2,
                api_requests: 3,
                successful_requests: 4,
                failed_requests: 5,
                cache_read_input_tokens: 6,
                cache_creation_input_tokens: 7,
                total_response_time_ms: 8,
                timed_requests: 9,
                spend: 0.125,
            },
            DailySpend {
                entity_id: user_id.to_owned(),
                date: "2026-01-15".to_owned(),
                api_key: api_key.to_owned(),
                model: model.to_owned(),
                custom_llm_provider: custom_llm_provider.to_owned(),
                mcp_namespaced_tool_name: String::new(),
                endpoint: "endpoint-sentinel".to_owned(),
                model_group: Some("sentinel-endpoint-group".to_owned()),
                prompt_tokens: 101,
                completion_tokens: 102,
                api_requests: 103,
                successful_requests: 104,
                failed_requests: 105,
                cache_read_input_tokens: 106,
                cache_creation_input_tokens: 107,
                total_response_time_ms: 108,
                timed_requests: 109,
                spend: 1.25,
            },
            DailySpend {
                entity_id: user_id.to_owned(),
                date: "2026-01-15".to_owned(),
                api_key: "api-key-sentinel".to_owned(),
                model: model.to_owned(),
                custom_llm_provider: custom_llm_provider.to_owned(),
                mcp_namespaced_tool_name: String::new(),
                endpoint: endpoint.to_owned(),
                model_group: Some("sentinel-key-group".to_owned()),
                prompt_tokens: 201,
                completion_tokens: 202,
                api_requests: 203,
                successful_requests: 204,
                failed_requests: 205,
                cache_read_input_tokens: 206,
                cache_creation_input_tokens: 207,
                total_response_time_ms: 208,
                timed_requests: 209,
                spend: 2.5,
            },
            DailySpend {
                entity_id: "entity-sentinel".to_owned(),
                date: "2026-01-15".to_owned(),
                api_key: api_key.to_owned(),
                model: model.to_owned(),
                custom_llm_provider: custom_llm_provider.to_owned(),
                mcp_namespaced_tool_name: String::new(),
                endpoint: endpoint.to_owned(),
                model_group: Some("sentinel-entity-group".to_owned()),
                prompt_tokens: 301,
                completion_tokens: 302,
                api_requests: 303,
                successful_requests: 304,
                failed_requests: 305,
                cache_read_input_tokens: 306,
                cache_creation_input_tokens: 307,
                total_response_time_ms: 308,
                timed_requests: 309,
                spend: 3.75,
            },
        ];
        upsert_daily_user_spend(&mut *tx, &first_batch).await?;

        #[expect(
            clippy::disallowed_methods,
            reason = "The integration test captures the inserted row identity before conflict handling"
        )]
        let (initial_id, initial_created_at): (String, NaiveDateTime) = sqlx::query_as(
            r#"SELECT id, created_at
               FROM "LiteLLM_DailyUserSpend"
               WHERE user_id = $1 AND date = $2 AND api_key = $3 AND model = $4
                 AND custom_llm_provider = $5 AND endpoint = $6"#,
        )
        .bind(user_id)
        .bind("2026-01-15")
        .bind(api_key)
        .bind(model)
        .bind(custom_llm_provider)
        .bind(endpoint)
        .fetch_one(&mut *tx)
        .await?;
        assert!(!initial_id.is_empty());

        #[expect(
            clippy::disallowed_methods,
            reason = "The integration test seeds updated_at before exercising the conflict update"
        )]
        let seeded_updated_at: NaiveDateTime = sqlx::query_scalar(
            r#"UPDATE "LiteLLM_DailyUserSpend"
               SET updated_at = TIMESTAMP '2000-01-01'
               WHERE id = $1
               RETURNING updated_at"#,
        )
        .bind(&initial_id)
        .fetch_one(&mut *tx)
        .await?;

        let increment = [DailySpend {
            entity_id: user_id.to_owned(),
            date: "2026-01-15".to_owned(),
            api_key: api_key.to_owned(),
            model: model.to_owned(),
            custom_llm_provider: custom_llm_provider.to_owned(),
            mcp_namespaced_tool_name: String::new(),
            endpoint: endpoint.to_owned(),
            model_group: Some("second-group".to_owned()),
            prompt_tokens: 10,
            completion_tokens: 20,
            api_requests: 30,
            successful_requests: 40,
            failed_requests: 50,
            cache_read_input_tokens: 60,
            cache_creation_input_tokens: 70,
            total_response_time_ms: 80,
            timed_requests: 90,
            spend: 0.625,
        }];
        upsert_daily_user_spend(&mut *tx, &increment).await?;
        upsert_daily_user_spend(&mut *tx, &[]).await?;

        #[expect(
            clippy::disallowed_methods,
            reason = "The integration test captures the transaction timestamp for assertions"
        )]
        let expected_updated_at: NaiveDateTime =
            sqlx::query_scalar("SELECT (NOW() AT TIME ZONE 'UTC')::timestamp(3)")
                .fetch_one(&mut *tx)
                .await?;

        #[expect(
            clippy::disallowed_methods,
            reason = "The integration test reads all rows to verify attribution and accumulation"
        )]
        let rows: Vec<DailyUserRow> = sqlx::query_as(
            r#"SELECT id, user_id, api_key, model, custom_llm_provider, endpoint,
                      model_group, mcp_namespaced_tool_name, prompt_tokens,
                      completion_tokens, api_requests, successful_requests,
                      failed_requests, cache_read_input_tokens,
                      cache_creation_input_tokens, total_response_time_ms,
                      timed_requests, spend, date, created_at, updated_at
               FROM "LiteLLM_DailyUserSpend"
               WHERE date = $1 AND model = $2 AND custom_llm_provider = $3
                 AND mcp_namespaced_tool_name = '' AND user_id = ANY($4)
               ORDER BY CASE
               WHEN user_id = $5 AND api_key = $6 AND endpoint = $7 THEN 0
                   WHEN user_id = $5 AND api_key = $6 AND endpoint = $9 THEN 1
                   WHEN user_id = $5 AND api_key = $8 AND endpoint = $7 THEN 2
                   WHEN user_id = $10 AND api_key = $6 AND endpoint = $7 THEN 3
                   ELSE 4
               END"#,
        )
        .bind("2026-01-15")
        .bind(model)
        .bind(custom_llm_provider)
        .bind(vec![user_id, "entity-sentinel"])
        .bind(user_id)
        .bind(api_key)
        .bind(endpoint)
        .bind("api-key-sentinel")
        .bind("endpoint-sentinel")
        .bind("entity-sentinel")
        .fetch_all(&mut *tx)
        .await?;

        assert_eq!(rows.len(), 4);
        assert_eq!(rows[0].id, initial_id);
        assert_eq!(rows[0].user_id.as_deref(), Some(user_id));
        assert_eq!(rows[0].api_key, api_key);
        assert_eq!(rows[0].model.as_deref(), Some(model));
        assert_eq!(
            rows[0].custom_llm_provider.as_deref(),
            Some(custom_llm_provider)
        );
        assert_eq!(rows[0].endpoint.as_deref(), Some(endpoint));
        assert_eq!(rows[0].model_group.as_deref(), first_model_group);
        assert_eq!(rows[0].mcp_namespaced_tool_name.as_deref(), Some(""));
        assert_eq!(
            (
                rows[0].prompt_tokens,
                rows[0].completion_tokens,
                rows[0].api_requests,
                rows[0].successful_requests,
                rows[0].failed_requests,
                rows[0].cache_read_input_tokens,
                rows[0].cache_creation_input_tokens,
                rows[0].total_response_time_ms,
                rows[0].timed_requests,
                rows[0].spend
            ),
            (11, 22, 33, 44, 55, 66, 77, 88, 99, 0.75)
        );
        assert_eq!(rows[0].date, "2026-01-15");
        assert_eq!(rows[0].created_at, initial_created_at);
        assert!(rows[0].updated_at > seeded_updated_at);
        assert_eq!(rows[0].updated_at, expected_updated_at);

        assert!(!rows[1].id.is_empty());
        assert_eq!(rows[1].user_id.as_deref(), Some(user_id));
        assert_eq!(rows[1].api_key, api_key);
        assert_eq!(rows[1].model.as_deref(), Some(model));
        assert_eq!(
            rows[1].custom_llm_provider.as_deref(),
            Some(custom_llm_provider)
        );
        assert_eq!(rows[1].endpoint.as_deref(), Some("endpoint-sentinel"));
        assert_eq!(
            rows[1].model_group.as_deref(),
            Some("sentinel-endpoint-group")
        );
        assert_eq!(rows[1].mcp_namespaced_tool_name.as_deref(), Some(""));
        assert_eq!(
            (
                rows[1].prompt_tokens,
                rows[1].completion_tokens,
                rows[1].api_requests,
                rows[1].successful_requests,
                rows[1].failed_requests,
                rows[1].cache_read_input_tokens,
                rows[1].cache_creation_input_tokens,
                rows[1].total_response_time_ms,
                rows[1].timed_requests,
                rows[1].spend
            ),
            (101, 102, 103, 104, 105, 106, 107, 108, 109, 1.25)
        );
        assert_eq!(rows[1].date, "2026-01-15");
        assert!(rows[1].updated_at > seeded_updated_at);
        assert_eq!(rows[1].updated_at, expected_updated_at);

        assert!(!rows[2].id.is_empty());
        assert_eq!(rows[2].user_id.as_deref(), Some(user_id));
        assert_eq!(rows[2].api_key, "api-key-sentinel");
        assert_eq!(rows[2].model.as_deref(), Some(model));
        assert_eq!(
            rows[2].custom_llm_provider.as_deref(),
            Some(custom_llm_provider)
        );
        assert_eq!(rows[2].endpoint.as_deref(), Some(endpoint));
        assert_eq!(rows[2].model_group.as_deref(), Some("sentinel-key-group"));
        assert_eq!(rows[2].mcp_namespaced_tool_name.as_deref(), Some(""));
        assert_eq!(
            (
                rows[2].prompt_tokens,
                rows[2].completion_tokens,
                rows[2].api_requests,
                rows[2].successful_requests,
                rows[2].failed_requests,
                rows[2].cache_read_input_tokens,
                rows[2].cache_creation_input_tokens,
                rows[2].total_response_time_ms,
                rows[2].timed_requests,
                rows[2].spend
            ),
            (201, 202, 203, 204, 205, 206, 207, 208, 209, 2.5)
        );
        assert_eq!(rows[2].date, "2026-01-15");
        assert!(rows[2].updated_at > seeded_updated_at);
        assert_eq!(rows[2].updated_at, expected_updated_at);

        assert!(!rows[3].id.is_empty());
        assert_eq!(rows[3].user_id.as_deref(), Some("entity-sentinel"));
        assert_eq!(rows[3].api_key, api_key);
        assert_eq!(rows[3].model.as_deref(), Some(model));
        assert_eq!(
            rows[3].custom_llm_provider.as_deref(),
            Some(custom_llm_provider)
        );
        assert_eq!(rows[3].endpoint.as_deref(), Some(endpoint));
        assert_eq!(
            rows[3].model_group.as_deref(),
            Some("sentinel-entity-group")
        );
        assert_eq!(rows[3].mcp_namespaced_tool_name.as_deref(), Some(""));
        assert_eq!(
            (
                rows[3].prompt_tokens,
                rows[3].completion_tokens,
                rows[3].api_requests,
                rows[3].successful_requests,
                rows[3].failed_requests,
                rows[3].cache_read_input_tokens,
                rows[3].cache_creation_input_tokens,
                rows[3].total_response_time_ms,
                rows[3].timed_requests,
                rows[3].spend
            ),
            (301, 302, 303, 304, 305, 306, 307, 308, 309, 3.75)
        );
        assert_eq!(rows[3].date, "2026-01-15");
        assert!(rows[3].updated_at > seeded_updated_at);
        assert_eq!(rows[3].updated_at, expected_updated_at);
        Ok(())
    }
}
