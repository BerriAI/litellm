use litellm_db::{
    DailySpend, DailyTable, EntitySpend, SpendLog, TeamMemberSpend, add_end_user_spend,
    add_key_spend, add_team_member_spend, add_team_spend, add_user_spend, insert_spend_logs,
    upsert_daily_spend,
};
use litellm_db_testing::MigratedPostgres;
use rstest::rstest;
use serde_json::{Value, json};
use sqlx::PgPool;

const FIXTURES: &str = r#"
INSERT INTO "LiteLLM_TeamTable" (team_id, members_with_roles, created_at, updated_at)
  VALUES ('team-1', '[{"user_id": "user-1", "role": "user"}]', now(), now());
INSERT INTO "LiteLLM_UserTable" (user_id, spend) VALUES ('user-1', 1.0);
INSERT INTO "LiteLLM_VerificationToken" (token, spend, total_spend, team_id, user_id)
  VALUES ('hash-1', 2.0, 2.0, 'team-1', 'user-1');
"#;

#[expect(
    clippy::disallowed_methods,
    reason = "test fixture rows and assertions; the writes under test use the checked macro"
)]
async fn setup(pool: &PgPool) {
    sqlx::raw_sql(FIXTURES).execute(pool).await.unwrap();
}

#[expect(
    clippy::disallowed_methods,
    reason = "test assertions read rows back; the writes under test use the checked macro"
)]
async fn read(pool: &PgPool, sql: &str) -> Value {
    sqlx::query_scalar::<_, Value>(sqlx::AssertSqlSafe(format!(
        "SELECT to_jsonb(r) FROM ({sql}) r"
    )))
    .fetch_one(pool)
    .await
    .unwrap()
}

#[expect(
    clippy::disallowed_methods,
    reason = "test assertions count rows; the writes under test use the checked macro"
)]
async fn count(pool: &PgPool, sql: &str) -> i64 {
    sqlx::query_scalar::<_, i64>(sqlx::AssertSqlSafe(sql.to_owned()))
        .fetch_one(pool)
        .await
        .unwrap()
}

fn spend(entries: &[(&str, f64)]) -> EntitySpend {
    entries
        .iter()
        .map(|(id, cost)| ((*id).to_owned(), *cost))
        .collect()
}

#[rstest]
#[tokio::test]
async fn entity_spend_adds_up_and_creates_only_what_the_proxy_creates() {
    let postgres = MigratedPostgres::start().await.unwrap();
    let pool = postgres.pool();
    setup(pool).await;

    for _ in 0..2 {
        add_key_spend(pool, &spend(&[("hash-1", 0.25), ("missing", 1.0)]))
            .await
            .unwrap();
        add_user_spend(pool, &spend(&[("user-1", 0.5), ("missing", 1.0)]))
            .await
            .unwrap();
        add_team_spend(pool, &spend(&[("team-1", 0.125)]))
            .await
            .unwrap();
        add_end_user_spend(pool, &spend(&[("end-1", 0.75)]))
            .await
            .unwrap();
        add_team_member_spend(
            pool,
            &[
                TeamMemberSpend {
                    team_id: "team-1".into(),
                    user_id: "user-1".into(),
                    cost: 0.5,
                },
                TeamMemberSpend {
                    team_id: "team-1".into(),
                    user_id: "stranger".into(),
                    cost: 9.0,
                },
            ],
        )
        .await
        .unwrap();
    }

    let key = read(
        pool,
        r#"SELECT spend, total_spend, last_active IS NOT NULL AS active
           FROM "LiteLLM_VerificationToken" WHERE token = 'hash-1'"#,
    )
    .await;
    assert_eq!(
        key,
        json!({"spend": 2.5, "total_spend": 2.5, "active": true})
    );
    let user = read(
        pool,
        r#"SELECT spend FROM "LiteLLM_UserTable" WHERE user_id = 'user-1'"#,
    )
    .await;
    assert_eq!(user, json!({"spend": 2}));
    let team = read(
        pool,
        r#"SELECT spend FROM "LiteLLM_TeamTable" WHERE team_id = 'team-1'"#,
    )
    .await;
    assert_eq!(team, json!({"spend": 0.25}));
    let end_user = read(
        pool,
        r#"SELECT spend, blocked FROM "LiteLLM_EndUserTable" WHERE user_id = 'end-1'"#,
    )
    .await;
    assert_eq!(end_user, json!({"spend": 1.5, "blocked": false}));
    let member = read(
        pool,
        r#"SELECT spend, total_spend FROM "LiteLLM_TeamMembership"
           WHERE team_id = 'team-1' AND user_id = 'user-1'"#,
    )
    .await;
    assert_eq!(member, json!({"spend": 1, "total_spend": 1}));
    assert_eq!(
        count(pool, r#"SELECT count(*) FROM "LiteLLM_UserTable""#).await,
        1
    );
    assert_eq!(
        count(
            pool,
            r#"SELECT count(*) FROM "LiteLLM_TeamMembership" WHERE user_id = 'stranger'"#
        )
        .await,
        0
    );
}

fn log(request_id: &str) -> SpendLog {
    SpendLog {
        request_id: request_id.into(),
        call_type: "anthropic_messages".into(),
        api_key: "hash-1".into(),
        spend: 0.5,
        total_tokens: 30,
        prompt_tokens: 10,
        completion_tokens: 20,
        start_time: "2026-10-17 01:02:03.456".into(),
        end_time: "2026-10-17 01:02:04.000".into(),
        completion_start_time: Some("2026-10-17 01:02:03.500".into()),
        request_duration_ms: Some(544),
        model: "claude-x".into(),
        model_group: "claude".into(),
        custom_llm_provider: "anthropic".into(),
        metadata: json!({"user_api_key": "hash-1", "status": null}),
        cache_hit: "None".into(),
        cache_key: "Cache OFF".into(),
        status: "success".into(),
        ..SpendLog::default()
    }
}

#[rstest]
#[tokio::test]
async fn spend_logs_are_written_once() {
    let postgres = MigratedPostgres::start().await.unwrap();
    let pool = postgres.pool();
    insert_spend_logs(pool, &[log("a"), log("b")])
        .await
        .unwrap();
    insert_spend_logs(pool, &[log("a")]).await.unwrap();

    assert_eq!(
        count(pool, r#"SELECT count(*) FROM "LiteLLM_SpendLogs""#).await,
        2
    );
    let row = read(
        pool,
        r#"SELECT "startTime", "completionStartTime", request_duration_ms, team_id, end_user,
                  metadata, request_tags, messages, response, cache_hit, status
           FROM "LiteLLM_SpendLogs" WHERE request_id = 'a'"#,
    )
    .await;
    assert_eq!(
        row,
        json!({
            "startTime": "2026-10-17T01:02:03.456",
            "completionStartTime": "2026-10-17T01:02:03.5",
            "request_duration_ms": 544,
            "team_id": "",
            "end_user": "",
            "metadata": {"user_api_key": "hash-1", "status": null},
            "request_tags": [],
            "messages": {},
            "response": {},
            "cache_hit": "None",
            "status": "success",
        })
    );
}

fn daily(entity: &str, model_group: &str) -> DailySpend {
    DailySpend {
        entity_id: entity.into(),
        date: "2026-10-17".into(),
        api_key: "hash-1".into(),
        model: "claude-x".into(),
        custom_llm_provider: "anthropic".into(),
        endpoint: String::new(),
        model_group: Some(model_group.into()),
        prompt_tokens: 10,
        completion_tokens: 20,
        api_requests: 1,
        successful_requests: 1,
        cache_read_input_tokens: 3,
        total_response_time_ms: 100,
        timed_requests: 1,
        spend: 0.5,
        ..DailySpend::default()
    }
}

#[rstest]
#[case::user(DailyTable::User, "LiteLLM_DailyUserSpend", "user_id")]
#[case::team(DailyTable::Team, "LiteLLM_DailyTeamSpend", "team_id")]
#[case::end_user(DailyTable::EndUser, "LiteLLM_DailyEndUserSpend", "end_user_id")]
#[tokio::test]
async fn daily_rows_add_up_and_keep_their_first_model_group(
    #[case] table: DailyTable,
    #[case] name: &str,
    #[case] entity: &str,
) {
    let postgres = MigratedPostgres::start().await.unwrap();
    let pool = postgres.pool();
    upsert_daily_spend(pool, table, &[daily("e-1", "first"), daily("e-2", "first")])
        .await
        .unwrap();
    upsert_daily_spend(pool, table, &[daily("e-1", "second")])
        .await
        .unwrap();

    let row = read(
        pool,
        &format!(
            r#"SELECT model_group, prompt_tokens, completion_tokens, api_requests,
                      successful_requests, failed_requests, cache_read_input_tokens,
                      total_response_time_ms, timed_requests, spend, endpoint,
                      mcp_namespaced_tool_name
               FROM "{name}" WHERE {entity} = 'e-1'"#
        ),
    )
    .await;
    assert_eq!(
        row,
        json!({
            "model_group": "first",
            "prompt_tokens": 20,
            "completion_tokens": 40,
            "api_requests": 2,
            "successful_requests": 2,
            "failed_requests": 0,
            "cache_read_input_tokens": 6,
            "total_response_time_ms": 200,
            "timed_requests": 2,
            "spend": 1,
            "endpoint": "",
            "mcp_namespaced_tool_name": "",
        })
    );
    assert_eq!(
        count(pool, &format!(r#"SELECT count(*) FROM "{name}""#)).await,
        2
    );
}
