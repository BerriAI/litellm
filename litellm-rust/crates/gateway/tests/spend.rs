//! The spend writer against a real proxy database: requests through the gateway end up in the
//! proxy's spend logs, entity spend and daily activity tables.

use axum::{
    Router,
    body::{Body, to_bytes},
    http::{Request, header::CONTENT_TYPE},
    response::IntoResponse,
};
use litellm_auth_types::SecretValue;
use litellm_config::Config;
use litellm_db::PgPool;
use litellm_db_testing::MigratedPostgres;
use litellm_gateway::spend::{SpendSettings, SpendWriter};
use litellm_gateway_auth::keys::KeyHash;
use rstest::rstest;
use serde_json::{Value, json};
use tokio::net::TcpListener;
use tower::ServiceExt;

const MESSAGE: &str = r#"{"id":"msg_1","type":"message","role":"assistant","model":"claude-x","content":[{"type":"text","text":"hi"}],"stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":3,"cache_read_input_tokens":2,"output_tokens":4}}"#;
const COMPLETION: &str = r#"{"id":"chatcmpl-1","object":"chat.completion","created":1,"model":"gpt-x","choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":6,"total_tokens":11}}"#;

async fn upstream() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let app = Router::new().fallback(|request: Request<Body>| async move {
        let payload = match request.uri().path().ends_with("/chat/completions") {
            true => COMPLETION,
            false => MESSAGE,
        };
        ([(CONTENT_TYPE, "application/json")], payload).into_response()
    });
    tokio::spawn(async move { axum::serve(listener, app).await });
    format!("http://{address}")
}

fn hash(token: &str) -> String {
    KeyHash::from_token(&SecretValue::new(token))
        .as_str()
        .to_owned()
}

#[expect(
    clippy::disallowed_methods,
    reason = "test fixture rows; the writes under test use the checked macros"
)]
async fn fixtures(pool: &PgPool) {
    let sql = format!(
        r#"
INSERT INTO "LiteLLM_TeamTable" (team_id, team_alias, members_with_roles, metadata, created_at, updated_at)
  VALUES ('team-1', 'research', '[{{"user_id": "user-1", "role": "user"}}]',
          '{{"spend_logs_metadata": {{"project": "team", "cost_center": "42"}}}}', now(), now());
INSERT INTO "LiteLLM_UserTable" (user_id, spend) VALUES ('user-1', 0);
INSERT INTO "LiteLLM_VerificationToken" (token, key_alias, spend, team_id, user_id, metadata)
  VALUES ('{}', 'team-key', 0, 'team-1', 'user-1',
          '{{"spend_logs_metadata": {{"project": "alpha"}}}}');
"#,
        hash("sk-team")
    );
    sqlx::raw_sql(sqlx::AssertSqlSafe(sql))
        .execute(pool)
        .await
        .unwrap();
}

#[expect(
    clippy::disallowed_methods,
    reason = "test assertions read rows back; the writes under test use the checked macros"
)]
async fn rows(pool: &PgPool, sql: &str) -> Vec<Value> {
    sqlx::query_scalar::<_, Value>(sqlx::AssertSqlSafe(format!(
        "SELECT to_jsonb(r) FROM ({sql}) r"
    )))
    .fetch_all(pool)
    .await
    .unwrap()
}

async fn send(app: &Router, path: &str, key: &str, body: Value) {
    let response = app
        .clone()
        .oneshot(
            Request::post(path)
                .header("content-type", "application/json")
                .header("authorization", format!("Bearer {key}"))
                .header("user-agent", "spend-test/1.0")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    let body = to_bytes(response.into_body(), 1 << 20).await.unwrap();
    assert_eq!(status, 200, "{body:?}");
}

fn close(left: &Value, right: f64) -> bool {
    (left.as_f64().unwrap() - right).abs() < 1e-9
}

#[rstest]
#[tokio::test]
async fn requests_are_written_as_the_proxy_writes_them() {
    let postgres = MigratedPostgres::start().await.unwrap();
    let pool = postgres.pool();
    fixtures(pool).await;
    let upstream = upstream().await;
    let config = Config::from_yaml(&format!(
        "model_list:
  - model_name: claude
    litellm_params:
      model: anthropic/claude-x
      api_base: {upstream}
      api_key: sk-upstream
      input_cost_per_token: 0.001
      output_cost_per_token: 0.002
      cache_read_input_token_cost: 0.0001
    model_info:
      id: claude-1
  - model_name: gpt
    litellm_params:
      model: openai/gpt-x
      api_base: {upstream}/v1
      api_key: sk-upstream
      input_cost_per_token: 1
      output_cost_per_token: 2
    model_info:
      id: gpt-1
router_settings:
  num_retries: 0
general_settings:
  master_key: sk-master
  database_url: '{}'
",
        postgres.url()
    ))
    .unwrap();
    let inference = litellm_gateway::build_inference(&config).unwrap();
    let database = litellm_gateway::build_database(&config, inference.secrets.as_ref())
        .await
        .unwrap()
        .unwrap();
    let keys = litellm_gateway::build_keys_on(&config, database.clone()).unwrap();
    let settings = SpendSettings::from_config(&config).unwrap();
    litellm_gateway::hooks::install(
        &inference,
        &config,
        Vec::new(),
        vec![SpendWriter::start(database, settings)],
    )
    .await
    .unwrap();
    let app = litellm_gateway::router_with_keys(inference.clone(), &config, None, None, Some(keys));

    let message = json!({
        "model": "claude",
        "max_tokens": 8,
        "messages": [{"role": "user", "content": "hi"}],
        "metadata": {"user_id": "customer-7"},
    });
    send(&app, "/v1/messages", "sk-team", message.clone()).await;
    send(&app, "/v1/messages", "sk-team", message).await;
    let chat = json!({"model": "gpt", "messages": [{"role": "user", "content": "hi"}]});
    send(&app, "/v1/chat/completions", "sk-master", chat).await;
    litellm_gateway::hooks::close(&inference).await;

    let message_cost = 3.0 * 0.001 + 2.0 * 0.0001 + 4.0 * 0.002;
    let logs = rows(
        pool,
        r#"SELECT call_type, api_key, spend, prompt_tokens, completion_tokens, total_tokens,
                  model, model_id, model_group, custom_llm_provider, "user", team_id, end_user,
                  cache_hit, cache_key, status, metadata
           FROM "LiteLLM_SpendLogs" ORDER BY call_type DESC, "startTime""#,
    )
    .await;
    assert_eq!(logs.len(), 3, "{logs:#?}");
    let [message_log, _, chat_log] = logs.as_slice() else {
        unreachable!()
    };
    assert!(
        close(&message_log["spend"], message_cost),
        "{message_log:#}"
    );
    let mut message_log = message_log.clone();
    let metadata = message_log
        .as_object_mut()
        .unwrap()
        .remove("metadata")
        .unwrap();
    message_log.as_object_mut().unwrap().remove("spend");
    assert_eq!(
        message_log,
        json!({
            "call_type": "anthropic_messages",
            "api_key": hash("sk-team"),
            "prompt_tokens": 5,
            "completion_tokens": 4,
            "total_tokens": 9,
            "model": "claude-x",
            "model_id": "claude-1",
            "model_group": "claude",
            "custom_llm_provider": "anthropic",
            "user": "user-1",
            "team_id": "team-1",
            "end_user": "customer-7",
            "cache_hit": "None",
            "cache_key": "Cache OFF",
            "status": "success",
        })
    );
    assert_eq!(metadata.as_object().unwrap().len(), 49, "{metadata:#}");
    assert_eq!(metadata["user_api_key_alias"], "team-key");
    assert_eq!(metadata["user_api_key_team_alias"], "research");
    assert_eq!(metadata["user_agent"], "spend-test/1.0");
    assert_eq!(metadata["status"], Value::Null);
    assert_eq!(
        metadata["spend_logs_metadata"],
        json!({"project": "alpha", "cost_center": "42"})
    );
    assert_eq!(chat_log["call_type"], "acompletion");
    assert_eq!(chat_log["user"], "default_user_id");
    assert_eq!(chat_log["spend"], 17.0);

    let key = rows(
        pool,
        &format!(
            r#"SELECT spend FROM "LiteLLM_VerificationToken" WHERE token = '{}'"#,
            hash("sk-team")
        ),
    )
    .await;
    assert!(close(&key[0]["spend"], 2.0 * message_cost), "{key:?}");
    for sql in [
        r#"SELECT spend FROM "LiteLLM_UserTable" WHERE user_id = 'user-1'"#,
        r#"SELECT spend FROM "LiteLLM_TeamTable" WHERE team_id = 'team-1'"#,
        r#"SELECT spend FROM "LiteLLM_TeamMembership" WHERE team_id = 'team-1'"#,
        r#"SELECT spend FROM "LiteLLM_EndUserTable" WHERE user_id = 'customer-7'"#,
    ] {
        let row = rows(pool, sql).await;
        assert!(
            close(&row[0]["spend"], 2.0 * message_cost),
            "{sql}: {row:?}"
        );
    }

    let daily = rows(
        pool,
        r#"SELECT user_id, api_key, model, model_group, endpoint, api_requests,
                  successful_requests, prompt_tokens, completion_tokens, cache_read_input_tokens
           FROM "LiteLLM_DailyUserSpend" ORDER BY user_id"#,
    )
    .await;
    assert_eq!(
        daily,
        [
            json!({
                "user_id": "default_user_id",
                "api_key": hash("sk-master"),
                "model": "gpt-x",
                "model_group": "gpt",
                "endpoint": "/chat/completions",
                "api_requests": 1,
                "successful_requests": 1,
                "prompt_tokens": 5,
                "completion_tokens": 6,
                "cache_read_input_tokens": 0,
            }),
            json!({
                "user_id": "user-1",
                "api_key": hash("sk-team"),
                "model": "claude-x",
                "model_group": "claude",
                "endpoint": "",
                "api_requests": 2,
                "successful_requests": 2,
                "prompt_tokens": 10,
                "completion_tokens": 8,
                "cache_read_input_tokens": 4,
            }),
        ]
    );
    let teams = rows(
        pool,
        r#"SELECT team_id, api_requests FROM "LiteLLM_DailyTeamSpend" ORDER BY team_id"#,
    )
    .await;
    assert_eq!(
        teams,
        [
            json!({"team_id": "", "api_requests": 1}),
            json!({"team_id": "team-1", "api_requests": 2}),
        ]
    );
    let end_users = rows(
        pool,
        r#"SELECT end_user_id, api_requests FROM "LiteLLM_DailyEndUserSpend""#,
    )
    .await;
    assert_eq!(
        end_users,
        [json!({"end_user_id": "customer-7", "api_requests": 2})]
    );
}
