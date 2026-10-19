//! The budgets hook on real requests: refusals as the proxy words them, reservations that keep
//! concurrent spend under a budget, and counters shared through Redis. Set
//! `LITELLM_TEST_REDIS_URL` to also run the shared cases against a live Redis.

use std::{
    collections::HashMap,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

use axum::{
    Router,
    body::{Body, to_bytes},
    http::{Request, header::CONTENT_TYPE},
    response::IntoResponse,
};
use litellm_auth_types::SecretValue;
use litellm_config::Config;
use litellm_gateway::{
    budgets::{BudgetSettings, Budgets},
    coordination::{Coordination, RedisSettings},
};
use litellm_gateway_auth::keys::{
    AllowedRoutes, KeyBudgets, KeyHash, KeyLookup, KeyModels, KeyProfile, KeyStatus, LookupFuture,
    MemberSpend, Spend,
};
use rstest::rstest;
use serde_json::{Value, json};
use tokio::net::TcpListener;
use tower::ServiceExt;

/// Costs 3 * 0.001 + 4 * 0.002 = 0.011 at the test prices.
const MESSAGE: &str = r#"{"id":"msg_1","type":"message","role":"assistant","model":"claude-x","content":[{"type":"text","text":"hi"}],"stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":3,"output_tokens":4}}"#;
const COST: f64 = 0.011;

async fn upstream() -> (String, Arc<AtomicUsize>) {
    let calls = Arc::new(AtomicUsize::new(0));
    let counted = calls.clone();
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let app = Router::new().fallback(move |_: Request<Body>| {
        let counted = counted.clone();
        async move {
            counted.fetch_add(1, Ordering::SeqCst);
            ([(CONTENT_TYPE, "application/json")], MESSAGE).into_response()
        }
    });
    tokio::spawn(async move { axum::serve(listener, app).await });
    (format!("http://{address}"), calls)
}

struct Keys(HashMap<KeyHash, KeyStatus>);

impl KeyLookup for Keys {
    fn lookup<'a>(&'a self, hash: &'a KeyHash) -> LookupFuture<'a> {
        Box::pin(async move { Ok(self.0.get(hash).cloned()) })
    }
}

fn hash(token: &str) -> KeyHash {
    KeyHash::from_token(&SecretValue::new(token))
}

fn key(team_id: Option<&str>, budgets: KeyBudgets) -> KeyStatus {
    KeyStatus::Active {
        expires_at: None,
        models: KeyModels::default(),
        allowed_routes: AllowedRoutes::default(),
        team_id: team_id.map(str::to_owned),
        profile: Arc::new(KeyProfile {
            alias: Some("prod".into()),
            user_id: Some("user-1".into()),
            budgets,
            ..KeyProfile::default()
        }),
    }
}

fn budget(spend: f64, max_budget: f64) -> Spend {
    Spend {
        spend,
        max_budget: Some(max_budget),
    }
}

struct Gateway {
    app: Router,
    calls: Arc<AtomicUsize>,
}

async fn gateway(
    keys: Vec<(&str, KeyStatus)>,
    redis: Option<Arc<Coordination>>,
    settings: &str,
) -> Gateway {
    gateway_with_database(keys, redis, None, settings).await
}

async fn gateway_with_database(
    keys: Vec<(&str, KeyStatus)>,
    redis: Option<Arc<Coordination>>,
    database: Option<litellm_db::PgPool>,
    settings: &str,
) -> Gateway {
    let (upstream, calls) = upstream().await;
    let config = Config::from_yaml(&format!(
        "model_list:
  - model_name: claude
    litellm_params:
      model: anthropic/claude-x
      api_base: {upstream}
      api_key: sk-upstream
      input_cost_per_token: 0.001
      output_cost_per_token: 0.002
    model_info:
      id: claude-1
  - model_name: free
    litellm_params:
      model: anthropic/claude-x
      api_base: {upstream}
      api_key: sk-upstream
      input_cost_per_token: 0
      output_cost_per_token: 0
    model_info:
      id: free-1
router_settings:
  num_retries: 0
general_settings:
  master_key: sk-master
{settings}"
    ))
    .unwrap();
    let inference = litellm_gateway::build_inference(&config).unwrap();
    let budgets = Budgets::for_gateway(&config, &inference, redis, database);
    litellm_gateway::hooks::install(&inference, &config, vec![Arc::new(budgets)], Vec::new())
        .await
        .unwrap();
    let keys: Arc<dyn KeyLookup> = Arc::new(Keys(
        keys.into_iter()
            .map(|(token, status)| (hash(token), status))
            .collect(),
    ));
    let app = litellm_gateway::router_with_keys(inference, &config, None, None, Some(keys));
    Gateway { app, calls }
}

async fn send(app: &Router, token: &str, model: &str) -> (u16, Value) {
    send_as(app, token, model, None).await
}

async fn send_as(app: &Router, token: &str, model: &str, end_user: Option<&str>) -> (u16, Value) {
    let mut body = json!({
        "model": model,
        "max_tokens": 8,
        "messages": [{"role": "user", "content": "hi"}],
    });
    if let Some(end_user) = end_user {
        body["metadata"] = json!({"user_id": end_user});
    }
    send_body(app, token, body).await
}

async fn send_body(app: &Router, token: &str, body: Value) -> (u16, Value) {
    let response = app
        .clone()
        .oneshot(
            Request::post("/v1/messages")
                .header("content-type", "application/json")
                .header("authorization", format!("Bearer {token}"))
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status().as_u16();
    let body = to_bytes(response.into_body(), 1 << 20).await.unwrap();
    (status, serde_json::from_slice(&body).unwrap_or(Value::Null))
}

/// Settling runs after the response, off the request path.
async fn settled() {
    tokio::time::sleep(Duration::from_millis(100)).await;
}

#[rstest]
#[case::key_at_budget(
    None,
    KeyBudgets { key: budget(10.0, 10.0), ..KeyBudgets::default() },
    "Budget has been exceeded! Key=prod Current cost: 10.0, Max budget: 10.0"
)]
#[case::zero_key_budget(
    None,
    KeyBudgets { key: budget(0.0, 0.0), ..KeyBudgets::default() },
    "Budget has been exceeded! Key=prod Current cost: 0.0, Max budget: 0.0"
)]
#[case::team_over_budget(
    Some("team-1"),
    KeyBudgets { team: Some(budget(5.25, 5.0)), ..KeyBudgets::default() },
    "Budget has been exceeded! Team=team-1 Current cost: 5.25, Max budget: 5.0"
)]
#[case::personal_user_over_budget(
    None,
    KeyBudgets { user: Some(budget(3.0, 3.0)), ..KeyBudgets::default() },
    "ExceededBudget: User=user-1 over budget. Spend=3.0, Budget=3.0"
)]
#[case::member_before_key(
    Some("team-1"),
    KeyBudgets {
        key: budget(10.0, 10.0),
        team_member: Some(MemberSpend { spend: 2.0, max_budget: Some(1.0), default_max_budget: None }),
        ..KeyBudgets::default()
    },
    "Budget has been exceeded! TeamMember=user-1:team-1 Current cost: 2.0, Max budget: 1.0"
)]
#[case::zero_member_budget(
    Some("team-1"),
    KeyBudgets {
        team_member: Some(MemberSpend { spend: 0.0, max_budget: Some(0.0), default_max_budget: None }),
        ..KeyBudgets::default()
    },
    "Budget has been exceeded! User=user-1 in Team=team-1 Current cost: 0.0, Max budget: 0.0"
)]
#[case::team_default_member_budget(
    Some("team-1"),
    KeyBudgets {
        team_member: Some(MemberSpend { spend: 4.0, max_budget: None, default_max_budget: Some(4.0) }),
        ..KeyBudgets::default()
    },
    "Budget has been exceeded! TeamMember=user-1:team-1 Current cost: 4.0, Max budget: 4.0"
)]
#[tokio::test]
async fn over_budget_requests_are_refused_like_the_proxy(
    #[case] team_id: Option<&str>,
    #[case] budgets: KeyBudgets,
    #[case] message: &str,
) {
    let gateway = gateway(vec![("sk-key", key(team_id, budgets))], None, "").await;
    let (status, body) = send(&gateway.app, "sk-key", "claude").await;
    assert_eq!(status, 422);
    assert_eq!(
        body,
        json!({"error": {"message": message, "type": "budget_exceeded", "param": null, "code": "422"}})
    );
    assert_eq!(gateway.calls.load(Ordering::SeqCst), 0);
}

#[rstest]
#[case::team_at_budget(
    Some("team-1"),
    KeyBudgets { team: Some(budget(5.0, 5.0)), ..KeyBudgets::default() }
)]
#[case::user_budget_on_team_key(
    Some("team-1"),
    KeyBudgets { user: Some(budget(3.0, 3.0)), ..KeyBudgets::default() }
)]
#[tokio::test]
async fn budgets_the_proxy_does_not_apply_pass(
    #[case] team_id: Option<&str>,
    #[case] budgets: KeyBudgets,
) {
    // A team exactly at its budget passes the proxy's check; only a reservation would refuse it.
    let gateway = gateway(
        vec![("sk-key", key(team_id, budgets))],
        None,
        "  disable_budget_reservation: true\n",
    )
    .await;
    assert_eq!(send(&gateway.app, "sk-key", "claude").await.0, 200);
}

#[rstest]
#[tokio::test]
async fn free_models_skip_budgets_and_the_status_code_is_configurable() {
    let budgets = KeyBudgets {
        key: budget(10.0, 10.0),
        ..KeyBudgets::default()
    };
    let gateway = gateway(
        vec![("sk-key", key(None, budgets))],
        None,
        "litellm_settings:\n  budget_exceeded_status_code: 429\n",
    )
    .await;
    assert_eq!(send(&gateway.app, "sk-key", "free").await.0, 200);
    assert_eq!(send(&gateway.app, "sk-key", "claude").await.0, 429);
}

/// Spends a key with a 0.02 budget down: the first request fits, the second's reservation
/// shrinks to what is left, and the third finds the budget spent.
async fn spend_down(app: &Router) {
    assert_eq!(send(app, "sk-key", "claude").await.0, 200);
    settled().await;
    assert_eq!(send(app, "sk-key", "claude").await.0, 200);
    settled().await;
    let (status, body) = send(app, "sk-key", "claude").await;
    assert_eq!(status, 422, "{body}");
    let message = body["error"]["message"].as_str().unwrap();
    assert!(
        message.starts_with("Budget has been exceeded! Key=prod Current cost: 0.02"),
        "{message}"
    );
}

fn budgeted_key() -> KeyStatus {
    key(
        None,
        KeyBudgets {
            key: budget(0.0, 0.02),
            ..KeyBudgets::default()
        },
    )
}

#[rstest]
#[tokio::test]
async fn spend_is_counted_on_the_replica_without_redis() {
    let gateway = gateway(vec![("sk-key", budgeted_key())], None, "").await;
    spend_down(&gateway.app).await;
    assert_eq!(gateway.calls.load(Ordering::SeqCst), 2);
}

#[rstest]
#[tokio::test]
async fn reservations_hold_back_concurrent_requests() {
    // The first request's reservation takes the whole budget, so the second, sent before the
    // first settles, finds it spent.
    let key = key(
        None,
        KeyBudgets {
            key: budget(0.0, 0.0001),
            ..KeyBudgets::default()
        },
    );
    let gateway = gateway(vec![("sk-key", key)], None, "").await;
    let (first, second) = tokio::join!(
        send(&gateway.app, "sk-key", "claude"),
        send(&gateway.app, "sk-key", "claude")
    );
    let mut statuses = [first.0, second.0];
    statuses.sort_unstable();
    assert_eq!(statuses, [200, 422]);
}

#[rstest]
#[tokio::test]
async fn requests_refused_after_admission_return_their_reservation() {
    // The reservation takes the whole budget; a body the gateway rejects after the hooks ran
    // must hand it back, or the key stays blocked.
    let key = key(
        None,
        KeyBudgets {
            key: budget(0.0, 0.0001),
            ..KeyBudgets::default()
        },
    );
    let gateway = gateway(vec![("sk-key", key)], None, "").await;
    let invalid = json!({
        "model": "claude",
        "max_tokens": 8,
        "messages": [{"role": "user", "content": "hi"}],
        "cache": {"ttl": -1},
    });
    let (status, body) = send_body(&gateway.app, "sk-key", invalid).await;
    assert_eq!(status, 400, "{body}");
    settled().await;
    let (status, body) = send(&gateway.app, "sk-key", "claude").await;
    assert_eq!(status, 200, "{body}");
}

fn redis_url() -> Option<String> {
    std::env::var("LITELLM_TEST_REDIS_URL").ok()
}

async fn redis(url: &str, namespace: &str) -> Arc<Coordination> {
    let coordination =
        Coordination::new(RedisSettings::new(url, Some(namespace.to_owned()))).unwrap();
    let router: Arc<dyn litellm_gateway::coordination::RouterHandle> = Arc::new(NoRouter(
        litellm_gateway_inference::ModelRouter::from_config(
            &Config::from_yaml("model_list: []").unwrap(),
        )
        .unwrap(),
    ));
    coordination.start(router, None, tokio_util::sync::CancellationToken::new());
    for _ in 0..100 {
        if coordination.connected() {
            return coordination;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    panic!("Redis did not connect");
}

struct NoRouter(litellm_gateway_inference::ModelRouter);

impl litellm_gateway::coordination::RouterHandle for NoRouter {
    fn router(&self) -> &litellm_gateway_inference::ModelRouter {
        &self.0
    }
}

#[rstest]
#[tokio::test]
async fn replicas_share_spend_through_the_proxy_counters() {
    let Some(url) = redis_url() else { return };
    let namespace = format!("budgets-{}", uuid::Uuid::new_v4());
    let first = gateway(
        vec![("sk-key", budgeted_key())],
        Some(redis(&url, &namespace).await),
        "",
    )
    .await;
    let second = gateway(
        vec![("sk-key", budgeted_key())],
        Some(redis(&url, &namespace).await),
        "",
    )
    .await;
    assert_eq!(send(&first.app, "sk-key", "claude").await.0, 200);
    settled().await;
    assert_eq!(send(&second.app, "sk-key", "claude").await.0, 200);
    settled().await;
    assert_eq!(send(&first.app, "sk-key", "claude").await.0, 422);

    let mut connection = redis::Client::open(url.as_str())
        .unwrap()
        .get_multiplexed_async_connection()
        .await
        .unwrap();
    let counter = format!("{namespace}:spend:key:{}", hash("sk-key").as_str());
    let spend: String = redis::cmd("GET")
        .arg(&counter)
        .query_async(&mut connection)
        .await
        .unwrap();
    assert!(
        (spend.parse::<f64>().unwrap() - 2.0 * COST).abs() < 1e-9,
        "{spend}"
    );
    let ttl: i64 = redis::cmd("TTL")
        .arg(&counter)
        .query_async(&mut connection)
        .await
        .unwrap();
    assert!((1..=60).contains(&ttl), "{ttl}");
    let user: String = redis::cmd("GET")
        .arg(format!("{namespace}:spend:user:user-1"))
        .query_async(&mut connection)
        .await
        .unwrap();
    assert!(
        (user.parse::<f64>().unwrap() - 2.0 * COST).abs() < 1e-9,
        "{user}"
    );
}

#[expect(
    clippy::disallowed_methods,
    reason = "test fixture rows; the lookups under test use the checked macro"
)]
async fn end_users(pool: &litellm_db::PgPool) {
    sqlx::raw_sql(
        r#"
INSERT INTO "LiteLLM_BudgetTable" (budget_id, max_budget, created_by, updated_by)
  VALUES ('small', 1.0, 'x', 'x'), ('default', 0.0001, 'x', 'x');
INSERT INTO "LiteLLM_EndUserTable" (user_id, spend, budget_id)
  VALUES ('over', 1.5, 'small'), ('at', 1.0, 'small'), ('unbudgeted', 9.0, NULL);
"#,
    )
    .execute(pool)
    .await
    .unwrap();
}

#[rstest]
#[tokio::test]
async fn end_users_are_held_to_their_budgets() {
    let postgres = litellm_db_testing::MigratedPostgres::start().await.unwrap();
    end_users(postgres.pool()).await;
    let key = key(None, KeyBudgets::default());
    let gateway = gateway_with_database(
        vec![("sk-key", key)],
        None,
        Some(postgres.pool().clone()),
        "  disable_budget_reservation: true\nlitellm_settings:\n  max_end_user_budget_id: default\n",
    )
    .await;
    let (status, body) = send_as(&gateway.app, "sk-key", "claude", Some("over")).await;
    assert_eq!(status, 422);
    assert_eq!(
        body["error"]["message"],
        "ExceededBudget: End User=over over budget. Spend=1.5, Budget=1.0"
    );
    // Strictly over: an end user exactly at budget passes the check.
    assert_eq!(
        send_as(&gateway.app, "sk-key", "claude", Some("at"))
            .await
            .0,
        200
    );
    // Without a row, only a reservation applies the default budget, and reservations are off.
    assert_eq!(
        send_as(&gateway.app, "sk-key", "claude", Some("new"))
            .await
            .0,
        200
    );
    // A row without a budget takes the default one.
    let (status, body) = send_as(&gateway.app, "sk-key", "claude", Some("unbudgeted")).await;
    assert_eq!(status, 422);
    assert_eq!(
        body["error"]["message"],
        "ExceededBudget: End User=unbudgeted over budget. Spend=9.0, Budget=0.0001"
    );
}

#[rstest]
fn settings_follow_the_proxy_defaults() {
    let settings = BudgetSettings::from_config(&Config::from_yaml("model_list: []").unwrap());
    assert_eq!(settings.status, 422);
    assert!(settings.reservations);
    assert_eq!(settings.ttl, Duration::from_secs(60));
}
