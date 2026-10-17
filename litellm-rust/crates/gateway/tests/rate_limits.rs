//! The rate limits hook on real requests: refusals and headers as the proxy sends them, tokens
//! reserved and settled to usage, in-flight slots, and counters shared through Redis. Set
//! `LITELLM_TEST_REDIS_URL` to also run the shared cases against a live Redis.

use std::{
    collections::HashMap,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

use axum::{
    Router,
    body::{Body, to_bytes},
    http::{HeaderMap, Request, header::CONTENT_TYPE},
    response::IntoResponse,
};
use litellm_auth_types::SecretValue;
use litellm_config::Config;
use litellm_gateway::{
    coordination::{Coordination, RedisSettings},
    rate_limits::{RateLimitSettings, RateLimiter},
};
use litellm_gateway_auth::keys::{
    AllowedRoutes, KeyHash, KeyLimits, KeyLookup, KeyModels, KeyProfile, KeyStatus, Limits,
    LookupFuture,
};
use rstest::rstest;
use serde_json::{Value, json};
use tokio::net::TcpListener;
use tower::ServiceExt;

/// 3 input and 4 output tokens: 7 count against a token limit.
const MESSAGE: &str = r#"{"id":"msg_1","type":"message","role":"assistant","model":"claude-x","content":[{"type":"text","text":"hi"}],"stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":3,"output_tokens":4}}"#;

#[derive(Clone, Default)]
struct Upstream {
    calls: Arc<AtomicUsize>,
    bodies: Arc<Mutex<Vec<Value>>>,
}

async fn upstream(delay: Duration) -> (String, Upstream) {
    let upstream = Upstream::default();
    let seen = upstream.clone();
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let app = Router::new().fallback(move |request: Request<Body>| {
        let seen = seen.clone();
        async move {
            seen.calls.fetch_add(1, Ordering::SeqCst);
            let body = to_bytes(request.into_body(), 1 << 20).await.unwrap();
            seen.bodies
                .lock()
                .unwrap()
                .push(serde_json::from_slice(&body).unwrap_or(Value::Null));
            tokio::time::sleep(delay).await;
            ([(CONTENT_TYPE, "application/json")], MESSAGE).into_response()
        }
    });
    tokio::spawn(async move { axum::serve(listener, app).await });
    (format!("http://{address}"), upstream)
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

fn key(team_id: Option<&str>, limits: KeyLimits) -> KeyStatus {
    KeyStatus::Active {
        expires_at: None,
        models: KeyModels::default(),
        allowed_routes: AllowedRoutes::default(),
        team_id: team_id.map(str::to_owned),
        profile: Arc::new(KeyProfile {
            user_id: Some("user-1".into()),
            limits,
            ..KeyProfile::default()
        }),
    }
}

fn rpm(limit: i64) -> Limits {
    Limits {
        rpm: Some(limit),
        tpm: None,
    }
}

fn tpm(limit: i64) -> Limits {
    Limits {
        rpm: None,
        tpm: Some(limit),
    }
}

struct Gateway {
    app: Router,
    upstream: Upstream,
}

struct Setup {
    redis: Option<Arc<Coordination>>,
    database: Option<litellm_db::PgPool>,
    delay: Duration,
}

impl Default for Setup {
    fn default() -> Self {
        Self {
            redis: None,
            database: None,
            delay: Duration::ZERO,
        }
    }
}

async fn gateway(keys: Vec<(&str, KeyStatus)>, setup: Setup) -> Gateway {
    let (address, upstream) = upstream(setup.delay).await;
    let config = Config::from_yaml(&format!(
        "model_list:
  - model_name: claude
    litellm_params:
      model: anthropic/claude-x
      api_base: {address}
      api_key: sk-upstream
router_settings:
  num_retries: 0
general_settings:
  master_key: sk-master
"
    ))
    .unwrap();
    let inference = litellm_gateway::build_inference(&config).unwrap();
    let limiter = RateLimiter::new(RateLimitSettings::default(), setup.redis, setup.database);
    litellm_gateway::hooks::install(&inference, &config, vec![Arc::new(limiter)], Vec::new())
        .await
        .unwrap();
    let keys: Arc<dyn KeyLookup> = Arc::new(Keys(
        keys.into_iter()
            .map(|(token, status)| (hash(token), status))
            .collect(),
    ));
    let app = litellm_gateway::router_with_keys(inference, &config, None, None, Some(keys));
    Gateway { app, upstream }
}

struct Reply {
    status: u16,
    headers: HeaderMap,
    body: Value,
}

async fn send(app: &Router, path: &str, body: Value) -> Reply {
    let response = app
        .clone()
        .oneshot(
            Request::post(path)
                .header("content-type", "application/json")
                .header("authorization", "Bearer sk-key")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status().as_u16();
    let headers = response.headers().clone();
    let body = to_bytes(response.into_body(), 1 << 20).await.unwrap();
    Reply {
        status,
        headers,
        body: serde_json::from_slice(&body).unwrap_or(Value::Null),
    }
}

fn message(extra: Value) -> Value {
    let mut body = json!({
        "model": "claude",
        "max_tokens": 8,
        "messages": [{"role": "user", "content": "hi"}],
    });
    for (field, value) in extra.as_object().unwrap() {
        body[field] = value.clone();
    }
    body
}

async fn messages(app: &Router) -> Reply {
    send(app, "/v1/messages", message(json!({}))).await
}

async fn chat(app: &Router) -> Reply {
    send(app, "/v1/chat/completions", message(json!({}))).await
}

/// Settling runs after the response, off the request path.
async fn settled() {
    tokio::time::sleep(Duration::from_millis(100)).await;
}

fn header<'a>(reply: &'a Reply, name: &str) -> Option<&'a str> {
    reply.headers.get(name).map(|value| value.to_str().unwrap())
}

/// The message up to the reset time, which moves with the clock.
fn refusal_message(reply: &Reply, path: &[&str]) -> String {
    let mut value = &reply.body;
    for part in path {
        value = &value[*part];
    }
    let message = value.as_str().unwrap();
    let (message, reset) = message.split_once(" Limit resets at: ").unwrap();
    assert!(reset.ends_with(" UTC"), "{reset}");
    message.to_owned()
}

#[rstest]
#[tokio::test]
async fn requests_over_the_key_limit_are_refused_like_the_proxy() {
    let limits = KeyLimits {
        key: rpm(2),
        ..KeyLimits::default()
    };
    let gateway = gateway(vec![("sk-key", key(None, limits))], Setup::default()).await;
    let first = messages(&gateway.app).await;
    assert_eq!(first.status, 200);
    assert_eq!(
        header(&first, "x-ratelimit-api_key-remaining-requests"),
        Some("1")
    );
    assert_eq!(
        header(&first, "x-ratelimit-api_key-limit-requests"),
        Some("2")
    );
    assert_eq!(header(&first, "x-litellm-key-rpm-limit"), Some("2"));
    assert_eq!(chat(&gateway.app).await.status, 200);

    let expected = format!(
        "Rate limit exceeded for api_key: {}. Limit type: requests. Current limit: 2, Remaining: 0.",
        hash("sk-key").as_str()
    );
    // Chat Completions: the proxy's error body and its retry headers.
    let refused = chat(&gateway.app).await;
    assert_eq!(refused.status, 429);
    assert_eq!(refusal_message(&refused, &["error", "message"]), expected);
    assert_eq!(refused.body["error"]["type"], "throttling_error");
    assert_eq!(refused.body["error"]["param"], Value::Null);
    assert_eq!(refused.body["error"]["code"], "429");
    assert_eq!(header(&refused, "retry-after"), Some("60"));
    assert_eq!(header(&refused, "rate_limit_type"), Some("requests"));
    assert!(header(&refused, "reset_at").unwrap().ends_with(" UTC"));
    assert_eq!(header(&refused, "x-litellm-key-rpm-limit"), Some("2"));
    // Messages: Anthropic's envelope, without the retry headers.
    let refused = messages(&gateway.app).await;
    assert_eq!(refused.status, 429);
    assert_eq!(refused.body["type"], "error");
    assert_eq!(refused.body["error"]["type"], "rate_limit_error");
    assert_eq!(refusal_message(&refused, &["error", "message"]), expected);
    assert_eq!(header(&refused, "retry-after"), None);
    assert_eq!(gateway.upstream.calls.load(Ordering::SeqCst), 2);
}

#[rstest]
#[case::user(KeyLimits { user: rpm(1), ..KeyLimits::default() }, "user: user-1")]
#[case::team(KeyLimits { team: rpm(1), ..KeyLimits::default() }, "team: team-1")]
#[case::team_member(KeyLimits { team_member: rpm(1), ..KeyLimits::default() }, "team_member: team-1:user-1")]
#[tokio::test]
async fn owners_are_limited_under_their_own_names(
    #[case] limits: KeyLimits,
    #[case] descriptor: &str,
) {
    let gateway = gateway(
        vec![("sk-key", key(Some("team-1"), limits))],
        Setup::default(),
    )
    .await;
    assert_eq!(messages(&gateway.app).await.status, 200);
    let refused = messages(&gateway.app).await;
    assert_eq!(refused.status, 429);
    assert_eq!(
        refusal_message(&refused, &["error", "message"]),
        format!(
            "Rate limit exceeded for {descriptor}. Limit type: requests. Current limit: 1, Remaining: 0."
        )
    );
}

#[rstest]
#[tokio::test]
async fn tokens_are_reserved_and_settled_to_usage() {
    // Each request reserves 1 input + 8 output tokens and settles to 7.
    let limits = KeyLimits {
        key: tpm(20),
        ..KeyLimits::default()
    };
    let gateway = gateway(vec![("sk-key", key(None, limits))], Setup::default()).await;
    let first = messages(&gateway.app).await;
    assert_eq!(first.status, 200);
    assert_eq!(
        header(&first, "x-ratelimit-api_key-remaining-tokens"),
        Some("11")
    );
    assert_eq!(header(&first, "x-litellm-key-tpm-limit"), Some("20"));
    settled().await;
    assert_eq!(messages(&gateway.app).await.status, 200);
    settled().await;
    let refused = messages(&gateway.app).await;
    assert_eq!(refused.status, 429);
    assert_eq!(
        refusal_message(&refused, &["error", "message"]),
        format!(
            "Rate limit exceeded for api_key: {}. Limit type: tokens. Current limit: 20, Remaining: 6.",
            hash("sk-key").as_str()
        )
    );
}

#[rstest]
#[tokio::test]
async fn small_token_limits_cap_uncapped_requests() {
    let limits = KeyLimits {
        key: tpm(400),
        ..KeyLimits::default()
    };
    let gateway = gateway(vec![("sk-key", key(None, limits))], Setup::default()).await;
    let body = json!({"model": "claude", "messages": [{"role": "user", "content": "hi"}]});
    assert_eq!(send(&gateway.app, "/v1/messages", body).await.status, 200);
    assert_eq!(
        gateway.upstream.bodies.lock().unwrap()[0]["max_tokens"],
        100
    );
}

#[rstest]
#[tokio::test]
async fn in_flight_requests_are_capped_per_key() {
    let limits = KeyLimits {
        max_parallel_requests: Some(1),
        ..KeyLimits::default()
    };
    let gateway = gateway(
        vec![("sk-key", key(None, limits))],
        Setup {
            delay: Duration::from_millis(300),
            ..Setup::default()
        },
    )
    .await;
    let (first, second) = tokio::join!(messages(&gateway.app), async {
        tokio::time::sleep(Duration::from_millis(100)).await;
        messages(&gateway.app).await
    });
    assert_eq!(first.status, 200);
    assert_eq!(
        header(
            &first,
            "x-ratelimit-api_key-remaining-max_parallel_requests"
        ),
        Some("0")
    );
    assert_eq!(second.status, 429);
    assert_eq!(
        refusal_message(&second, &["error", "message"]),
        format!(
            "Rate limit exceeded for api_key: {}. Limit type: max_parallel_requests. Current limit: 1, Remaining: 0.",
            hash("sk-key").as_str()
        )
    );
    settled().await;
    assert_eq!(messages(&gateway.app).await.status, 200);
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
async fn replicas_share_the_proxy_counters() {
    let Some(url) = redis_url() else { return };
    let namespace = format!("rate-limits-{}", uuid::Uuid::new_v4());
    let limits = KeyLimits {
        key: Limits {
            rpm: Some(3),
            tpm: Some(1_000),
        },
        max_parallel_requests: Some(5),
        ..KeyLimits::default()
    };
    let mut replicas = Vec::new();
    for _ in 0..2 {
        let setup = Setup {
            redis: Some(redis(&url, &namespace).await),
            ..Setup::default()
        };
        replicas.push(gateway(vec![("sk-key", key(None, limits))], setup).await);
    }
    assert_eq!(messages(&replicas[0].app).await.status, 200);
    assert_eq!(messages(&replicas[1].app).await.status, 200);
    let third = messages(&replicas[0].app).await;
    assert_eq!(
        header(&third, "x-ratelimit-api_key-remaining-requests"),
        Some("0")
    );
    settled().await;
    assert_eq!(messages(&replicas[1].app).await.status, 429);

    let mut connection = redis::Client::open(url.as_str())
        .unwrap()
        .get_multiplexed_async_connection()
        .await
        .unwrap();
    let prefix = format!("{namespace}:{{api_key:{}}}", hash("sk-key").as_str());
    let get = async |connection: &mut redis::aio::MultiplexedConnection, suffix: &str| {
        redis::cmd("GET")
            .arg(format!("{prefix}:{suffix}"))
            .query_async::<Option<String>>(connection)
            .await
            .unwrap()
    };
    // The refused request still counted; three settled requests used 7 tokens each.
    assert_eq!(get(&mut connection, "requests").await.as_deref(), Some("4"));
    assert_eq!(get(&mut connection, "tokens").await.as_deref(), Some("21"));
    assert!(get(&mut connection, "window").await.is_some());
    let in_flight: i64 = redis::cmd("ZCARD")
        .arg(format!("{prefix}:max_parallel_requests"))
        .query_async(&mut connection)
        .await
        .unwrap();
    assert_eq!(in_flight, 0);
}

#[rstest]
#[tokio::test]
async fn shared_refusals_count_and_hold_as_the_proxy() {
    let Some(url) = redis_url() else { return };
    let mut connection = redis::Client::open(url.as_str())
        .unwrap()
        .get_multiplexed_async_connection()
        .await
        .unwrap();

    // Over the tokens limit: the request still counts, and its in-flight slot is given back.
    let namespace = format!("rate-limits-{}", uuid::Uuid::new_v4());
    let limits = KeyLimits {
        key: Limits {
            rpm: Some(10),
            tpm: Some(1),
        },
        max_parallel_requests: Some(1),
        ..KeyLimits::default()
    };
    let setup = Setup {
        redis: Some(redis(&url, &namespace).await),
        ..Setup::default()
    };
    let single = gateway(vec![("sk-key", key(None, limits))], setup).await;
    let refused = messages(&single.app).await;
    assert_eq!(refused.status, 429);
    assert!(
        refusal_message(&refused, &["error", "message"])
            .contains("Limit type: tokens. Current limit: 1, Remaining: 1.")
    );
    let prefix = format!("{namespace}:{{api_key:{}}}", hash("sk-key").as_str());
    let requests: Option<String> = redis::cmd("GET")
        .arg(format!("{prefix}:requests"))
        .query_async(&mut connection)
        .await
        .unwrap();
    assert_eq!(requests.as_deref(), Some("1"));
    let in_flight: i64 = redis::cmd("ZCARD")
        .arg(format!("{prefix}:max_parallel_requests"))
        .query_async(&mut connection)
        .await
        .unwrap();
    assert_eq!(in_flight, 0);

    // A full gauge on one replica refuses on another.
    let namespace = format!("rate-limits-{}", uuid::Uuid::new_v4());
    let limits = KeyLimits {
        max_parallel_requests: Some(1),
        ..KeyLimits::default()
    };
    let mut replicas = Vec::new();
    for _ in 0..2 {
        let setup = Setup {
            redis: Some(redis(&url, &namespace).await),
            delay: Duration::from_millis(300),
            ..Setup::default()
        };
        replicas.push(gateway(vec![("sk-key", key(None, limits))], setup).await);
    }
    let (first, second) = tokio::join!(messages(&replicas[0].app), async {
        tokio::time::sleep(Duration::from_millis(100)).await;
        messages(&replicas[1].app).await
    });
    assert_eq!(first.status, 200);
    assert_eq!(second.status, 429);
    assert!(
        refusal_message(&second, &["error", "message"])
            .contains("Limit type: max_parallel_requests. Current limit: 1, Remaining: 0.")
    );
    settled().await;
    assert_eq!(messages(&replicas[1].app).await.status, 200);
}

#[expect(
    clippy::disallowed_methods,
    reason = "test fixture rows; the lookups under test use the checked macro"
)]
async fn end_users(pool: &litellm_db::PgPool) {
    sqlx::raw_sql(
        r#"
INSERT INTO "LiteLLM_BudgetTable" (budget_id, rpm_limit, created_by, updated_by)
  VALUES ('one-a-minute', 1, 'x', 'x');
INSERT INTO "LiteLLM_EndUserTable" (user_id, spend, budget_id)
  VALUES ('limited', 0, 'one-a-minute'), ('free', 0, NULL);
"#,
    )
    .execute(pool)
    .await
    .unwrap();
}

#[rstest]
#[tokio::test]
async fn end_users_are_held_to_their_budget_limits() {
    let postgres = litellm_db_testing::MigratedPostgres::start().await.unwrap();
    end_users(postgres.pool()).await;
    let gateway = gateway(
        vec![("sk-key", key(None, KeyLimits::default()))],
        Setup {
            database: Some(postgres.pool().clone()),
            ..Setup::default()
        },
    )
    .await;
    let as_end_user = |user: &str| message(json!({"metadata": {"user_id": user}}));
    for user in ["limited", "free", "free"] {
        assert_eq!(
            send(&gateway.app, "/v1/messages", as_end_user(user))
                .await
                .status,
            200
        );
    }
    let refused = send(&gateway.app, "/v1/messages", as_end_user("limited")).await;
    assert_eq!(refused.status, 429);
    assert_eq!(
        refusal_message(&refused, &["error", "message"]),
        "Rate limit exceeded for end_user: limited. Limit type: requests. Current limit: 1, Remaining: 0."
    );
}
