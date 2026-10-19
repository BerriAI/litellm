//! The model capacity hook on real requests: a group held to its deployments' summed capacity,
//! priorities held to their share, refusals and headers as the proxy sends them, and counters
//! shared through Redis. Set `LITELLM_TEST_REDIS_URL` to also run the shared case.

use std::{collections::HashMap, sync::Arc, time::Duration};

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
    model_capacity::{CapacitySettings, ModelCapacity},
};
use litellm_gateway_auth::keys::{
    AllowedRoutes, KeyHash, KeyLookup, KeyModels, KeyProfile, KeyStatus, LookupFuture,
};
use rstest::rstest;
use serde_json::{Value, json};
use tokio::net::TcpListener;
use tower::ServiceExt;

/// 3 input and 4 output tokens: 7 count against the group.
const MESSAGE: &str = r#"{"id":"msg_1","type":"message","role":"assistant","model":"claude-x","content":[{"type":"text","text":"hi"}],"stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":3,"output_tokens":4}}"#;

async fn upstream() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let app = Router::new().fallback(|_: Request<Body>| async {
        ([(CONTENT_TYPE, "application/json")], MESSAGE).into_response()
    });
    tokio::spawn(async move { axum::serve(listener, app).await });
    format!("http://{address}")
}

struct Keys(HashMap<KeyHash, KeyStatus>);

impl KeyLookup for Keys {
    fn lookup<'a>(&'a self, hash: &'a KeyHash) -> LookupFuture<'a> {
        Box::pin(async move { Ok(self.0.get(hash).cloned()) })
    }
}

/// Two deployments of `claude`, each with `limits` (such as `rpm: 1`).
async fn gateway(limits: &str, settings: &str, redis: Option<Arc<Coordination>>) -> Router {
    let address = upstream().await;
    let config = Config::from_yaml(&format!(
        "model_list:
  - model_name: claude
    litellm_params: {{model: anthropic/claude-x, api_base: {address}, api_key: sk-a, {limits}}}
  - model_name: claude
    litellm_params: {{model: anthropic/claude-y, api_base: {address}, api_key: sk-b, {limits}}}
router_settings:
  num_retries: 0
general_settings:
  master_key: sk-master
litellm_settings:
  callbacks: [dynamic_rate_limiter_v3]
{settings}"
    ))
    .unwrap();
    let inference = litellm_gateway::build_inference(&config).unwrap();
    let hook = ModelCapacity::for_gateway(&config, redis).unwrap();
    litellm_gateway::hooks::install(&inference, &config, vec![Arc::new(hook)], Vec::new())
        .await
        .unwrap();
    let mut metadata = serde_json::Map::new();
    metadata.insert("priority".into(), json!("prod"));
    let key = KeyStatus::Active {
        expires_at: None,
        models: KeyModels::default(),
        allowed_routes: AllowedRoutes::default(),
        team_id: None,
        profile: Arc::new(KeyProfile {
            metadata: Arc::new(metadata),
            ..KeyProfile::default()
        }),
    };
    let keys: Arc<dyn KeyLookup> = Arc::new(Keys(HashMap::from([(
        KeyHash::from_token(&SecretValue::new("sk-prod")),
        key,
    )])));
    litellm_gateway::router_with_keys(inference, &config, None, None, Some(keys))
}

struct Reply {
    status: u16,
    headers: HeaderMap,
    body: Value,
}

async fn send(app: &Router, path: &str, token: &str) -> Reply {
    let body = json!({
        "model": "claude",
        "max_tokens": 8,
        "messages": [{"role": "user", "content": "hi"}],
    });
    let response = app
        .clone()
        .oneshot(
            Request::post(path)
                .header("content-type", "application/json")
                .header("authorization", format!("Bearer {token}"))
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

fn header<'a>(reply: &'a Reply, name: &str) -> Option<&'a str> {
    reply.headers.get(name).map(|value| value.to_str().unwrap())
}

async fn settled() {
    tokio::time::sleep(Duration::from_millis(100)).await;
}

#[rstest]
#[tokio::test]
async fn a_group_is_held_to_its_deployments_capacity() {
    let app = gateway("rpm: 1", "", None).await;
    let first = send(&app, "/v1/messages", "sk-master").await;
    assert_eq!(first.status, 200);
    assert_eq!(
        header(
            &first,
            "x-ratelimit-model_saturation_check-remaining-requests"
        ),
        Some("1")
    );
    assert_eq!(
        header(&first, "x-ratelimit-model_saturation_check-limit-requests"),
        Some("2")
    );
    assert_eq!(header(&first, "x-litellm-rate-limiter-version"), Some("v3"));
    assert_eq!(header(&first, "x-litellm-priority"), Some("default"));
    assert_eq!(send(&app, "/v1/messages", "sk-master").await.status, 200);

    let message = "Model capacity reached for claude. Priority: None, Rate limit type: requests, Model TPM: not configured, Model RPM: 2, Remaining: 0";
    let refused = send(&app, "/v1/chat/completions", "sk-master").await;
    assert_eq!(refused.status, 429);
    assert_eq!(
        refused.body,
        json!({"error": {"message": message, "type": "throttling_error", "param": null, "code": "429"}})
    );
    assert_eq!(header(&refused, "retry-after"), Some("60"));
    assert_eq!(header(&refused, "rate_limit_type"), Some("requests"));
    assert_eq!(header(&refused, "x-litellm-priority"), Some("default"));
    let refused = send(&app, "/v1/messages", "sk-master").await;
    assert_eq!(
        refused.body,
        json!({"type": "error", "error": {"type": "rate_limit_error", "message": message}})
    );
    assert_eq!(header(&refused, "retry-after"), None);
}

#[rstest]
#[tokio::test]
async fn tokens_count_after_the_response() {
    // 2 x 5 tokens of capacity; each response uses 7.
    let app = gateway("tpm: 5", "", None).await;
    assert_eq!(send(&app, "/v1/messages", "sk-master").await.status, 200);
    settled().await;
    assert_eq!(send(&app, "/v1/messages", "sk-master").await.status, 200);
    settled().await;
    let refused = send(&app, "/v1/messages", "sk-master").await;
    assert_eq!(refused.status, 429);
    assert_eq!(
        refused.body["error"]["message"],
        "Model capacity reached for claude. Priority: None, Rate limit type: tokens, Model TPM: 10, Model RPM: not configured, Remaining: -4"
    );
}

#[rstest]
#[tokio::test]
async fn busy_groups_hold_priorities_to_their_share() {
    // 20 requests a window; `prod` gets a fifth once the group is busy, here from the start.
    let app = gateway(
        "rpm: 10",
        "  priority_reservation: {prod: 0.1}\n  priority_reservation_settings: {saturation_threshold: 0.0}\n",
        None,
    )
    .await;
    assert_eq!(
        send(&app, "/v1/chat/completions", "sk-prod").await.status,
        200
    );
    let second = send(&app, "/v1/chat/completions", "sk-prod").await;
    assert_eq!(second.status, 200);
    assert_eq!(header(&second, "x-litellm-priority"), Some("prod"));
    assert_eq!(
        header(&second, "x-ratelimit-priority_model-remaining-requests"),
        Some("0")
    );
    let refused = send(&app, "/v1/chat/completions", "sk-prod").await;
    assert_eq!(refused.status, 429);
    assert_eq!(
        refused.body["error"]["message"],
        "Priority-based rate limit exceeded. Model: claude, Priority: prod, Rate limit type: requests, Model TPM: not configured, Model RPM: 20, Remaining: 0, Model saturation: 0.0%"
    );
    assert_eq!(header(&refused, "x-litellm-saturation"), Some("0.00%"));
    // Other callers share the default pool and the rest of the group.
    assert_eq!(
        send(&app, "/v1/chat/completions", "sk-master").await.status,
        200
    );
}

#[rstest]
fn the_hook_is_on_only_when_listed() {
    let config = Config::from_yaml("model_list: []").unwrap();
    assert_eq!(CapacitySettings::from_config(&config), None);
    let config =
        Config::from_yaml("model_list: []\nlitellm_settings:\n  callbacks: model_capacity\n")
            .unwrap();
    assert_eq!(
        CapacitySettings::from_config(&config),
        Some(CapacitySettings::default())
    );
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
async fn replicas_share_the_group_counters() {
    let Ok(url) = std::env::var("LITELLM_TEST_REDIS_URL") else {
        return;
    };
    let namespace = format!("model-capacity-{}", uuid::Uuid::new_v4());
    let limits = "rpm: 2, tpm: 1000";
    let first = gateway(limits, "", Some(redis(&url, &namespace).await)).await;
    let second = gateway(limits, "", Some(redis(&url, &namespace).await)).await;
    for app in [&first, &second, &first, &second] {
        assert_eq!(send(app, "/v1/messages", "sk-master").await.status, 200);
    }
    settled().await;
    assert_eq!(send(&second, "/v1/messages", "sk-master").await.status, 429);

    let mut connection = redis::Client::open(url.as_str())
        .unwrap()
        .get_multiplexed_async_connection()
        .await
        .unwrap();
    let get = async |connection: &mut redis::aio::MultiplexedConnection, key: String| {
        redis::cmd("GET")
            .arg(key)
            .query_async::<Option<String>>(connection)
            .await
            .unwrap()
    };
    let group = format!("{namespace}:{{model_saturation_check:claude}}");
    // The refused request is not counted; four responses used 7 tokens each.
    assert_eq!(
        get(&mut connection, format!("{group}:requests"))
            .await
            .as_deref(),
        Some("4")
    );
    assert_eq!(
        get(&mut connection, format!("{group}:tokens"))
            .await
            .as_deref(),
        Some("28")
    );
    let pool = format!("{namespace}:{{priority_model:claude:default_pool}}:tokens");
    assert_eq!(get(&mut connection, pool).await.as_deref(), Some("28"));
}
