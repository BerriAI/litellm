//! Replicas sharing state through a live Redis. Set `LITELLM_TEST_REDIS_URL` (for example
//! `redis://127.0.0.1:6379`) to run them; without it they pass without checking anything.

use std::{
    collections::HashMap,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

use axum::{
    Json, Router,
    body::Body,
    http::{Request, StatusCode},
    response::IntoResponse,
};
use litellm_auth_types::SecretValue;
use litellm_config::Config;
use litellm_gateway::coordination::{Coordination, RedisHealth, RedisSettings};
use litellm_gateway_auth::keys::{KeyHash, KeyLookup, KeyStatus, LookupFuture, ModelAccess};
use rstest::rstest;
use serde_json::json;
use tokio::net::TcpListener;
use tokio_util::sync::CancellationToken;
use tower::ServiceExt;

fn redis_url() -> Option<String> {
    std::env::var("LITELLM_TEST_REDIS_URL").ok()
}

fn namespace() -> String {
    format!("test-{}", uuid::Uuid::new_v4())
}

/// An Anthropic-shaped upstream that answers 429 when `limited`, counting its requests.
async fn upstream(limited: bool) -> (String, Arc<AtomicUsize>) {
    let hits = Arc::new(AtomicUsize::new(0));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let counted = hits.clone();
    let app = Router::new().fallback(move || {
        let hits = counted.clone();
        async move {
            hits.fetch_add(1, Ordering::SeqCst);
            if limited {
                return (
                    StatusCode::TOO_MANY_REQUESTS,
                    Json(json!({"type": "error", "error": {"type": "rate_limit_error", "message": "slow down"}})),
                )
                    .into_response();
            }
            Json(json!({
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "claude-x",
                "content": [{"type": "text", "text": "hi"}],
                "stop_reason": "end_turn",
                "stop_sequence": null,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }))
            .into_response()
        }
    });
    tokio::spawn(async move { axum::serve(listener, app).await });
    (format!("http://{address}"), hits)
}

/// One gateway replica sharing state under `namespace`, with its key cache when `keys` is set.
async fn replica(
    url: &str,
    namespace: &str,
    config: &Config,
    keys: Option<Arc<dyn KeyLookup>>,
    shutdown: &CancellationToken,
) -> (Router, Arc<Coordination>) {
    let inference = litellm_gateway::build_inference(config).unwrap();
    let coordination = Coordination::new(
        RedisSettings::new(url, Some(namespace.to_owned()))
            .with_cooldown_read_interval(Duration::from_millis(50)),
    )
    .unwrap();
    inference
        .models
        .share_health(Arc::new(RedisHealth(coordination.clone())));
    let keys = keys.map(|keys| litellm_gateway::key_cache(keys, config));
    coordination.start(inference.clone(), keys.clone(), shutdown.clone());
    let app = litellm_gateway::router_with_key_cache(inference, config, None, None, keys);
    (app, coordination)
}

async fn until(mut done: impl AsyncFnMut() -> bool) {
    for _ in 0..100 {
        if done().await {
            return;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    panic!("condition not met within 5 s");
}

async fn send(app: &Router, key: &str) -> StatusCode {
    app.clone()
        .oneshot(
            Request::post("/v1/messages")
                .header("authorization", format!("Bearer {key}"))
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({"model": "claude", "max_tokens": 1, "messages": [{"role": "user", "content": "hi"}]})
                        .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap()
        .status()
}

#[rstest]
#[tokio::test]
async fn a_deployment_one_replica_cools_down_is_skipped_by_the_others() {
    let Some(url) = redis_url() else { return };
    let namespace = namespace();
    let (limited, limited_hits) = upstream(true).await;
    let (healthy, _) = upstream(false).await;
    let config = Config::from_yaml(&format!(
        "model_list:
  - model_name: claude
    litellm_params: {{model: anthropic/claude-x, api_base: '{limited}', api_key: sk-upstream}}
  - model_name: claude
    litellm_params: {{model: anthropic/claude-x, api_base: '{healthy}', api_key: sk-upstream}}
router_settings:
  num_retries: 1
  cooldown_time: 30
general_settings:
  master_key: sk-master
"
    ))
    .unwrap();
    let shutdown = CancellationToken::new();
    let (first, first_redis) = replica(&url, &namespace, &config, None, &shutdown).await;
    let (second, second_redis) = replica(&url, &namespace, &config, None, &shutdown).await;
    until(async || first_redis.connected() && second_redis.connected()).await;

    // The first replica meets the 429, cools that deployment and retries on the other.
    until(async || {
        assert_eq!(send(&first, "sk-master").await, StatusCode::OK);
        limited_hits.load(Ordering::SeqCst) > 0
    })
    .await;
    tokio::time::sleep(Duration::from_millis(300)).await;

    let hits = limited_hits.load(Ordering::SeqCst);
    for _ in 0..50 {
        assert_eq!(send(&second, "sk-master").await, StatusCode::OK);
    }
    assert_eq!(limited_hits.load(Ordering::SeqCst), hits);
    shutdown.cancel();
}

/// Keys whose statuses a test changes, all on `team-1`.
#[derive(Default)]
struct Keys(Mutex<HashMap<KeyHash, KeyStatus>>);

impl Keys {
    fn put(&self, token: &str, status: KeyStatus) {
        self.0
            .lock()
            .unwrap()
            .insert(KeyHash::from_token(&SecretValue::new(token)), status);
    }
}

impl KeyLookup for Keys {
    fn lookup<'a>(&'a self, hash: &'a KeyHash) -> LookupFuture<'a> {
        Box::pin(async move { Ok(self.0.lock().unwrap().get(hash).cloned()) })
    }
}

fn on_team() -> KeyStatus {
    match KeyStatus::active(None, ModelAccess::default()) {
        KeyStatus::Active {
            expires_at,
            models,
            allowed_routes,
            ..
        } => KeyStatus::Active {
            expires_at,
            models,
            allowed_routes,
            team_id: Some("team-1".into()),
            profile: Default::default(),
        },
        status => status,
    }
}

#[rstest]
#[case::key(None)]
#[case::team(Some("team_id:team-1"))]
#[tokio::test]
async fn a_key_the_proxy_invalidates_is_reread_on_every_replica(#[case] cache_key: Option<&str>) {
    let Some(url) = redis_url() else { return };
    let namespace = namespace();
    let (healthy, _) = upstream(false).await;
    let config = Config::from_yaml(&format!(
        "model_list:
  - model_name: claude
    litellm_params: {{model: anthropic/claude-x, api_base: '{healthy}', api_key: sk-upstream}}
general_settings:
  master_key: sk-master
"
    ))
    .unwrap();
    let keys = Arc::new(Keys::default());
    keys.put("sk-user", on_team());
    let shutdown = CancellationToken::new();
    let (app, redis) = replica(&url, &namespace, &config, Some(keys.clone()), &shutdown).await;
    // Subscribing drops every cached key, so cache this one after it.
    until(async || redis.subscribed()).await;
    assert_eq!(send(&app, "sk-user").await, StatusCode::OK);

    keys.put("sk-user", KeyStatus::Blocked);
    // Still served from the cache: the key TTL is a minute.
    assert_eq!(send(&app, "sk-user").await, StatusCode::OK);

    // What the proxy's `publish_auth_cache_invalidation` sends after a key or team update.
    let hash = KeyHash::from_token(&SecretValue::new("sk-user"));
    let message = json!({"cache_key": cache_key.unwrap_or(hash.as_str())}).to_string();
    let channel = format!("{namespace}:litellm_proxy.auth_cache_invalidation");
    let mut publisher = redis::Client::open(url.as_str())
        .unwrap()
        .get_multiplexed_async_connection()
        .await
        .unwrap();
    until(async || {
        let _: i64 = redis::cmd("PUBLISH")
            .arg(&channel)
            .arg(&message)
            .query_async(&mut publisher)
            .await
            .unwrap();
        send(&app, "sk-user").await == StatusCode::UNAUTHORIZED
    })
    .await;
    shutdown.cancel();
}
