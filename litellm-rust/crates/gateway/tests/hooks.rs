//! Hooks and observers around real requests: a hook that can refuse, add a response header and
//! change the upstream body, and an observer that records what each request cost.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use axum::{
    Router,
    body::{Body, Bytes, to_bytes},
    http::{HeaderName, HeaderValue, Request, header::CONTENT_TYPE},
    response::IntoResponse,
};
use futures_util::future::BoxFuture;
use litellm_config::Config;
use litellm_gateway_auth::keys::{
    AllowedRoutes, KeyHash, KeyLookup, KeyModels, KeyProfile, KeyStatus, LookupFuture,
};
use litellm_gateway_hooks::{
    Facts, Hook, Observer, Record, Refusal, RequestEdit, RequestFacts, Settle, Targets,
};
use rstest::rstest;
use serde_json::{Value, json};
use tokio::net::TcpListener;
use tower::ServiceExt;

const MESSAGE: &str = r#"{"id":"msg_1","type":"message","role":"assistant","model":"claude-x","content":[{"type":"text","text":"hi"}],"stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":3,"output_tokens":4}}"#;
const MESSAGE_STREAM: &str = concat!(
    "event: message_start\n",
    "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[],\"model\":\"claude-x\",\"usage\":{\"input_tokens\":10,\"cache_read_input_tokens\":20,\"output_tokens\":1}}}\n\n",
    "event: content_block_delta\n",
    "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n",
    "event: message_delta\n",
    "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":42}}\n\n",
    "event: message_stop\n",
    "data: {\"type\":\"message_stop\"}\n\n",
);
const COMPLETION: &str = r#"{"id":"chatcmpl-1","object":"chat.completion","created":1,"model":"gpt-x","choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":6,"total_tokens":11}}"#;

type Bodies = Arc<Mutex<Vec<Value>>>;

/// Answers Anthropic and OpenAI calls, keeping each request body.
async fn upstream(bodies: Bodies) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let app = Router::new().fallback(move |request: Request<Body>| {
        let bodies = bodies.clone();
        async move {
            let path = request.uri().path().to_owned();
            let body: Value =
                serde_json::from_slice(&to_bytes(request.into_body(), 1 << 20).await.unwrap())
                    .unwrap();
            let stream = body["stream"].as_bool().unwrap_or(false);
            bodies.lock().unwrap().push(body);
            let (content_type, payload) = match (path.ends_with("/chat/completions"), stream) {
                (true, _) => ("application/json", COMPLETION),
                (false, true) => ("text/event-stream", MESSAGE_STREAM),
                (false, false) => ("application/json", MESSAGE),
            };
            ([(CONTENT_TYPE, content_type)], payload).into_response()
        }
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

fn keys() -> Arc<dyn KeyLookup> {
    let status = KeyStatus::Active {
        expires_at: None,
        models: KeyModels::default(),
        allowed_routes: AllowedRoutes::default(),
        team_id: Some("team-1".into()),
        profile: Arc::new(KeyProfile {
            alias: Some("team-key".into()),
            user_id: Some("user-1".into()),
            team_alias: Some("research".into()),
            ..KeyProfile::default()
        }),
    };
    let hash = KeyHash::from_token(&litellm_auth_types::SecretValue::new("sk-team"));
    Arc::new(Keys(HashMap::from([(hash, status)])))
}

struct Released(Arc<Mutex<Vec<u16>>>);

impl Settle for Released {
    fn settle(self: Box<Self>, record: &Record) {
        self.0.lock().unwrap().push(record.outcome.status);
    }
}

/// Refuses requests carrying `x-refuse`; otherwise tags the upstream body and the response.
struct GateHook(Arc<Mutex<Vec<u16>>>);

impl Hook for GateHook {
    fn name(&self) -> &'static str {
        "gate"
    }

    fn reads(&self) -> Facts {
        Facts::CLIENT_HEADERS
    }

    fn writes(&self) -> Targets {
        Targets::ADMISSION | Targets::UPSTREAM_BODY | Targets::RESPONSE_HEADERS
    }

    fn run<'a>(
        &'a self,
        request: &'a Arc<RequestFacts>,
        edit: &'a mut RequestEdit,
    ) -> BoxFuture<'a, Result<Option<Box<dyn Settle>>, Refusal>> {
        Box::pin(async move {
            if request.client_headers.contains_key("x-refuse") {
                return Err(
                    Refusal::new(429, "rate_limit_exceeded", "slow down").with_header(
                        HeaderName::from_static("retry-after"),
                        HeaderValue::from_static("7"),
                    ),
                );
            }
            edit.set_upstream_body_field("metadata", json!({"user_id": "hooked"}));
            edit.add_response_header(
                HeaderName::from_static("x-hooked"),
                HeaderValue::from_static("1"),
            );
            Ok(Some(Box::new(Released(self.0.clone())) as Box<dyn Settle>))
        })
    }
}

#[derive(Debug, PartialEq)]
struct Seen {
    status: u16,
    key_alias: Option<String>,
    team_id: Option<String>,
    deployment: Option<String>,
    usage: Option<(u64, u64, u64)>,
    cost: Option<f64>,
    interrupted: bool,
}

struct Collect(Mutex<Vec<Seen>>);

impl Observer for Collect {
    fn name(&self) -> &'static str {
        "collect"
    }

    fn reads(&self) -> Facts {
        Facts::KEY | Facts::DEPLOYMENT | Facts::USAGE | Facts::COST
    }

    fn observe<'a>(&'a self, record: &'a Record) -> BoxFuture<'a, ()> {
        Box::pin(async move {
            self.0.lock().unwrap().push(Seen {
                status: record.outcome.status,
                key_alias: record.key().and_then(|key| key.alias.clone()),
                team_id: record.key().and_then(|key| key.team_id.clone()),
                deployment: record
                    .outcome
                    .deployment
                    .as_ref()
                    .map(|deployment| deployment.id.to_string()),
                usage: record.outcome.usage.map(|usage| {
                    (
                        usage.input_tokens,
                        usage.cache_read_tokens,
                        usage.output_tokens,
                    )
                }),
                cost: record.outcome.cost,
                interrupted: record.outcome.interrupted,
            });
        })
    }
}

struct Harness {
    app: Router,
    inference: Arc<litellm_gateway_inference::Gateway>,
    bodies: Bodies,
    settled: Arc<Mutex<Vec<u16>>>,
    seen: Arc<Collect>,
}

async fn harness() -> Harness {
    let bodies = Bodies::default();
    let upstream = upstream(bodies.clone()).await;
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
"
    ))
    .unwrap();
    let inference = litellm_gateway::build_inference(&config).unwrap();
    let settled = Arc::new(Mutex::new(Vec::new()));
    let seen = Arc::new(Collect(Mutex::new(Vec::new())));
    litellm_gateway::hooks::install(
        &inference,
        &config,
        vec![Arc::new(GateHook(settled.clone()))],
        vec![seen.clone()],
    )
    .await
    .unwrap();
    let app =
        litellm_gateway::router_with_keys(inference.clone(), &config, None, None, Some(keys()));
    Harness {
        app,
        inference,
        bodies,
        settled,
        seen,
    }
}

async fn send(
    app: &Router,
    path: &str,
    key: &str,
    body: Value,
    refuse: bool,
) -> (u16, axum::http::HeaderMap, Bytes) {
    let mut request = Request::post(path)
        .header("content-type", "application/json")
        .header("authorization", format!("Bearer {key}"));
    if refuse {
        request = request.header("x-refuse", "1");
    }
    let response = app
        .clone()
        .oneshot(request.body(Body::from(body.to_string())).unwrap())
        .await
        .unwrap();
    let status = response.status().as_u16();
    let headers = response.headers().clone();
    let body = to_bytes(response.into_body(), 1 << 20).await.unwrap();
    (status, headers, body)
}

fn messages(stream: bool) -> Value {
    json!({
        "model": "claude",
        "max_tokens": 8,
        "stream": stream,
        "messages": [{"role": "user", "content": "hi"}],
    })
}

fn close(left: f64, right: f64) -> bool {
    (left - right).abs() < 1e-9
}

#[rstest]
#[case::unary(false, (3, 0, 4), 3.0 * 0.001 + 4.0 * 0.002)]
#[case::stream(true, (10, 20, 42), 10.0 * 0.001 + 20.0 * 0.0001 + 42.0 * 0.002)]
#[tokio::test]
async fn messages_are_hooked_and_priced(
    #[case] stream: bool,
    #[case] usage: (u64, u64, u64),
    #[case] cost: f64,
) {
    let harness = harness().await;
    let (status, headers, body) = send(
        &harness.app,
        "/v1/messages",
        "sk-team",
        messages(stream),
        false,
    )
    .await;
    assert_eq!(status, 200, "{body:?}");
    assert_eq!(headers["x-hooked"], "1");
    if !stream {
        assert_eq!(headers["content-length"], body.len().to_string());
    }
    assert_eq!(
        harness.bodies.lock().unwrap()[0]["metadata"],
        json!({"user_id": "hooked"})
    );

    litellm_gateway::hooks::close(&harness.inference).await;
    assert_eq!(*harness.settled.lock().unwrap(), [200]);
    let seen = harness.seen.0.lock().unwrap();
    let [seen] = seen.as_slice() else {
        panic!("{seen:?}")
    };
    assert_eq!(seen.status, 200);
    assert_eq!(seen.key_alias.as_deref(), Some("team-key"));
    assert_eq!(seen.team_id.as_deref(), Some("team-1"));
    assert_eq!(seen.deployment.as_deref(), Some("claude-1"));
    assert_eq!(seen.usage, Some(usage));
    assert!(close(seen.cost.unwrap(), cost), "{seen:?}");
    assert!(!seen.interrupted);
}

#[rstest]
#[tokio::test]
async fn chat_completions_are_priced() {
    let harness = harness().await;
    let request = json!({"model": "gpt", "messages": [{"role": "user", "content": "hi"}]});
    let (status, _, body) = send(
        &harness.app,
        "/v1/chat/completions",
        "sk-master",
        request,
        false,
    )
    .await;
    assert_eq!(status, 200, "{body:?}");

    litellm_gateway::hooks::close(&harness.inference).await;
    let seen = harness.seen.0.lock().unwrap();
    assert_eq!(seen[0].deployment.as_deref(), Some("gpt-1"));
    assert_eq!(seen[0].usage, Some((5, 0, 6)));
    assert_eq!(seen[0].cost, Some(17.0));
    assert_eq!(seen[0].key_alias, None);
}

#[rstest]
#[case::messages("/v1/messages")]
#[case::chat("/v1/chat/completions")]
#[tokio::test]
async fn a_refusal_stops_the_request_with_its_headers(#[case] path: &str) {
    let harness = harness().await;
    let (status, headers, body) = send(&harness.app, path, "sk-team", messages(false), true).await;
    assert_eq!(status, 429);
    assert_eq!(headers["retry-after"], "7");
    let body: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(body["error"]["message"], "slow down");
    assert_eq!(body["error"]["type"], "rate_limit_exceeded");
    assert!(harness.bodies.lock().unwrap().is_empty());

    litellm_gateway::hooks::close(&harness.inference).await;
    let seen = harness.seen.0.lock().unwrap();
    assert_eq!(seen[0].status, 429);
    assert_eq!(seen[0].deployment, None);
}

#[rstest]
#[tokio::test]
async fn a_stream_dropped_early_is_recorded_as_interrupted() {
    let harness = harness().await;
    let response = harness
        .app
        .clone()
        .oneshot(
            Request::post("/v1/messages")
                .header("content-type", "application/json")
                .header("authorization", "Bearer sk-team")
                .body(Body::from(messages(true).to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    drop(response);

    litellm_gateway::hooks::close(&harness.inference).await;
    let seen = harness.seen.0.lock().unwrap();
    assert!(seen[0].interrupted);
    assert_eq!(*harness.settled.lock().unwrap(), [200]);
}
