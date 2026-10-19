//! The configurable request hooks on real requests: headers forwarded and stripped, provider
//! metadata set from the caller, and spend metadata handed to observers.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use axum::{
    Router,
    body::{Body, Bytes, to_bytes},
    http::{HeaderMap, Request, header::CONTENT_TYPE},
    response::IntoResponse,
};

use futures_util::future::BoxFuture;
use litellm_auth_types::SecretValue;
use litellm_config::Config;
use litellm_gateway_auth::keys::{
    AllowedRoutes, KeyHash, KeyLookup, KeyModels, KeyProfile, KeyStatus, LookupFuture,
};
use litellm_gateway_hooks::{
    Facts, KeyFacts, Observer, Observers, Pipeline, Record, RequestFacts, Route,
};
use rstest::rstest;
use serde_json::{Map, Value, json};
use tokio::net::TcpListener;
use tower::ServiceExt;

const MESSAGE: &str = r#"{"id":"msg_1","type":"message","role":"assistant","model":"claude-x","content":[{"type":"text","text":"hi"}],"stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":3,"output_tokens":4}}"#;

type Seen = Arc<Mutex<Vec<(HeaderMap, Value)>>>;

async fn upstream() -> (String, Seen) {
    let seen = Seen::default();
    let recorded = seen.clone();
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let app = Router::new().fallback(move |request: Request<Body>| {
        let recorded = recorded.clone();
        async move {
            let headers = request.headers().clone();
            let body = to_bytes(request.into_body(), 1 << 20).await.unwrap();
            recorded.lock().unwrap().push((
                headers,
                serde_json::from_slice(&body).unwrap_or(Value::Null),
            ));
            ([(CONTENT_TYPE, "application/json")], MESSAGE).into_response()
        }
    });
    tokio::spawn(async move { axum::serve(listener, app).await });
    (format!("http://{address}"), seen)
}

struct Keys(HashMap<KeyHash, KeyStatus>);

impl KeyLookup for Keys {
    fn lookup<'a>(&'a self, hash: &'a KeyHash) -> LookupFuture<'a> {
        Box::pin(async move { Ok(self.0.get(hash).cloned()) })
    }
}

const HOOKS: &str = r#"
  hooks:
    - forward_headers:
        headers: ["x-trace-*", "x-cost-center", "authorization"]
    - strip_headers:
        headers: ["x-cost-center", "anthropic-beta"]
    - provider_metadata:
        body:
          metadata.user_id: "{end_user|key.user_id}"
        headers:
          x-caller-team: "team-{key.team_alias}"
"#;

async fn gateway(hooks: &str) -> (Router, Seen) {
    let (address, seen) = upstream().await;
    let config = Config::from_yaml(&format!(
        "model_list:
  - model_name: claude
    litellm_params:
      model: anthropic/claude-x
      api_base: {address}
      api_key: sk-upstream
general_settings:
  master_key: sk-master
litellm_settings:{hooks}"
    ))
    .unwrap();
    let inference = litellm_gateway::build_inference(&config).unwrap();
    let hooks = litellm_gateway::request_hooks::from_config(&config).unwrap();
    litellm_gateway::hooks::install(&inference, &config, hooks, Vec::new())
        .await
        .unwrap();
    let key = KeyStatus::Active {
        expires_at: None,
        models: KeyModels::default(),
        allowed_routes: AllowedRoutes::default(),
        team_id: Some("team-1".into()),
        profile: Arc::new(KeyProfile {
            user_id: Some("user-1".into()),
            team_alias: Some("research".into()),
            ..KeyProfile::default()
        }),
    };
    let keys: Arc<dyn KeyLookup> = Arc::new(Keys(HashMap::from([(
        KeyHash::from_token(&SecretValue::new("sk-key")),
        key,
    )])));
    let app = litellm_gateway::router_with_keys(inference, &config, None, None, Some(keys));
    (app, seen)
}

async fn send(app: &Router, path: &str, body: Value) -> u16 {
    let response = app
        .clone()
        .oneshot(
            Request::post(path)
                .header("content-type", "application/json")
                .header("authorization", "Bearer sk-key")
                .header("x-trace-id", "trace-1")
                .header("x-cost-center", "cc-9")
                .header("anthropic-beta", "some-beta")
                .header("x-other", "kept-out")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    response.status().as_u16()
}

fn header<'a>(headers: &'a HeaderMap, name: &str) -> Option<&'a str> {
    headers.get(name).map(|value| value.to_str().unwrap())
}

#[rstest]
#[tokio::test]
async fn messages_carry_the_configured_headers_and_metadata() {
    let (app, seen) = gateway(HOOKS).await;
    let body = json!({
        "model": "claude",
        "max_tokens": 8,
        "metadata": {"tag": "t"},
        "messages": [{"role": "user", "content": "hi"}],
    });
    assert_eq!(send(&app, "/v1/messages", body).await, 200);
    let (headers, body) = seen.lock().unwrap().pop().unwrap();
    assert_eq!(header(&headers, "x-trace-id"), Some("trace-1"));
    assert_eq!(header(&headers, "x-caller-team"), Some("team-research"));
    // Stripped after forwarding, and the gateway's own pass-through is stripped too.
    assert_eq!(header(&headers, "x-cost-center"), None);
    assert_eq!(header(&headers, "anthropic-beta"), None);
    assert_eq!(header(&headers, "x-other"), None);
    // Credentials are never forwarded, even when listed.
    assert_ne!(header(&headers, "authorization"), Some("Bearer sk-key"));
    assert_eq!(header(&headers, "x-api-key"), Some("sk-upstream"));
    // The provider keeps only the `metadata` fields it knows.
    assert_eq!(body["metadata"]["user_id"], "user-1");
}

#[rstest]
#[tokio::test]
async fn the_end_user_comes_before_the_key_user() {
    let (app, seen) = gateway(HOOKS).await;
    let body = json!({
        "model": "claude",
        "max_tokens": 8,
        "litellm_metadata": {"user": "end-7"},
        "messages": [{"role": "user", "content": "hi"}],
    });
    assert_eq!(send(&app, "/v1/messages", body).await, 200);
    let (_, body) = seen.lock().unwrap().pop().unwrap();
    assert_eq!(body["metadata"]["user_id"], "end-7");
}

#[rstest]
#[tokio::test]
async fn chat_completions_take_the_headers_only() {
    let (app, seen) = gateway(HOOKS).await;
    let body = json!({
        "model": "claude",
        "max_tokens": 8,
        "messages": [{"role": "user", "content": "hi"}],
    });
    assert_eq!(send(&app, "/v1/chat/completions", body).await, 200);
    let (headers, body) = seen.lock().unwrap().pop().unwrap();
    assert_eq!(header(&headers, "x-trace-id"), Some("trace-1"));
    assert_eq!(header(&headers, "x-caller-team"), Some("team-research"));
    assert_eq!(header(&headers, "x-cost-center"), None);
    assert_eq!(body.get("metadata"), None);
}

#[rstest]
#[tokio::test]
async fn without_hooks_the_default_pass_through_is_unchanged() {
    let (app, seen) = gateway("\n  drop_params: true\n").await;
    let body = json!({
        "model": "claude",
        "max_tokens": 8,
        "messages": [{"role": "user", "content": "hi"}],
    });
    assert_eq!(send(&app, "/v1/messages", body).await, 200);
    let (headers, _) = seen.lock().unwrap().pop().unwrap();
    assert_eq!(header(&headers, "anthropic-beta"), Some("some-beta"));
    assert_eq!(header(&headers, "x-trace-id"), None);
}

struct Annotations(Mutex<Vec<Option<Value>>>);

impl Observer for Annotations {
    fn name(&self) -> &'static str {
        "annotations"
    }

    fn reads(&self) -> Facts {
        Facts::ANNOTATIONS
    }

    fn observe<'a>(&'a self, record: &'a Record) -> BoxFuture<'a, ()> {
        Box::pin(async move {
            self.0.lock().unwrap().push(
                record
                    .annotations
                    .get(litellm_gateway::spend::SPEND_LOGS_METADATA)
                    .cloned(),
            );
        })
    }
}

#[rstest]
#[tokio::test]
async fn spend_metadata_reaches_observers() {
    let config = Config::from_yaml(
        r#"
model_list: []
litellm_settings:
  hooks:
    - spend_metadata:
        from_headers:
          x-litellm-spend-logs-metadata: json
          x-cost-center: cost_center
        from_key_metadata: [owner]
"#,
    )
    .unwrap();
    let observed = Arc::new(Annotations(Mutex::default()));
    let pipeline = Pipeline::new(
        litellm_gateway::request_hooks::from_config(&config).unwrap(),
        Observers::start(vec![observed.clone()], 8),
        Arc::default(),
    );
    let run = async |headers: &[(&'static str, &'static str)]| {
        let mut map = HeaderMap::new();
        for (name, value) in headers {
            map.insert(*name, value.parse().unwrap());
        }
        let Value::Object(metadata) = json!({"owner": "platform"}) else {
            unreachable!()
        };
        let key = KeyFacts {
            metadata: Arc::new(metadata),
            ..KeyFacts::default()
        };
        let request = RequestFacts::new(
            "r".into(),
            Route::Messages,
            Some(Arc::new(key)),
            None,
            "claude".into(),
            Arc::new(map),
            false,
            None,
            Bytes::from_static(b"{}"),
        );
        let admission = pipeline.admit(Arc::new(request)).await.unwrap();
        pipeline.finish(
            admission,
            litellm_gateway_hooks::Outcome {
                deployment: None,
                status: 200,
                usage: None,
                cost: None,
                time_to_first_byte: None,
                ended_at: std::time::SystemTime::now(),
                interrupted: false,
            },
            None,
        );
    };
    run(&[
        (
            "x-litellm-spend-logs-metadata",
            r#"{"project": "p1", "owner": "caller"}"#,
        ),
        ("x-cost-center", "cc-9"),
    ])
    .await;
    run(&[("x-litellm-spend-logs-metadata", "not json")]).await;
    pipeline.close().await;
    let mut expected = Map::new();
    expected.insert("project".into(), json!("p1"));
    expected.insert("owner".into(), json!("platform"));
    expected.insert("cost_center".into(), json!("cc-9"));
    assert_eq!(
        *observed.0.lock().unwrap(),
        [
            Some(Value::Object(expected)),
            Some(json!({"owner": "platform"}))
        ]
    );
}

#[rstest]
#[case::unknown_hook("    - rewrite_everything: {}\n", "unknown hook rewrite_everything")]
#[case::bad_template(
    "    - provider_metadata:\n        headers:\n          x-a: \"{key.secret}\"\n",
    "provider_metadata: unknown fact key.secret"
)]
#[case::credential_header(
    "    - provider_metadata:\n        headers:\n          authorization: \"{key.user_id}\"\n",
    "provider_metadata: authorization cannot be set by a hook"
)]
#[case::unknown_setting(
    "    - strip_headers:\n        names: [x-a]\n",
    "strip_headers: unknown field `names`"
)]
fn misconfigured_hooks_stop_the_gateway(#[case] hooks: &str, #[case] message: &str) {
    let config = Config::from_yaml(&format!(
        "model_list: []\nlitellm_settings:\n  hooks:\n{hooks}"
    ))
    .unwrap();
    let error = litellm_gateway::request_hooks::from_config(&config)
        .err()
        .unwrap()
        .to_string();
    assert!(error.contains(message), "{error}");
}
