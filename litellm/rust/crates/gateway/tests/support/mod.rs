// Shared across integration-test targets; each target uses a different subset.
#![allow(dead_code)]

use std::sync::Arc;

use axum::{Extension, Router, body::Body, http::Request, response::Response};
use litellm_config::{Config, Environment};
use litellm_gateway::{AppState, KeyLimiter, host::auth::AuthStore};
use serde_json::{Value, json};
use tower::ServiceExt;

use crate::keys;
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{body_partial_json, method},
};

pub const MASTER_KEY: &str = "sk-master";
/// 8 input, 14 output, 100 read from cache and 5 written to it, which Python charges as 27.
const USAGE: &str = r#"{"input_tokens":8,"output_tokens":14,"cache_read_input_tokens":100,"cache_creation_input_tokens":5}"#;

/// The provider every deployment but `broken` calls: it answers a stream when asked for one,
/// and a complete message otherwise, each with [`USAGE`].
async fn provider() -> MockServer {
    let upstream = MockServer::start().await;
    let stream = concat!(
        "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-test\",\"content\":[],\"usage\":{\"input_tokens\":8,\"cache_read_input_tokens\":100,\"cache_creation_input_tokens\":5}}}\n\n",
        "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":14}}\n\n",
        "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
    );
    Mock::given(method("POST"))
        .and(body_partial_json(json!({"stream": true})))
        .respond_with(ResponseTemplate::new(200).set_body_raw(stream, "text/event-stream"))
        .mount(&upstream)
        .await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(
            format!(
                r#"{{"id":"msg_1","type":"message","role":"assistant","model":"claude-test","content":[{{"type":"text","text":"hi"}}],"stop_reason":"end_turn","usage":{USAGE}}}"#
            ),
            "application/json",
        ))
        .mount(&upstream)
        .await;
    upstream
}

/// The inference routes as the gateway mounts them: the groups `claude` (also asked for as
/// `sonnet`, serving 10 requests and 100000 tokens a minute) and `gpt` answer, and `broken` is
/// refused upstream.
pub struct Served {
    pub app: Router,
    pub limiter: Arc<KeyLimiter>,
    pub upstream: MockServer,
    _refusing: MockServer,
}

/// The gateway over `database`, with `general_settings` set besides its master key and
/// `litellm_settings` as its own section.
pub async fn served(
    database: keys::Held,
    general_settings: &str,
    litellm_settings: &str,
) -> Served {
    let upstream = provider().await;
    let refusing = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(400).set_body_json(json!({"type": "error",
            "error": {"type": "invalid_request_error", "message": "refused upstream"}})),
        )
        .mount(&refusing)
        .await;
    let deployment = |name: &str, base: &str, serves: &str| {
        format!(
            "  - model_name: {name}\n    litellm_params: {{model: anthropic/{name}-test, api_base: \"{base}\", api_key: sk-upstream{serves}}}\n"
        )
    };
    let config = Config::from_yaml(&format!(
        "model_list:\n{}{}{}router_settings:\n  model_group_alias: {{sonnet: claude}}\nenvironment_variables:\n  LITELLM_LICENSE: any\ngeneral_settings:\n  master_key: {MASTER_KEY}\n{general_settings}{litellm_settings}",
        deployment("claude", &upstream.uri(), ", rpm: 10, tpm: 100000"),
        deployment("gpt", &upstream.uri(), ""),
        deployment("broken", &refusing.uri(), ""),
    ))
    .unwrap();
    let state = AppState {
        keys: Some(Arc::new(AuthStore::new(
            keys::memory_cache(),
            database.database(),
            database.teams(),
        ))),
        ..AppState::build(&config, &Environment::default())
            .await
            .unwrap()
    };
    let limiter = state.key_limiter.clone();
    Served {
        // The database lives as long as the routes that read it, whichever fields a test binds.
        app: litellm_gateway::router(state, &config).layer(Extension(Arc::new(database))),
        limiter,
        upstream,
        _refusing: refusing,
    }
}

/// A body asking `model` for at most `max_tokens`, whose prompt is estimated at one token.
pub fn asking(model: &str, max_tokens: Option<u64>) -> Value {
    let body = json!({"model": model, "messages": [{"role": "user", "content": "hi"}]});
    match (body, max_tokens) {
        (Value::Object(mut body), Some(max_tokens)) => {
            body.insert("max_tokens".to_owned(), json!(max_tokens));
            Value::Object(body)
        }
        (body, _) => body,
    }
}

pub async fn send_as(app: &Router, token: &str, body: Value) -> Response {
    app.clone()
        .oneshot(
            Request::post("/v1/messages")
                .header("authorization", format!("Bearer {token}"))
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap()
}
