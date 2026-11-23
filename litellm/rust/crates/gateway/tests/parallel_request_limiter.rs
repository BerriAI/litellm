mod keys;

use std::{future::Future, sync::Arc, time::Duration};

use axum::{
    Extension, Router,
    body::{Body, to_bytes},
    http::{Request, StatusCode, header},
    response::Response,
};
use litellm_config::{Config, Environment};
use litellm_gateway::{
    AppState, KeyLimiter,
    host::auth::AuthStore,
    parallel_request_limiter::{KeyLimits, ModelLimits},
};
use litellm_gateway_auth_core::Token;
use rstest::rstest;
use serde_json::{Value, json};
use tower::ServiceExt;
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{body_partial_json, method},
};

const MASTER_KEY: &str = "sk-master";

/// The hash a key's row is stored under, which names the key in every counter.
fn hash_token(token: &str) -> String {
    Token::new(token).hash().to_string()
}
const KEY: &str = "sk-limited";
/// 8 input, 14 output, 100 read from cache and 5 written to it, which Python charges as 27.
const USAGE: &str = r#"{"input_tokens":8,"output_tokens":14,"cache_read_input_tokens":100,"cache_creation_input_tokens":5}"#;

/// What the database holds for [`KEY`]: a plain row with these limits, and the owner the native
/// app does not serve a key for.
struct Holding {
    limits: KeyLimits,
    owner: Option<&'static str>,
}

impl Holding {
    async fn stored(&self) -> keys::Held {
        let KeyLimits {
            rpm,
            tpm,
            max_parallel_requests,
            model_rpm,
            model_tpm,
        } = &self.limits;
        let columns = json!({
            "rpm_limit": rpm,
            "tpm_limit": tpm,
            "max_parallel_requests": max_parallel_requests,
            "metadata": {"model_rpm_limit": **model_rpm, "model_tpm_limit": **model_tpm},
            "user_id": self.owner,
        });
        keys::Held::holding(KEY, columns).await
    }
}

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

/// The inference routes as the gateway mounts them, for a key holding `limits`: the groups
/// `claude` (also asked for as `sonnet`) and `gpt` answer, and `broken` is refused upstream.
struct Served {
    app: Router,
    limiter: Arc<KeyLimiter>,
    upstream: MockServer,
    _refusing: MockServer,
}

async fn served(limits: KeyLimits) -> Served {
    configured(limits, "").await
}

async fn configured(limits: KeyLimits, general_settings: &str) -> Served {
    held(
        Holding {
            limits,
            owner: None,
        },
        general_settings,
    )
    .await
}

async fn held(holding: Holding, general_settings: &str) -> Served {
    let upstream = provider().await;
    let refusing = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(400).set_body_json(json!({"type": "error",
            "error": {"type": "invalid_request_error", "message": "refused upstream"}})),
        )
        .mount(&refusing)
        .await;
    let deployment = |name: &str, base: &str| {
        format!(
            "  - model_name: {name}\n    litellm_params: {{model: anthropic/{name}-test, api_base: \"{base}\", api_key: sk-upstream}}\n"
        )
    };
    let config = Config::from_yaml(&format!(
        "model_list:\n{}{}{}router_settings:\n  model_group_alias: {{sonnet: claude}}\ngeneral_settings:\n  master_key: {MASTER_KEY}\n{general_settings}",
        deployment("claude", &upstream.uri()),
        deployment("gpt", &upstream.uri()),
        deployment("broken", &refusing.uri()),
    ))
    .unwrap();
    let database = holding.stored().await;
    let state = AppState {
        keys: Some(Arc::new(AuthStore::new(
            keys::memory_cache(),
            database.database(),
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

fn charged_to(limiter: &KeyLimiter, descriptor: &str, value: &str) -> Option<f64> {
    limiter.counted("tokens", descriptor, value)
}

/// What `descriptor`'s token counter holds once it reads `expected`, or what it last held.
async fn settled(
    limiter: &KeyLimiter,
    descriptor: &str,
    value: &str,
    expected: f64,
) -> Option<f64> {
    for _ in 0..200 {
        if charged_to(limiter, descriptor, value) == Some(expected) {
            break;
        }
        tokio::time::sleep(Duration::from_millis(2)).await;
    }
    charged_to(limiter, descriptor, value)
}

/// A body asking `model` for at most `max_tokens`, whose prompt is estimated at one token.
fn asking(model: &str, max_tokens: Option<u64>) -> Value {
    let body = json!({"model": model, "messages": [{"role": "user", "content": "hi"}]});
    match (body, max_tokens) {
        (Value::Object(mut body), Some(max_tokens)) => {
            body.insert("max_tokens".to_owned(), json!(max_tokens));
            Value::Object(body)
        }
        (body, _) => body,
    }
}

fn complete() -> Value {
    asking("claude", Some(16))
}

fn streamed() -> Value {
    json!({"model": "claude", "max_tokens": 16, "stream": true,
        "messages": [{"role": "user", "content": "hi"}]})
}

fn failing() -> Value {
    asking("broken", Some(16))
}

async fn send_as(app: &Router, token: &str, body: Value) -> Response {
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

async fn send(app: &Router, body: Value) -> Response {
    send_as(app, KEY, body).await
}

fn by_model(limits: &[(&str, i64)]) -> ModelLimits {
    Arc::new(
        limits
            .iter()
            .map(|(model, limit)| ((*model).to_owned(), *limit))
            .collect(),
    )
}

async fn json_body(response: Response) -> Value {
    serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap()
}

fn ratelimit_headers(response: &Response) -> Vec<(String, String)> {
    let mut found: Vec<(String, String)> = response
        .headers()
        .iter()
        .filter(|(name, _)| name.as_str().starts_with("x-ratelimit-api_key"))
        .map(|(name, value)| (name.to_string(), value.to_str().unwrap().to_owned()))
        .collect();
    found.sort();
    found
}

fn headers(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
    pairs
        .iter()
        .map(|(name, value)| ((*name).to_owned(), (*value).to_owned()))
        .collect()
}

/// A request that goes to Python over its key is counted there, so it must count nothing here.
#[tokio::test(flavor = "multi_thread")]
async fn a_key_the_native_app_does_not_serve_counts_nothing_against_its_limits() {
    let Served {
        app,
        limiter,
        upstream,
        ..
    } = held(
        Holding {
            limits: KeyLimits {
                rpm: Some(1),
                tpm: Some(100),
                ..KeyLimits::default()
            },
            owner: Some("owner"),
        },
        "",
    )
    .await;

    let first = send(&app, complete()).await;
    let second = send(&app, complete()).await;

    assert_eq!(
        (first.status(), second.status()),
        (StatusCode::FORBIDDEN, StatusCode::FORBIDDEN)
    );
    assert_eq!(charged_to(&limiter, "api_key", &hash_token(KEY)), None);
    assert!(upstream.received_requests().await.unwrap().is_empty());
}

#[tokio::test(flavor = "multi_thread")]
async fn a_keys_rpm_limit_counts_each_call_and_refuses_the_one_past_it() {
    let Served { app, .. } = served(KeyLimits {
        rpm: Some(2),
        ..KeyLimits::default()
    })
    .await;

    let first = send(&app, complete()).await;
    let second = send(&app, complete()).await;
    let third = send(&app, complete()).await;

    assert_eq!(
        ratelimit_headers(&first),
        headers(&[
            ("x-ratelimit-api_key-limit-requests", "2"),
            ("x-ratelimit-api_key-remaining-requests", "1"),
        ])
    );
    assert_eq!(second.status(), StatusCode::OK);
    assert_eq!(
        ratelimit_headers(&second),
        headers(&[
            ("x-ratelimit-api_key-limit-requests", "2"),
            ("x-ratelimit-api_key-remaining-requests", "0"),
        ])
    );
    assert_eq!(third.status(), StatusCode::TOO_MANY_REQUESTS);
    let refusal: Vec<(&str, &str)> = third
        .headers()
        .iter()
        .filter(|(name, _)| *name != header::CONTENT_TYPE && *name != header::CONTENT_LENGTH)
        .map(|(name, value)| (name.as_str(), value.to_str().unwrap()))
        .collect();
    assert_eq!(
        refusal.iter().map(|(name, _)| *name).collect::<Vec<_>>(),
        [
            "retry-after",
            "rate_limit_type",
            "rate_limited_by",
            "reset_at",
            "x-litellm-call-id"
        ]
    );
    assert_eq!(
        &refusal[..3],
        [
            ("retry-after", "60"),
            ("rate_limit_type", "requests"),
            ("rate_limited_by", "api_key"),
        ]
    );
    let body = json_body(third).await;
    assert_eq!(
        (&body["error"]["type"], &body["error"]["litellm_error_type"]),
        (&json!("rate_limit_error"), &json!("throttling_error"))
    );
    let message = body["error"]["message"].as_str().unwrap();
    let expected = format!(
        "Rate limit exceeded for api_key: {}. Limit type: requests. Current limit: 2, Remaining: 0. Limit resets at: ",
        hash_token(KEY)
    );
    assert!(message.starts_with(&expected), "{message}");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_complete_answer_gives_its_parallel_slot_back_before_it_goes_out() {
    let Served { app, .. } = served(KeyLimits {
        max_parallel_requests: Some(1),
        ..KeyLimits::default()
    })
    .await;

    let answered = send(&app, complete()).await;
    let again = send(&app, complete()).await;
    let failed = send(&app, failing()).await;
    let after_the_failure = send(&app, complete()).await;

    assert_eq!(
        (
            answered.status(),
            ratelimit_headers(&answered),
            again.status(),
            failed.status(),
            ratelimit_headers(&failed),
            after_the_failure.status()
        ),
        (
            StatusCode::OK,
            headers(&[
                ("x-ratelimit-api_key-limit-max_parallel_requests", "1"),
                ("x-ratelimit-api_key-remaining-max_parallel_requests", "0"),
            ]),
            StatusCode::OK,
            StatusCode::BAD_REQUEST,
            vec![],
            StatusCode::OK
        )
    );
}

async fn read_to_its_end(response: Response) {
    to_bytes(response.into_body(), usize::MAX).await.unwrap();
}

async fn dropped(response: Response) {
    drop(response);
}

async fn served_once_released(app: &Router) -> bool {
    for _ in 0..200 {
        if send(app, complete()).await.status() == StatusCode::OK {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(2)).await;
    }
    false
}

#[rstest]
#[case::read(read_to_its_end)]
#[case::dropped(dropped)]
#[tokio::test(flavor = "multi_thread")]
async fn a_streamed_answer_holds_its_parallel_slot_until_its_body_is_done<F>(
    #[case] finish: fn(Response) -> F,
) where
    F: Future<Output = ()>,
{
    let Served { app, .. } = served(KeyLimits {
        max_parallel_requests: Some(1),
        ..KeyLimits::default()
    })
    .await;

    let streaming = send(&app, streamed()).await;
    let refused = send(&app, complete()).await;

    assert_eq!(
        (
            streaming.status(),
            ratelimit_headers(&streaming),
            refused.status(),
            refused
                .headers()
                .get("rate_limit_type")
                .map(|kind| kind.to_str().unwrap().to_owned())
        ),
        (
            StatusCode::OK,
            vec![],
            StatusCode::TOO_MANY_REQUESTS,
            Some("max_parallel_requests".to_owned())
        )
    );
    finish(streaming).await;
    assert!(served_once_released(&app).await);
}

#[rstest]
#[case::the_master_key(MASTER_KEY, KeyLimits { rpm: Some(0), max_parallel_requests: Some(0), ..KeyLimits::default() })]
#[case::a_key_without_limits(KEY, KeyLimits::default())]
#[tokio::test(flavor = "multi_thread")]
async fn a_caller_without_limits_of_its_own_is_never_refused(
    #[case] token: &str,
    #[case] key_limits: KeyLimits,
) {
    let Served { app, .. } = served(key_limits).await;

    let first = send_as(&app, token, complete()).await;
    let second = send_as(&app, token, complete()).await;

    assert_eq!(
        (first.status(), second.status(), ratelimit_headers(&second)),
        (StatusCode::OK, StatusCode::OK, vec![])
    );
}

#[rstest]
#[case::complete(complete())]
#[case::streamed_to_its_end(streamed())]
#[tokio::test(flavor = "multi_thread")]
async fn a_served_call_charges_its_tokens_less_the_cached_reads_to_its_key_and_model(
    #[case] body: Value,
) {
    let Served { app, limiter, .. } = served(KeyLimits {
        rpm: Some(5),
        ..KeyLimits::default()
    })
    .await;
    let hash = hash_token(KEY);

    read_to_its_end(send(&app, body).await).await;

    assert_eq!(
        (
            settled(&limiter, "api_key", &hash, 27.0).await,
            settled(&limiter, "model_per_key", &format!("{hash}:claude"), 27.0).await
        ),
        (Some(27.0), Some(27.0))
    );
}

#[rstest]
#[case::its_key("api_key", "litellm_proxy_master_key")]
#[case::its_user("user", "default_user_id")]
#[case::its_model("model_per_key", "litellm_proxy_master_key:claude")]
#[tokio::test(flavor = "multi_thread")]
async fn the_master_key_is_charged_as_the_proxy_names_it(
    #[case] descriptor: &str,
    #[case] value: &str,
) {
    let Served { app, limiter, .. } = served(KeyLimits::default()).await;

    send_as(&app, MASTER_KEY, complete()).await;

    assert_eq!(settled(&limiter, descriptor, value, 27.0).await, Some(27.0));
}

#[rstest]
#[case::a_dropped_stream(streamed())]
#[case::a_failed_call(failing())]
#[tokio::test(flavor = "multi_thread")]
async fn a_call_that_tells_no_usage_charges_nothing(#[case] body: Value) {
    let Served { app, limiter, .. } = served(KeyLimits {
        rpm: Some(5),
        max_parallel_requests: Some(1),
        ..KeyLimits::default()
    })
    .await;

    drop(send(&app, body).await);

    assert!(served_once_released(&app).await);
    assert_eq!(
        settled(&limiter, "api_key", &hash_token(KEY), 27.0).await,
        Some(27.0),
        "the call that told no usage was charged too"
    );
}

#[rstest]
#[case::complete(complete(), StatusCode::OK)]
#[case::streamed(streamed(), StatusCode::OK)]
#[case::failed(failing(), StatusCode::BAD_REQUEST)]
#[tokio::test(flavor = "multi_thread")]
async fn every_answer_of_the_handler_names_the_keys_rpm_limit_and_authentications_refusal_never(
    #[case] body: Value,
    #[case] status: StatusCode,
) {
    let Served { app, .. } = served(KeyLimits {
        rpm: Some(1),
        ..KeyLimits::default()
    })
    .await;
    let unlimited = served(KeyLimits::default()).await;

    let admitted = send(&app, body.clone()).await;
    let refused = send(&app, body.clone()).await;
    let without_a_limit = send(&unlimited.app, body).await;

    let named = |response: &Response| response.headers().get("x-litellm-key-rpm-limit").cloned();
    assert_eq!(
        (
            admitted.status(),
            named(&admitted),
            refused.status(),
            named(&refused),
            named(&without_a_limit)
        ),
        (
            status,
            Some("1".parse().unwrap()),
            StatusCode::TOO_MANY_REQUESTS,
            None,
            None
        )
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn a_keys_tpm_limit_reserves_what_the_body_asks_for_and_settles_what_was_used() {
    let Served { app, limiter, .. } = served(KeyLimits {
        tpm: Some(60),
        ..KeyLimits::default()
    })
    .await;
    let hash = hash_token(KEY);

    let admitted = send(&app, asking("claude", Some(50))).await;

    assert_eq!(admitted.status(), StatusCode::OK);
    assert_eq!(admitted.headers()["x-litellm-key-tpm-limit"], "60");
    assert_eq!(
        ratelimit_headers(&admitted),
        headers(&[
            ("x-ratelimit-api_key-limit-tokens", "60"),
            ("x-ratelimit-api_key-remaining-tokens", "9"),
        ])
    );
    assert_eq!(
        settled(&limiter, "api_key", &hash, 27.0).await,
        Some(27.0),
        "the 51 reserved were not settled to the 27 used"
    );
    let over = send(&app, asking("claude", Some(50))).await;
    assert_eq!(over.status(), StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(over.headers()["x-litellm-key-tpm-limit"], "60");
    assert!(!over.headers().contains_key("retry-after"));
    let error = json_body(over).await["error"].clone();
    assert_eq!(error["type"], "rate_limit_error");
    assert_eq!(error.get("litellm_error_type"), None);
    assert!(
        error["message"]
            .as_str()
            .unwrap()
            .starts_with(&format!("Rate limit exceeded for api_key: {hash}. Limit type: tokens. Current limit: 60, Remaining: 33."))
    );
    assert_eq!(
        send(&app, asking("claude", Some(20))).await.status(),
        StatusCode::OK,
        "the refused call kept its reservation"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn a_failed_call_gives_its_reserved_tokens_back() {
    let Served { app, .. } = served(KeyLimits {
        tpm: Some(60),
        ..KeyLimits::default()
    })
    .await;

    let failed = send(&app, asking("broken", Some(50))).await;
    let next = send(&app, asking("claude", Some(50))).await;

    assert_eq!(
        (failed.status(), next.status()),
        (StatusCode::BAD_REQUEST, StatusCode::OK)
    );
}

#[rstest]
#[case::none_sent(None, json!(25))]
#[case::one_sent(Some(50), json!(50))]
#[tokio::test(flavor = "multi_thread")]
async fn a_body_that_sets_no_output_cap_is_capped_at_a_quarter_of_the_token_limit(
    #[case] sent: Option<u64>,
    #[case] reached: Value,
) {
    let Served { app, upstream, .. } = served(KeyLimits {
        tpm: Some(100),
        rpm: Some(5),
        ..KeyLimits::default()
    })
    .await;

    send(&app, asking("claude", sent)).await;

    let received = upstream.received_requests().await.unwrap();
    let provider_saw: Option<Value> = received.last().map(|request| {
        serde_json::from_slice::<Value>(&request.body).unwrap()["max_tokens"].clone()
    });
    assert_eq!(provider_saw, Some(reached));
}

#[tokio::test(flavor = "multi_thread")]
async fn a_body_only_python_estimates_reserves_the_least() {
    let Served { app, .. } = served(KeyLimits {
        tpm: Some(60),
        ..KeyLimits::default()
    })
    .await;
    let body = json!({"model": "claude", "max_tokens": 1000, "messages": [{"role": "user", "content": [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AA=="}},
    ]}]});

    assert_eq!(send(&app, body).await.status(), StatusCode::OK);
}

#[rstest]
#[case::requests(
    KeyLimits { model_rpm: by_model(&[("claude", 1)]), ..KeyLimits::default() },
    "requests"
)]
#[case::tokens(
    KeyLimits { model_tpm: by_model(&[("claude", 60)]), ..KeyLimits::default() },
    "tokens"
)]
#[tokio::test(flavor = "multi_thread")]
async fn a_keys_model_limit_counts_only_the_model_it_names(
    #[case] limits: KeyLimits,
    #[case] limit_type: &str,
) {
    let Served { app, limiter, .. } = served(limits).await;

    let admitted = send(&app, asking("claude", Some(50))).await;
    assert_eq!(admitted.status(), StatusCode::OK);
    settled(
        &limiter,
        "model_per_key",
        &format!("{}:claude", hash_token(KEY)),
        27.0,
    )
    .await;
    let refused = send(&app, asking("claude", Some(50))).await;
    let other = send(&app, asking("gpt", Some(50))).await;

    assert_eq!(refused.status(), StatusCode::TOO_MANY_REQUESTS);
    assert!(!refused.headers().contains_key("retry-after"));
    let error = json_body(refused).await["error"].clone();
    assert_eq!(error.get("litellm_error_type"), None);
    assert!(error["message"].as_str().unwrap().starts_with(&format!(
        "Rate limit exceeded for model_per_key: {}:claude. Limit type: {limit_type}.",
        hash_token(KEY)
    )));
    assert_eq!(other.status(), StatusCode::OK);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_model_limit_is_counted_under_the_models_group() {
    let Served { app, .. } = served(KeyLimits {
        model_rpm: by_model(&[("claude", 1)]),
        ..KeyLimits::default()
    })
    .await;

    let by_alias = send(&app, asking("sonnet", Some(50))).await;
    let by_group = send(&app, asking("claude", Some(50))).await;

    assert_eq!(
        (by_alias.status(), by_group.status()),
        (StatusCode::OK, StatusCode::TOO_MANY_REQUESTS)
    );
}

/// The proxy logs the group an alias names as the call's model group, so a call that reserved
/// no tokens settles under the name its model limit is counted by.
#[tokio::test(flavor = "multi_thread")]
async fn a_call_through_an_alias_is_charged_to_the_group_the_alias_names() {
    let Served { app, limiter, .. } = served(KeyLimits::default()).await;
    let hash = hash_token(KEY);

    send(&app, asking("sonnet", Some(50))).await;

    assert_eq!(
        (
            settled(&limiter, "model_per_key", &format!("{hash}:claude"), 27.0).await,
            charged_to(&limiter, "model_per_key", &format!("{hash}:sonnet")),
        ),
        (Some(27.0), None)
    );
}

#[rstest]
#[case::output("output", 14.0)]
#[case::input("input", 13.0)]
#[case::anything_else("tokens", 27.0)]
#[tokio::test(flavor = "multi_thread")]
async fn a_served_call_is_charged_the_tokens_the_configured_type_counts(
    #[case] token_rate_limit_type: &str,
    #[case] charged: f64,
) {
    let Served { app, limiter, .. } = configured(
        KeyLimits::default(),
        &format!("  token_rate_limit_type: {token_rate_limit_type}\n"),
    )
    .await;

    send(&app, complete()).await;

    assert_eq!(
        settled(&limiter, "api_key", &hash_token(KEY), charged).await,
        Some(charged)
    );
}
