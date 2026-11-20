mod keys;
mod support;

use std::{sync::Arc, time::Duration};

use axum::{
    Router,
    body::{Body, to_bytes},
    http::{Request, StatusCode},
    response::Response,
};
use keys::plain;
use litellm_gateway::{
    KeyLimiter,
    host::auth::{KeyRows, RowFuture},
};
use litellm_gateway_auth::{
    hash_token,
    keys::{KeyLimits, ModelLimits},
};
use litellm_repositories::{KeyView, RawJson};
use rstest::rstest;
use serde_json::{Value, json};
use support::{MASTER_KEY, Served, asking, send_as};
use tower::ServiceExt;
use wiremock::{Mock, MockServer, ResponseTemplate, matchers::body_partial_json};

const RESERVED: &str = "litellm_settings:\n  callbacks: [dynamic_rate_limiter_v3]\n  priority_reservation: {prod: 0.9, dev: 0.1}\n";
const SPLIT_BY_KEY: &str = "litellm_settings:\n  callbacks: [dynamic_rate_limiter_v3]\n  priority_reservation: {prod: 0.9, dev: 0.1}\n  priority_reservation_settings:\n    priority_class_share_config:\n      split_equally_among_active: [key]\n";

/// Each key active under the priority its name carries: `sk-prod`, `sk-dev-1`, `sk-dev-2`.
/// `sk-dev-1` also holds one request a minute of its own.
struct Prioritized;

impl KeyRows for Prioritized {
    fn find<'a>(&'a self, hash: &'a str) -> RowFuture<'a> {
        Box::pin(async move {
            let named = |token: &str| hash == hash_token(token);
            let metadata = ["sk-prod", "sk-dev-1", "sk-dev-2", "sk-limited-dev"]
                .into_iter()
                .find(|token| named(token))
                .map(|token| if token == "sk-prod" { "prod" } else { "dev" })
                .map_or_else(|| json!({}), |priority| json!({"priority": priority}));
            Ok(Some(KeyView {
                token: hash.to_owned(),
                rpm_limit: named("sk-limited-dev").then_some(1),
                metadata: RawJson(serde_json::value::to_raw_value(&metadata).unwrap()),
                ..plain("")
            }))
        })
    }
}

const BORROWING: &str = "litellm_settings:\n  callbacks: [dynamic_rate_limiter_v3]\n  priority_reservation_settings:\n    model_limit_borrowing_config:\n      borrow_idle_capacity: [key]\n";
const BORROWING_TWICE_ITS_LIMIT: &str = "litellm_settings:\n  callbacks: [dynamic_rate_limiter_v3]\n  priority_reservation_settings:\n    model_limit_borrowing_config:\n      borrow_idle_capacity: [key]\n      max_borrow_multiple: 2\n";
const KEY: &str = "sk-borrower";

/// A database holding, under every hash, a plain row with these limits.
struct Holding(KeyLimits);

impl KeyRows for Holding {
    fn find<'a>(&'a self, hash: &'a str) -> RowFuture<'a> {
        Box::pin(async move {
            let KeyLimits {
                rpm,
                tpm,
                max_parallel_requests,
                model_rpm,
                model_tpm,
            } = &self.0;
            let metadata = json!({"model_rpm_limit": **model_rpm, "model_tpm_limit": **model_tpm});
            Ok(Some(KeyView {
                token: hash.to_owned(),
                rpm_limit: *rpm,
                tpm_limit: *tpm,
                max_parallel_requests: max_parallel_requests
                    .map(|limit| i32::try_from(limit).unwrap()),
                metadata: RawJson(serde_json::value::to_raw_value(&metadata).unwrap()),
                ..plain("")
            }))
        })
    }
}

fn by_model(model: &str, limit: i64) -> ModelLimits {
    Arc::new([(model.to_owned(), limit)].into_iter().collect())
}

/// The gateway for a key that holds `limits`, over the same groups as [`served`].
async fn holding(limits: KeyLimits, litellm_settings: &str) -> Served {
    support::served(Arc::new(Holding(limits)), "", litellm_settings).await
}

/// The gateway over the group `claude`, which serves 10 requests and 100000 tokens a minute.
async fn served(litellm_settings: &str) -> Served {
    support::served(Arc::new(Prioritized), "", litellm_settings).await
}

async fn call(app: &Router, token: &str, model: &str) -> Response {
    send_as(app, token, asking(model, Some(16))).await
}

async fn statuses(app: &Router, tokens: impl IntoIterator<Item = &str>) -> Vec<StatusCode> {
    let mut answered = Vec::new();
    for token in tokens {
        answered.push(call(app, token, "claude").await.status());
    }
    answered
}

#[rstest]
#[case::a_reserved_priority("sk-prod", "prod")]
#[case::no_priority(MASTER_KEY, "default")]
#[tokio::test]
async fn a_complete_answer_names_the_priority_it_was_counted_under(
    #[case] token: &str,
    #[case] priority: &str,
) {
    let Served { app, .. } = served(RESERVED).await;

    let answer = call(&app, token, "claude").await;

    assert_eq!(
        (
            answer.status(),
            answer.headers()["x-litellm-priority"].to_str().unwrap(),
            answer.headers()["x-litellm-rate-limiter-version"]
                .to_str()
                .unwrap(),
        ),
        (StatusCode::OK, priority, "v3")
    );
}

#[tokio::test]
async fn once_its_group_is_saturated_a_priority_is_held_to_what_it_reserves() {
    let Served { app, .. } = served(RESERVED).await;

    let answered = statuses(
        &app,
        [
            "sk-dev-1", "sk-dev-1", "sk-dev-1", "sk-dev-1", "sk-prod", "sk-dev-1", "sk-prod",
        ],
    )
    .await;

    let ok = StatusCode::OK;
    assert_eq!(
        answered,
        [ok, ok, ok, ok, ok, StatusCode::TOO_MANY_REQUESTS, ok]
    );
}

#[rstest]
#[case::one_pool_for_the_class(RESERVED, StatusCode::TOO_MANY_REQUESTS)]
#[case::a_slice_for_each_key(SPLIT_BY_KEY, StatusCode::OK)]
#[tokio::test]
async fn a_key_new_to_a_saturated_group_is_refused_only_when_its_class_shares_one_pool(
    #[case] config: &str,
    #[case] expected: StatusCode,
) {
    let Served { app, .. } = served(config).await;
    statuses(
        &app,
        ["sk-dev-1", "sk-dev-1", "sk-dev-1", "sk-dev-1", "sk-prod"],
    )
    .await;

    assert_eq!(call(&app, "sk-dev-2", "claude").await.status(), expected);
}

#[tokio::test]
async fn a_refusal_is_the_proxys_rate_limit_error_as_the_routes_handler_answers_it() {
    let Served { app, upstream, .. } = served(RESERVED).await;
    statuses(&app, ["sk-dev-1"; 5]).await;

    let refused = call(&app, "sk-dev-1", "claude").await;

    let status = refused.status();
    let body: Value =
        serde_json::from_slice(&to_bytes(refused.into_body(), 4096).await.unwrap()).unwrap();
    assert_eq!(
        (
            status,
            &body["type"],
            &body["error"]["type"],
            upstream.received_requests().await.unwrap().len()
        ),
        (
            StatusCode::TOO_MANY_REQUESTS,
            &json!("error"),
            &json!("rate_limit_error"),
            5
        )
    );
}

/// The limiter's headers of a response, in the order they were sent.
fn limit_headers(response: &Response) -> Vec<(&str, &str)> {
    const SENT: [&str; 5] = [
        "retry-after",
        "rate_limit_type",
        "rate_limited_by",
        "x-litellm-priority",
        "x-litellm-saturation",
    ];
    response
        .headers()
        .iter()
        .filter(|(name, _)| SENT.contains(&name.as_str()))
        .map(|(name, value)| (name.as_str(), value.to_str().unwrap()))
        .collect()
}

/// The proxy answers its `ProxyRateLimitError` with the headers in the order the limiter
/// lists them, which is not the order of their names.
#[tokio::test]
async fn an_openai_shaped_route_answers_a_refusal_with_its_headers_in_the_proxys_order() {
    let Served { app, .. } = served(RESERVED).await;
    statuses(&app, ["sk-dev-1"; 5]).await;

    let refused = post_as(&app, "sk-dev-1", "/v1/chat/completions", chat(false)).await;

    assert_eq!(
        (refused.status(), limit_headers(&refused)),
        (
            StatusCode::TOO_MANY_REQUESTS,
            vec![
                ("retry-after", "60"),
                ("rate_limit_type", "requests"),
                ("x-litellm-priority", "dev"),
                ("x-litellm-saturation", "50.00%"),
            ]
        )
    );
}

/// The proxy's failure hook and failure log only give back what a share that borrows held,
/// so a failed call stays counted as a request of its group and of its priority.
#[tokio::test]
async fn a_failed_call_stays_counted_as_a_request_and_adds_no_tokens() {
    let Served {
        app,
        limiter,
        upstream,
        ..
    } = served(RESERVED).await;
    refusing_forty_nine(&upstream).await;

    let failed = send_as(&app, "sk-prod", asking("claude", Some(49))).await;

    assert_eq!(
        (
            failed.status(),
            limiter.counted("requests", "model_saturation_check", "claude"),
            limiter.counted("requests", "priority_model", "claude:prod"),
            limiter.counted("tokens", "model_saturation_check", "claude"),
            limiter.counted("tokens", "priority_model", "claude:prod"),
        ),
        (
            StatusCode::BAD_REQUEST,
            Some(1.0),
            Some(1.0),
            Some(0.0),
            Some(0.0)
        )
    );
}

/// 40 input tokens of which 15 were read from cache, and 9 output, which Python charges as 34.
const RESPONSE: &str = r#"{"id":"response-1","model":"responses-test","output":[],"usage":{"input_tokens":40,"output_tokens":9,"total_tokens":49,"input_tokens_details":{"cached_tokens":15}}}"#;

/// An OpenAI provider that answers a Responses call, complete or streamed, with [`RESPONSE`].
async fn responding() -> MockServer {
    let upstream = MockServer::start().await;
    let completed = format!(
        "event: response.completed\ndata: {{\"type\":\"response.completed\",\"response\":{RESPONSE}}}\n\n"
    );
    Mock::given(body_partial_json(json!({"stream": true})))
        .respond_with(ResponseTemplate::new(200).set_body_raw(completed, "text/event-stream"))
        .mount(&upstream)
        .await;
    Mock::given(body_partial_json(json!({"model": "responses-test"})))
        .respond_with(ResponseTemplate::new(200).set_body_raw(RESPONSE, "application/json"))
        .mount(&upstream)
        .await;
    upstream
}

async fn post_as(app: &Router, token: &str, path: &str, body: Value) -> Response {
    app.clone()
        .oneshot(
            Request::post(path)
                .header("authorization", format!("Bearer {token}"))
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap()
}

fn chat(stream: bool) -> Value {
    json!({"model": "claude", "max_tokens": 16, "stream": stream,
        "messages": [{"role": "user", "content": "hi"}]})
}

fn response(stream: bool) -> Value {
    json!({"model": "responses", "input": "hello", "stream": stream})
}

/// The gateway with one more group, `responses`, which serves 10 requests and 100000 tokens a
/// minute at the OpenAI provider handed back with it.
async fn with_responses(keys: Arc<dyn KeyRows>, litellm_settings: &str) -> (Served, MockServer) {
    let provider = responding().await;
    let responses = format!(
        "  - model_name: responses\n    litellm_params: {{model: openai/responses-test, api_base: \"{}\", api_key: sk-upstream, rpm: 10, tpm: 100000}}\n",
        provider.uri()
    );
    let served = support::serving(keys, "", litellm_settings, &responses).await;
    (served, provider)
}

/// The success log settles what the route says the call used, less the prompt tokens read
/// from cache, whichever route served it and whether it streamed. The chat completions route
/// serves no stream.
#[rstest]
#[case::a_message("/v1/messages", chat(false), "claude", 27.0)]
#[case::a_streamed_message("/v1/messages", chat(true), "claude", 27.0)]
#[case::a_chat_completion("/v1/chat/completions", chat(false), "claude", 27.0)]
#[case::a_response("/v1/responses", response(false), "responses", 34.0)]
#[case::a_streamed_response("/v1/responses", response(true), "responses", 34.0)]
#[tokio::test]
async fn the_tokens_a_call_used_are_added_to_its_group_and_to_its_priority(
    #[case] path: &str,
    #[case] body: Value,
    #[case] group: &str,
    #[case] used: f64,
) {
    let (Served { app, limiter, .. }, _provider) =
        with_responses(Arc::new(Prioritized), RESERVED).await;

    let answer = post_as(&app, "sk-prod", path, body).await;
    let status = answer.status();
    to_bytes(answer.into_body(), usize::MAX).await.unwrap();

    let tokens = |descriptor: &str, value: &str| limiter.counted("tokens", descriptor, value);
    let priority = format!("{group}:prod");
    for _ in 0..200 {
        if tokens("priority_model", &priority) == Some(used) {
            break;
        }
        tokio::time::sleep(Duration::from_millis(2)).await;
    }
    assert_eq!(
        (
            status,
            tokens("model_saturation_check", group),
            tokens("priority_model", &priority),
            limiter.counted("requests", "model_saturation_check", group)
        ),
        (StatusCode::OK, Some(used), Some(used), Some(1.0))
    );
}

#[tokio::test]
async fn a_model_group_that_sets_no_limit_is_answered_uncounted() {
    let Served { app, limiter, .. } = served(RESERVED).await;

    let answer = call(&app, "sk-prod", "gpt").await;

    assert_eq!(
        (
            answer.status(),
            answer.headers()["x-litellm-priority"].to_str().unwrap(),
            limiter.counted("requests", "model_saturation_check", "gpt")
        ),
        (StatusCode::OK, "prod", None)
    );
}

/// The proxy reads an alias's group for its limits and counts the call under the name asked
/// for.
#[tokio::test]
async fn a_call_under_an_alias_is_counted_against_its_groups_limits() {
    let Served { app, limiter, .. } = served(RESERVED).await;

    call(&app, "sk-prod", "sonnet").await;

    assert_eq!(
        limiter.counted("requests", "model_saturation_check", "sonnet"),
        Some(1.0)
    );
}

#[tokio::test]
async fn a_key_that_borrows_is_held_to_its_limit_on_the_group_under_an_alias_of_it() {
    let limits = KeyLimits {
        model_rpm: by_model("claude", 1),
        ..KeyLimits::default()
    };
    let Served { app, .. } = holding(limits, BORROWING_TWICE_ITS_LIMIT).await;

    let first = call(&app, KEY, "sonnet").await.status();
    let second = call(&app, KEY, "sonnet").await.status();
    let third = call(&app, KEY, "sonnet").await.status();

    assert_eq!(
        (first, second, third),
        (
            StatusCode::OK,
            StatusCode::OK,
            StatusCode::TOO_MANY_REQUESTS
        )
    );
}

/// Python's authentication refuses the key before any hook, so the group never counts a call
/// its key was already over the limit for.
#[tokio::test]
async fn a_caller_over_its_own_limit_is_refused_before_its_model_group_counts_the_call() {
    let Served { app, limiter, .. } = served(RESERVED).await;

    let answered = statuses(&app, ["sk-limited-dev", "sk-limited-dev"]).await;

    assert_eq!(
        (
            answered,
            limiter.counted("requests", "model_saturation_check", "claude")
        ),
        (
            vec![StatusCode::OK, StatusCode::TOO_MANY_REQUESTS],
            Some(1.0)
        )
    );
}

/// The group is saturated from its sixth call on, at half its 10 requests a minute.
#[rstest]
#[case::until_its_group_saturates(BORROWING, 7, 6)]
#[case::up_to_its_borrow_ceiling(BORROWING_TWICE_ITS_LIMIT, 3, 2)]
#[tokio::test]
async fn a_key_that_borrows_runs_past_its_own_model_limit_on_idle_capacity(
    #[case] config: &str,
    #[case] calls: usize,
    #[case] answered: usize,
) {
    let limits = KeyLimits {
        model_rpm: by_model("claude", 1),
        ..KeyLimits::default()
    };
    let Served { app, upstream, .. } = holding(limits, config).await;

    let last = statuses(&app, vec![KEY; calls]).await.pop();

    assert_eq!(
        (last, upstream.received_requests().await.unwrap().len()),
        (Some(StatusCode::TOO_MANY_REQUESTS), answered)
    );
}

/// The tokens the key's borrow ceiling on a group holds.
struct Held<'a>(&'a KeyLimiter, &'a str);

fn held_under_the_ceiling(limiter: &KeyLimiter) -> Held<'_> {
    Held(limiter, "claude")
}

impl Held<'_> {
    fn now(&self) -> Option<f64> {
        self.0.counted(
            "tokens",
            "model_per_key_max_borrow_multiple",
            &format!("{}:{}", hash_token(KEY), self.1),
        )
    }

    /// What it holds once it reads `expected`, or what it last held.
    async fn once(&self, expected: f64) -> Option<f64> {
        for _ in 0..200 {
            if self.now() == Some(expected) {
                break;
            }
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
        self.now()
    }
}

/// A key that may use 30 tokens a minute of `claude`, and 60 while it borrows.
async fn borrowing_up_to_sixty_tokens() -> Served {
    let limits = KeyLimits {
        model_tpm: by_model("claude", 30),
        ..KeyLimits::default()
    };
    holding(limits, BORROWING_TWICE_ITS_LIMIT).await
}

#[tokio::test]
async fn a_share_that_holds_tokens_reserves_what_the_body_asks_for_and_settles_what_was_used() {
    let Served { app, limiter, .. } = borrowing_up_to_sixty_tokens().await;

    let admitted = send_as(&app, KEY, asking("claude", Some(50)))
        .await
        .status();
    let settled = held_under_the_ceiling(&limiter).once(27.0).await;
    let over = send_as(&app, KEY, asking("claude", Some(50)))
        .await
        .status();
    let fitting = send_as(&app, KEY, asking("claude", Some(20)))
        .await
        .status();

    assert_eq!(
        (admitted, settled, over, fitting),
        (
            StatusCode::OK,
            Some(27.0),
            StatusCode::TOO_MANY_REQUESTS,
            StatusCode::OK
        ),
        "the 51 reserved settle to the 27 used, and a refused call reserves nothing"
    );
}

/// The estimate a share reserves for a route is settled to what that route says the call
/// used, as it is for `/v1/messages`.
#[tokio::test]
async fn a_share_settles_what_it_reserved_to_what_a_response_used() {
    let limits = KeyLimits {
        model_tpm: by_model("responses", 1000),
        ..KeyLimits::default()
    };
    let (Served { app, limiter, .. }, _provider) =
        with_responses(Arc::new(Holding(limits)), BORROWING_TWICE_ITS_LIMIT).await;

    let answer = post_as(&app, KEY, "/v1/responses", response(false)).await;

    assert_eq!(
        (
            answer.status(),
            Held(&limiter, "responses").once(34.0).await
        ),
        (StatusCode::OK, Some(34.0))
    );
}

/// The provider refuses a call that asks for exactly 49 tokens.
async fn refusing_forty_nine(upstream: &MockServer) {
    Mock::given(body_partial_json(json!({"max_tokens": 49})))
        .respond_with(
            ResponseTemplate::new(400).set_body_json(json!({"type": "error",
            "error": {"type": "invalid_request_error", "message": "refused upstream"}})),
        )
        .with_priority(1)
        .mount(upstream)
        .await;
}

/// The failure hook gives them back before the client is answered, so a retry finds them.
#[tokio::test]
async fn a_failed_call_has_given_back_the_tokens_its_share_held_once_it_is_answered() {
    let Served {
        app,
        limiter,
        upstream,
        ..
    } = borrowing_up_to_sixty_tokens().await;
    refusing_forty_nine(&upstream).await;

    let failed = send_as(&app, KEY, asking("claude", Some(49))).await;

    assert_eq!(
        (failed.status(), held_under_the_ceiling(&limiter).now()),
        (StatusCode::BAD_REQUEST, Some(0.0))
    );
}

#[tokio::test]
async fn a_stream_the_client_leaves_gives_back_the_tokens_its_share_held() {
    let Served { app, limiter, .. } = borrowing_up_to_sixty_tokens().await;
    let streamed = json!({"model": "claude", "max_tokens": 50, "stream": true,
        "messages": [{"role": "user", "content": "hi"}]});

    drop(send_as(&app, KEY, streamed).await);

    assert_eq!(held_under_the_ceiling(&limiter).once(0.0).await, Some(0.0));
}

/// A share holds tokens before its group saturates only under a borrow ceiling. The route
/// refuses a body left without `max_tokens`, so nothing reaches the provider.
#[rstest]
#[case::a_share_that_holds_tokens(BORROWING_TWICE_ITS_LIMIT, Some(json!(25)))]
#[case::a_share_that_holds_none(BORROWING, None)]
#[tokio::test]
async fn a_body_that_sets_no_output_cap_is_capped_at_a_quarter_of_the_share_that_holds_tokens(
    #[case] config: &str,
    #[case] reached: Option<Value>,
) {
    let limits = KeyLimits {
        model_tpm: by_model("claude", 100),
        ..KeyLimits::default()
    };
    let Served { app, upstream, .. } = holding(limits, config).await;

    send_as(&app, KEY, asking("claude", None)).await;

    let received = upstream.received_requests().await.unwrap();
    let provider_saw: Option<Value> = received.last().map(|request| {
        serde_json::from_slice::<Value>(&request.body).unwrap()["max_tokens"].clone()
    });
    assert_eq!(provider_saw, reached);
}

/// `gpt` sets no `tpm` or `rpm`, so the dynamic rate limiter does not cover it.
#[tokio::test]
async fn a_limit_on_a_group_the_dynamic_rate_limiter_does_not_cover_stays_the_key_limiters() {
    let limits = KeyLimits {
        model_rpm: by_model("gpt", 1),
        ..KeyLimits::default()
    };
    let Served { app, .. } = holding(limits, BORROWING).await;

    let first = call(&app, KEY, "gpt").await.status();
    let second = call(&app, KEY, "gpt").await.status();

    assert_eq!(
        (first, second),
        (StatusCode::OK, StatusCode::TOO_MANY_REQUESTS)
    );
}
