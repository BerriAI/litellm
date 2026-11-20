mod keys;

use std::{future::Future, sync::Arc, time::Duration};

use axum::{
    Router,
    body::{Body, to_bytes},
    http::{Request, StatusCode, header},
    response::Response,
};
use keys::plain;
use litellm_config::{Config, Environment};
use litellm_gateway::{
    AppState, KeyLimiter,
    host::auth::{KeyRows, RowFuture, authenticator},
};
use litellm_gateway_auth::{
    hash_token,
    keys::{KeyLimits, ModelLimits},
};
use litellm_model_catalog::{BUNDLED_REGISTRY, Catalog};
use litellm_repositories::{KeyView, RawJson};
use rstest::rstest;
use serde_json::{Value, json};
use tower::ServiceExt;
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{body_partial_json, method},
};

const MASTER_KEY_ALIAS: &str = "litellm_proxy_master_key";

const MASTER_KEY: &str = "sk-master";
const KEY: &str = "sk-limited";
/// 8 input, 14 output, 100 read from cache and 5 written to it, which Python charges as 27.
const USAGE: &str = r#"{"input_tokens":8,"output_tokens":14,"cache_read_input_tokens":100,"cache_creation_input_tokens":5}"#;

/// A database holding, under every hash, a plain row with these limits, and the owner the
/// native app does not serve a key for.
struct Holding {
    limits: KeyLimits,
    owner: Option<&'static str>,
}

impl KeyRows for Holding {
    fn find<'a>(&'a self, hash: &'a str) -> RowFuture<'a> {
        Box::pin(async move {
            let KeyLimits {
                rpm,
                tpm,
                max_parallel_requests,
                model_rpm,
                model_tpm,
            } = &self.limits;
            let metadata = json!({"model_rpm_limit": **model_rpm, "model_tpm_limit": **model_tpm});
            Ok(Some(KeyView {
                token: hash.to_owned(),
                rpm_limit: *rpm,
                tpm_limit: *tpm,
                max_parallel_requests: max_parallel_requests
                    .map(|limit| i32::try_from(limit).unwrap()),
                metadata: RawJson(serde_json::value::to_raw_value(&metadata).unwrap()),
                user_id: self.owner.map(str::to_owned),
                ..plain("")
            }))
        })
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

/// 30 input tokens of which 10 were read from cache, and 7 output, which Python charges as 27.
const RESPONSES_USAGE: &str = r#"{"input_tokens":30,"output_tokens":7,"total_tokens":37,"input_tokens_details":{"cached_tokens":10}}"#;

/// The OpenAI provider `responder` calls: it answers a Responses stream when asked for one,
/// and a complete response otherwise, each with [`RESPONSES_USAGE`].
async fn responses_provider() -> MockServer {
    let upstream = MockServer::start().await;
    let response = format!(
        r#"{{"id":"resp_1","object":"response","model":"responder-test","status":"completed","output":[],"usage":{RESPONSES_USAGE}}}"#
    );
    Mock::given(method("POST"))
        .and(body_partial_json(json!({"stream": true})))
        .respond_with(ResponseTemplate::new(200).set_body_raw(
            format!(
                "event: response.completed\ndata: {{\"type\":\"response.completed\",\"response\":{response}}}\n\n"
            ),
            "text/event-stream",
        ))
        .mount(&upstream)
        .await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(response, "application/json"))
        .mount(&upstream)
        .await;
    upstream
}

/// The inference routes as the gateway mounts them, for a key holding `limits`: the groups
/// `claude` (also asked for as `sonnet`) and `gpt` answer, `responder` answers the Responses
/// route, and `broken` is refused upstream.
struct Served {
    app: Router,
    limiter: Arc<KeyLimiter>,
    upstream: MockServer,
    responding: MockServer,
    _refusing: MockServer,
}

impl Served {
    /// What the provider behind `path` was last sent as `param`.
    async fn provider_saw(&self, path: &str, param: &str) -> Option<Value> {
        let provider = match path {
            RESPONSES => &self.responding,
            _ => &self.upstream,
        };
        let received = provider.received_requests().await.unwrap();
        received
            .last()
            .map(|request| serde_json::from_slice::<Value>(&request.body).unwrap()[param].clone())
    }
}

async fn served(limits: KeyLimits) -> Served {
    configured(limits, "").await
}

async fn configured(limits: KeyLimits, general_settings: &str) -> Served {
    Setup {
        limits,
        general_settings,
        ..Setup::default()
    }
    .serve()
    .await
}

async fn deployed(limits: KeyLimits, groups: &[(&str, &str)]) -> Served {
    Setup {
        limits,
        groups,
        ..Setup::default()
    }
    .serve()
    .await
}

/// What [`Served`] is built from. `groups` are further groups, each a name and the
/// `litellm_params` its deployment sets besides its model.
#[derive(Default)]
struct Setup<'a> {
    limits: KeyLimits,
    /// The owner of every key, which the native app does not serve a key for.
    owner: Option<&'static str>,
    general_settings: &'a str,
    groups: &'a [(&'a str, &'a str)],
    environment: Environment,
}

impl Setup<'_> {
    async fn serve(self) -> Served {
        let Self {
            limits,
            owner,
            general_settings,
            groups,
            environment,
        } = self;
        let upstream = provider().await;
        let responding = responses_provider().await;
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
        let further: String = groups
            .iter()
            .map(|(name, params)| {
                format!(
                    "  - model_name: {name}\n    litellm_params: {{model: anthropic/further-test, api_base: \"{}\", api_key: sk-upstream{params}}}\n",
                    upstream.uri()
                )
            })
            .collect();
        let responder = format!(
            "  - model_name: responder\n    litellm_params: {{model: openai/responder-test, api_base: \"{}\", api_key: sk-upstream}}\n",
            responding.uri()
        );
        let config = Config::from_yaml(&format!(
            "model_list:\n{}{}{}{responder}{further}router_settings:\n  model_group_alias: {{sonnet: claude}}\ngeneral_settings:\n  master_key: {MASTER_KEY}\n{general_settings}",
            deployment("claude", &upstream.uri()),
            deployment("gpt", &upstream.uri()),
            deployment("broken", &refusing.uri()),
        ))
        .unwrap();
        let state = AppState {
            keys: Some(authenticator(
                &config,
                Arc::new(Holding { limits, owner }),
                None,
            )),
            ..AppState::build(&config, &environment).await.unwrap()
        };
        let limiter = state.key_limiter.clone();
        Served {
            app: litellm_gateway::router(state, &config),
            limiter,
            upstream,
            responding,
            _refusing: refusing,
        }
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

const MESSAGES: &str = "/v1/messages";
const CHAT: &str = "/v1/chat/completions";
const RESPONSES: &str = "/v1/responses";

/// A Responses body asking `responder` for at most `max_output_tokens`.
fn responding(max_output_tokens: Option<u64>, stream: bool) -> Value {
    let body = json!({"model": "responder", "input": "hi", "stream": stream});
    match (body, max_output_tokens) {
        (Value::Object(mut body), Some(max_output_tokens)) => {
            body.insert("max_output_tokens".to_owned(), json!(max_output_tokens));
            Value::Object(body)
        }
        (body, _) => body,
    }
}

async fn post(app: &Router, token: &str, path: &str, body: Value) -> Response {
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

async fn send_as(app: &Router, token: &str, body: Value) -> Response {
    post(app, token, MESSAGES, body).await
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
#[tokio::test]
async fn a_key_the_native_app_does_not_serve_counts_nothing_against_its_limits() {
    let Served {
        app,
        limiter,
        upstream,
        ..
    } = Setup {
        limits: KeyLimits {
            rpm: Some(1),
            tpm: Some(100),
            ..KeyLimits::default()
        },
        owner: Some("owner"),
        ..Setup::default()
    }
    .serve()
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

#[tokio::test]
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

#[tokio::test]
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
#[tokio::test]
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
#[tokio::test]
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
#[case::complete(MESSAGES, complete(), "claude")]
#[case::streamed_to_its_end(MESSAGES, streamed(), "claude")]
#[case::a_chat_completion(CHAT, complete(), "claude")]
#[case::a_response(RESPONSES, responding(None, false), "responder")]
#[case::a_streamed_response(RESPONSES, responding(None, true), "responder")]
#[tokio::test]
async fn a_served_call_charges_its_tokens_less_the_cached_reads_to_its_key_and_model(
    #[case] path: &str,
    #[case] body: Value,
    #[case] group: &str,
) {
    let Served { app, limiter, .. } = served(KeyLimits {
        rpm: Some(5),
        ..KeyLimits::default()
    })
    .await;
    let hash = hash_token(KEY);

    let answered = post(&app, KEY, path, body).await;
    let status = answered.status();
    read_to_its_end(answered).await;

    assert_eq!(
        (
            status,
            settled(&limiter, "api_key", &hash, 27.0).await,
            settled(&limiter, "model_per_key", &format!("{hash}:{group}"), 27.0).await
        ),
        (StatusCode::OK, Some(27.0), Some(27.0))
    );
}

#[rstest]
#[case::a_chat_completion(CHAT, asking("claude", Some(50)))]
#[case::a_response(RESPONSES, responding(Some(50), false))]
#[case::a_streamed_response(RESPONSES, responding(Some(50), true))]
#[tokio::test]
async fn a_token_limit_reserves_what_a_call_of_any_route_asks_for_and_settles_what_it_used(
    #[case] path: &str,
    #[case] body: Value,
) {
    let Served { app, limiter, .. } = served(KeyLimits {
        tpm: Some(60),
        ..KeyLimits::default()
    })
    .await;

    let admitted = post(&app, KEY, path, body.clone()).await;
    let status = admitted.status();
    read_to_its_end(admitted).await;
    let used = settled(&limiter, "api_key", &hash_token(KEY), 27.0).await;
    let over = post(&app, KEY, path, body).await;

    assert_eq!(
        (status, used, over.status()),
        (StatusCode::OK, Some(27.0), StatusCode::TOO_MANY_REQUESTS)
    );
}

#[rstest]
#[case::its_key("api_key", "litellm_proxy_master_key")]
#[case::its_user("user", "default_user_id")]
#[case::its_model("model_per_key", "litellm_proxy_master_key:claude")]
#[tokio::test]
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
#[tokio::test]
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
#[tokio::test]
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

#[tokio::test]
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

#[tokio::test]
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
#[case::none_sent(MESSAGES, asking("claude", None), "max_tokens", json!(25))]
#[case::one_sent(MESSAGES, asking("claude", Some(50)), "max_tokens", json!(50))]
#[case::none_sent_to_chat(CHAT, asking("claude", None), "max_tokens", json!(25))]
#[case::none_sent_to_responses(
    RESPONSES,
    responding(None, false),
    "max_output_tokens",
    json!(25)
)]
#[tokio::test]
async fn a_body_that_sets_no_output_cap_is_capped_at_a_quarter_of_the_token_limit(
    #[case] path: &str,
    #[case] body: Value,
    #[case] cap: &str,
    #[case] reached: Value,
) {
    let served = served(KeyLimits {
        tpm: Some(100),
        rpm: Some(5),
        ..KeyLimits::default()
    })
    .await;

    post(&served.app, KEY, path, body).await;

    assert_eq!(served.provider_saw(path, cap).await, Some(reached));
}

#[rstest]
#[case::set("0.5", json!(50))]
#[case::beyond_the_whole_limit("1.5", json!(25))]
#[case::not_a_number("half", json!(25))]
#[tokio::test]
async fn the_output_cap_is_the_fraction_of_the_token_limit_the_environment_sets(
    #[case] fraction: &'static str,
    #[case] reached: Value,
) {
    let variables = move |name: &str| {
        (name == "NO_MAX_TOKENS_TPM_RESERVATION_FRACTION").then(|| fraction.to_owned())
    };
    let Served { app, upstream, .. } = Setup {
        limits: KeyLimits {
            tpm: Some(100),
            ..KeyLimits::default()
        },
        environment: Environment::read(&variables).unwrap(),
        ..Setup::default()
    }
    .serve()
    .await;

    send(&app, asking("claude", None)).await;

    let received = upstream.received_requests().await.unwrap();
    let provider_saw: Option<Value> = received.last().map(|request| {
        serde_json::from_slice::<Value>(&request.body).unwrap()["max_tokens"].clone()
    });
    assert_eq!(provider_saw, Some(reached));
}

#[tokio::test]
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
#[tokio::test]
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

#[tokio::test]
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
#[tokio::test]
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

/// The first alias row of the bundled cost map that `picked` takes, with the model it stands
/// for.
fn alias_row(picked: fn(&str, &str) -> bool) -> (String, String) {
    let catalog = Catalog::parse(BUNDLED_REGISTRY).unwrap().value;
    catalog
        .aliases()
        .iter()
        .find(|(alias, model)| picked(alias, model))
        .map(|(alias, model)| (alias.clone(), model.clone()))
        .unwrap()
}

fn a_bare_alias() -> (String, String) {
    alias_row(|alias, _| !alias.contains('/'))
}

fn an_alias_under_its_models_provider() -> (String, String) {
    alias_row(|alias, model| {
        alias
            .split_once('/')
            .is_some_and(|(provider, _)| model.starts_with(&format!("{provider}/")))
    })
}

/// An alias asked for through a provider its model's own name does not carry, as a Bedrock
/// alias of a `global.` model is. Its limit is found under that provider.
fn an_alias_under_a_provider_hosting_its_model() -> (String, String) {
    let (alias, model) = alias_row(|alias, model| {
        alias
            .split_once('/')
            .is_some_and(|(provider, _)| !model.starts_with(&format!("{provider}/")))
    });
    let limited = format!("{}/{model}", alias.split_once('/').unwrap().0);
    (alias, limited)
}

#[rstest]
#[case::bare(a_bare_alias())]
#[case::under_its_models_provider(an_alias_under_its_models_provider())]
#[case::under_a_provider_hosting_its_model(an_alias_under_a_provider_hosting_its_model())]
#[tokio::test]
async fn a_limit_on_the_model_a_latest_alias_stands_for_counts_a_call_to_the_alias(
    #[case] (alias, limited): (String, String),
) {
    let Served { app, limiter, .. } = deployed(
        KeyLimits {
            model_tpm: by_model(&[(&limited, 60)]),
            ..KeyLimits::default()
        },
        &[(&alias, "")],
    )
    .await;

    let admitted = send(&app, asking(&alias, Some(50))).await;
    let charged = settled(
        &limiter,
        "model_per_key",
        &format!("{}:{limited}", hash_token(KEY)),
        27.0,
    )
    .await;
    let refused = send(&app, asking(&alias, Some(50))).await;

    assert_eq!(
        (admitted.status(), charged, refused.status()),
        (StatusCode::OK, Some(27.0), StatusCode::TOO_MANY_REQUESTS)
    );
}

#[rstest]
#[case::a_key_without_limits(KEY, KeyLimits::default(), &[", default_api_key_rpm_limit: 1"], 1)]
#[case::the_master_key(MASTER_KEY, KeyLimits::default(), &[", default_api_key_rpm_limit: 1"], 1)]
#[case::the_least_of_its_groups_deployments(
    KEY,
    KeyLimits::default(),
    &[", default_api_key_rpm_limit: 3", ", default_api_key_rpm_limit: 1", ""],
    1
)]
#[case::a_key_with_its_own_limit_for_the_model(
    KEY,
    KeyLimits { model_rpm: by_model(&[("limited", 2)]), ..KeyLimits::default() },
    &[", default_api_key_rpm_limit: 1"],
    2
)]
#[tokio::test]
async fn a_deployments_default_request_limit_counts_a_key_with_no_limit_for_its_model(
    #[case] token: &str,
    #[case] limits: KeyLimits,
    #[case] deployments: &[&str],
    #[case] admitted: usize,
) {
    let groups: Vec<(&str, &str)> = deployments
        .iter()
        .map(|params| ("limited", *params))
        .collect();
    let Served { app, .. } = deployed(limits, &groups).await;

    let served = futures_util::future::join_all(
        std::iter::repeat_n((), admitted)
            .map(|()| send_as(&app, token, asking("limited", Some(16)))),
    )
    .await;
    let refused = send_as(&app, token, asking("limited", Some(16))).await;
    let another_group = send_as(&app, token, asking("claude", Some(16))).await;

    assert_eq!(
        served.iter().map(Response::status).collect::<Vec<_>>(),
        vec![StatusCode::OK; admitted]
    );
    assert_eq!(
        (refused.status(), another_group.status()),
        (StatusCode::TOO_MANY_REQUESTS, StatusCode::OK)
    );
    let caller = match token {
        MASTER_KEY => MASTER_KEY_ALIAS.to_owned(),
        _ => hash_token(token),
    };
    let message = json_body(refused).await["error"]["message"].clone();
    assert!(
        message.as_str().unwrap().starts_with(&format!(
            "Rate limit exceeded for model_per_key: {caller}:limited. Limit type: requests. Current limit: {admitted}"
        )),
        "{message}"
    );
}

#[tokio::test]
async fn a_deployments_default_token_limit_reserves_what_the_body_asks_for() {
    let Served { app, upstream, .. } = deployed(
        KeyLimits::default(),
        &[("limited", ", default_api_key_tpm_limit: 60")],
    )
    .await;

    let uncapped = send(&app, asking("limited", None)).await;
    let received = upstream.received_requests().await.unwrap();
    let provider_saw = received.last().map(|request| {
        serde_json::from_slice::<Value>(&request.body).unwrap()["max_tokens"].clone()
    });
    let over = send(&app, asking("limited", Some(50))).await;

    assert_eq!(
        (uncapped.status(), provider_saw, over.status()),
        (
            StatusCode::OK,
            Some(json!(15)),
            StatusCode::TOO_MANY_REQUESTS
        )
    );
}

#[rstest]
#[case::output(MESSAGES, complete(), "output", 14.0)]
#[case::input(MESSAGES, complete(), "input", 13.0)]
#[case::anything_else(MESSAGES, complete(), "tokens", 27.0)]
#[case::a_chat_completions_input(CHAT, complete(), "input", 13.0)]
#[case::a_responses_output(RESPONSES, responding(None, false), "output", 7.0)]
#[case::a_responses_input(RESPONSES, responding(None, false), "input", 20.0)]
#[case::a_streamed_responses_input(RESPONSES, responding(None, true), "input", 20.0)]
#[tokio::test]
async fn a_served_call_is_charged_the_tokens_the_configured_type_counts(
    #[case] path: &str,
    #[case] body: Value,
    #[case] token_rate_limit_type: &str,
    #[case] charged: f64,
) {
    let Served { app, limiter, .. } = configured(
        KeyLimits::default(),
        &format!("  token_rate_limit_type: {token_rate_limit_type}\n"),
    )
    .await;

    read_to_its_end(post(&app, KEY, path, body).await).await;

    assert_eq!(
        settled(&limiter, "api_key", &hash_token(KEY), charged).await,
        Some(charged)
    );
}
