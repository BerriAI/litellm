mod keys;
mod support;

use std::time::Duration;

use axum::{Router, body::to_bytes, http::StatusCode, response::Response};
use rstest::rstest;
use serde_json::{Value, json};
use support::{MASTER_KEY, Served, asking, send_as};

const RESERVED: &str = "litellm_settings:\n  callbacks: [dynamic_rate_limiter_v3]\n  priority_reservation: {prod: 0.9, dev: 0.1}\n";
const SPLIT_BY_KEY: &str = "litellm_settings:\n  callbacks: [dynamic_rate_limiter_v3]\n  priority_reservation: {prod: 0.9, dev: 0.1}\n  priority_reservation_settings:\n    priority_class_share_config:\n      split_equally_among_active: [key]\n";

/// Each key active under the priority its name carries: `sk-prod`, `sk-dev-1`, `sk-dev-2`.
/// `sk-limited-dev` also holds one request a minute of its own.
async fn prioritized() -> keys::Held {
    let held = keys::Held::start().await;
    for (token, columns) in [
        ("sk-prod", json!({"metadata": {"priority": "prod"}})),
        ("sk-dev-1", json!({"metadata": {"priority": "dev"}})),
        ("sk-dev-2", json!({"metadata": {"priority": "dev"}})),
        (
            "sk-limited-dev",
            json!({"metadata": {"priority": "dev"}, "rpm_limit": 1}),
        ),
    ] {
        held.key(token, columns).await;
    }
    held
}

/// The gateway over the group `claude`, which serves 10 requests and 100000 tokens a minute.
async fn served(litellm_settings: &str) -> Served {
    support::served(prioritized().await, "", litellm_settings).await
}

async fn call(app: &Router, token: &str, model: &str) -> Response {
    send_as(app, token, asking(model, Some(16))).await
}

async fn statuses<const N: usize>(app: &Router, tokens: [&str; N]) -> Vec<StatusCode> {
    let mut answered = Vec::with_capacity(N);
    for token in tokens {
        answered.push(call(app, token, "claude").await.status());
    }
    answered
}

#[rstest]
#[case::a_reserved_priority("sk-prod", "prod")]
#[case::no_priority(MASTER_KEY, "default")]
#[tokio::test(flavor = "multi_thread")]
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

#[tokio::test(flavor = "multi_thread")]
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
#[tokio::test(flavor = "multi_thread")]
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

#[tokio::test(flavor = "multi_thread")]
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

#[tokio::test(flavor = "multi_thread")]
async fn the_tokens_a_call_used_are_added_to_its_groups_saturation() {
    let Served { app, limiter, .. } = served(RESERVED).await;

    call(&app, "sk-prod", "claude").await;

    let tokens = || limiter.counted("tokens", "model_saturation_check", "claude");
    for _ in 0..200 {
        if tokens() == Some(27.0) {
            break;
        }
        tokio::time::sleep(Duration::from_millis(2)).await;
    }
    assert_eq!(
        (
            tokens(),
            limiter.counted("requests", "model_saturation_check", "claude")
        ),
        (Some(27.0), Some(1.0))
    );
}

#[tokio::test(flavor = "multi_thread")]
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

/// Python's authentication refuses the key before any hook, so the group never counts a call
/// its key was already over the limit for.
#[tokio::test(flavor = "multi_thread")]
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
