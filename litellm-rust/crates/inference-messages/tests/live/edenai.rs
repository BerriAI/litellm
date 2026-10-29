use litellm_host::interceptors::Cost;
use litellm_inference_messages::MessagesRoute;
use litellm_inference_testing::live::within_deadline;
use rstest::rstest;
use serde_json::{Number, Value, json};

use super::support::{LiveCall, assert_text, call, complete, request, route, stream_events};

const PROVIDER: &str = "edenai";

#[rstest]
#[case::basic(json!({"messages": [{"role": "user", "content": "Say hello."}]}))]
#[case::system_text(json!({
    "system": "You are a helpful assistant.",
    "messages": [{"role": "user", "content": "Say hello."}]
}))]
#[case::portable_cache_hint(json!({
    "system": [{"type": "text", "text": "You are a helpful assistant.", "cache_control": {"type": "ephemeral", "ttl": "1h"}}],
    "messages": [{"role": "user", "content": "Say hello."}]
}))]
#[ignore = "calls a real provider, requires credentials and a live model"]
#[tokio::test]
async fn completion_and_cost_handoff(route: MessagesRoute, #[case] payload: Value) {
    within_deadline(async {
        let host = LiveCall::new(PROVIDER);
        let message = complete(request(&route, call(PROVIDER, payload), &host).await);
        assert_text(&message);
        host.assert_provider_result();
        let expected_cost = message
            .extra
            .get("cost")
            .and_then(|value| match value {
                Value::Number(number) => number.as_f64(),
                Value::String(text) => text.parse::<f64>().ok(),
                _ => None,
            })
            .filter(|cost| *cost >= 0.0)
            .and_then(Number::from_f64)
            .map_or(Cost::Deferred, |amount| Cost::Reported { amount });
        assert_eq!(host.facts().cost, expected_cost);
        println!("{}", serde_json::to_string(&message).unwrap());
    })
    .await;
}

#[rstest]
#[ignore = "calls a real provider, requires credentials and a live model"]
#[tokio::test]
async fn streaming(route: MessagesRoute) {
    within_deadline(async {
        let host = LiveCall::new(PROVIDER);
        let events = stream_events(
            request(
                &route,
                call(
                    PROVIDER,
                    json!({
                        "stream": true, "messages": [{"role": "user", "content": "Say hello."}]
                    }),
                ),
                &host,
            )
            .await,
        )
        .await;
        assert!(events.iter().any(|event| {
            event["delta"]["text"]
                .as_str()
                .is_some_and(|text| !text.is_empty())
        }));
        host.assert_provider_result();
        assert_eq!(host.facts().cost, Cost::Deferred);
    })
    .await;
}

#[rstest]
#[case::complete(false)]
#[case::streaming(true)]
#[ignore = "calls a real provider, requires credentials and a live model"]
#[tokio::test]
async fn tool_round_trip(route: MessagesRoute, #[case] stream: bool) {
    super::support::tool_round_trip(
        route,
        PROVIDER,
        stream,
        json!({
            "max_tokens": 512, "tool_choice": {"type": "auto"}
        }),
    )
    .await;
}
