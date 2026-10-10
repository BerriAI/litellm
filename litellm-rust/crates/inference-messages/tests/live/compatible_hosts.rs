use litellm_host::interceptors::Cost;
use litellm_inference_messages::MessagesRoute;
use litellm_inference_testing::live::within_deadline;
use litellm_llms::{
    base_llm::messages::transformation::BaseMessagesConfig,
    edenai::messages::transformation::EDENAI_MESSAGES_CONFIG,
    openrouter::messages::transformation::OPENROUTER_MESSAGES_CONFIG,
};
use rstest::rstest;
use serde_json::{Value, json};

use super::support::{LiveCall, assert_text, call, complete, request, route, stream_events};

fn config(provider: &str) -> &'static dyn BaseMessagesConfig {
    match provider {
        "edenai" => &EDENAI_MESSAGES_CONFIG,
        "openrouter" => &OPENROUTER_MESSAGES_CONFIG,
        other => panic!("{other} is not an Anthropic-compatible host case"),
    }
}

#[rstest]
#[case::basic(json!({"messages": [{"role": "user", "content": "Say hello."}]}))]
#[case::system_text(json!({
    "system": "You are a helpful assistant.",
    "messages": [{"role": "user", "content": "Say hello."}]
}))]
#[case::cache_hint(json!({
    "system": [{"type": "text", "text": "You are a helpful assistant.", "cache_control": {"type": "ephemeral", "ttl": "1h"}}],
    "messages": [{"role": "user", "content": "Say hello."}]
}))]
#[ignore = "calls a real provider, requires credentials and a live model"]
#[tokio::test]
async fn completion_and_cost_handoff(
    route: MessagesRoute,
    #[case] payload: Value,
    #[values("edenai", "openrouter")] provider: &'static str,
) {
    within_deadline(async {
        let host = LiveCall::new(provider);
        let message = complete(request(&route, call(provider, payload), &host).await);
        assert_text(&message);
        host.assert_provider_result();
        let expected_cost = config(provider)
            .reported_cost(&message)
            .map_or(Cost::Deferred, |amount| Cost::Reported { amount });
        assert!(
            matches!(expected_cost, Cost::Reported { .. }),
            "{provider} must report a cost"
        );
        assert_eq!(host.facts().cost, expected_cost);
        println!("{}", serde_json::to_string(&message).unwrap());
    })
    .await;
}

#[rstest]
#[ignore = "calls a real provider, requires credentials and a live model"]
#[tokio::test]
async fn streaming(route: MessagesRoute, #[values("edenai", "openrouter")] provider: &'static str) {
    within_deadline(async {
        let host = LiveCall::new(provider);
        let events = stream_events(
            request(
                &route,
                call(
                    provider,
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
async fn tool_round_trip(
    route: MessagesRoute,
    #[case] stream: bool,
    #[values("edenai", "openrouter")] provider: &'static str,
) {
    super::support::tool_round_trip(route, provider, stream, tool_params(provider)).await;
}

fn tool_params(provider: &str) -> Value {
    match provider {
        "edenai" | "openrouter" => json!({"max_tokens": 512, "tool_choice": {"type": "auto"}}),
        other => panic!("{other} is not an Anthropic-compatible host case"),
    }
}
