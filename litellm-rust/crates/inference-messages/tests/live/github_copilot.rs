use litellm_inference_messages::{MessagesCall, MessagesRoute};
use litellm_inference_testing::live::within_deadline;
use rstest::rstest;
use serde_json::{Value, json};

use super::support::{LiveCall, assert_text, call, complete, request, route, stream_events};

const PROVIDER: &str = "github_copilot";

#[rstest]
#[case::basic(json!({"messages": [{"role": "user", "content": "Say hello."}]}), false)]
#[case::system_text(json!({
    "system": "You are a helpful assistant.",
    "messages": [{"role": "user", "content": "Say hello."}]
}), false)]
#[case::caller_credentials_ignored(json!({
    "messages": [{"role": "user", "content": "Say hello."}]
}), true)]
#[ignore = "calls a real provider, requires credentials and a live model"]
#[tokio::test]
async fn completion(
    route: MessagesRoute,
    #[case] payload: Value,
    #[case] caller_credentials: bool,
) {
    within_deadline(async {
        let host = LiveCall::new(PROVIDER);
        let base_call = call(PROVIDER, payload);
        let live_call = if caller_credentials {
            MessagesCall {
                api_key: Some("must-not-be-forwarded".into()),
                api_base: Some("https://example.invalid/v1".into()),
                ..base_call
            }
        } else {
            base_call
        };
        let message = complete(request(&route, live_call, &host).await);
        assert_text(&message);
        host.assert_provider_result();
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
        json!({"max_tokens": 512, "tool_choice": {"type": "auto"}}),
    )
    .await;
}
