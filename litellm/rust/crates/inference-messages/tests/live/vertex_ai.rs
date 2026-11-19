use litellm_inference_messages::MessagesRoute;
use litellm_inference_testing::live::within_deadline;
use rstest::rstest;
use serde_json::{Value, json};

use super::support::{LiveCall, assert_text, call, complete, request, route, stream_events};

const PROVIDER: &str = "vertex_ai";

#[rstest]
#[case::basic(json!({"messages": [{"role": "user", "content": "Say hello."}]}))]
#[case::system_text(json!({
    "system": "You are a helpful assistant.",
    "messages": [{"role": "user", "content": "Say hello."}]
}))]
#[ignore = "calls a real provider, requires credentials and a live model"]
#[tokio::test]
async fn completion(route: MessagesRoute, #[case] payload: Value) {
    within_deadline(async {
        let host = LiveCall::new(PROVIDER);
        let message = complete(request(&route, call(PROVIDER, payload), &host).await);
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
        let response = request(
            &route,
            call(
                PROVIDER,
                json!({"stream": true, "messages": [{"role": "user", "content": "Say hello."}]}),
            ),
            &host,
        )
        .await;
        let events = stream_events(response).await;
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
        json!({"tool_choice": {"type": "auto"}}),
    )
    .await;
}
