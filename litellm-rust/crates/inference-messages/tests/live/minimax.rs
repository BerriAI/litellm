use litellm_inference_messages::MessagesRoute;
use litellm_inference_testing::live::within_deadline;
use rstest::rstest;
use serde_json::{Value, json};

use super::support::{LiveCall, assert_text, call, complete, request, route, stream_events};

#[rstest]
#[case::basic(json!({"messages": [{"role": "user", "content": "Reply with only OK."}]}))]
#[case::system(json!({"system": "Keep replies short.", "messages": [{"role": "user", "content": "Say hello."}]}))]
#[ignore = "calls a real provider, requires credentials and a live model"]
#[tokio::test]
async fn completion(route: MessagesRoute, #[case] payload: Value) {
    within_deadline(async {
        let host = LiveCall::new("minimax");
        let message = complete(request(&route, call("minimax", payload), &host).await);
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
        let host = LiveCall::new("minimax");
        let events = stream_events(request(&route, call("minimax", json!({"stream": true, "messages": [{"role": "user", "content": "Reply with only OK."}]})), &host).await).await;
        assert!(events.iter().any(|event| event["delta"]["text"].as_str().is_some_and(|text| !text.is_empty())));
        host.assert_provider_result();
    }).await;
}

#[rstest]
#[case::complete(false)]
#[case::streaming(true)]
#[ignore = "calls a real provider, requires credentials and a live model"]
#[tokio::test]
async fn tool_round_trip(route: MessagesRoute, #[case] stream: bool) {
    super::support::tool_round_trip(
        route,
        "minimax",
        stream,
        json!({"max_tokens": 1024, "tool_choice": {"type": "tool", "name": "echo"}}),
    )
    .await;
}
