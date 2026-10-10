use litellm_inference_messages::{MessagesCall, MessagesRoute};
use litellm_inference_testing::live::within_deadline;
use litellm_llms_types::formats::messages::MessagesRequest;
use rstest::rstest;
use serde_json::{Value, json};

use super::support::{LiveCall, assert_text, call, complete, request, route, stream_events};

const PROVIDER: &str = "bedrock";

#[rstest]
#[case::basic(json!({"messages": [{"role": "user", "content": "Say hello."}]}))]
#[case::system_text(json!({
    "system": "You are a helpful assistant.",
    "messages": [{"role": "user", "content": "Say hello."}]
}))]
#[case::system_role(json!({"messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Say hello."}
]}))]
#[case::unknown_field(json!({
    "unknown_extension": true,
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
#[case::model(false)]
#[case::invoke_prefix(true)]
#[ignore = "calls a real provider, requires credentials and a live model"]
#[tokio::test]
async fn streaming(route: MessagesRoute, #[case] invoke_prefix: bool) {
    within_deadline(async {
        let host = LiveCall::new(PROVIDER);
        let original = call(
            PROVIDER,
            json!({
                "stream": true,
                "messages": [{"role": "user", "content": "Say hello."}]
            }),
        );
        let model_id = original
            .body
            .model
            .strip_prefix("bedrock/")
            .unwrap_or(&original.body.model);
        let model = match invoke_prefix {
            false => original.body.model.clone(),
            true => format!(
                "bedrock/invoke/{}",
                model_id.strip_prefix("invoke/").unwrap_or(model_id)
            ),
        };
        let request_call = MessagesCall {
            body: MessagesRequest {
                model,
                ..original.body
            },
            ..original
        };
        let events = stream_events(request(&route, request_call, &host).await).await;
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
        json!({
            "tool_choice": {"type": "tool", "name": "echo"}
        }),
    )
    .await;
}
