//! Claude on Bedrock InvokeModel. Runs when `LITELLM_LIVE_MESSAGES_BEDROCK_MODEL` names a Claude
//! model id (for example a `us.anthropic.*` inference profile) and AWS credentials resolve from the
//! environment or default chain, or `AWS_BEARER_TOKEN_BEDROCK` is set.

use litellm_inference_messages::MessagesRoute;
use litellm_inference_testing::live::within_deadline;
use rstest::{fixture, rstest};
use serde_json::json;

use super::support::{LiveCall, assert_text, call, complete, request, route, stream_events};

const PROVIDER: &str = "bedrock";

#[fixture]
fn model() -> Option<String> {
    std::env::var("LITELLM_LIVE_MESSAGES_BEDROCK_MODEL")
        .ok()
        .filter(|model| !model.trim().is_empty())
}

#[rstest]
#[tokio::test]
async fn completion(route: MessagesRoute, model: Option<String>) {
    if model.is_none() {
        return;
    }
    within_deadline(async {
        let host = LiveCall::new(PROVIDER);
        let message = complete(
            request(
                &route,
                call(
                    PROVIDER,
                    json!({"messages": [{"role": "user", "content": "Say hello."}]}),
                ),
                &host,
            )
            .await,
        );
        assert_text(&message);
        host.assert_provider_result();
    })
    .await;
}

#[rstest]
#[tokio::test]
async fn streaming(route: MessagesRoute, model: Option<String>) {
    if model.is_none() {
        return;
    }
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
#[tokio::test]
async fn tool_round_trip(route: MessagesRoute, model: Option<String>, #[case] stream: bool) {
    if model.is_none() {
        return;
    }
    super::support::tool_round_trip(route, PROVIDER, stream, json!({})).await;
}
