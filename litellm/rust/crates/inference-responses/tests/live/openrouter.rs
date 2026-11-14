use std::{sync::Arc, time::Duration};

use futures_util::TryStreamExt;
use litellm_inference_responses::types::{ResponsesCall, ResponsesOutput};
use litellm_inference_testing::RecordingSecrets;
use litellm_llms::base_llm::responses::streaming::decode;
use litellm_llms_types::formats::responses::streaming::ResponsesEventType;
use rstest::{fixture, rstest};
use serde_json::json;

use crate::support::responses_route;

const DEFAULT_MODEL: &str = "openrouter/openai/gpt-5.5";

/// `OPENROUTER_API_KEY` as a secret source; tests skip when it is unset.
#[fixture]
fn secrets() -> Option<Arc<RecordingSecrets>> {
    let key = std::env::var("OPENROUTER_API_KEY").ok()?;
    Some(Arc::new(RecordingSecrets::new([(
        "OPENROUTER_API_KEY",
        key.as_str(),
    )])))
}

fn call(stream: bool) -> ResponsesCall {
    ResponsesCall {
        model: std::env::var("OPENROUTER_LIVE_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.into()),
        input: json!("Reply with the single word: pong"),
        optional_params: json!({"max_output_tokens": 32, "stream": stream})
            .as_object()
            .unwrap()
            .clone(),
        api_key: None,
        api_base: None,
        custom_llm_provider: None,
        extra_headers: None,
        timeout: Some(Duration::from_secs(60)),
    }
}

#[rstest]
#[tokio::test]
async fn a_complete_response_carries_output_and_usage(secrets: Option<Arc<RecordingSecrets>>) {
    let Some(secrets) = secrets else { return };
    let output = responses_route(secrets)
        .execute(call(false), &(), None)
        .await
        .unwrap();
    let ResponsesOutput::Complete(response) = output else {
        panic!("expected a complete response");
    };
    let body = serde_json::to_value(response).unwrap();
    let text = body["output"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|item| item["content"].as_array())
        .flatten()
        .filter_map(|part| part["text"].as_str())
        .collect::<String>();
    assert!(text.to_ascii_lowercase().contains("pong"), "{body}");
    assert!(
        body["usage"]["total_tokens"].as_u64().unwrap() > 0,
        "{body}"
    );
}

#[rstest]
#[tokio::test]
async fn a_stream_ends_with_response_completed(secrets: Option<Arc<RecordingSecrets>>) {
    let Some(secrets) = secrets else { return };
    let output = responses_route(secrets)
        .execute(call(true), &(), None)
        .await
        .unwrap();
    let ResponsesOutput::Stream { chunks, .. } = output else {
        panic!("expected SSE");
    };
    let wire = chunks.try_collect::<Vec<_>>().await.unwrap().concat();
    let events = decode(&wire).unwrap();
    assert_eq!(
        events.first().map(|event| &event.event_type),
        Some(&ResponsesEventType::ResponseCreated)
    );
    assert!(
        events
            .iter()
            .any(|event| event.event_type == ResponsesEventType::OutputTextDelta)
    );
    let last = events.last().unwrap();
    assert_eq!(last.event_type, ResponsesEventType::ResponseCompleted);
    assert!(
        last.data["response"]["usage"]["total_tokens"]
            .as_u64()
            .unwrap()
            > 0
    );
}
