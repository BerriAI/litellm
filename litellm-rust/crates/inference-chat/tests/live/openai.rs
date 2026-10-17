use std::time::Duration;

use futures_util::TryStreamExt;
use litellm_host::call::CallOutput;
use litellm_inference_chat::{
    ChatCompletionsRoute,
    openai::{OpenAiChatCall, OpenAiChatOutput},
};
use litellm_inference_testing::live::{LiveResources, within_deadline};
use rstest::{fixture, rstest};
use serde_json::{Map, Value, json};

const DEFAULT_MODEL: &str = "openai/gpt-6-astra";

/// A route reading `OPENAI_API_KEY` from the environment; tests skip when it is unset.
#[fixture]
fn route() -> Option<ChatCompletionsRoute> {
    std::env::var("OPENAI_API_KEY").ok()?;
    let resources = LiveResources::default();
    Some(ChatCompletionsRoute::new(
        resources.http,
        resources.auth,
        resources.secrets,
    ))
}

fn call(extra: Value) -> OpenAiChatCall {
    let Value::Object(extra) = extra else {
        panic!("extra params are an object");
    };
    let body: Map<String, Value> = json!({
        "model": "public",
        "messages": [{"role": "user", "content": "Reply with the single word: pong"}],
        "max_completion_tokens": 256,
    })
    .as_object()
    .unwrap()
    .clone()
    .into_iter()
    .chain(extra)
    .collect();
    OpenAiChatCall {
        model: std::env::var("OPENAI_LIVE_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.into()),
        body,
        api_key: None,
        api_base: None,
        custom_llm_provider: None,
        extra_headers: None,
        timeout: Some(Duration::from_secs(60)),
        ..OpenAiChatCall::default()
    }
}

async fn execute(route: &ChatCompletionsRoute, call: OpenAiChatCall) -> OpenAiChatOutput {
    within_deadline(route.execute_openai(call, &(), None))
        .await
        .unwrap()
}

#[rstest]
#[tokio::test]
async fn a_completion_returns_openai_json(route: Option<ChatCompletionsRoute>) {
    let Some(route) = route else { return };
    let CallOutput::Complete(body) = execute(&route, call(json!({}))).await else {
        panic!("expected a complete response");
    };
    let body: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(body["object"], "chat.completion", "{body}");
    assert!(
        body["choices"][0]["message"]["content"]
            .as_str()
            .is_some_and(|text| text.to_ascii_lowercase().contains("pong")),
        "{body}"
    );
    assert!(
        body["usage"]["total_tokens"].as_u64().unwrap() > 0,
        "{body}"
    );
}

#[rstest]
#[tokio::test]
async fn a_stream_ends_with_usage_and_done(route: Option<ChatCompletionsRoute>) {
    let Some(route) = route else { return };
    let output = execute(
        &route,
        call(json!({"stream": true, "stream_options": {"include_usage": true}})),
    )
    .await;
    let CallOutput::Stream { chunks, .. } = output else {
        panic!("expected SSE");
    };
    let wire = within_deadline(chunks.try_collect::<Vec<_>>())
        .await
        .unwrap()
        .concat();
    let text = std::str::from_utf8(&wire).unwrap();
    let events: Vec<&str> = text
        .split("\n\n")
        .filter_map(|event| event.trim().strip_prefix("data:"))
        .map(str::trim)
        .collect();
    assert_eq!(events.last(), Some(&"[DONE]"), "{text}");
    let usage: Value = serde_json::from_str(events[events.len() - 2]).unwrap();
    assert!(
        usage["usage"]["total_tokens"].as_u64().unwrap() > 0,
        "{usage}"
    );
}
