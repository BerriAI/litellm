use futures_util::TryStreamExt;
use litellm_host::{
    interceptors::{RetriedRequest, RetryReason},
    lifecycle::{CallEvent, ExecutionEvent},
};
use litellm_inference_messages::MessagesCallResponse;
use litellm_inference_testing::RecordingSecrets;
use litellm_llms::base_llm::messages::context::MessagesModelCapabilities;
use rstest::rstest;

use super::*;

const INVALID_SIGNATURE: &str = "messages.1.content.0: Invalid `signature` in `thinking` block";
const BOUND_ELSEWHERE: &str = "messages.1.content.0: Invalid `signature` in `thinking` block. The block is bound to a different conversation.";

fn rejection(status: u16, message: &str) -> ResponseTemplate {
    status_response(
        status,
        json!({"type": "error", "error": {"type": "invalid_request_error", "message": message}}),
    )
}

fn sse_response() -> ResponseTemplate {
    ResponseTemplate::new(200).set_body_raw(
        "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
        "text/event-stream",
    )
}

fn replay(call: MessagesCall, api_base: String, stream: bool, thinking: Value) -> MessagesCall {
    MessagesCall {
        api_key: Some("sk-ant".into()),
        api_base: Some(api_base),
        ..with_fields(
            call,
            json!({
                "max_tokens": 4096,
                "stream": stream,
                "thinking": thinking,
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": [
                        {"type": "thinking", "thinking": "plan", "signature": "sig"},
                        {"type": "text", "text": "hello"}
                    ]},
                    {"role": "assistant", "content": [{"type": "redacted_thinking", "data": "opaque"}]},
                    {"role": "user", "content": "again"}
                ]
            }),
        )
    }
}

fn enabled() -> Value {
    json!({"type": "enabled", "budget_tokens": 2048})
}

async fn execute(
    host: &RecordingCall<Messages>,
) -> Result<(MessagesCallResponse, Vec<bytes::Bytes>), Error> {
    let route = messages_route(Arc::new(RecordingSecrets::empty()));
    let response = route
        .execute(host.request()?, host, Some(host.events.0.sender.clone()))
        .await?;
    match response {
        MessagesCallResponse::Stream { head, chunks } => Ok((
            MessagesCallResponse::Stream {
                head,
                chunks: Box::pin(futures_util::stream::empty()),
            },
            chunks.try_collect().await?,
        )),
        complete => Ok((complete, Vec::new())),
    }
}

fn observed_retries(host: &RecordingCall<Messages>) -> Vec<RetriedRequest> {
    host.events
        .0
        .lock()
        .unwrap()
        .iter()
        .filter_map(|event| match event {
            CallEvent::Execution(ExecutionEvent::ProviderRequestRetried { retry }) => {
                Some(retry.clone())
            }
            _ => None,
        })
        .collect()
}

#[rstest]
#[case::invalid_signature(INVALID_SIGNATURE)]
#[case::missing_signature(
    "messages.1.content.0.thinking.signature.str: Input should be a valid string"
)]
#[case::empty_thinking("messages.1.content.0.thinking: each thinking block must contain thinking")]
#[case::unpinned_binding_mismatch(BOUND_ELSEWHERE)]
#[tokio::test]
async fn a_rejected_thinking_replay_is_sent_once_more_without_thinking(
    call: MessagesCall,
    #[case] message: &str,
    #[values(false, true)] stream: bool,
    #[values("anthropic", "azure_ai", "deepseek", "minimax")] provider: &str,
) {
    let success = if stream {
        sse_response()
    } else {
        message_response()
    };
    let upstream = upstream([rejection(400, message), success]).await;
    let host = RecordingCall::<Messages>::new(replay(
        MessagesCall {
            custom_llm_provider: Some(provider.into()),
            ..call
        },
        upstream.uri(),
        stream,
        enabled(),
    ));

    let (response, chunks) = execute(&host).await.expect("the retry succeeds");

    let [first, second] = <[_; 2]>::try_from(received(&upstream).await).unwrap();
    assert_eq!(first.json()["thinking"], enabled());
    assert_eq!(
        second.json()["messages"],
        json!([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [{"type": "text", "text": "hello"}]},
            {"role": "user", "content": "again"}
        ])
    );
    let mut expected = first.json();
    expected.as_object_mut().unwrap().shift_remove("thinking");
    expected["messages"] = second.json()["messages"].clone();
    assert_eq!(second.json(), expected);
    assert_eq!(first.header("x-hook"), second.header("x-hook"));
    let expected_retry = RetriedRequest {
        reasons: vec![RetryReason::ThinkingSignature],
        body: second.json(),
    };
    assert_eq!(*host.retried.lock().unwrap(), vec![expected_retry.clone()]);
    assert_eq!(observed_retries(&host), vec![expected_retry]);
    assert_eq!(
        matches!(response, MessagesCallResponse::Stream { .. }),
        stream
    );
    assert_eq!(chunks.is_empty(), !stream);
}

#[rstest]
#[tokio::test]
async fn the_raw_head_of_a_retried_call_carries_the_retried_bytes(
    call: MessagesCall,
    #[values(false, true)] stream: bool,
) {
    let success = if stream {
        sse_response()
    } else {
        message_response()
    };
    let upstream = upstream([rejection(400, INVALID_SIGNATURE), success]).await;
    let host = RecordingCall::<Messages>::new(replay(call, upstream.uri(), stream, enabled()));

    execute(&host).await.expect("the retry succeeds");

    let [_, retried] = <[_; 2]>::try_from(received(&upstream).await).unwrap();
    let [raw] = <[_; 1]>::try_from(host.events.0.raw_heads()).unwrap();
    assert_eq!(
        raw.head.expect("the head is forwarded").sent_body,
        retried.body
    );
}

#[rstest]
#[tokio::test]
async fn a_second_rejection_is_the_error_the_caller_sees(
    call: MessagesCall,
    #[values(false, true)] stream: bool,
) {
    let upstream = upstream([
        rejection(400, INVALID_SIGNATURE),
        rejection(
            400,
            "messages.0.content.0: Invalid `signature` in `thinking` block, again",
        ),
    ])
    .await;
    let host = RecordingCall::<Messages>::new(replay(call, upstream.uri(), stream, enabled()));

    let error = execute(&host).await.err().expect("the retry fails");

    assert_eq!(received(&upstream).await.len(), 2);
    let (status, body) = error.upstream_response().unwrap();
    assert_eq!(status, 400);
    assert!(body.contains("again"), "{body}");
    assert_eq!(host.retried.lock().unwrap().len(), 1);
}

#[rstest]
#[case::not_a_400(429, INVALID_SIGNATURE, json!({"type": "adaptive"}))]
#[case::unrelated_400(400, "max_tokens: 4096 > 64000", json!({"type": "adaptive"}))]
#[case::signature_without_thinking(400, "Invalid `signature` in `tool_use` block", json!({"type": "adaptive"}))]
#[case::caller_pinned_binding(
    400,
    BOUND_ELSEWHERE,
    json!({"type": "adaptive", "block_binding": {"prefix_mismatch_behavior": "error"}})
)]
#[tokio::test]
async fn other_failures_reach_the_caller_without_a_retry(
    call: MessagesCall,
    #[case] status: u16,
    #[case] message: &str,
    #[case] thinking: Value,
) {
    let upstream = upstream([rejection(status, message), message_response()]).await;
    let adaptive = MessagesShaping {
        capabilities: MessagesModelCapabilities {
            supports_adaptive_thinking: true,
            ..MessagesModelCapabilities::default()
        },
        ..MessagesShaping::default()
    };
    let host = RecordingCall::<Messages>::new(replay(
        MessagesCall {
            shaping: adaptive,
            ..call
        },
        upstream.uri(),
        false,
        thinking.clone(),
    ));

    let error = execute(&host).await.err().expect("the call fails");

    let sent = only_request(&upstream).await;
    assert_eq!(sent.json()["thinking"], thinking);
    assert_eq!(error.upstream_response().unwrap().0, status);
    assert!(host.retried.lock().unwrap().is_empty());
    assert!(observed_retries(&host).is_empty());
}

const STRICT_REJECTION: &str = "Too many strict tools (21). The maximum is 20.";

fn strict_tools(
    call: MessagesCall,
    api_base: String,
    stream: bool,
    drop_params: bool,
) -> MessagesCall {
    MessagesCall {
        api_key: Some("sk-ant".into()),
        api_base: Some(api_base),
        shaping: MessagesShaping {
            settings: MessagesSettings {
                drop_params,
                drops_unsupported_params: drop_params,
                ..MessagesSettings::default()
            },
            ..MessagesShaping::default()
        },
        ..with_fields(
            call,
            json!({
                "max_tokens": 64,
                "stream": stream,
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [
                    {"name": "a", "input_schema": {"type": "object"}, "strict": true},
                    {"name": "b", "input_schema": {"type": "object"}, "strict": false},
                    {"name": "c", "input_schema": {"type": "object"}}
                ]
            }),
        )
    }
}

#[rstest]
#[tokio::test]
async fn a_rejected_strict_tool_is_sent_once_more_without_strict_under_drop_params(
    call: MessagesCall,
    #[values(false, true)] stream: bool,
) {
    let success = if stream {
        sse_response()
    } else {
        message_response()
    };
    let upstream = upstream([rejection(400, STRICT_REJECTION), success]).await;
    let host = RecordingCall::<Messages>::new(strict_tools(call, upstream.uri(), stream, true));

    let (response, _) = execute(&host).await.expect("the retry succeeds");

    let [first, second] = <[_; 2]>::try_from(received(&upstream).await).unwrap();
    assert_eq!(first.json()["tools"][0]["strict"], json!(true));
    let mut expected = first.json();
    expected["tools"] = json!([
        {"name": "a", "input_schema": {"type": "object"}},
        {"name": "b", "input_schema": {"type": "object"}},
        {"name": "c", "input_schema": {"type": "object"}}
    ]);
    assert_eq!(second.json(), expected);
    let expected_retry = RetriedRequest {
        reasons: vec![RetryReason::StrictTools { dropped: 1 }],
        body: second.json(),
    };
    assert_eq!(*host.retried.lock().unwrap(), vec![expected_retry.clone()]);
    assert_eq!(observed_retries(&host), vec![expected_retry]);
    assert_eq!(
        matches!(response, MessagesCallResponse::Stream { .. }),
        stream
    );
}

#[rstest]
#[tokio::test]
async fn a_rejected_strict_tool_reaches_the_caller_without_drop_params(call: MessagesCall) {
    let upstream = upstream([rejection(400, STRICT_REJECTION), message_response()]).await;
    let host = RecordingCall::<Messages>::new(strict_tools(call, upstream.uri(), false, false));

    let error = execute(&host).await.err().expect("the call fails");

    assert_eq!(
        only_request(&upstream).await.json()["tools"][0]["strict"],
        json!(true)
    );
    assert_eq!(error.upstream_response().unwrap().0, 400);
    assert!(observed_retries(&host).is_empty());
}

#[rstest]
#[tokio::test]
async fn strict_tools_and_a_thinking_replay_share_the_one_retry(call: MessagesCall) {
    let upstream = upstream([
        rejection(400, STRICT_REJECTION),
        rejection(400, INVALID_SIGNATURE),
        message_response(),
    ])
    .await;
    let host = RecordingCall::<Messages>::new(replay(
        strict_tools(call, upstream.uri(), false, true),
        upstream.uri(),
        false,
        enabled(),
    ));

    let error = execute(&host)
        .await
        .err()
        .expect("the second rejection is raised");

    let [first, second] = <[_; 2]>::try_from(received(&upstream).await).unwrap();
    assert_eq!(first.json()["thinking"], enabled());
    assert_eq!(second.json()["thinking"], enabled());
    assert_eq!(
        second.json()["tools"][0],
        json!({"name": "a", "input_schema": {"type": "object"}})
    );
    assert_eq!(received(&upstream).await.len(), 2);
    assert!(error.upstream_response().unwrap().1.contains("signature"));
}

#[rstest]
#[tokio::test]
async fn a_truthy_global_drop_params_that_is_not_true_does_not_drop_strict(call: MessagesCall) {
    let upstream = upstream([rejection(400, STRICT_REJECTION), message_response()]).await;
    let strict = strict_tools(call, upstream.uri(), false, true);
    let host = RecordingCall::<Messages>::new(MessagesCall {
        shaping: MessagesShaping {
            settings: MessagesSettings {
                drops_unsupported_params: false,
                ..strict.shaping.settings.clone()
            },
            ..strict.shaping.clone()
        },
        ..strict
    });

    let error = execute(&host).await.err().expect("the call fails");

    assert_eq!(
        only_request(&upstream).await.json()["tools"][0]["strict"],
        json!(true)
    );
    assert_eq!(error.upstream_response().unwrap().0, 400);
    assert!(observed_retries(&host).is_empty());
}
