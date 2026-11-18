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

fn replayed_turns() -> Value {
    json!([
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "plan", "signature": "sig"},
            {"type": "text", "text": "hello"}
        ]},
        {"role": "assistant", "content": [{"type": "redacted_thinking", "data": "opaque"}]},
        {"role": "user", "content": "again"}
    ])
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
                "messages": replayed_turns()
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
    let head = raw.head.expect("the head is forwarded");
    assert_eq!(head.sent_body, retried.body);
    assert_eq!(head.retry_headers, None);
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

const BINDING_BETA: &str = "thinking-binding-controls-2026-08-01";
const CALLER_BETA: &str = "context-1m-2025-08-07";

fn with_adaptive_thinking(call: MessagesCall) -> MessagesCall {
    MessagesCall {
        shaping: MessagesShaping {
            capabilities: MessagesModelCapabilities {
                supports_adaptive_thinking: true,
                ..MessagesModelCapabilities::default()
            },
            ..call.shaping
        },
        ..call
    }
}

fn drop_block(mut thinking: Value) -> Value {
    thinking["block_binding"] = json!({"prefix_mismatch_behavior": "drop_block"});
    thinking
}

fn betas(header: Option<&str>) -> std::collections::BTreeSet<&str> {
    header
        .into_iter()
        .flat_map(|value| value.split(','))
        .collect()
}

fn binding_replay(
    call: MessagesCall,
    api_base: String,
    provider: &str,
    stream: bool,
) -> MessagesCall {
    MessagesCall {
        custom_llm_provider: Some(provider.into()),
        extra_headers: headers([("anthropic-beta", CALLER_BETA)]),
        ..replay(
            with_adaptive_thinking(call),
            api_base,
            stream,
            json!({"type": "adaptive", "display": "summarized"}),
        )
    }
}

fn assert_only_thinking_changed(first: &Value, second: &Value) {
    let mut expected = first.clone();
    expected["thinking"] = drop_block(first["thinking"].clone());
    expected["anthropic_beta"] = second.get("anthropic_beta").cloned().unwrap_or(Value::Null);
    let mut retried = second.clone();
    retried["anthropic_beta"] = expected["anthropic_beta"].clone();
    assert_eq!(retried, expected);
}

#[rstest]
#[tokio::test]
async fn a_binding_mismatch_keeps_the_history_and_adds_the_beta_header(
    call: MessagesCall,
    #[values("anthropic", "azure_ai")] provider: &str,
    #[values(false, true)] stream: bool,
) {
    let success = if stream {
        sse_response()
    } else {
        message_response()
    };
    let upstream = upstream([rejection(400, BOUND_ELSEWHERE), success]).await;
    let host =
        RecordingCall::<Messages>::new(binding_replay(call, upstream.uri(), provider, stream));

    execute(&host).await.expect("the retry succeeds");

    let [first, second] = <[_; 2]>::try_from(received(&upstream).await).unwrap();
    assert_only_thinking_changed(&first.json(), &second.json());
    assert_eq!(
        second.json()["thinking"],
        drop_block(json!({"type": "adaptive", "display": "summarized"}))
    );
    let sent = betas(first.header("anthropic-beta"));
    assert!(sent.contains(CALLER_BETA), "{sent:?}");
    assert!(!sent.contains(BINDING_BETA), "{sent:?}");
    assert_eq!(
        betas(second.header("anthropic-beta")),
        sent.iter().copied().chain([BINDING_BETA]).collect()
    );
    let expected_retry = RetriedRequest {
        reasons: vec![RetryReason::ThinkingBinding],
        body: second.json(),
    };
    assert_eq!(*host.retried.lock().unwrap(), vec![expected_retry.clone()]);
    assert_eq!(observed_retries(&host), vec![expected_retry]);
    let [raw] = <[_; 1]>::try_from(host.events.0.raw_heads()).unwrap();
    let logged_headers = raw
        .head
        .expect("the head is forwarded")
        .retry_headers
        .expect("the retry changed the headers");
    assert_eq!(
        betas(
            logged_headers
                .iter()
                .find(|(name, _)| name.eq_ignore_ascii_case("anthropic-beta"))
                .map(|(_, value)| value.as_str())
        ),
        betas(second.header("anthropic-beta"))
    );
}

#[rstest]
#[tokio::test]
async fn a_bad_signature_after_a_binding_retry_recovers_without_thinking(
    call: MessagesCall,
    #[values("anthropic", "azure_ai")] provider: &str,
) {
    let upstream = upstream([
        rejection(400, BOUND_ELSEWHERE),
        rejection(400, INVALID_SIGNATURE),
        message_response(),
    ])
    .await;
    let host =
        RecordingCall::<Messages>::new(binding_replay(call, upstream.uri(), provider, false));

    execute(&host).await.expect("the third attempt succeeds");

    let [first, dropping, stripped] = <[_; 3]>::try_from(received(&upstream).await).unwrap();
    assert_only_thinking_changed(&first.json(), &dropping.json());
    assert!(stripped.json().get("thinking").is_none());
    assert!(
        !stripped.json()["messages"]
            .to_string()
            .contains("\"thinking\"")
    );
    assert_eq!(
        host.retried
            .lock()
            .unwrap()
            .iter()
            .map(|retried| retried.reasons.clone())
            .collect::<Vec<_>>(),
        vec![
            vec![RetryReason::ThinkingBinding],
            vec![RetryReason::ThinkingSignature]
        ]
    );
}

#[rstest]
#[tokio::test]
async fn a_bad_signature_after_a_binding_retry_stops_at_the_third_attempt(call: MessagesCall) {
    let upstream = upstream([
        rejection(400, BOUND_ELSEWHERE),
        rejection(400, INVALID_SIGNATURE),
        rejection(400, INVALID_SIGNATURE),
        message_response(),
    ])
    .await;
    let host =
        RecordingCall::<Messages>::new(binding_replay(call, upstream.uri(), "anthropic", false));

    assert!(execute(&host).await.is_err());

    assert_eq!(received(&upstream).await.len(), 3);
}

#[rstest]
#[tokio::test]
async fn a_binding_mismatch_on_bedrock_invoke_names_the_beta_in_the_body(call: MessagesCall) {
    let upstream = upstream([rejection(400, BOUND_ELSEWHERE), message_response()]).await;
    let host = RecordingCall::<Messages>::new(MessagesCall {
        extra_headers: None,
        ..binding_replay(
            with_model(call, "anthropic.claude-sonnet-5-5"),
            upstream.uri(),
            "bedrock",
            false,
        )
    });

    execute(&host).await.expect("the retry succeeds");

    let [first, second] = <[_; 2]>::try_from(received(&upstream).await).unwrap();
    assert_only_thinking_changed(&first.json(), &second.json());
    assert!(
        !first.json()["anthropic_beta"]
            .as_array()
            .is_some_and(|sent| sent.contains(&json!(BINDING_BETA)))
    );
    assert!(
        second.json()["anthropic_beta"]
            .as_array()
            .is_some_and(|sent| sent.contains(&json!(BINDING_BETA)))
    );
    assert!(!betas(second.header("anthropic-beta")).contains(BINDING_BETA));
    assert_eq!(
        *host.retried.lock().unwrap(),
        vec![RetriedRequest {
            reasons: vec![RetryReason::ThinkingBinding],
            body: second.json(),
        }]
    );
}

#[rstest]
#[tokio::test]
async fn a_binding_mismatch_without_thinking_turns_on_adaptive_thinking(call: MessagesCall) {
    let upstream = upstream([rejection(400, BOUND_ELSEWHERE), message_response()]).await;
    let host = RecordingCall::<Messages>::new(MessagesCall {
        api_key: Some("sk-ant".into()),
        api_base: Some(upstream.uri()),
        ..with_fields(
            call,
            json!({"max_tokens": 4096, "messages": replayed_turns()}),
        )
    });

    execute(&host).await.expect("the retry succeeds");

    let [first, second] = <[_; 2]>::try_from(received(&upstream).await).unwrap();
    assert_eq!(first.json().get("thinking"), None);
    assert_eq!(second.json()["messages"], first.json()["messages"]);
    assert_eq!(
        second.json()["thinking"],
        drop_block(json!({"type": "adaptive"}))
    );
    assert!(betas(second.header("anthropic-beta")).contains(BINDING_BETA));
}

#[rstest]
#[tokio::test]
async fn a_rejected_binding_retry_is_the_error_the_caller_sees(call: MessagesCall) {
    let upstream = upstream([
        rejection(400, BOUND_ELSEWHERE),
        rejection(
            400,
            "messages.1.content.0: bound to a different conversation, again",
        ),
        message_response(),
    ])
    .await;
    let host =
        RecordingCall::<Messages>::new(binding_replay(call, upstream.uri(), "anthropic", false));

    let error = execute(&host).await.err().expect("the retry fails");

    let [_, second] = <[_; 2]>::try_from(received(&upstream).await).unwrap();
    assert_eq!(
        second.json()["thinking"]["block_binding"]["prefix_mismatch_behavior"],
        "drop_block"
    );
    let (status, body) = error.upstream_response().unwrap();
    assert_eq!(status, 400);
    assert!(body.contains("again"), "{body}");
    assert_eq!(host.retried.lock().unwrap().len(), 1);
}

#[rstest]
#[case::enabled_thinking("anthropic", false, enabled())]
#[case::a_caller_chosen_binding(
    "anthropic",
    true,
    json!({"type": "adaptive", "block_binding": {"prefix_mismatch_behavior": "drop_block"}})
)]
#[case::a_host_that_is_not_claude("deepseek", true, json!({"type": "adaptive"}))]
#[tokio::test]
async fn other_binding_mismatches_keep_the_strip_retry(
    call: MessagesCall,
    #[case] provider: &str,
    #[case] adaptive_model: bool,
    #[case] thinking: Value,
) {
    let upstream = upstream([rejection(400, BOUND_ELSEWHERE), message_response()]).await;
    let shaped = if adaptive_model {
        with_adaptive_thinking(call)
    } else {
        call
    };
    let host = RecordingCall::<Messages>::new(replay(
        MessagesCall {
            custom_llm_provider: Some(provider.into()),
            ..shaped
        },
        upstream.uri(),
        false,
        thinking,
    ));

    execute(&host).await.expect("the retry succeeds");

    let [first, second] = <[_; 2]>::try_from(received(&upstream).await).unwrap();
    assert_eq!(second.json().get("thinking"), None);
    assert_eq!(second.json()["messages"].as_array().unwrap().len(), 3);
    assert_eq!(
        betas(second.header("anthropic-beta")),
        betas(first.header("anthropic-beta"))
    );
    assert_eq!(
        host.retried.lock().unwrap()[0].reasons,
        vec![RetryReason::ThinkingSignature]
    );
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
