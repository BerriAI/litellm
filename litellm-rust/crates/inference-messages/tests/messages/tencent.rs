use std::ops::ControlFlow;

use bytes::Bytes;
use litellm_llms::base_llm::messages::context::MessagesModelCapabilities;
use litellm_llms_types::providers::anthropic::DEFAULT_HEADERS;
use rstest::{fixture, rstest};

use super::*;

#[fixture]
fn tencent_call(call: MessagesCall) -> MessagesCall {
    MessagesCall {
        custom_llm_provider: Some("tencent".into()),
        api_key: Some("deployment-key".into()),
        ..call
    }
}

#[rstest]
#[case::deployment_key(&[], "x-api-key", "deployment-key")]
#[case::forwarded_key(&[("X-Api-Key", "caller-key")], "x-api-key", "caller-key")]
#[case::forwarded_bearer(&[("Authorization", "Bearer caller")], "authorization", "Bearer caller")]
#[tokio::test]
async fn credentials_reach_the_selected_provider(
    tencent_call: MessagesCall,
    #[case] forwarded: &[(&str, &str)],
    #[case] header: &str,
    #[case] credential: &str,
) {
    let upstream = upstream([message_response()]).await;
    run_message(MessagesCall {
        api_base: Some(upstream.uri()),
        extra_headers: headers(forwarded.iter().copied()),
        ..tencent_call
    })
    .await;
    let sent = only_request(&upstream).await;
    assert_eq!(sent.header_values(header), [credential]);
    let version = DEFAULT_HEADERS
        .iter()
        .find_map(|(name, value)| (*name == "anthropic-version").then_some(*value))
        .unwrap();
    assert_eq!(sent.header_values("anthropic-version"), [version]);
    assert_eq!(sent.json()["model"], MODEL);
}

#[rstest]
#[tokio::test]
async fn credentials_and_base_resolve_from_the_secret_source(tencent_call: MessagesCall) {
    let upstream = upstream([message_response()]).await;
    let base = format!("{}/v1/chat/completions/", upstream.uri());
    let secrets = Arc::new(RecordingSecrets::new([
        ("TENCENT_API_KEY", "secret-key"),
        ("TENCENT_ANTHROPIC_API_BASE", base.as_str()),
        ("TENCENT_API_BASE", UNREACHABLE_BASE),
    ]));
    run_with(
        secrets,
        MessagesCall {
            api_key: None,
            ..tencent_call
        },
    )
    .await
    .unwrap();
    let sent = only_request(&upstream).await;
    assert_eq!(sent.url.path(), "/v1/messages");
    assert_eq!(sent.header("x-api-key"), Some("secret-key"));
}

#[rstest]
#[tokio::test]
async fn missing_credentials_fail_before_sending(tencent_call: MessagesCall) {
    let upstream = upstream([message_response()]).await;
    let error = run(MessagesCall {
        api_key: None,
        api_base: Some(upstream.uri()),
        ..tencent_call
    })
    .await
    .unwrap_err();
    assert!(matches!(
        error,
        Error::Auth(litellm_auth::Error::MissingApiKey {
            environment_variable: "TENCENT_API_KEY",
            ..
        })
    ));
    assert!(received(&upstream).await.is_empty());
}

#[rstest]
#[case::missing_max_tokens(json!({"max_tokens": null}))]
#[case::invalid_metadata(json!({"metadata": {"user_id": 7}}))]
#[case::invalid_reasoning_effort(json!({"reasoning_effort": "unknown"}))]
#[tokio::test]
async fn malformed_requests_fail_before_sending(tencent_call: MessagesCall, #[case] fields: Value) {
    let upstream = upstream([message_response()]).await;
    let error = run(with_fields(
        MessagesCall {
            api_base: Some(upstream.uri()),
            ..tencent_call
        },
        fields,
    ))
    .await
    .unwrap_err();
    assert!(error.is_request(), "{error:?}");
    assert!(received(&upstream).await.is_empty());
}

#[rstest]
#[case::legacy(false, json!({"type": "enabled", "budget_tokens": 4095}))]
#[case::adaptive(true, json!({"type": "adaptive", "display": "summarized"}))]
#[tokio::test]
async fn reasoning_effort_becomes_model_appropriate_thinking(
    tencent_call: MessagesCall,
    #[case] adaptive: bool,
    #[case] thinking: Value,
) {
    let upstream = upstream([message_response()]).await;
    run_message(with_fields(
        MessagesCall {
            api_base: Some(upstream.uri()),
            shaping: MessagesShaping {
                capabilities: MessagesModelCapabilities {
                    supports_reasoning: true,
                    supports_adaptive_thinking: adaptive,
                    ..Default::default()
                },
                ..Default::default()
            },
            ..tencent_call
        },
        json!({"max_tokens": 4096, "reasoning_effort": "high"}),
    ))
    .await;
    let sent = only_request(&upstream).await.json();
    assert_eq!(sent["thinking"], thinking);
    assert_eq!(sent.get("reasoning_effort"), None);
    assert_eq!(
        sent.get("output_config"),
        adaptive.then(|| json!({"effort": "high"})).as_ref()
    );
}

#[rstest]
#[case::encrypted_reasoning(json!({"type": "thinking", "thinking": "private", "signature": "litellm_encrypted_reasoning:private"}))]
#[case::orphaned_advisor(json!({"type": "server_tool_use", "name": "advisor", "id": "srvtoolu_advisor", "input": {}}))]
#[tokio::test]
async fn replayed_history_is_cleaned_without_losing_text(
    tencent_call: MessagesCall,
    #[case] block: Value,
) {
    let upstream = upstream([message_response()]).await;
    run_message(with_fields(
        MessagesCall {
            api_base: Some(upstream.uri()),
            ..tencent_call
        },
        json!({"messages": [
            {"role": "assistant", "content": [block, {"type": "text", "text": "keep"}]},
            {"role": "user", "content": "continue"}
        ]}),
    ))
    .await;
    assert_eq!(
        only_request(&upstream).await.json()["messages"],
        json!([
            {"role": "assistant", "content": [{"type": "text", "text": "keep"}]},
            {"role": "user", "content": "continue"}
        ])
    );
}

#[rstest]
#[case::billing_string(json!("x-anthropic-billing-header: private"), None)]
#[case::mixed_blocks(json!([
    {"type": "text", "text": "x-anthropic-billing-header: private"},
    {"type": "text", "text": "keep"}
]), Some(json!([{ "type": "text", "text": "keep" }])))]
#[case::ordinary_system(json!("keep"), Some(json!("keep")))]
#[tokio::test]
async fn billing_is_removed_and_tools_and_cache_hints_are_preserved(
    tencent_call: MessagesCall,
    #[case] system: Value,
    #[case] expected: Option<Value>,
) {
    let upstream = upstream([message_response()]).await;
    let fields = json!({
        "system": system,
        "tools": [{"name": "echo", "input_schema": {"type": "object"}}],
        "messages": [{"role": "user", "content": [{
            "type": "text", "text": "hi", "cache_control": {"type": "ephemeral", "ttl": "1h"}
        }]}]
    });
    run_message(with_fields(
        MessagesCall {
            api_base: Some(upstream.uri()),
            ..tencent_call
        },
        fields.clone(),
    ))
    .await;
    let sent = only_request(&upstream).await.json();
    assert_eq!(sent.get("system"), expected.as_ref());
    assert_eq!(sent["tools"], fields["tools"]);
    assert_eq!(sent["messages"], fields["messages"]);
}

#[derive(Default)]
struct StreamSink(Mutex<Vec<u8>>);

impl litellm_host_native::in_process::StreamConsumer<Messages> for StreamSink {
    async fn open_stream(
        &self,
        _: litellm_inference_messages::route::MessagesStreamHead,
    ) -> Result<ControlFlow<()>, Error> {
        Ok(ControlFlow::Continue(()))
    }

    async fn send_chunk(&self, chunk: Bytes) -> Result<ControlFlow<()>, Error> {
        self.0.lock().unwrap().extend_from_slice(&chunk);
        Ok(ControlFlow::Continue(()))
    }
}

#[rstest]
#[tokio::test]
async fn streaming_relays_provider_events_unchanged(tencent_call: MessagesCall) {
    let events = concat!(
        "event: message_start\ndata: {\"type\":\"message_start\"}\n\n",
        "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"OK\"}}\n\n",
        "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"
    );
    let upstream =
        upstream([ResponseTemplate::new(200).set_body_raw(events, "text/event-stream")]).await;
    let host = LocalMessagesHost::new(with_fields(
        MessagesCall {
            api_base: Some(upstream.uri()),
            ..tencent_call
        },
        json!({"stream": true}),
    ));
    let sink = StreamSink::default();
    let outcome = litellm_host_native::in_process::run_hosted(
        machine(Arc::new(RecordingSecrets::empty()))(host.request().unwrap()),
        litellm_host_native::in_process::Host {
            services: &(),
            interceptors: &host,
            stream: &sink,
            observers: None,
        },
    )
    .await
    .unwrap();
    assert!(matches!(outcome, MessagesOutput::StreamEnded));
    assert_eq!(sink.0.into_inner().unwrap(), events.as_bytes());
    assert_eq!(only_request(&upstream).await.json()["stream"], true);
}
