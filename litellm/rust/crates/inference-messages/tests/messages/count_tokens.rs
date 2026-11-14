use litellm_llms::base_llm::messages::context::{MessagesModelCapabilities, SupportedEffortTiers};
use rstest::rstest;

use super::*;

fn without_max_tokens(model: &str, fields: Value) -> MessagesRequest {
    let messages = json!({"model": model, "messages": [{"role": "user", "content": "hi"}]});
    body(Value::Object(
        object(messages).into_iter().chain(object(fields)).collect(),
    ))
}

async fn count_on_the_wire(call: MessagesCall) -> Value {
    let upstream = upstream([json_response(json!({"input_tokens": 3}))]).await;
    messages_route(Arc::new(RecordingSecrets::empty()))
        .count_tokens(MessagesCall {
            api_key: Some("sk-ant".into()),
            api_base: Some(upstream.uri()),
            ..call
        })
        .await
        .expect("count succeeds");
    only_request(&upstream).await.json()
}

#[rstest]
#[tokio::test]
async fn vertex_ai_counts_at_the_publisher_count_endpoint_with_the_unversioned_model(
    call: MessagesCall,
) {
    let upstream = upstream([json_response(json!({"input_tokens": 14}))]).await;
    let messages = json!([{"role": "user", "content": "hi"}]);

    let counted = messages_route(Arc::new(RecordingSecrets::empty()))
        .count_tokens(MessagesCall {
            body: body(json!({"model": "claude-sonnet-4-5@20250929", "messages": messages})),
            custom_llm_provider: Some("vertex_ai".into()),
            litellm_params: serde_json::from_value(json!({
                "model": "vertex_ai/claude-sonnet-4-5@20250929",
                "vertex_project": "proj",
                "vertex_location": "europe-west1",
            }))
            .unwrap(),
            api_base: Some(upstream.uri()),
            extra_headers: headers([
                ("Authorization", "Bearer caller-token"),
                ("anthropic-version", "2023-06-01"),
            ]),
            ..call
        })
        .await
        .expect("count succeeds");

    assert_eq!(counted.input_tokens, 14);
    let request = only_request(&upstream).await;
    assert_eq!(
        request.url.path(),
        "/v1/projects/proj/locations/europe-west1/publishers/anthropic/models/count-tokens:rawPredict"
    );
    assert_eq!(request.header("authorization"), Some("Bearer caller-token"));
    assert_eq!(request.header("anthropic-version"), None);
    assert_eq!(
        request.json(),
        json!({"model": "claude-sonnet-4-5", "messages": messages})
    );
}

#[rstest]
#[tokio::test]
async fn a_count_without_input_tokens_is_an_invalid_response(call: MessagesCall) {
    let upstream = upstream([json_response(json!({"tokens": 14}))]).await;

    let error = messages_route(Arc::new(RecordingSecrets::empty()))
        .count_tokens(MessagesCall {
            api_key: Some("sk-ant".into()),
            api_base: Some(upstream.uri()),
            ..call
        })
        .await
        .unwrap_err();

    assert!(error.is_response(), "{error:?}");
}

#[rstest]
#[case::max_tokens(json!({"max_tokens": 1024}))]
#[case::stream(json!({"stream": true}))]
#[case::temperature(json!({"temperature": 0.5}))]
#[case::top_p(json!({"top_p": 0.5}))]
#[case::top_k(json!({"top_k": 5}))]
#[case::stop_sequences(json!({"stop_sequences": ["end"]}))]
#[case::metadata_the_messages_route_would_refuse(json!({"metadata": {"user_id": 7}}))]
#[case::speed(json!({"speed": "fast"}))]
#[case::service_tier(json!({"service_tier": "auto"}))]
#[case::container(json!({"container": "container_1"}))]
#[case::inference_geo(json!({"inference_geo": "us"}))]
#[case::mcp_servers(json!({"mcp_servers": [{"type": "url", "name": "docs", "url": "https://mcp.test"}]}))]
#[case::compaction(json!({"compaction": {"type": "summarize"}}))]
#[case::output_format(json!({"output_format": {"type": "json_schema"}}))]
#[case::field_no_provider_knows(json!({"litellm_metadata": {"tags": ["a"]}}))]
#[tokio::test]
async fn a_count_sends_only_what_the_count_endpoint_reads(
    call: MessagesCall,
    #[case] fields: Value,
) {
    let upstream = upstream([json_response(json!({"input_tokens": 3}))]).await;

    messages_route(Arc::new(RecordingSecrets::empty()))
        .count_tokens(MessagesCall {
            body: without_max_tokens(MODEL, fields),
            api_key: Some("sk-ant".into()),
            api_base: Some(upstream.uri()),
            shaping: MessagesShaping {
                capabilities: MessagesModelCapabilities {
                    supports_sampling_params: false,
                    ..MessagesModelCapabilities::default()
                },
                ..MessagesShaping::default()
            },
            ..call
        })
        .await
        .expect("count succeeds");

    let request = only_request(&upstream).await;
    assert_eq!(request.header("anthropic-beta"), None);
    assert_eq!(
        request.json(),
        json!({"model": MODEL, "messages": [{"role": "user", "content": "hi"}]})
    );
}

#[rstest]
#[case::system(json!({"system": "be brief"}))]
#[case::tools(json!({"tools": [{"name": "lookup", "input_schema": {"type": "object"}}]}))]
#[case::tool_choice(json!({"tool_choice": {"type": "auto"}}))]
#[case::thinking(json!({"thinking": {"type": "enabled", "budget_tokens": 2048}}))]
#[case::output_config(json!({"output_config": {"format": {"type": "json_schema"}}}))]
#[case::context_management(json!({"context_management": {"edits": [{"type": "clear_tool_uses_20250919"}]}}))]
#[tokio::test]
async fn a_count_keeps_what_the_count_endpoint_reads(call: MessagesCall, #[case] fields: Value) {
    let expected = without_max_tokens(MODEL, fields.clone());

    let sent = count_on_the_wire(MessagesCall {
        body: without_max_tokens(MODEL, fields),
        ..call
    })
    .await;

    assert_eq!(sent, serde_json::to_value(expected).unwrap());
}

#[rstest]
#[tokio::test]
async fn a_count_asks_for_the_betas_of_the_fields_it_sends(call: MessagesCall) {
    let upstream = upstream([json_response(json!({"input_tokens": 3}))]).await;

    messages_route(Arc::new(RecordingSecrets::empty()))
        .count_tokens(MessagesCall {
            body: without_max_tokens(
                MODEL,
                json!({"context_management": {"edits": [{"type": "clear_tool_uses_20250919"}]}}),
            ),
            api_key: Some("sk-ant".into()),
            api_base: Some(upstream.uri()),
            ..call
        })
        .await
        .expect("count succeeds");

    assert_eq!(
        only_request(&upstream).await.header("anthropic-beta"),
        Some("context-management-2025-06-27")
    );
}

#[rstest]
#[case::no_ceiling_keeps_the_whole_budget(json!({"reasoning_effort": "high"}), 4096)]
#[case::the_callers_ceiling_caps_the_budget(
    json!({"reasoning_effort": "high", "max_tokens": 3000}),
    2999
)]
#[tokio::test]
async fn a_count_translates_reasoning_like_the_messages_route(
    call: MessagesCall,
    #[case] fields: Value,
    #[case] budget_tokens: u64,
) {
    let sent = count_on_the_wire(MessagesCall {
        body: without_max_tokens(MODEL, fields),
        shaping: MessagesShaping {
            capabilities: MessagesModelCapabilities {
                supports_reasoning: true,
                ..MessagesModelCapabilities::default()
            },
            ..MessagesShaping::default()
        },
        ..call
    })
    .await;

    assert_eq!(
        sent,
        json!({
            "model": MODEL,
            "messages": [{"role": "user", "content": "hi"}],
            "thinking": {"type": "enabled", "budget_tokens": budget_tokens},
        })
    );
}

#[rstest]
#[tokio::test]
async fn a_count_on_an_adaptive_model_sends_the_effort_the_messages_route_sends(
    call: MessagesCall,
) {
    let sent = count_on_the_wire(MessagesCall {
        body: without_max_tokens(MODEL, json!({"reasoning_effort": "high"})),
        shaping: MessagesShaping {
            capabilities: MessagesModelCapabilities {
                supports_reasoning: true,
                supports_adaptive_thinking: true,
                supports_output_config: true,
                effort_tiers: SupportedEffortTiers {
                    high: true,
                    ..SupportedEffortTiers::default()
                },
                ..MessagesModelCapabilities::default()
            },
            ..MessagesShaping::default()
        },
        ..call
    })
    .await;

    assert_eq!(sent["thinking"]["type"], "adaptive");
    assert_eq!(sent["output_config"]["effort"], "high");
    assert_eq!(sent.get("reasoning_effort"), None);
}

#[rstest]
#[tokio::test]
async fn a_vertex_count_lifts_the_system_role_and_strips_the_scope_vertex_refuses(
    call: MessagesCall,
) {
    let upstream = upstream([json_response(json!({"input_tokens": 3}))]).await;

    messages_route(Arc::new(RecordingSecrets::empty()))
        .count_tokens(MessagesCall {
            body: body(json!({
                "model": "claude-sonnet-4-5",
                "messages": [
                    {"role": "system", "content": "be brief"},
                    {"role": "user", "content": [{
                        "type": "text",
                        "text": "hi",
                        "cache_control": {"type": "ephemeral", "scope": "global"},
                    }]},
                ],
            })),
            custom_llm_provider: Some("vertex_ai".into()),
            litellm_params: serde_json::from_value(json!({
                "model": "vertex_ai/claude-sonnet-4-5",
                "vertex_project": "proj",
            }))
            .unwrap(),
            api_base: Some(upstream.uri()),
            extra_headers: headers([("Authorization", "Bearer caller-token")]),
            ..call
        })
        .await
        .expect("count succeeds");

    let sent = only_request(&upstream).await.json();
    assert_eq!(
        sent["system"],
        json!([{"type": "text", "text": "be brief"}])
    );
    assert_eq!(
        sent["messages"],
        json!([{"role": "user", "content": [{
            "type": "text",
            "text": "hi",
            "cache_control": {"type": "ephemeral"},
        }]}])
    );
}

#[rstest]
#[tokio::test]
async fn azure_ai_counts_at_its_anthropic_count_endpoint(call: MessagesCall) {
    let upstream = upstream([json_response(json!({"input_tokens": 9}))]).await;

    let counted = messages_route(Arc::new(RecordingSecrets::empty()))
        .count_tokens(MessagesCall {
            custom_llm_provider: Some("azure_ai".into()),
            api_key: Some("azure-key".into()),
            api_base: Some(upstream.uri()),
            ..call
        })
        .await
        .expect("count succeeds");

    assert_eq!(counted.input_tokens, 9);
    let request = only_request(&upstream).await;
    assert_eq!(request.url.path(), "/anthropic/v1/messages/count_tokens");
    assert_eq!(request.header("x-api-key"), Some("azure-key"));
    assert_eq!(
        request.json(),
        json!({"model": MODEL, "messages": [{"role": "user", "content": "hi"}]})
    );
}
