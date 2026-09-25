use litellm_llms::anthropic::common_utils::{
    ANTHROPIC_ADVISOR_TOOL_TYPE, ANTHROPIC_OAUTH_BETA_HEADER, AnthropicModelCapabilities,
    SupportedEffortTiers, beta,
};
use litellm_types::utils::{ProviderSpecificHeader, ProviderSpecificHeaders};
use rstest::rstest;

use super::*;

#[rstest]
#[case::anthropic_key("anthropic", Some("sk-ant"), &[], ("x-api-key", "sk-ant"), &["authorization"])]
#[case::azure_key("azure_ai", Some("sk-azure"), &[], ("x-api-key", "sk-azure"), &["authorization"])]
#[case::caller_x_api_key_wins(
    "azure_ai",
    Some("rust-fallback-key"),
    &[("x-api-key", "from-python")],
    ("x-api-key", "from-python"),
    &["authorization"]
)]
#[case::entra_bearer_without_key(
    "azure_ai",
    None,
    &[("Authorization", "Bearer entra-token")],
    ("authorization", "Bearer entra-token"),
    &["x-api-key"]
)]
#[case::empty_bearer_falls_back_to_key(
    "azure_ai",
    Some("sk-azure"),
    &[("Authorization", "Bearer ")],
    ("x-api-key", "sk-azure"),
    &[]
)]
#[case::anthropic_forwards_caller_authorization(
    "anthropic",
    Some("sk-ant"),
    &[("Authorization", "Bearer caller")],
    ("authorization", "Bearer caller"),
    &["x-api-key"]
)]
#[case::anthropic_oauth_key_becomes_bearer(
    "anthropic",
    Some("sk-ant-oat01-token"),
    &[],
    ("authorization", "Bearer sk-ant-oat01-token"),
    &["x-api-key"]
)]
#[tokio::test]
async fn credentials_become_exactly_one_auth_header(
    call: MessagesCall,
    #[case] provider: &str,
    #[case] api_key: Option<&str>,
    #[case] extra_headers: &[(&str, &str)],
    #[case] expected: (&str, &str),
    #[case] absent: &[&str],
) {
    let upstream = upstream([message_response()]).await;

    run_message(MessagesCall {
        custom_llm_provider: Some(provider.into()),
        api_key: api_key.map(Into::into),
        api_base: Some(upstream.uri()),
        extra_headers: headers(extra_headers.iter().copied()),
        ..call
    })
    .await;

    let request = only_request(&upstream).await;
    let (name, value) = expected;
    assert_eq!(request.header_values(name), [value]);
    for name in absent {
        assert_eq!(request.header(name), None, "{name} must not be sent");
    }
}

#[rstest]
#[case::anthropic("anthropic")]
#[case::azure_ai("azure_ai")]
#[tokio::test]
async fn a_call_without_credentials_fails_before_sending(
    call: MessagesCall,
    #[case] provider: &str,
) {
    let upstream = upstream([message_response()]).await;

    let error = run(MessagesCall {
        custom_llm_provider: Some(provider.into()),
        api_base: Some(upstream.uri()),
        ..call
    })
    .await
    .err()
    .expect("a call without credentials fails");

    assert!(
        matches!(
            error,
            Error::Auth(litellm_auth::Error::MissingApiKey { .. })
        ),
        "{error:?}"
    );
    assert!(received(&upstream).await.is_empty());
}

#[rstest]
#[case::anthropic(MODEL, Some("anthropic"), "", "/v1/messages")]
#[case::anthropic_base_with_trailing_slash(MODEL, Some("anthropic"), "/", "/v1/messages")]
#[case::anthropic_base_with_the_messages_path(
    MODEL,
    Some("anthropic"),
    "/v1/messages",
    "/v1/messages"
)]
#[case::azure_ai(MODEL, Some("azure_ai"), "", "/anthropic/v1/messages")]
#[case::provider_from_model_prefix("anthropic/claude-sonnet-4-5", None, "", "/v1/messages")]
#[tokio::test]
async fn each_provider_posts_to_its_messages_endpoint(
    call: MessagesCall,
    #[case] model: &str,
    #[case] provider: Option<&str>,
    #[case] base_suffix: &str,
    #[case] path: &str,
) {
    let upstream = upstream([message_response()]).await;

    run_message(MessagesCall {
        model: model.into(),
        custom_llm_provider: provider.map(Into::into),
        api_key: Some("sk".into()),
        api_base: Some(format!("{}{base_suffix}", upstream.uri())),
        ..call
    })
    .await;

    let request = only_request(&upstream).await;
    assert_eq!(request.method.as_str(), "POST");
    assert_eq!(request.url.path(), path);
    assert_eq!(request.json()["model"], MODEL);
    assert_eq!(request.header_values("anthropic-version"), ["2023-06-01"]);
    assert_eq!(request.header_values("content-type"), ["application/json"]);
}

#[rstest]
#[case::unknown_provider(MODEL, Some("openai"), "openai")]
#[case::unresolvable_model(
    "no-such-model",
    None,
    "unable to resolve custom_llm_provider for messages request"
)]
#[tokio::test]
async fn unsupported_providers_are_rejected_before_sending(
    call: MessagesCall,
    #[case] model: &str,
    #[case] provider: Option<&str>,
    #[case] reported: &str,
) {
    let error = run(MessagesCall {
        model: model.into(),
        custom_llm_provider: provider.map(Into::into),
        api_key: Some("sk".into()),
        api_base: Some(UNREACHABLE_BASE.into()),
        ..call
    })
    .await
    .err()
    .expect("unsupported provider errors");

    assert_eq!(error, Error::InvalidProvider(reported.into()));
}

#[rstest]
#[tokio::test]
async fn caller_headers_and_provider_scoped_headers_are_forwarded(call: MessagesCall) {
    let upstream = upstream([message_response()]).await;
    let scoped = |provider: &str, value: &str| ProviderSpecificHeader {
        custom_llm_provider: provider.into(),
        extra_headers: object(json!({"x-scoped": value})),
    };

    run_message(MessagesCall {
        api_key: Some("sk".into()),
        api_base: Some(upstream.uri()),
        extra_headers: headers([("anthropic-beta", "token-efficient-tools-2025-02-19")]),
        provider_specific_header: Some(ProviderSpecificHeaders::Many(vec![
            scoped("bedrock", "other-provider"),
            scoped("azure_ai, anthropic", "this-provider"),
        ])),
        ..call
    })
    .await;

    let request = only_request(&upstream).await;
    assert_eq!(
        request.header("anthropic-beta"),
        Some("token-efficient-tools-2025-02-19")
    );
    assert_eq!(request.header_values("x-scoped"), ["this-provider"]);
}

#[rstest]
#[tokio::test]
async fn azure_strips_the_cache_control_scope_anthropic_rejects(call: MessagesCall) {
    let upstream = upstream([message_response()]).await;

    run_message(MessagesCall {
        custom_llm_provider: Some("azure_ai".into()),
        api_key: Some("sk-azure".into()),
        api_base: Some(upstream.uri()),
        body: object(json!({
            "model": MODEL,
            "max_tokens": 16,
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": "hi",
                    "cache_control": {"type": "ephemeral", "scope": "global"}
                }]
            }]
        })),
        ..call
    })
    .await;

    assert_eq!(
        only_request(&upstream).await.json()["messages"][0]["content"][0]["cache_control"],
        json!({"type": "ephemeral"})
    );
}

#[rstest]
#[tokio::test]
async fn additional_drop_params_remove_fields_before_sending(call: MessagesCall) {
    let upstream = upstream([message_response()]).await;
    let mut body = call.body.clone();
    body.insert("temperature".into(), json!(0.5));
    body.insert("top_k".into(), json!(3));

    run_message(MessagesCall {
        api_key: Some("sk".into()),
        api_base: Some(upstream.uri()),
        body,
        shaping: MessagesShaping {
            additional_drop_params: vec!["temperature".into()],
            ..MessagesShaping::default()
        },
        ..call
    })
    .await;

    let sent = only_request(&upstream).await.json();
    assert_eq!(sent.get("temperature"), None);
    assert_eq!(sent["top_k"], 3);
}

fn with_fields(call: MessagesCall, fields: Value) -> MessagesCall {
    let body: Map<String, Value> = call.body.into_iter().chain(object(fields)).collect();
    MessagesCall { body, ..call }
}

fn sent_betas(request: &wiremock::Request) -> Vec<String> {
    let [header] = <[&str; 1]>::try_from(request.header_values("anthropic-beta"))
        .unwrap_or_else(|values| panic!("expected one anthropic-beta header, got {values:?}"));
    header
        .split(',')
        .map(str::trim)
        .map(str::to_string)
        .collect()
}

#[rstest]
#[case::structured_output(json!({"output_format": {"type": "json_schema"}}), &[beta::STRUCTURED_OUTPUT])]
#[case::fast_mode(json!({"speed": "fast"}), &[beta::FAST_MODE_2026_02_01])]
#[case::compaction(json!({"compaction": {"enabled": true}}), &[beta::COMPACT_2026_09_04])]
#[case::context_management_edits(
    json!({"context_management": {"edits": [{"type": "clear_tool_uses_20250919"}]}}),
    &[beta::CONTEXT_MANAGEMENT_2025_06_27]
)]
#[case::per_message_output_config(
    json!({"messages": [{"role": "user", "content": "hi", "output_config": {"effort": "low"}}]}),
    &[beta::PER_TURN_CONTROL_2026_07_01]
)]
#[case::advisor_tool(
    json!({"tools": [{"type": ANTHROPIC_ADVISOR_TOOL_TYPE, "name": "advisor", "model": MODEL}]}),
    &[beta::ADVISOR_TOOL_2026_03_01]
)]
#[case::several_features_at_once(
    json!({"speed": "fast", "output_format": {"type": "json_schema"}}),
    &[beta::STRUCTURED_OUTPUT, beta::FAST_MODE_2026_02_01]
)]
#[tokio::test]
async fn feature_betas_join_the_callers_betas_in_one_sorted_header(
    call: MessagesCall,
    #[case] fields: Value,
    #[case] features: &[&str],
) {
    let upstream = upstream([message_response()]).await;
    let capabilities = AnthropicModelCapabilities {
        supports_speed: true,
        ..AnthropicModelCapabilities::default()
    };

    run_message(with_fields(
        MessagesCall {
            api_key: Some("sk".into()),
            api_base: Some(upstream.uri()),
            extra_headers: headers([("Anthropic-Beta", "caller-beta-2025-01-01")]),
            shaping: MessagesShaping {
                capabilities,
                ..MessagesShaping::default()
            },
            ..call
        },
        fields,
    ))
    .await;

    let sent = sent_betas(&only_request(&upstream).await);
    let mut expected: Vec<String> = features
        .iter()
        .map(|feature| feature.to_string())
        .chain(["caller-beta-2025-01-01".to_string()])
        .collect();
    expected.sort();
    assert_eq!(sent, expected);
}

#[rstest]
#[tokio::test]
async fn an_oauth_key_sends_the_browser_access_header_and_the_oauth_beta(call: MessagesCall) {
    let upstream = upstream([message_response()]).await;

    run_message(MessagesCall {
        api_key: Some("sk-ant-oat01-token".into()),
        api_base: Some(upstream.uri()),
        ..call
    })
    .await;

    let request = only_request(&upstream).await;
    assert_eq!(
        request.header("anthropic-dangerous-direct-browser-access"),
        Some("true")
    );
    assert_eq!(sent_betas(&request), [ANTHROPIC_OAUTH_BETA_HEADER]);
    assert_eq!(request.header("x-api-key"), None);
}

#[rstest]
#[case::anthropic("anthropic")]
#[case::azure_ai("azure_ai")]
#[tokio::test]
async fn caller_protocol_headers_win_over_the_defaults(call: MessagesCall, #[case] provider: &str) {
    let upstream = upstream([message_response()]).await;

    run_message(MessagesCall {
        custom_llm_provider: Some(provider.into()),
        api_key: Some("sk".into()),
        api_base: Some(upstream.uri()),
        extra_headers: headers([
            ("Anthropic-Version", "2024-01-01"),
            ("Content-Type", "application/json; charset=utf-8"),
        ]),
        ..call
    })
    .await;

    let request = only_request(&upstream).await;
    assert_eq!(request.header_values("anthropic-version"), ["2024-01-01"]);
    assert_eq!(
        request.header_values("content-type"),
        ["application/json; charset=utf-8"]
    );
}

fn sampling_removed() -> AnthropicModelCapabilities {
    AnthropicModelCapabilities {
        supports_sampling_params: false,
        ..AnthropicModelCapabilities::default()
    }
}

#[rstest]
#[case::sampling_params(sampling_removed(), json!({"temperature": 0.2, "top_p": 0.9, "top_k": 5}), &["temperature", "top_p", "top_k"], "temperature=0.2")]
#[case::speed(AnthropicModelCapabilities::default(), json!({"speed": "fast"}), &["speed"], "speed='fast'")]
#[tokio::test]
async fn unsupported_params_are_dropped_under_drop_params_and_rejected_without_it(
    call: MessagesCall,
    #[case] capabilities: AnthropicModelCapabilities,
    #[case] fields: Value,
    #[case] dropped: &[&str],
    #[case] rejected_as: &str,
) {
    let upstream = upstream([message_response(), message_response()]).await;
    let shaped = |drop_params: bool| {
        with_fields(
            MessagesCall {
                api_key: Some("sk".into()),
                api_base: Some(upstream.uri()),
                shaping: MessagesShaping {
                    capabilities: capabilities.clone(),
                    drop_params,
                    ..MessagesShaping::default()
                },
                body: call.body.clone(),
                custom_llm_provider: call.custom_llm_provider.clone(),
                extra_headers: None,
                provider_specific_header: None,
                model: call.model.clone(),
                timeout: call.timeout,
            },
            fields.clone(),
        )
    };

    let error = run(shaped(false))
        .await
        .err()
        .expect("an unsupported param is rejected without drop_params");
    assert!(
        matches!(&error, Error::InvalidRequest(message) if message.contains(rejected_as)),
        "{error:?}"
    );
    assert!(received(&upstream).await.is_empty());

    run_message(shaped(true)).await;
    let sent = only_request(&upstream).await.json();
    for name in dropped {
        assert_eq!(sent.get(*name), None, "{name} must be dropped");
    }
    assert_eq!(sent["max_tokens"], 16);
}

#[rstest]
#[case::adaptive_thinking(json!({"type": "adaptive"}), json!({"type": "adaptive", "display": "summarized"}))]
#[case::disabled_thinking(json!({"type": "disabled"}), json!({"type": "disabled"}))]
#[tokio::test]
async fn reasoning_auto_summary_marks_active_thinking_on_the_wire(
    call: MessagesCall,
    #[case] thinking: Value,
    #[case] expected: Value,
) {
    let upstream = upstream([message_response()]).await;

    run_message(with_fields(
        MessagesCall {
            api_key: Some("sk".into()),
            api_base: Some(upstream.uri()),
            shaping: MessagesShaping {
                capabilities: AnthropicModelCapabilities {
                    supports_reasoning: true,
                    supports_adaptive_thinking: true,
                    ..AnthropicModelCapabilities::default()
                },
                reasoning_auto_summary: true,
                ..MessagesShaping::default()
            },
            ..call
        },
        json!({"thinking": thinking}),
    ))
    .await;

    assert_eq!(only_request(&upstream).await.json()["thinking"], expected);
}

#[rstest]
#[case::reasoning_effort_on_an_adaptive_model(
    AnthropicModelCapabilities {
        supports_reasoning: true,
        supports_adaptive_thinking: true,
        supports_output_config: true,
        effort_tiers: SupportedEffortTiers { high: true, ..SupportedEffortTiers::default() },
        ..AnthropicModelCapabilities::default()
    },
    json!({"reasoning_effort": "high"}),
    json!({"thinking": {"type": "adaptive", "display": "summarized"}, "output_config": {"effort": "high"}})
)]
#[case::reasoning_effort_on_a_legacy_model_caps_the_budget_below_max_tokens(
    AnthropicModelCapabilities {
        supports_reasoning: true,
        ..AnthropicModelCapabilities::default()
    },
    json!({"reasoning_effort": "high"}),
    json!({"thinking": {"type": "enabled", "budget_tokens": 2999}})
)]
#[case::adaptive_payload_on_a_legacy_model_becomes_a_capped_budget(
    AnthropicModelCapabilities {
        supports_reasoning: true,
        ..AnthropicModelCapabilities::default()
    },
    json!({"thinking": {"type": "adaptive"}, "output_config": {"effort": "high"}, "temperature": 0}),
    json!({"thinking": {"type": "enabled", "budget_tokens": 2999}})
)]
#[case::adaptive_payload_on_a_model_without_reasoning_is_dropped(
    AnthropicModelCapabilities::default(),
    json!({"thinking": {"type": "adaptive"}, "output_config": {"effort": "high"}}),
    json!({})
)]
#[tokio::test]
async fn reasoning_is_translated_by_the_model_capabilities(
    call: MessagesCall,
    #[case] capabilities: AnthropicModelCapabilities,
    #[case] fields: Value,
    #[case] expected: Value,
) {
    let upstream = upstream([message_response()]).await;

    run_message(with_fields(
        MessagesCall {
            api_key: Some("sk".into()),
            api_base: Some(upstream.uri()),
            shaping: MessagesShaping {
                capabilities,
                ..MessagesShaping::default()
            },
            ..call
        },
        [("max_tokens".to_string(), json!(3000))]
            .into_iter()
            .chain(object(fields))
            .collect(),
    ))
    .await;

    let sent = only_request(&upstream).await.json();
    assert_eq!(sent.get("reasoning_effort"), None);
    assert_eq!(sent.get("temperature"), None);
    let reasoning: Map<String, Value> = ["thinking", "output_config"]
        .into_iter()
        .filter_map(|name| Some((name.to_string(), sent.get(name)?.clone())))
        .collect();
    assert_eq!(Value::Object(reasoning), expected);
}

#[rstest]
#[case::empty_text_blocks(
    json!([{"role": "assistant", "content": [{"type": "text", "text": "  "}, {"type": "text", "text": "kept"}]}]),
    json!([{"role": "assistant", "content": [{"type": "text", "text": "kept"}]}])
)]
#[case::provider_specific_fields(
    json!([{"role": "assistant", "content": [{"type": "text", "text": "kept", "provider_specific_fields": {"x": 1}}]}]),
    json!([{"role": "assistant", "content": [{"type": "text", "text": "kept"}]}])
)]
#[case::unencrypted_web_search_results_become_text(
    json!([{"role": "assistant", "content": [{
        "type": "web_search_tool_result",
        "tool_use_id": "srvtoolu_1",
        "content": [{"type": "web_search_result", "title": "T", "url": "https://e.x", "page_age": null}]
    }]}]),
    json!([{"role": "assistant", "content": [{"type": "text", "text": "Web search results:\n\nTitle: T\nURL: https://e.x"}]}])
)]
#[tokio::test]
async fn replayed_history_is_cleaned_before_sending(
    call: MessagesCall,
    #[case] history: Value,
    #[case] expected: Value,
) {
    let upstream = upstream([message_response()]).await;

    run_message(with_fields(
        MessagesCall {
            api_key: Some("sk".into()),
            api_base: Some(upstream.uri()),
            ..call
        },
        json!({"messages": history}),
    ))
    .await;

    assert_eq!(only_request(&upstream).await.json()["messages"], expected);
}

#[rstest]
#[tokio::test]
async fn metadata_is_reduced_to_the_user_id(call: MessagesCall) {
    let upstream = upstream([message_response()]).await;

    run_message(with_fields(
        MessagesCall {
            api_key: Some("sk".into()),
            api_base: Some(upstream.uri()),
            ..call
        },
        json!({"metadata": {"user_id": "u-1", "trace_id": "internal", "tags": ["a"]}}),
    ))
    .await;

    assert_eq!(
        only_request(&upstream).await.json()["metadata"],
        json!({"user_id": "u-1"})
    );
}

#[rstest]
#[case::numeric_user_id(json!({"metadata": {"user_id": 7}}))]
#[case::missing_max_tokens(json!({"max_tokens": null}))]
#[tokio::test]
async fn an_invalid_request_fails_before_sending(call: MessagesCall, #[case] fields: Value) {
    let upstream = upstream([message_response()]).await;

    let error = run(with_fields(
        MessagesCall {
            api_key: Some("sk".into()),
            api_base: Some(upstream.uri()),
            ..call
        },
        fields,
    ))
    .await
    .err()
    .expect("the request is rejected");

    assert!(error.is_request(), "{error:?}");
    assert!(received(&upstream).await.is_empty());
}

#[rstest]
#[tokio::test]
async fn azure_folds_system_role_messages_into_the_system_prompt(call: MessagesCall) {
    let upstream = upstream([message_response()]).await;

    run_message(with_fields(
        MessagesCall {
            custom_llm_provider: Some("azure_ai".into()),
            api_key: Some("sk-azure".into()),
            api_base: Some(upstream.uri()),
            ..call
        },
        json!({
            "system": "top level",
            "messages": [
                {"role": "system", "content": "from a message"},
                {"role": "user", "content": "hi"}
            ]
        }),
    ))
    .await;

    let sent = only_request(&upstream).await.json();
    assert_eq!(
        sent["system"],
        json!([
            {"type": "text", "text": "top level"},
            {"type": "text", "text": "from a message"}
        ])
    );
    assert_eq!(sent["messages"], json!([{"role": "user", "content": "hi"}]));
}

#[rstest]
#[case::bare_model(MODEL, MODEL)]
#[case::one_prefix("anthropic/claude-sonnet-4-5", MODEL)]
#[case::doubled_prefix_loses_one_segment(
    "anthropic/anthropic/claude-sonnet-4-5",
    "anthropic/claude-sonnet-4-5"
)]
#[tokio::test]
async fn the_provider_prefix_is_stripped_exactly_once(
    call: MessagesCall,
    #[case] model: &str,
    #[case] sent_model: &str,
) {
    let upstream = upstream([message_response()]).await;

    run_message(MessagesCall {
        model: model.into(),
        api_key: Some("sk".into()),
        api_base: Some(upstream.uri()),
        ..call
    })
    .await;

    assert_eq!(only_request(&upstream).await.json()["model"], sent_model);
}
