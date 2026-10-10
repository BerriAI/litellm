use litellm_auth::VertexParams;
use litellm_llms::base_llm::messages::context::{MessagesModelCapabilities, SupportedEffortTiers};
use litellm_llms_types::{
    headers::{ProviderSpecificHeader, ProviderSpecificHeaders},
    providers::anthropic::{AnthropicBeta, BetaSet},
};
use litellm_router_types::LitellmParams;
use rstest::rstest;

use super::*;

#[rstest]
#[case::anthropic_key("anthropic", Some("sk-ant"), &[], ("x-api-key", "sk-ant"), &["authorization"])]
#[case::azure_key("azure_ai", Some("sk-azure"), &[], ("x-api-key", "sk-azure"), &["authorization"])]
#[case::deepseek_key("deepseek", Some("sk-deepseek"), &[], ("x-api-key", "sk-deepseek"), &["authorization"])]
#[case::deepseek_forwards_caller_authorization(
    "deepseek",
    Some("sk-deepseek"),
    &[("Authorization", "Bearer caller")],
    ("authorization", "Bearer caller"),
    &["x-api-key"]
)]
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
#[case::openrouter_key_becomes_bearer(
    "openrouter",
    Some("sk-or"),
    &[],
    ("authorization", "Bearer sk-or"),
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
        litellm_params: Default::default(),
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
#[case::deepseek("deepseek")]
#[case::edenai("edenai")]
#[case::openrouter("openrouter")]
#[tokio::test]
async fn a_call_without_credentials_fails_before_sending(
    call: MessagesCall,
    #[case] provider: &str,
) {
    let upstream = upstream([message_response()]).await;

    let error = run(MessagesCall {
        custom_llm_provider: Some(provider.into()),
        litellm_params: Default::default(),
        api_base: Some(upstream.uri()),
        ..call
    })
    .await
    .expect_err("a call without credentials fails");

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
#[case::deepseek(MODEL, Some("deepseek"), "", "/anthropic/v1/messages")]
#[case::deepseek_openai_compatible_base(MODEL, Some("deepseek"), "/beta", "/anthropic/v1/messages")]
#[case::edenai(MODEL, Some("edenai"), "/v3", "/v3/v1/messages")]
#[case::openrouter(MODEL, Some("openrouter"), "/api/v1", "/api/v1/messages")]
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
        custom_llm_provider: provider.map(Into::into),
        litellm_params: Default::default(),
        api_key: Some("sk".into()),
        api_base: Some(format!("{}{base_suffix}", upstream.uri())),
        ..with_model(call, model)
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
        custom_llm_provider: provider.map(Into::into),
        litellm_params: Default::default(),
        api_key: Some("sk".into()),
        api_base: Some(UNREACHABLE_BASE.into()),
        ..with_model(call, model)
    })
    .await
    .expect_err("unsupported provider errors");

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
#[case::azure("azure_ai", json!({"type": "ephemeral", "ttl": "1h", "future": "kept"}))]
#[case::anthropic("anthropic", json!({"type": "ephemeral", "ttl": "1h", "scope": "global", "future": "kept"}))]
#[tokio::test]
async fn cache_scope_removal_is_selected_by_the_provider(
    call: MessagesCall,
    #[case] provider: &str,
    #[case] expected: Value,
) {
    let upstream = upstream([message_response()]).await;

    run_message(MessagesCall {
        custom_llm_provider: Some(provider.into()),
        litellm_params: Default::default(),
        api_key: Some("sk-azure".into()),
        api_base: Some(upstream.uri()),
        body: body(json!({
            "model": MODEL,
            "max_tokens": 16,
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": "hi",
                    "cache_control": {"type": "ephemeral", "ttl": "1h", "scope": "global", "future": "kept"}
                }]
            }]
        })),
        ..call
    })
    .await;

    assert_eq!(
        only_request(&upstream).await.json()["messages"][0]["content"][0]["cache_control"],
        expected
    );
}

#[rstest]
#[tokio::test]
async fn additional_drop_params_remove_fields_before_sending(call: MessagesCall) {
    let upstream = upstream([message_response()]).await;

    run_message(MessagesCall {
        api_key: Some("sk".into()),
        api_base: Some(upstream.uri()),
        shaping: MessagesShaping {
            settings: MessagesSettings {
                additional_drop_params: vec!["temperature".into()],
                ..MessagesSettings::default()
            },
            ..MessagesShaping::default()
        },
        ..with_fields(call, json!({"temperature": 0.5, "top_k": 3}))
    })
    .await;

    let sent = only_request(&upstream).await.json();
    assert_eq!(sent.get("temperature"), None);
    assert_eq!(sent["top_k"], 3);
}

fn sent_betas(request: &wiremock::Request) -> BetaSet {
    let [header] = <[&str; 1]>::try_from(request.header_values("anthropic-beta"))
        .unwrap_or_else(|values| panic!("expected one anthropic-beta header, got {values:?}"));
    header.parse().unwrap()
}

#[rstest]
#[case::structured_output(json!({"output_format": {"type": "json_schema"}}), &[AnthropicBeta::StructuredOutputs20251113])]
#[case::fast_mode(json!({"speed": "fast"}), &[AnthropicBeta::FastMode20260201])]
#[case::compaction(json!({"compaction": {"enabled": true}}), &[AnthropicBeta::Compact20260904])]
#[case::context_management_edits(
    json!({"context_management": {"edits": [{"type": "clear_tool_uses_20250919"}]}}),
    &[AnthropicBeta::ContextManagement20250627]
)]
#[case::per_message_output_config(
    json!({"messages": [{"role": "user", "content": "hi", "output_config": {"effort": "low"}}]}),
    &[AnthropicBeta::PerTurnControl20260701]
)]
#[case::advisor_tool(
    json!({"tools": [{"type": "advisor_20260301", "name": "advisor", "model": MODEL}]}),
    &[AnthropicBeta::AdvisorTool20260301]
)]
#[case::several_features_at_once(
    json!({"speed": "fast", "output_format": {"type": "json_schema"}}),
    &[AnthropicBeta::StructuredOutputs20251113, AnthropicBeta::FastMode20260201]
)]
#[tokio::test]
async fn feature_betas_join_the_callers_betas_in_one_sorted_header(
    call: MessagesCall,
    #[case] fields: Value,
    #[case] features: &[AnthropicBeta],
) {
    let upstream = upstream([message_response()]).await;
    let capabilities = MessagesModelCapabilities {
        supports_speed: true,
        ..MessagesModelCapabilities::default()
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
    let expected: BetaSet = features
        .iter()
        .cloned()
        .chain([AnthropicBeta::Other("caller-beta-2025-01-01".to_string())])
        .collect();
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
    assert_eq!(
        sent_betas(&request),
        BetaSet::from_iter([AnthropicBeta::Oauth20250420])
    );
    assert_eq!(request.header("x-api-key"), None);
}

#[rstest]
#[case::anthropic("anthropic")]
#[case::azure_ai("azure_ai")]
#[case::deepseek("deepseek")]
#[tokio::test]
async fn caller_protocol_headers_win_over_the_defaults(call: MessagesCall, #[case] provider: &str) {
    let upstream = upstream([message_response()]).await;

    run_message(MessagesCall {
        custom_llm_provider: Some(provider.into()),
        litellm_params: Default::default(),
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

fn sampling_removed() -> MessagesModelCapabilities {
    MessagesModelCapabilities {
        supports_sampling_params: false,
        ..MessagesModelCapabilities::default()
    }
}

#[rstest]
#[case::sampling_params(sampling_removed(), json!({"temperature": 0.2, "top_p": 0.9, "top_k": 5}), &["temperature", "top_p", "top_k"], "temperature=0.2")]
#[case::speed(MessagesModelCapabilities::default(), json!({"speed": "fast"}), &["speed"], "speed='fast'")]
#[tokio::test]
async fn unsupported_params_are_dropped_under_drop_params_and_rejected_without_it(
    call: MessagesCall,
    #[case] capabilities: MessagesModelCapabilities,
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
                    capabilities,
                    settings: MessagesSettings {
                        drop_params,
                        ..MessagesSettings::default()
                    },
                },
                body: call.body.clone(),
                custom_llm_provider: call.custom_llm_provider.clone(),
                litellm_params: Default::default(),
                extra_headers: None,
                provider_specific_header: None,
                timeout: call.timeout,
            },
            fields.clone(),
        )
    };

    let error = run(shaped(false))
        .await
        .expect_err("an unsupported param is rejected without drop_params");
    assert!(
        matches!(&error, Error::InvalidRequest(message) if message.to_string().contains(rejected_as)),
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
                settings: MessagesSettings {
                    reasoning_auto_summary: true,
                    ..MessagesSettings::default()
                },
                capabilities: MessagesModelCapabilities {
                    supports_reasoning: true,
                    supports_adaptive_thinking: true,
                    ..MessagesModelCapabilities::default()
                },
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
    MessagesModelCapabilities {
        supports_reasoning: true,
        supports_adaptive_thinking: true,
        supports_output_config: true,
        effort_tiers: SupportedEffortTiers { high: true, ..SupportedEffortTiers::default() },
        ..MessagesModelCapabilities::default()
    },
    json!({"reasoning_effort": "high"}),
    json!({"thinking": {"type": "adaptive", "display": "summarized"}, "output_config": {"effort": "high"}})
)]
#[case::reasoning_effort_on_a_legacy_model_caps_the_budget_below_max_tokens(
    MessagesModelCapabilities {
        supports_reasoning: true,
        ..MessagesModelCapabilities::default()
    },
    json!({"reasoning_effort": "high"}),
    json!({"thinking": {"type": "enabled", "budget_tokens": 2999}})
)]
#[case::adaptive_payload_on_a_legacy_model_becomes_a_capped_budget(
    MessagesModelCapabilities {
        supports_reasoning: true,
        ..MessagesModelCapabilities::default()
    },
    json!({"thinking": {"type": "adaptive"}, "output_config": {"effort": "high"}, "temperature": 0}),
    json!({"thinking": {"type": "enabled", "budget_tokens": 2999}})
)]
#[case::adaptive_payload_on_a_model_without_reasoning_is_dropped(
    MessagesModelCapabilities::default(),
    json!({"thinking": {"type": "adaptive"}, "output_config": {"effort": "high"}}),
    json!({})
)]
#[tokio::test]
async fn reasoning_is_translated_by_the_model_capabilities(
    call: MessagesCall,
    #[case] capabilities: MessagesModelCapabilities,
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
#[case::anthropic("anthropic")]
#[case::azure("azure_ai")]
#[case::deepseek("deepseek")]
#[tokio::test]
async fn metadata_is_reduced_to_the_user_id(call: MessagesCall, #[case] provider: &str) {
    let upstream = upstream([message_response()]).await;

    run_message(with_fields(
        MessagesCall {
            custom_llm_provider: Some(provider.into()),
            litellm_params: Default::default(),
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
#[tokio::test]
async fn deepseek_sends_neither_billing_blocks_nor_the_custom_tool_discriminator(
    call: MessagesCall,
) {
    let upstream = upstream([message_response()]).await;

    run_message(with_fields(
        MessagesCall {
            custom_llm_provider: Some("deepseek".into()),
            litellm_params: Default::default(),
            api_key: Some("sk".into()),
            api_base: Some(upstream.uri()),
            ..call
        },
        json!({
            "system": [
                {"type": "text", "text": "x-anthropic-billing-header: cc_version=1"},
                {"type": "text", "text": "be terse"}
            ],
            "tools": [{"type": "custom", "name": "get_weather", "input_schema": {"type": "object"}}]
        }),
    ))
    .await;

    let body = only_request(&upstream).await.json();
    assert_eq!(
        body["system"],
        json!([{"type": "text", "text": "be terse"}])
    );
    assert_eq!(
        body["tools"],
        json!([{"name": "get_weather", "input_schema": {"type": "object"}}])
    );
}

#[rstest]
#[case::not_streaming(false, ":rawPredict")]
#[case::streaming(true, ":streamRawPredict?alt=sse")]
#[tokio::test]
async fn vertex_ai_addresses_the_model_in_the_url_and_not_in_the_body(
    call: MessagesCall,
    #[case] stream: bool,
    #[case] suffix: &str,
) {
    let streamed = ResponseTemplate::new(200).set_body_raw(
        "event: message_start\ndata: {\"type\":\"message_start\"}\n\nevent: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
        "text/event-stream",
    );
    let upstream = upstream([if stream { streamed } else { message_response() }]).await;

    run(with_fields(
        MessagesCall {
            custom_llm_provider: Some("vertex_ai".into()),
            litellm_params: LitellmParams {
                vertex: VertexParams {
                    vertex_project: Some("proj".into()),
                    vertex_location: Some("us-east5".into()),
                    ..VertexParams::default()
                },
                ..LitellmParams::default()
            },
            api_base: Some(upstream.uri()),
            extra_headers: headers([("Authorization", "Bearer caller-token")]),
            ..with_model(call, "claude-sonnet-4-5@20250929")
        },
        json!({"stream": stream}),
    ))
    .await
    .expect("messages call succeeds");

    let request = only_request(&upstream).await;
    assert_eq!(
        request.url.path().to_string() + request.url.query().map_or("", |_| "?alt=sse"),
        format!(
            "/v1/projects/proj/locations/us-east5/publishers/anthropic/models/claude-sonnet-4-5@20250929{suffix}"
        )
    );
    assert_eq!(request.header("authorization"), Some("Bearer caller-token"));
    assert_eq!(request.header("anthropic-version"), None);
    let body = request.json();
    assert_eq!(body.get("model"), None);
    assert_eq!(body["anthropic_version"], json!("vertex-2023-10-16"));
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
    .expect_err("the request is rejected");

    assert!(error.is_request(), "{error:?}");
    assert!(received(&upstream).await.is_empty());
}

#[rstest]
#[case::azure("azure_ai", &[], json!([
    {"type": "text", "text": "top level"},
    {"type": "text", "text": "from a message"}
]), json!([{"role": "user", "content": "hi"}]))]
#[case::anthropic("anthropic", &[], json!("top level"), json!([
    {"role": "system", "content": "from a message"},
    {"role": "user", "content": "hi"}
]))]
#[case::azure_folds_after_caller_drops("azure_ai", &["system"], json!([
    {"type": "text", "text": "from a message"}
]), json!([{"role": "user", "content": "hi"}]))]
#[tokio::test]
async fn system_message_folding_is_selected_by_the_provider(
    call: MessagesCall,
    #[case] provider: &str,
    #[case] drop_params: &[&str],
    #[case] expected_system: Value,
    #[case] expected_messages: Value,
) {
    let upstream = upstream([message_response()]).await;

    run_message(with_fields(
        MessagesCall {
            custom_llm_provider: Some(provider.into()),
            litellm_params: Default::default(),
            api_key: Some("sk-azure".into()),
            api_base: Some(upstream.uri()),
            shaping: MessagesShaping {
                settings: MessagesSettings {
                    additional_drop_params: drop_params.iter().map(ToString::to_string).collect(),
                    ..call.shaping.settings
                },
                ..call.shaping
            },
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
    assert_eq!(sent["system"], expected_system);
    assert_eq!(sent["messages"], expected_messages);
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
        api_key: Some("sk".into()),
        api_base: Some(upstream.uri()),
        ..with_model(call, model)
    })
    .await;

    assert_eq!(only_request(&upstream).await.json()["model"], sent_model);
}

#[rstest]
#[case::anthropic("anthropic")]
#[case::azure("azure_ai")]
#[case::bedrock("bedrock")]
#[tokio::test]
async fn provider_validation_runs_before_caller_parameter_removal(
    call: MessagesCall,
    #[case] provider: &str,
) {
    let upstream = upstream([message_response()]).await;
    let result = run(with_fields(
        MessagesCall {
            custom_llm_provider: Some(provider.into()),
            litellm_params: Default::default(),
            api_key: Some("sk-test".into()),
            api_base: Some(upstream.uri()),
            shaping: MessagesShaping {
                settings: MessagesSettings {
                    additional_drop_params: vec!["metadata".into()],
                    ..call.shaping.settings
                },
                ..call.shaping
            },
            ..call
        },
        json!({"metadata": {"user_id": 7}}),
    ))
    .await;
    let error = result.expect_err("metadata is validated before removal");
    assert!(
        error
            .to_string()
            .contains("metadata.user_id must be a string"),
        "{error}"
    );
    assert!(received(&upstream).await.is_empty());
}

#[rstest]
#[tokio::test]
async fn edenai_preserves_native_thinking_and_strips_billing_and_cache_extensions(
    call: MessagesCall,
) {
    let upstream = upstream([message_response()]).await;
    let thinking = json!({"type": "enabled", "budget_tokens": 512, "native_field": true});
    run_message(MessagesCall {
        custom_llm_provider: Some("edenai".into()),
        api_base: Some(format!("{}/v3", upstream.uri())),
        api_key: Some("eden-key".into()),
        extra_headers: headers([("Anthropic-Beta", "unregistered-native-beta")]),
        ..with_fields(call, json!({
            "thinking": thinking, "temperature": 0.25,
            "system": [{"type": "text", "text": "x-anthropic-billing-header: hidden"},
                       {"type": "text", "text": "keep", "cache_control": {"type": "ephemeral", "ttl": "1h", "scope": "global"}}],
        }))
    }).await;
    let sent = only_request(&upstream).await;
    assert_eq!(sent.url.path(), "/v3/v1/messages");
    assert_eq!(sent.header("authorization"), Some("Bearer eden-key"));
    assert_eq!(
        sent.header("anthropic-beta"),
        Some("unregistered-native-beta")
    );
    assert_eq!(
        sent.json(),
        json!({
            "model": MODEL, "max_tokens": 16, "messages": [{"role": "user", "content": "hi"}],
            "thinking": thinking, "temperature": 0.25,
            "system": [{"type": "text", "text": "keep", "cache_control": {"type": "ephemeral"}}],
        })
    );
}

struct CopilotToken {
    acquisitions: std::sync::atomic::AtomicUsize,
    api_base: &'static str,
}

impl litellm_auth_copilot::CopilotSessionSource for CopilotToken {
    fn acquire(&self, _: u64) -> litellm_auth_copilot::SessionFuture<'_> {
        self.acquisitions
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Box::pin(async move {
            litellm_auth_copilot::CopilotSession::new(
                litellm_auth::SecretValue::new("session-token"),
                Some(self.api_base),
                200,
            )
        })
    }
}

struct CopilotWire {
    upstream: String,
    seen: Mutex<Vec<litellm_host::interceptors::WireRequest>>,
}

impl litellm_host::interceptors::Interceptors<Error> for CopilotWire {
    async fn before_provider_request(
        &self,
        wire: litellm_host::interceptors::WireRequest,
        _: litellm_host::interceptors::RequestContext,
    ) -> Result<litellm_host::interceptors::WireRequest, Error> {
        self.seen.lock().unwrap().push(wire.clone());
        Ok(litellm_host::interceptors::WireRequest {
            url: format!("{}/v1/messages", self.upstream),
            ..wire
        })
    }

    async fn after_provider_response(
        &self,
        _: litellm_host::interceptors::RawResponse,
    ) -> Result<(), Error> {
        Ok(())
    }
}

#[rstest]
#[case::complete(false, "https://tenant.githubcopilot.com")]
#[case::complete_endpoint(false, "https://tenant.githubcopilot.com/v1/messages")]
#[case::stream(true, "https://tenant.githubcopilot.com")]
#[tokio::test]
async fn copilot_binds_session_auth_to_its_endpoint_and_preserves_native_messages(
    call: MessagesCall,
    #[case] stream: bool,
    #[case] api_base: &'static str,
) {
    use futures_util::TryStreamExt;
    use litellm_inference_messages::MessagesCallResponse;

    const SSE: &str = "event: message_start\ndata: {\"type\":\"message_start\"}\n\nevent: message_stop\ndata: {\"type\":\"message_stop\"}\n\n";
    let template = if stream {
        ResponseTemplate::new(200)
            .insert_header("content-type", "text/event-stream")
            .set_body_string(SSE)
    } else {
        message_response()
    };
    let attacker = upstream([]).await;
    let upstream = upstream([template.clone(), template]).await;
    let tokens = Arc::new(CopilotToken {
        acquisitions: std::sync::atomic::AtomicUsize::new(0),
        api_base,
    });
    let route = copilot_route(tokens.clone());
    let intercept = CopilotWire {
        upstream: upstream.uri(),
        seen: Mutex::new(Vec::new()),
    };
    let request_body = with_fields(call, json!({"model": "claude-test", "stream": stream})).body;
    let request = || MessagesCall {
        custom_llm_provider: Some("github_copilot".into()),
        api_key: Some("caller-key".into()),
        api_base: Some(attacker.uri()),
        extra_headers: headers([
            ("Authorization", "Bearer caller-one"),
            ("authorization", "Bearer caller-two"),
            ("openai-intent", "caller-intent"),
            ("x-interaction-type", "caller-type"),
            ("x-github-api-version", "caller-version"),
            ("editor-version", "caller-editor"),
            ("anthropic-beta", "caller-beta"),
        ]),
        body: request_body.clone(),
        litellm_params: Default::default(),
        provider_specific_header: None,
        timeout: Some(Duration::from_secs(5)),
        shaping: Default::default(),
    };
    let first = route.execute(request(), &intercept, None).await.unwrap();
    let second = route.execute(request(), &intercept, None).await.unwrap();
    match (first, second) {
        (MessagesCallResponse::Complete(first), MessagesCallResponse::Complete(second)) => {
            assert_eq!(
                *first,
                serde_json::from_value::<MessagesResponse>(message_body()).unwrap()
            );
            assert_eq!(first, second);
        }
        (
            MessagesCallResponse::Stream { chunks: first, .. },
            MessagesCallResponse::Stream { chunks: second, .. },
        ) => {
            assert_eq!(
                first.try_collect::<Vec<_>>().await.unwrap().concat(),
                SSE.as_bytes()
            );
            assert_eq!(
                second.try_collect::<Vec<_>>().await.unwrap().concat(),
                SSE.as_bytes()
            );
        }
        _ => panic!("the requested delivery mode must survive authentication"),
    }
    assert!(received(&attacker).await.is_empty());
    let sent = received(&upstream).await;
    assert_eq!(sent.len(), 2);
    assert_eq!(
        sent[0].header_values("authorization"),
        ["Bearer session-token"]
    );
    assert_eq!(sent[0].header("editor-version"), Some("caller-editor"));
    assert_eq!(sent[0].header("openai-intent"), Some("messages-proxy"));
    assert_eq!(sent[0].header("x-interaction-type"), Some("messages-proxy"));
    assert_ne!(
        sent[0].header("x-github-api-version"),
        Some("caller-version")
    );
    assert_eq!(sent[0].header("anthropic-beta"), Some("caller-beta"));
    assert_eq!(
        sent[0].json(),
        json!({
            "model": "claude-test", "max_tokens": 16, "stream": stream,
            "messages": [{"role": "user", "content": "hi"}]
        })
    );
    let first_id = sent[0].header("x-request-id").expect("request identity");
    let second_id = sent[1].header("x-request-id").expect("request identity");
    assert!(!first_id.is_empty());
    assert_ne!(first_id, second_id);
    assert_eq!(
        tokens
            .acquisitions
            .load(std::sync::atomic::Ordering::SeqCst),
        1
    );
    let seen = intercept.seen.into_inner().unwrap();
    assert_eq!(seen.len(), 2);
    assert!(
        seen.iter()
            .all(|wire| wire.url == "https://tenant.githubcopilot.com/v1/messages")
    );
}

fn copilot_route(
    source: Arc<dyn litellm_auth_copilot::CopilotSessionSource>,
) -> litellm_inference_messages::MessagesRoute {
    let resources = litellm_inference_testing::resources();
    litellm_inference_messages::MessagesRoute::new(
        litellm_inference_testing::provider_http(
            &resources,
            &litellm_inference_testing::http_config(),
        ),
        Arc::new(litellm_auth::AuthServices {
            copilot: litellm_auth_copilot::CopilotAuthService::new(source, Arc::new(|| 100)),
            ..Default::default()
        }),
        litellm_inference_testing::no_secrets(),
    )
}

struct CopilotLoginRequired;

impl litellm_auth_copilot::CopilotSessionSource for CopilotLoginRequired {
    fn acquire(&self, _: u64) -> litellm_auth_copilot::SessionFuture<'_> {
        Box::pin(async {
            Err(litellm_auth::Error::ProviderAuthentication(
                "login required".into(),
            ))
        })
    }
}

#[rstest]
#[tokio::test]
async fn caller_credentials_cannot_bypass_a_failed_copilot_session(call: MessagesCall) {
    let upstream = upstream([message_response()]).await;
    let error = copilot_route(Arc::new(CopilotLoginRequired))
        .execute(
            MessagesCall {
                custom_llm_provider: Some("github_copilot".into()),
                api_key: Some("caller-key".into()),
                api_base: Some(upstream.uri()),
                extra_headers: headers([("Authorization", "Bearer caller")]),
                ..call
            },
            &(),
            None,
        )
        .await
        .err()
        .expect("missing session cannot send inference");
    assert_eq!(
        error,
        Error::Auth(litellm_auth::Error::ProviderAuthentication(
            "login required".into()
        ))
    );
    assert!(received(&upstream).await.is_empty());
}

#[rstest]
#[tokio::test]
async fn unprojected_per_user_sessions_cannot_fall_back_to_shared_credentials(call: MessagesCall) {
    let upstream = upstream([message_response()]).await;
    let tokens = Arc::new(CopilotToken {
        acquisitions: std::sync::atomic::AtomicUsize::new(0),
        api_base: "https://tenant.githubcopilot.com",
    });
    let intercept = CopilotWire {
        upstream: upstream.uri(),
        seen: Mutex::new(Vec::new()),
    };
    let error = copilot_route(tokens.clone())
        .execute(MessagesCall {
            custom_llm_provider: Some("github_copilot".into()),
            litellm_params: serde_json::from_value(json!({
                "model": "claude-test",
                "github_copilot_user_session": {"token": "caller-session", "api_base": upstream.uri()}
            })).unwrap(),
            ..call
        }, &intercept, None).await.err().expect("a session must be projected through a trusted boundary");
    assert_eq!(
        error,
        Error::Unsupported("native Copilot per-user session projection")
    );
    assert_eq!(
        tokens
            .acquisitions
            .load(std::sync::atomic::Ordering::SeqCst),
        0
    );
    assert!(received(&upstream).await.is_empty());
}
