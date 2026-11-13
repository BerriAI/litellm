use litellm_auth::VertexParams;
use litellm_llms::base_llm::messages::context::{MessagesModelCapabilities, SupportedEffortTiers};
use litellm_llms_types::{
    headers::{ProviderSpecificHeader, ProviderSpecificHeaders},
    providers::anthropic::{AnthropicBeta, BetaSet},
};
use litellm_router_types::LitellmParams;
use litellm_secret_value::SecretValue;
use rstest::rstest;

use super::*;

#[rstest]
#[case::anthropic_key("anthropic", Some("sk-ant"), &[], ("x-api-key", "sk-ant"), &["authorization"])]
#[case::minimax_key("minimax", Some("sk-test"), &[], ("x-api-key", "sk-test"), &["authorization"])]
#[case::minimax_forwarded_key("minimax", Some("deployment"), &[("X-Api-Key", "caller")], ("x-api-key", "caller"), &["authorization"])]
#[case::minimax_forwarded_bearer("minimax", None, &[("Authorization", "Bearer caller")], ("authorization", "Bearer caller"), &["x-api-key"])]
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
#[case::caller_x_api_key_without_key(
    "azure_ai",
    None,
    &[("X-Api-Key", "caller")],
    ("x-api-key", "caller"),
    &["authorization"]
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
#[case::a06_caller_x_api_key_beside_the_operator_api_key(
    Some("sk-operator"),
    &[("x-api-key", "caller")],
    json!({}),
    None,
    &[],
    &[("x-api-key", "caller"), ("api-key", "sk-operator")],
)]
#[case::a07_other_casing_x_api_key_joined_with_the_operator_key(
    Some("sk-operator"),
    &[("X-Api-Key", "caller")],
    json!({}),
    None,
    &[],
    &[("x-api-key", "caller, sk-operator")],
)]
#[case::a08_caller_bearer_beside_the_operator_key(
    Some("sk-operator"),
    &[("Authorization", "Bearer caller")],
    json!({}),
    None,
    &[],
    &[("x-api-key", "sk-operator"), ("authorization", "Bearer caller")],
)]
#[case::a09_exact_caller_api_key_replaces_the_operator_key(
    Some("sk-operator"),
    &[("api-key", "caller")],
    json!({}),
    None,
    &[],
    &[("x-api-key", "caller")],
)]
#[case::a09b_title_case_api_key_beside_the_operator_key(
    Some("sk-operator"),
    &[("Api-Key", "caller")],
    json!({}),
    None,
    &[],
    &[("x-api-key", "sk-operator"), ("api-key", "caller")],
)]
#[case::module_key_after_an_empty_call_key(Some(""), &[], json!({}), Some("sk-module"), &[], &[("x-api-key", "sk-module")])]
#[case::call_key_over_the_module_key(Some("sk-call"), &[], json!({}), Some("sk-module"), &[], &[("x-api-key", "sk-call")])]
#[case::azure_openai_key_before_azure_key(
    None,
    &[],
    json!({}),
    None,
    &[("AZURE_OPENAI_API_KEY", "sk-openai"), ("AZURE_API_KEY", "sk-azure")],
    &[("x-api-key", "sk-openai")],
)]
#[case::blank_call_key_is_sent_instead_of_the_environment_key(
    Some(" "),
    &[],
    json!({}),
    None,
    &[("AZURE_API_KEY", "sk-azure")],
    &[("x-api-key", "")],
)]
#[case::entra_token_without_a_key(
    None,
    &[],
    json!({"azure_ad_token": "entra-token"}),
    None,
    &[],
    &[("authorization", "Bearer entra-token")],
)]
#[case::entra_token_from_the_secret_source(
    None,
    &[],
    json!({}),
    None,
    &[("AZURE_AD_TOKEN", "secret-token")],
    &[("authorization", "Bearer secret-token")],
)]
#[case::client_secret_settings_only_in_the_secret_source_are_not_used(
    None,
    &[],
    json!({"azure_ad_token": "entra-token"}),
    None,
    &[("AZURE_TENANT_ID", "t"), ("AZURE_CLIENT_ID", "c"), ("AZURE_CLIENT_SECRET", "s")],
    &[("authorization", "Bearer entra-token")],
)]
#[case::username_and_password_only_in_the_secret_source_are_not_used(
    None,
    &[],
    json!({"client_id": "c"}),
    None,
    &[("AZURE_USERNAME", "u"), ("AZURE_PASSWORD", "p")],
    &[],
)]
#[case::a10_entra_token_overwrites_the_caller_bearer(
    None,
    &[("Authorization", "Bearer caller")],
    json!({"azure_ad_token": "entra-token"}),
    None,
    &[],
    &[("authorization", "Bearer entra-token")],
)]
#[case::a10b_lowercase_caller_authorization_joined_with_entra(
    None,
    &[("authorization", "Bearer caller")],
    json!({"azure_ad_token": "entra-token"}),
    None,
    &[],
    &[("authorization", "Bearer caller, Bearer entra-token")],
)]
#[case::a10c_caller_x_api_key_beside_entra(
    None,
    &[("x-api-key", "caller")],
    json!({"azure_ad_token": "entra-token"}),
    None,
    &[],
    &[("x-api-key", "caller"), ("authorization", "Bearer entra-token")],
)]
#[case::a09b_upper_case_api_key_beside_the_operator_key(
    Some("sk-operator"),
    &[("API-Key", "caller")],
    json!({}),
    None,
    &[],
    &[("x-api-key", "sk-operator"), ("api-key", "caller")],
)]
#[case::caller_basic_authorization_beside_the_operator_key(
    Some("sk-operator"),
    &[("Authorization", "Basic caller")],
    json!({}),
    None,
    &[],
    &[("x-api-key", "sk-operator"), ("authorization", "Basic caller")],
)]
#[case::a_key_outranks_an_entra_token(
    Some("sk-call"),
    &[],
    json!({"azure_ad_token": "entra-token"}),
    None,
    &[],
    &[("x-api-key", "sk-call")],
)]
#[case::no_credential_is_sent_unauthenticated(None, &[], json!({}), None, &[], &[])]
#[tokio::test]
async fn azure_credentials_resolve_in_pythons_order(
    call: MessagesCall,
    #[case] api_key: Option<&str>,
    #[case] extra_headers: &[(&str, &str)],
    #[case] params: Value,
    #[case] module_api_key: Option<&str>,
    #[case] secrets: &[(&str, &str)],
    #[case] expected: &[(&str, &str)],
) {
    let _serialized = PROCESS_ENVIRONMENT.lock().await;
    let upstream = upstream([message_response()]).await;
    let Value::Object(params) = params else {
        panic!("case params are an object")
    };

    run_hosted_with(
        Arc::new(RecordingSecrets::new(secrets.iter().copied())),
        MessagesCall {
            custom_llm_provider: Some("azure_ai".into()),
            litellm_params: litellm_inference_messages::litellm_params(
                [("model".to_string(), json!(MODEL))]
                    .into_iter()
                    .chain(params)
                    .collect(),
            )
            .expect("valid params"),
            api_key: api_key.map(Into::into),
            api_base: Some(upstream.uri()),
            extra_headers: headers(extra_headers.iter().copied()),
            shaping: MessagesShaping {
                settings: MessagesSettings {
                    module_api_key: module_api_key.map(SecretValue::new),
                    ..MessagesSettings::default()
                },
                ..MessagesShaping::default()
            },
            ..call
        },
    )
    .await
    .expect("messages call succeeds");

    let request = only_request(&upstream).await;
    let sent: Vec<(String, String)> = ["x-api-key", "api-key", "authorization"]
        .into_iter()
        .flat_map(|name| {
            request
                .header_values(name)
                .into_iter()
                .map(move |value| (name.to_string(), value.to_string()))
        })
        .collect();
    let expected: Vec<(String, String)> = expected
        .iter()
        .map(|(name, value)| (name.to_string(), value.to_string()))
        .collect();
    assert_eq!(sent, expected);
}

/// Without a Python host, as in the Rust gateway, a deployment without a key fails before sending
/// as on main, whatever Entra ID settings it holds.
#[rstest]
#[case::nothing(json!({}), &[])]
#[case::a_static_token(json!({"azure_ad_token": "entra-token"}), &[])]
#[case::a_token_in_the_secret_source(json!({}), &[("AZURE_AD_TOKEN", "secret-token")])]
#[case::a_provider_named_callable(json!({"azure_ad_token_provider": "callable"}), &[])]
#[case::client_secret(json!({"tenant_id": "t", "client_id": "c", "client_secret": "s"}), &[])]
#[case::username_and_password(
    json!({"azure_username": "u", "azure_password": "p", "client_id": "c"}),
    &[]
)]
#[case::token_refresh(json!({"enable_azure_ad_token_refresh": true}), &[])]
#[tokio::test]
async fn an_azure_deployment_without_a_key_or_a_python_host_is_a_missing_key(
    call: MessagesCall,
    #[case] params: Value,
    #[case] secrets: &[(&str, &str)],
) {
    let upstream = upstream([message_response()]).await;
    let Value::Object(params) = params else {
        panic!("case params are an object")
    };

    let error = run_with(
        Arc::new(RecordingSecrets::new(secrets.iter().copied())),
        MessagesCall {
            custom_llm_provider: Some("azure_ai".into()),
            litellm_params: litellm_inference_messages::litellm_params(
                [("model".to_string(), json!(MODEL))]
                    .into_iter()
                    .chain(params)
                    .collect(),
            )
            .expect("valid params"),
            api_base: Some(upstream.uri()),
            ..call
        },
    )
    .await
    .expect_err("no key and no Python host");

    assert_eq!(
        error,
        Error::from(litellm_llms::Error::from(
            litellm_auth::Error::MissingApiKey {
                provider: "Azure",
                environment_variable: "AZURE_API_KEY",
            }
        ))
    );
    assert!(received(&upstream).await.is_empty());
}

/// Python reads `AZURE_USERNAME` and `AZURE_PASSWORD` with `os.getenv`, so only the process
/// environment sends the call to the host's `get_azure_ad_token`, which fails in these tests.
static PROCESS_ENVIRONMENT: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

#[rstest]
#[case::from_the_process_environment(&[("AZURE_USERNAME", "u"), ("AZURE_PASSWORD", "p")], true)]
#[case::without_them(&[], false)]
#[tokio::test]
async fn azure_username_and_password_come_from_the_process_environment(
    call: MessagesCall,
    #[case] process_env: &[(&'static str, &'static str)],
    #[case] asks_the_host: bool,
) {
    let _serialized = PROCESS_ENVIRONMENT.lock().await;
    for (name, value) in process_env {
        unsafe { std::env::set_var(name, value) };
    }
    let upstream = upstream([message_response()]).await;

    let outcome = run_hosted_with(
        Arc::new(RecordingSecrets::empty()),
        MessagesCall {
            custom_llm_provider: Some("azure_ai".into()),
            litellm_params: litellm_inference_messages::litellm_params(
                [
                    ("model".to_string(), json!(MODEL)),
                    ("client_id".to_string(), json!("c")),
                ]
                .into_iter()
                .collect(),
            )
            .expect("valid params"),
            api_base: Some(upstream.uri()),
            ..call
        },
    )
    .await;
    for (name, _) in process_env {
        unsafe { std::env::remove_var(name) };
    }

    assert_eq!(outcome.is_err(), asks_the_host, "{outcome:?}");
    assert_eq!(received(&upstream).await.is_empty(), asks_the_host);
}

#[rstest]
#[case::anthropic("anthropic")]
#[case::deepseek("deepseek")]
#[case::minimax("minimax")]
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

    assert!(error.is_missing_credential(), "{error:?}");
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
#[case::minimax(MODEL, Some("minimax"), "", "/v1/messages")]
#[case::minimax_prefix("minimax/claude-sonnet-4-5", None, "/v1/messages", "/v1/messages")]
#[case::minimax_endpoint_slash(MODEL, Some("minimax"), "/v1/messages/", "/v1/messages")]
#[case::azure_ai(MODEL, Some("azure_ai"), "", "/anthropic/v1/messages")]
#[case::deepseek(MODEL, Some("deepseek"), "", "/anthropic/v1/messages")]
#[case::deepseek_openai_compatible_base(MODEL, Some("deepseek"), "/beta", "/anthropic/v1/messages")]
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

    assert!(error.is_request(), "{error:?}");
    let report = litellm_error::Report::new(&error).to_string();
    assert!(report.contains(reported), "{report}");
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
    #[values("anthropic", "minimax")] provider: &str,
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
            custom_llm_provider: Some(provider.into()),
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
#[case::anthropic("anthropic", &[AnthropicBeta::DangerousToolUse20260903])]
#[case::deepseek("deepseek", &[])]
#[case::minimax("minimax", &[])]
#[tokio::test]
async fn safeguards_add_the_dangerous_tool_use_beta_only_where_the_beta_map_has_a_column(
    call: MessagesCall,
    #[case] provider: &str,
    #[case] derived: &[AnthropicBeta],
) {
    let upstream = upstream([message_response()]).await;
    let safeguards = json!([{"type": "dangerous_tool_use", "classifier_context": {"v": 1}}]);

    run_message(with_fields(
        MessagesCall {
            custom_llm_provider: Some(provider.into()),
            api_key: Some("sk".into()),
            api_base: Some(upstream.uri()),
            extra_headers: headers([("Anthropic-Beta", "caller-beta-2025-01-01")]),
            ..call
        },
        json!({"safeguards": safeguards}),
    ))
    .await;

    let request = only_request(&upstream).await;
    let expected: BetaSet = derived
        .iter()
        .cloned()
        .chain([AnthropicBeta::Other("caller-beta-2025-01-01".to_string())])
        .collect();
    assert_eq!(sent_betas(&request), expected);
    assert_eq!(
        request.body_json::<Value>().unwrap()["safeguards"],
        safeguards
    );
}

#[rstest]
#[tokio::test]
async fn null_safeguards_are_dropped_instead_of_sent_without_their_beta(
    call: MessagesCall,
    #[values("anthropic", "azure_ai", "bedrock_mantle", "deepseek", "minimax")] provider: &str,
) {
    let upstream = upstream([message_response()]).await;

    run_message(with_fields(
        MessagesCall {
            custom_llm_provider: Some(provider.into()),
            api_key: Some("sk".into()),
            api_base: Some(upstream.uri()),
            extra_headers: headers([("Anthropic-Beta", "context-1m-2025-08-07")]),
            ..call
        },
        json!({"safeguards": null}),
    ))
    .await;

    let request = only_request(&upstream).await;
    assert_eq!(
        sent_betas(&request),
        BetaSet::from_iter([AnthropicBeta::Context1m20250807])
    );
    assert!(
        !request
            .body_json::<Map<String, Value>>()
            .unwrap()
            .contains_key("safeguards")
    );
}

#[rstest]
#[case::unknown_and_unsupported_caller_betas(
    json!({}),
    "mcp-servers-2025-12-04, caller-beta-2025-01-01 ,effort-2025-11-24",
    Some("effort-2025-11-24")
)]
#[case::nothing_left(json!({}), "mcp-servers-2025-12-04,caller-beta-2025-01-01", None)]
#[case::an_unsupported_feature_beta(
    json!({"tools": [{"type": "advisor_20260301", "name": "advisor", "model": MODEL}]}),
    "caller-beta-2025-01-01",
    None
)]
#[case::a_supported_feature_beta(
    json!({"output_format": {"type": "json_schema"}}),
    "caller-beta-2025-01-01",
    Some("structured-outputs-2025-11-13")
)]
#[tokio::test]
async fn azure_ai_sends_only_the_betas_its_row_accepts(
    call: MessagesCall,
    #[case] fields: Value,
    #[case] caller: &str,
    #[case] sent: Option<&str>,
) {
    let upstream = upstream([message_response()]).await;

    run_message(with_fields(
        MessagesCall {
            custom_llm_provider: Some("azure_ai".into()),
            api_key: Some("sk".into()),
            api_base: Some(upstream.uri()),
            extra_headers: headers([("Anthropic-Beta", caller)]),
            ..call
        },
        fields,
    ))
    .await;

    assert_eq!(only_request(&upstream).await.header("anthropic-beta"), sent);
}

#[rstest]
#[tokio::test]
async fn azure_ai_picks_the_betas_before_it_converts_a_system_turn(call: MessagesCall) {
    let upstream = upstream([message_response()]).await;

    run_message(with_fields(
        MessagesCall {
            custom_llm_provider: Some("azure_ai".into()),
            api_key: Some("sk".into()),
            api_base: Some(upstream.uri()),
            ..call
        },
        json!({"messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "system", "content": "be brief", "output_config": {"effort": "low"}},
            {"role": "user", "content": "go on"}
        ]}),
    ))
    .await;

    let request = only_request(&upstream).await;
    assert_eq!(
        request.header("anthropic-beta"),
        Some("per-turn-control-2026-07-01")
    );
    assert_eq!(request.json()["messages"][2]["role"], "user");
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
#[case::minimax("minimax")]
#[tokio::test]
/// Python adds a default unless that exact lowercase name is present, then folds the casings
/// into one header, so a capitalised caller header is sent beside the default.
async fn capitalised_caller_protocol_headers_are_joined_with_the_defaults(
    call: MessagesCall,
    #[case] provider: &str,
) {
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
    assert_eq!(
        request.header_values("anthropic-version"),
        ["2024-01-01, 2023-06-01"]
    );
    assert_eq!(
        request.header_values("content-type"),
        ["application/json; charset=utf-8, application/json"]
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
    assert!(error.is_request(), "{error:?}");
    let report = litellm_error::Report::new(&error).to_string();
    assert!(report.contains(rejected_as), "{report}");
    assert!(received(&upstream).await.is_empty());

    run_message(shaped(true)).await;
    let sent = only_request(&upstream).await.json();
    for name in dropped {
        assert_eq!(sent.get(*name), None, "{name} must be dropped");
    }
    assert_eq!(sent["max_tokens"], 16);
}

#[derive(Clone, Default)]
struct Warnings(Arc<Mutex<Vec<String>>>);

impl litellm_tracing::Sink for Warnings {
    fn enabled(&self, metadata: &litellm_tracing::Metadata<'_>) -> bool {
        *metadata.level() == litellm_tracing::Level::WARN
    }

    fn emit(&self, record: &litellm_tracing::Record) {
        self.0.lock().unwrap().push(record.message.clone());
    }
}

#[rstest]
#[case::anthropic_logs_as_main("anthropic", false)]
#[case::azure_ai_logs_python_s_warning("azure_ai", true)]
#[tokio::test]
async fn a_speed_dropped_under_drop_params_is_logged_only_on_azure_ai(
    call: MessagesCall,
    #[case] provider: &str,
    #[case] logged: bool,
) {
    let upstream = upstream([message_response()]).await;
    let warnings = Warnings::default();

    litellm_tracing::Logger::new(warnings.clone())
        .instrument(run_message(MessagesCall {
            custom_llm_provider: Some(provider.into()),
            api_key: Some("sk".into()),
            api_base: Some(upstream.uri()),
            shaping: MessagesShaping {
                settings: MessagesSettings {
                    drop_params: true,
                    ..MessagesSettings::default()
                },
                ..MessagesShaping::default()
            },
            ..with_fields(call, json!({"speed": "fast"}))
        }))
        .await;

    assert_eq!(only_request(&upstream).await.json().get("speed"), None);
    let speed_warnings: Vec<String> = warnings
        .0
        .lock()
        .unwrap()
        .iter()
        .filter(|message| message.starts_with("Dropping unsupported `speed`"))
        .cloned()
        .collect();
    assert_eq!(
        speed_warnings.len(),
        usize::from(logged),
        "{speed_warnings:?}"
    );
}

#[rstest]
#[tokio::test]
async fn additional_drop_params_clear_a_param_the_model_does_not_support(call: MessagesCall) {
    let upstream = upstream([message_response()]).await;

    run_message(MessagesCall {
        api_key: Some("sk".into()),
        api_base: Some(upstream.uri()),
        shaping: MessagesShaping {
            capabilities: sampling_removed(),
            settings: MessagesSettings {
                additional_drop_params: vec!["top_k".into()],
                ..MessagesSettings::default()
            },
        },
        ..with_fields(call, json!({"top_k": 5}))
    })
    .await;

    let sent = only_request(&upstream).await.json();
    assert_eq!(sent.get("top_k"), None);
    assert_eq!(sent["max_tokens"], 16);
}

#[rstest]
#[case::adaptive_thinking(json!({"type": "adaptive"}), json!({"type": "adaptive", "display": "summarized"}))]
#[case::disabled_thinking(json!({"type": "disabled"}), json!({"type": "disabled"}))]
#[tokio::test]
async fn reasoning_auto_summary_marks_active_thinking_on_the_wire(
    call: MessagesCall,
    #[values("anthropic", "minimax")] provider: &str,
    #[case] thinking: Value,
    #[case] expected: Value,
) {
    let upstream = upstream([message_response()]).await;

    run_message(with_fields(
        MessagesCall {
            custom_llm_provider: Some(provider.into()),
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
    #[values("anthropic", "minimax")] provider: &str,
    #[case] capabilities: MessagesModelCapabilities,
    #[case] fields: Value,
    #[case] expected: Value,
) {
    let upstream = upstream([message_response()]).await;

    run_message(with_fields(
        MessagesCall {
            custom_llm_provider: Some(provider.into()),
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
#[case::minimax("minimax")]
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
async fn an_invalid_request_fails_before_sending(
    call: MessagesCall,
    #[values("anthropic", "minimax")] provider: &str,
    #[case] fields: Value,
) {
    let upstream = upstream([message_response()]).await;

    let error = run(with_fields(
        MessagesCall {
            custom_llm_provider: Some(provider.into()),
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
#[case::billing_string(json!("x-anthropic-billing-header: private"), None)]
#[case::billing_blocks(json!([{ "type": "text", "text": "x-anthropic-billing-header: private" }]), None)]
#[case::mixed_blocks(
    json!([{ "type": "text", "text": "x-anthropic-billing-header: private" }, { "type": "text", "text": "keep" }]),
    Some(json!([{ "type": "text", "text": "keep" }]))
)]
#[case::ordinary_system(json!("keep"), Some(json!("keep")))]
#[tokio::test]
async fn minimax_strips_only_billing_system_blocks(
    call: MessagesCall,
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
            custom_llm_provider: Some("minimax".into()),
            api_key: Some("sk".into()),
            api_base: Some(upstream.uri()),
            ..call
        },
        fields.clone(),
    ))
    .await;

    let sent = only_request(&upstream).await.json();
    assert_eq!(sent.get("system"), expected.as_ref());
    assert_eq!(sent["tools"], fields["tools"]);
    assert_eq!(sent["messages"], fields["messages"]);
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
    let report = litellm_error::Report::new(&error).to_string();
    assert!(
        report.contains("metadata.user_id must be a string"),
        "{report}"
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
