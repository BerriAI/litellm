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
    assert_eq!(request.header("anthropic-version"), Some("2023-06-01"));
    assert_eq!(request.header("content-type"), Some("application/json"));
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

#[rstest]
#[case::anthropic_streams(MODEL, Some("anthropic"), true, true)]
#[case::anthropic_prefix_streams("anthropic/claude-sonnet-4-5", None, true, true)]
#[case::azure_without_stream(MODEL, Some("azure_ai"), false, true)]
#[case::azure_stream(MODEL, Some("azure_ai"), true, false)]
#[case::other_provider(MODEL, Some("openai"), false, false)]
#[case::unresolvable_model("no-such-model", None, false, false)]
fn supports_matches_what_the_route_can_serve(
    #[case] model: &str,
    #[case] provider: Option<&str>,
    #[case] stream: bool,
    #[case] supported: bool,
) {
    assert_eq!(
        litellm_core::messages::route::supports(model, provider, stream),
        supported
    );
}
