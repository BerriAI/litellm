use litellm_llms::{
    anthropic::experimental_pass_through::messages::transformation::ANTHROPIC_MESSAGES_CONFIG,
    azure_ai::anthropic::messages_transformation::AZURE_ANTHROPIC_MESSAGES_CONFIG,
    base_llm::anthropic_messages::transformation::BaseAnthropicMessagesConfig,
};
use rstest::rstest;

use super::*;

#[rstest]
#[case::anthropic("anthropic", &ANTHROPIC_MESSAGES_CONFIG, "ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL", "/v1/messages")]
#[case::azure_ai("azure_ai", &AZURE_ANTHROPIC_MESSAGES_CONFIG, "AZURE_API_KEY", "AZURE_API_BASE", "/anthropic/v1/messages")]
#[tokio::test]
async fn the_credential_and_base_come_from_the_secret_source(
    call: MessagesCall,
    #[case] provider: &str,
    #[case] config: &dyn BaseAnthropicMessagesConfig,
    #[case] key_name: &str,
    #[case] base_name: &str,
    #[case] path: &str,
) {
    let upstream = upstream([message_response()]).await;
    let base = upstream.uri();
    let secrets = Arc::new(RecordingSecrets::new([
        (key_name, "sk-from-manager"),
        (base_name, base.as_str()),
    ]));

    let output = run_with(
        secrets.clone(),
        MessagesCall {
            custom_llm_provider: Some(provider.into()),
            ..call
        },
    )
    .await
    .expect("messages call succeeds");

    assert!(matches!(output, MessagesOutput::Message(_)));
    let request = only_request(&upstream).await;
    assert_eq!(request.url.path(), path);
    assert_eq!(request.header("x-api-key"), Some("sk-from-manager"));
    assert_eq!(secrets.requested(), config.secret_names());
}

#[rstest]
#[tokio::test]
async fn call_arguments_win_over_the_secret_source(call: MessagesCall) {
    let upstream = upstream([message_response()]).await;
    let secrets = Arc::new(RecordingSecrets::new([
        ("ANTHROPIC_API_KEY", "sk-from-manager"),
        ("ANTHROPIC_BASE_URL", UNREACHABLE_BASE),
    ]));

    run_with(
        secrets,
        MessagesCall {
            api_key: Some("sk-from-call".into()),
            api_base: Some(upstream.uri()),
            ..call
        },
    )
    .await
    .expect("messages call succeeds");

    assert_eq!(
        only_request(&upstream).await.header("x-api-key"),
        Some("sk-from-call")
    );
}

#[rstest]
#[tokio::test]
async fn a_secret_manager_failure_fails_the_call_before_sending(call: MessagesCall) {
    let upstream = upstream([message_response()]).await;

    let error = run_with(
        Arc::new(RecordingSecrets::failing()),
        MessagesCall {
            api_key: Some("sk".into()),
            api_base: Some(upstream.uri()),
            ..call
        },
    )
    .await
    .err()
    .expect("a secret manager failure fails the call");

    assert!(
        matches!(&error, Error::Secret(source) if matches!(source.source_error(), litellm_secrets::Error::ManagedSecretMissing)),
        "{error:?}"
    );
    assert!(received(&upstream).await.is_empty());
}
