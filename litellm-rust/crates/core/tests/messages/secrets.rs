use rstest::rstest;

use super::*;

#[rstest]
#[case::anthropic(
    "anthropic",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_BASE_URL",
    "/v1/messages",
    &["ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_API_BASE", "ANTHROPIC_BASE_URL"]
)]
#[case::azure_ai(
    "azure_ai",
    "AZURE_API_KEY",
    "AZURE_API_BASE",
    "/anthropic/v1/messages",
    &["AZURE_API_KEY", "AZURE_API_BASE"]
)]
#[tokio::test]
async fn the_credential_and_base_come_from_the_secret_source(
    call: MessagesCall,
    #[case] provider: &str,
    #[case] key_name: &str,
    #[case] base_name: &str,
    #[case] path: &str,
    #[case] looked_up: &[&str],
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
    assert_eq!(secrets.requested(), looked_up);
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

#[derive(Clone, Copy)]
enum Base {
    Upstream,
    Unreachable,
    Blank,
    Absent,
}

fn base_value(base: Base, upstream: &str) -> Option<String> {
    match base {
        Base::Upstream => Some(upstream.to_string()),
        Base::Unreachable => Some(UNREACHABLE_BASE.to_string()),
        Base::Blank => Some("  ".to_string()),
        Base::Absent => None,
    }
}

#[rstest]
#[case::api_base_beats_base_url(Base::Upstream, Base::Unreachable)]
#[case::blank_api_base_falls_through_to_base_url(Base::Blank, Base::Upstream)]
#[case::base_url_alone(Base::Absent, Base::Upstream)]
#[tokio::test]
async fn the_anthropic_base_env_precedence_picks_the_upstream(
    call: MessagesCall,
    #[case] api_base: Base,
    #[case] base_url: Base,
) {
    let upstream = upstream([message_response()]).await;
    let uri = upstream.uri();
    let values: Vec<(&str, &str)> = [
        ("ANTHROPIC_API_KEY", Some("sk-env".to_string())),
        ("ANTHROPIC_API_BASE", base_value(api_base, &uri)),
        ("ANTHROPIC_BASE_URL", base_value(base_url, &uri)),
    ]
    .iter()
    .filter_map(|(name, value)| Some((*name, value.as_deref()?)))
    .map(|(name, value)| (name, Box::leak(value.to_string().into_boxed_str()) as &str))
    .collect();

    run_with(Arc::new(RecordingSecrets::new(values)), call)
        .await
        .expect("messages call reaches the upstream the precedence picks");

    assert_eq!(only_request(&upstream).await.url.path(), "/v1/messages");
}

#[rstest]
#[case::auth_token_alone(
    &[("ANTHROPIC_AUTH_TOKEN", "tok")],
    ("authorization", "Bearer tok"),
    "x-api-key"
)]
#[case::api_key_beats_the_auth_token(
    &[("ANTHROPIC_API_KEY", "sk-env"), ("ANTHROPIC_AUTH_TOKEN", "tok")],
    ("x-api-key", "sk-env"),
    "authorization"
)]
#[tokio::test]
async fn the_auth_token_env_is_a_bearer_only_without_a_key(
    call: MessagesCall,
    #[case] values: &[(&str, &str)],
    #[case] expected: (&str, &str),
    #[case] absent: &str,
) {
    let upstream = upstream([message_response()]).await;

    run_with(
        Arc::new(RecordingSecrets::new(values.iter().copied())),
        MessagesCall {
            api_base: Some(upstream.uri()),
            ..call
        },
    )
    .await
    .expect("messages call succeeds");

    let request = only_request(&upstream).await;
    let (name, value) = expected;
    assert_eq!(request.header_values(name), [value]);
    assert_eq!(request.header(absent), None);
}

#[rstest]
#[tokio::test]
async fn azure_without_a_base_anywhere_fails_before_sending(call: MessagesCall) {
    let error = run_with(
        Arc::new(RecordingSecrets::new([("AZURE_API_KEY", "sk-azure")])),
        MessagesCall {
            custom_llm_provider: Some("azure_ai".into()),
            ..call
        },
    )
    .await
    .err()
    .expect("azure needs a base");

    assert_eq!(error, Error::Auth(litellm_auth::Error::MissingAzureApiBase));
}
