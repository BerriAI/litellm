use std::sync::Arc;

use litellm_secrets::{
    AccessMode, KeyManagementSettings, KeyManagementSystem, OidcResolver, Secret, SecretManager,
    SecretManagerState, SecretResolver, SecretValue, secret_manager_would_be_consulted,
};

fn resolver(value: Option<&str>, readable: bool) -> SecretResolver {
    let state = if readable {
        SecretManagerState::new(
            Some(KeyManagementSystem::Local),
            Some(KeyManagementSettings::default()),
            Some(SecretManager::Local),
        )
        .unwrap()
    } else {
        SecretManagerState::default()
    };
    let value = value.map(str::to_owned);
    SecretResolver::new(
        Arc::new(state),
        Arc::new(move |_: &str| value.clone()),
        OidcResolver::default(),
    )
}

#[rstest::rstest]
#[case::lowercase_true("true", Some(true), None)]
#[case::whitespace_lowercase_false(" FALSE ", Some(false), None)]
#[case::python_true("True", Some(true), Some(true))]
#[case::python_false("False", Some(false), Some(false))]
#[case::parenthesized_python_true("(True)", None, Some(true))]
#[case::commented_python_false("False # comment", None, Some(false))]
#[case::integer("1", None, None)]
#[case::yes("yes", None, None)]
#[case::plain_string("secret", None, None)]
#[tokio::test]
async fn boolean_conversion_preserves_local_and_manager_differences(
    #[case] input: &str,
    #[case] local: Option<bool>,
    #[case] manager: Option<bool>,
    #[values(false, true)] readable: bool,
) {
    let boolean = if readable { manager } else { local };
    let resolver = resolver(Some(input), readable);
    let expected = boolean
        .map(Secret::Bool)
        .unwrap_or_else(|| Secret::String(SecretValue::new(input)));
    assert_eq!(
        resolver.get_secret("key", None).await.unwrap(),
        Some(expected)
    );
    assert_eq!(
        resolver
            .get_secret_str("key", None)
            .await
            .unwrap()
            .map(|v| v.expose().to_owned()),
        boolean.is_none().then(|| input.to_owned())
    );
}

#[tokio::test]
async fn manager_boolean_conversion_trims_whitespace() {
    assert_eq!(
        resolver(Some(" true "), true)
            .get_secret_bool("key", None)
            .await
            .unwrap(),
        Some(true)
    );
}

#[tokio::test]
async fn missing_values_ignore_defaults_and_prefix_is_removed_before_lookup() {
    let missing = resolver(None, false);
    assert_eq!(
        missing
            .get_secret("missing", Some(Secret::Bool(true)))
            .await
            .unwrap(),
        None
    );
    assert_eq!(
        missing
            .get_secret_bool("missing", Some(true))
            .await
            .unwrap(),
        None
    );
    let resolver = SecretResolver::new(
        Arc::new(SecretManagerState::default()),
        Arc::new(|name: &str| (name == "KEY").then(|| "value".into())),
        OidcResolver::default(),
    );
    assert_eq!(
        resolver
            .get_secret_str("os.environ/KEY", None)
            .await
            .unwrap()
            .unwrap()
            .expose(),
        "value"
    );
}

#[rstest::rstest]
#[case::all_keys(None)]
#[case::no_keys(Some(Vec::new()))]
#[case::allowlisted_key(Some(vec!["KEY".into()]))]
fn manager_gating_requires_client_readable_settings_and_allowlisted_name(
    #[values(AccessMode::ReadOnly, AccessMode::WriteOnly, AccessMode::ReadAndWrite)]
    access_mode: AccessMode,
    #[values(false, true)] client: bool,
    #[case] keys: Option<Vec<String>>,
) {
    let expected = client
        && access_mode.readable()
        && keys
            .as_ref()
            .is_none_or(|keys| keys.iter().any(|key| key == "KEY"));
    let state = SecretManagerState::new(
        Some(KeyManagementSystem::Local),
        Some(KeyManagementSettings {
            access_mode,
            hosted_keys: keys,
            ..Default::default()
        }),
        client.then_some(SecretManager::Local),
    )
    .unwrap();
    assert_eq!(
        secret_manager_would_be_consulted(&state, "os.environ/KEY"),
        expected
    );
}

#[test]
fn manager_gating_requires_settings() {
    let no_settings = SecretManagerState::new(None, None, Some(SecretManager::Local)).unwrap();
    assert!(!secret_manager_would_be_consulted(&no_settings, "KEY"));
}

#[cfg(feature = "aws")]
#[rstest::rstest]
#[case::missing_value(None, None)]
#[case::lookup_error(Some("primary".to_owned()), Some("environment-value"))]
#[tokio::test]
async fn aws_missing_values_do_not_fallback_but_lookup_errors_do(
    #[case] primary: Option<String>,
    #[case] expected: Option<&str>,
) {
    use litellm_secrets::aws::AwsSecretsManagerV2;
    use wiremock::{Mock, MockServer, ResponseTemplate, matchers::body_partial_json};
    let server = MockServer::start().await;
    Mock::given(body_partial_json(serde_json::json!({"SecretId":"KEY"})))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({})))
        .expect(u64::from(primary.is_none()))
        .mount(&server)
        .await;
    Mock::given(body_partial_json(serde_json::json!({"SecretId":"primary"})))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(serde_json::json!({"SecretString":"invalid-json"})),
        )
        .expect(u64::from(primary.is_some()))
        .mount(&server)
        .await;
    let endpoint = server.uri();
    let environment: Arc<dyn litellm_core_utils::settings::Lookup + Send + Sync> =
        Arc::new(move |name: &str| match name {
            "AWS_REGION_NAME" => Some("us-east-1".into()),
            "AWS_ACCESS_KEY_ID" | "AWS_SECRET_ACCESS_KEY" => Some("test".into()),
            "AWS_BEDROCK_RUNTIME_ENDPOINT" => Some(endpoint.clone()),
            "KEY" => Some("environment-value".into()),
            _ => None,
        });
    let settings = KeyManagementSettings {
        primary_secret_name: primary,
        ..Default::default()
    };
    let manager = AwsSecretsManagerV2::load_aws_secret_manager(
        Some(true),
        settings.clone(),
        environment.clone(),
    )
    .unwrap()
    .unwrap();
    let state = SecretManagerState::new(
        Some(KeyManagementSystem::AwsSecretManager),
        Some(settings),
        Some(SecretManager::AwsSecretsManagerV2(manager)),
    )
    .unwrap();
    let resolver = SecretResolver::new(
        Arc::new(state),
        environment.clone(),
        OidcResolver::default(),
    );
    assert_eq!(
        resolver
            .get_secret_str("os.environ/KEY", None)
            .await
            .unwrap()
            .map(|v| v.expose().to_owned())
            .as_deref(),
        expected
    );
}

#[cfg(feature = "google")]
#[rstest::rstest]
#[case::hosted_filter(Some(Vec::new()), Some(KeyManagementSystem::GoogleSecretManager))]
#[case::negative_cache(None, Some(KeyManagementSystem::GoogleSecretManager))]
#[case::missing_system(None, None)]
#[case::hosted_nested_prefix(Some(vec!["os.environ/KEY".into()]), Some(KeyManagementSystem::GoogleSecretManager))]
#[tokio::test]
async fn google_negative_cache_still_falls_back_and_hosted_filter_avoids_io(
    #[case] hosted_keys: Option<Vec<String>>,
    #[case] system: Option<KeyManagementSystem>,
) {
    use litellm_secrets::google::GoogleSecretManager;
    use std::time::Duration;
    use wiremock::{Mock, MockServer, ResponseTemplate, matchers::path};
    let server = MockServer::start().await;
    Mock::given(path(
        "/v1/projects/project/secrets/os%2Eenviron%2FKEY/versions/latest:access",
    ))
    .respond_with(ResponseTemplate::new(404))
    .expect(u64::from(
        hosted_keys.as_ref().is_none_or(|keys| !keys.is_empty()) && system.is_some(),
    ))
    .mount(&server)
    .await;
    let environment: Arc<dyn litellm_core_utils::settings::Lookup + Send + Sync> =
        Arc::new(|name: &str| match name {
            "VERTEX_AI_API_KEY" => Some("token".into()),
            "os.environ/KEY" => Some("environment-value".into()),
            _ => None,
        });
    let manager = GoogleSecretManager::with_client(
        reqwest::Client::new(),
        server.uri().parse().unwrap(),
        "project".into(),
        environment.clone(),
        Some(Duration::from_secs(60)),
        false,
    )
    .unwrap();
    let settings = KeyManagementSettings {
        hosted_keys,
        ..Default::default()
    };
    let state = SecretManagerState::new(
        system,
        Some(settings),
        Some(SecretManager::GoogleSecretManager(manager)),
    )
    .unwrap();
    let resolver = SecretResolver::new(
        Arc::new(state),
        environment.clone(),
        OidcResolver::default(),
    );
    for _ in 0..2 {
        assert_eq!(
            resolver
                .get_secret_str("os.environ/os.environ/KEY", None)
                .await
                .unwrap()
                .unwrap()
                .expose(),
            "environment-value"
        );
    }
}

#[tokio::test]
async fn resolver_future_can_run_on_a_tokio_worker() {
    let resolver = resolver(Some("worker-value"), false);
    let result = tokio::spawn(async move { resolver.get_secret_str("KEY", None).await })
        .await
        .unwrap()
        .unwrap();
    assert_eq!(result.unwrap().expose(), "worker-value");
}

#[rstest::rstest]
#[case::nested_true("((True)) # comment", Some(true))]
#[case::commented_false("(False # comment\n)", Some(false))]
#[case::boolean_expression("True and False", None)]
#[case::string_literal("'True'", None)]
#[case::tuple("(True,)", None)]
#[case::unary_expression("not False", None)]
#[case::multiple_expressions("True\nFalse", None)]
#[case::incomplete_expression("(True", None)]
#[tokio::test]
async fn manager_boolean_literals_follow_python_syntax(
    #[case] input: &str,
    #[case] expected: Option<bool>,
) {
    let value = resolver(Some(input), true)
        .get_secret("key", None)
        .await
        .unwrap();
    assert_eq!(
        value,
        Some(
            expected
                .map(Secret::Bool)
                .unwrap_or_else(|| Secret::String(SecretValue::new(input)))
        )
    );
}

#[tokio::test]
async fn environment_prefix_is_removed_only_once_and_gating_uses_the_same_name() {
    let name = "os.environ/folder/os.environ/KEY";
    let state = SecretManagerState::new(
        Some(KeyManagementSystem::Local),
        Some(KeyManagementSettings {
            hosted_keys: Some(vec!["folder/os.environ/KEY".into()]),
            ..Default::default()
        }),
        Some(SecretManager::Local),
    )
    .unwrap();
    assert!(secret_manager_would_be_consulted(&state, name));
    let resolver = SecretResolver::new(
        Arc::new(state),
        Arc::new(|name: &str| (name == "folder/os.environ/KEY").then(|| "value".into())),
        OidcResolver::default(),
    );
    assert_eq!(
        resolver
            .get_secret_str(name, None)
            .await
            .unwrap()
            .unwrap()
            .expose(),
        "value"
    );
}
