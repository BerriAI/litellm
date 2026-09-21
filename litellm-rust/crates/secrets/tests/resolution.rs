use std::sync::Arc;

use litellm_secrets::{
    Error, KeyManagementSettings, OidcResolver, Secret, SecretManager, SecretManagerState,
    SecretResolver, SecretValue, secret_manager_would_be_consulted,
};

fn resolver(value: Option<&str>, configured: bool) -> SecretResolver {
    let state = if configured {
        SecretManagerState::new(SecretManager::Local, KeyManagementSettings::default())
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
#[case("true", Some(true))]
#[case(" FALSE ", Some(false))]
#[case("(True)", None)]
#[case("False # comment", None)]
#[case("1", None)]
#[case("secret", None)]
#[tokio::test]
async fn conversion_is_explicit_and_independent_of_manager_configuration(
    #[case] input: &str,
    #[case] boolean: Option<bool>,
    #[values(false, true)] configured: bool,
) {
    let resolver = resolver(Some(input), configured);
    assert_eq!(
        resolver.get_secret("key", None).await.unwrap(),
        Some(Secret::String(SecretValue::new(input)))
    );
    assert_eq!(
        resolver
            .get_secret_str("key", None)
            .await
            .unwrap()
            .unwrap()
            .expose(),
        input
    );
    match boolean {
        Some(value) => assert_eq!(
            resolver.get_secret_bool("key", None).await.unwrap(),
            Some(value)
        ),
        None => assert!(matches!(
            resolver.get_secret_bool("key", Some(true)).await,
            Err(Error::TypeMismatch {
                expected: "boolean"
            })
        )),
    }
}

#[rstest::rstest]
#[tokio::test]
async fn defaults_apply_only_to_absence(#[values(false, true)] configured: bool) {
    let missing = resolver(None, configured);
    assert_eq!(missing.get_secret("key", None).await.unwrap(), None);
    assert_eq!(
        missing.get_secret_bool("key", Some(false)).await.unwrap(),
        Some(false)
    );
    assert_eq!(
        missing
            .get_secret_str("key", Some(SecretValue::new("default")))
            .await
            .unwrap()
            .unwrap()
            .expose(),
        "default"
    );
    for value in [
        Secret::Bool(false),
        Secret::from_json(serde_json::json!({"key":1})),
        Secret::from_json(serde_json::Value::Null),
    ] {
        assert_eq!(
            missing
                .get_secret("key", Some(value.clone()))
                .await
                .unwrap(),
            Some(value)
        );
    }
    assert_eq!(
        resolver(Some(""), configured)
            .get_secret_str("key", Some(SecretValue::new("default")))
            .await
            .unwrap()
            .unwrap()
            .expose(),
        ""
    );
}

#[tokio::test]
async fn prefix_is_removed_once_and_local_manager_is_not_consulted() {
    let state = SecretManagerState::new(SecretManager::Local, KeyManagementSettings::default());
    assert_eq!(
        state.system(),
        Some(litellm_secrets::KeyManagementSystem::Local)
    );
    assert!(!secret_manager_would_be_consulted(
        &state,
        "os.environ/os.environ/KEY"
    ));
    let resolver = SecretResolver::new(
        Arc::new(state),
        Arc::new(|name: &str| (name == "os.environ/KEY").then(|| "value".into())),
        OidcResolver::default(),
    );
    assert_eq!(
        resolver
            .get_secret_str("os.environ/os.environ/KEY", None)
            .await
            .unwrap()
            .unwrap()
            .expose(),
        "value"
    );
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

#[cfg(feature = "aws")]
mod aws {
    use super::*;
    use litellm_secrets::{AccessMode, FailurePolicy, aws::AwsSecretsManagerV2};
    use wiremock::{Mock, MockServer, ResponseTemplate, matchers::method};

    fn state(server: &MockServer, settings: KeyManagementSettings) -> SecretManagerState {
        let endpoint = server.uri();
        let environment = Arc::new(move |name: &str| match name {
            "AWS_REGION_NAME" => Some("us-east-1".into()),
            "AWS_ACCESS_KEY_ID" | "AWS_SECRET_ACCESS_KEY" => Some("test".into()),
            "AWS_BEDROCK_RUNTIME_ENDPOINT" => Some(endpoint.clone()),
            _ => None,
        });
        let manager =
            AwsSecretsManagerV2::load_aws_secret_manager(Some(true), settings.clone(), environment)
                .unwrap()
                .unwrap();
        SecretManagerState::new(SecretManager::AwsSecretsManagerV2(manager), settings)
    }

    #[rstest::rstest]
    #[case::missing(400, serde_json::json!({"__type":"ResourceNotFoundException"}), false)]
    #[case::denied(400, serde_json::json!({"__type":"AccessDeniedException"}), true)]
    #[case::malformed(200, serde_json::json!({}), true)]
    #[tokio::test]
    async fn failure_policy_preserves_errors_and_fallback_precedence(
        #[case] status: u16,
        #[case] body: serde_json::Value,
        #[case] fails: bool,
        #[values(FailurePolicy::Propagate, FailurePolicy::EnvironmentFallback)]
        policy: FailurePolicy,
        #[values(None, Some("environment"))] environment: Option<&'static str>,
        #[values(None, Some("default"))] default: Option<&str>,
    ) {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(status).set_body_json(body))
            .expect(1)
            .mount(&server)
            .await;
        let resolver = SecretResolver::new(
            Arc::new(state(&server, KeyManagementSettings::default())),
            Arc::new(move |_: &str| environment.map(str::to_owned)),
            OidcResolver::default(),
        )
        .with_failure_policy(policy);
        let result = resolver
            .get_secret_str("KEY", default.map(SecretValue::new))
            .await;
        let fallback = environment.or(default);
        if fails && (policy == FailurePolicy::Propagate || fallback.is_none()) {
            assert!(matches!(result, Err(Error::Aws(_))));
        } else {
            assert_eq!(result.unwrap().as_ref().map(SecretValue::expose), fallback);
        }
    }

    #[rstest::rstest]
    #[case::boolean(serde_json::json!(false))]
    #[case::object(serde_json::json!({"key":1}))]
    #[case::null(serde_json::Value::Null)]
    #[case::string(serde_json::json!("true"))]
    #[tokio::test]
    async fn typed_values_survive_resolution_and_accessors_reject_wrong_types(
        #[case] value: serde_json::Value,
    ) {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(
                serde_json::json!({"SecretString":serde_json::json!({"KEY":value}).to_string()}),
            ))
            .expect(3)
            .mount(&server)
            .await;
        let settings = KeyManagementSettings {
            primary_secret_name: Some("primary".into()),
            ..Default::default()
        };
        let resolver = SecretResolver::new(
            Arc::new(state(&server, settings)),
            Arc::new(|_: &str| Some("fallback".into())),
            OidcResolver::default(),
        );
        assert_eq!(
            resolver
                .get_secret("KEY", Some(Secret::Bool(true)))
                .await
                .unwrap(),
            Some(Secret::from_json(value.clone()))
        );
        match &value {
            serde_json::Value::String(text) => assert_eq!(
                resolver
                    .get_secret_str("KEY", None)
                    .await
                    .unwrap()
                    .unwrap()
                    .expose(),
                text
            ),
            _ => assert!(matches!(
                resolver.get_secret_str("KEY", None).await,
                Err(Error::TypeMismatch { expected: "string" })
            )),
        }
        match value {
            serde_json::Value::Bool(boolean) => assert_eq!(
                resolver.get_secret_bool("KEY", None).await.unwrap(),
                Some(boolean)
            ),
            serde_json::Value::String(_) => assert_eq!(
                resolver.get_secret_bool("KEY", None).await.unwrap(),
                Some(true)
            ),
            _ => assert!(matches!(
                resolver.get_secret_bool("KEY", None).await,
                Err(Error::TypeMismatch {
                    expected: "boolean"
                })
            )),
        }
    }

    #[rstest::rstest]
    #[tokio::test]
    async fn gating_prediction_matches_actual_lookup(
        #[values(AccessMode::ReadOnly, AccessMode::WriteOnly, AccessMode::ReadAndWrite)]
        access_mode: AccessMode,
        #[values(None, Some(vec![]), Some(vec!["KEY".into()]))] hosted_keys: Option<Vec<String>>,
        #[values("os.environ/KEY", "os.environ/oidc/env/KEY")] name: &str,
    ) {
        let server = MockServer::start().await;
        let expected = name == "os.environ/KEY"
            && access_mode.readable()
            && hosted_keys
                .as_ref()
                .is_none_or(|keys| keys.iter().any(|key| key == "KEY"));
        Mock::given(method("POST"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(serde_json::json!({"SecretString":"remote"})),
            )
            .expect(u64::from(expected))
            .mount(&server)
            .await;
        let state = state(
            &server,
            KeyManagementSettings {
                access_mode,
                hosted_keys,
                ..Default::default()
            },
        );
        assert!(state.backend().is_some());
        assert_eq!(state.settings().unwrap().access_mode, access_mode);
        assert_eq!(secret_manager_would_be_consulted(&state, name), expected);
        let resolver = SecretResolver::new(
            Arc::new(state),
            Arc::new(|_: &str| Some("environment".into())),
            OidcResolver::default(),
        );
        assert_eq!(
            resolver
                .get_secret_str(name, None)
                .await
                .unwrap()
                .unwrap()
                .expose(),
            if expected { "remote" } else { "environment" }
        );
    }
}

#[cfg(feature = "google")]
#[rstest::rstest]
#[case::missing(404)]
#[case::failure(503)]
#[tokio::test]
async fn google_resolver_distinguishes_absence_from_failure(#[case] status: u16) {
    use litellm_secrets::{FailurePolicy, google::GoogleSecretManager};
    use wiremock::{Mock, MockServer, ResponseTemplate, matchers::method};
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .respond_with(ResponseTemplate::new(status))
        .expect(2)
        .mount(&server)
        .await;
    let environment: Arc<dyn litellm_core_utils::settings::Lookup + Send + Sync> =
        Arc::new(|name: &str| match name {
            "VERTEX_AI_API_KEY" => Some("token".into()),
            "KEY" => Some("environment".into()),
            _ => None,
        });
    let manager = GoogleSecretManager::with_client(
        reqwest::Client::new(),
        server.uri().parse().unwrap(),
        "project".into(),
        environment.clone(),
        None,
        false,
    )
    .unwrap();
    let state = SecretManagerState::new(
        SecretManager::GoogleSecretManager(manager),
        KeyManagementSettings::default(),
    );
    let resolver = SecretResolver::new(Arc::new(state), environment, OidcResolver::default());
    let result = resolver.get_secret_str("KEY", None).await;
    if status == 404 {
        assert_eq!(result.unwrap().unwrap().expose(), "environment");
    } else {
        assert!(
            matches!(result, Err(Error::Google(litellm_secrets::google::Error::Status(actual))) if actual == status)
        );
    }
    assert_eq!(
        resolver
            .with_failure_policy(FailurePolicy::EnvironmentFallback)
            .get_secret_str("KEY", None)
            .await
            .unwrap()
            .unwrap()
            .expose(),
        "environment"
    );
}
