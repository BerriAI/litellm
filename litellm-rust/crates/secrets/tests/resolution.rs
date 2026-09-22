use std::{future::Future, pin::Pin, sync::Arc};

use litellm_core_utils::settings::Lookup;
use litellm_secrets::{
    Error, ExternalSecretManager, FailurePolicy, KeyManagementSettings, KeyManagementSystem,
    OidcResolver, Secret, SecretManager, SecretManagerState, SecretResolver, SecretValue,
    secret_manager_would_be_consulted,
};

fn resolver(value: Option<&str>) -> SecretResolver {
    let value = value.map(str::to_owned);
    SecretResolver::new(
        Arc::new(SecretManagerState::default()),
        Arc::new(move |_: &str| value.clone()),
        OidcResolver::default(),
    )
}

#[rstest::rstest]
#[case::lowercase_true("true", Some(true))]
#[case::padded_false(" FALSE ", Some(false))]
#[case::capitalized_true("True", Some(true))]
#[case::parenthesized("(True)", None)]
#[case::commented("False # comment", None)]
#[case::number("1", None)]
#[case::text("secret", None)]
#[tokio::test]
async fn environment_values_are_coerced_like_str_to_bool(
    #[case] input: &str,
    #[case] boolean: Option<bool>,
) {
    let resolver = resolver(Some(input));
    assert_eq!(
        resolver.get_secret("key", None).await.unwrap(),
        Some(boolean.map_or_else(|| Secret::String(SecretValue::new(input)), Secret::Bool))
    );
    assert_eq!(
        resolver
            .get_secret_str("key", None)
            .await
            .unwrap()
            .as_ref()
            .map(SecretValue::expose),
        boolean.is_none().then_some(input)
    );
    assert_eq!(
        resolver.get_secret_bool("key", Some(true)).await.unwrap(),
        boolean
    );
}

#[tokio::test]
async fn defaults_never_replace_an_absent_secret() {
    let missing = resolver(None);
    assert_eq!(
        missing
            .get_secret("key", Some(Secret::Bool(false)))
            .await
            .unwrap(),
        None
    );
    assert_eq!(
        missing.get_secret_bool("key", Some(false)).await.unwrap(),
        None
    );
    assert_eq!(
        missing
            .get_secret_str("key", Some(SecretValue::new("default")))
            .await
            .unwrap(),
        None
    );
    assert_eq!(
        resolver(Some(""))
            .get_secret_str("key", Some(SecretValue::new("default")))
            .await
            .unwrap()
            .unwrap()
            .expose(),
        ""
    );
}

struct FixedManager(Result<Option<Secret>, ()>);

impl ExternalSecretManager for FixedManager {
    fn system(&self) -> KeyManagementSystem {
        KeyManagementSystem::Custom
    }

    fn read_secret<'a>(
        &'a self,
        _name: &'a str,
        _settings: &'a KeyManagementSettings,
        _environment: &'a (dyn Lookup + Send + Sync),
    ) -> Pin<Box<dyn Future<Output = Result<Option<Secret>, Error>> + Send + 'a>> {
        Box::pin(async move { self.0.clone().map_err(|()| Error::MissingCiphertext) })
    }
}

fn managed(reply: Result<Option<Secret>, ()>, environment: Option<&'static str>) -> SecretResolver {
    SecretResolver::new(
        Arc::new(SecretManagerState::new(
            SecretManager::External(Arc::new(FixedManager(reply))),
            KeyManagementSettings::default(),
        )),
        Arc::new(move |_: &str| environment.map(str::to_owned)),
        OidcResolver::default(),
    )
    .with_failure_policy(FailurePolicy::EnvironmentFallback)
}

#[tokio::test]
async fn manager_absence_is_final_even_with_environment_and_default() {
    assert_eq!(
        managed(Ok(None), Some("environment"))
            .get_secret("key", Some(Secret::Bool(true)))
            .await
            .unwrap(),
        None
    );
}

#[rstest::rstest]
#[case::capitalized_true("True", Some(Secret::Bool(true)), Some(true))]
#[case::parenthesized_false("(False)", Some(Secret::Bool(false)), Some(false))]
#[case::lowercase_true("true", None, Some(true))]
#[case::number("1", None, None)]
#[case::text("secret", None, None)]
#[tokio::test]
async fn manager_strings_are_coerced_like_literal_eval(
    #[case] input: &'static str,
    #[case] literal: Option<Secret>,
    #[case] boolean: Option<bool>,
) {
    let resolver = managed(Ok(Some(Secret::String(SecretValue::new(input)))), None);
    assert_eq!(
        resolver.get_secret("key", None).await.unwrap(),
        Some(
            literal
                .clone()
                .unwrap_or_else(|| Secret::String(SecretValue::new(input)))
        )
    );
    assert_eq!(
        resolver
            .get_secret_str("key", None)
            .await
            .unwrap()
            .as_ref()
            .map(SecretValue::expose),
        literal.is_none().then_some(input)
    );
    assert_eq!(
        resolver.get_secret_bool("key", None).await.unwrap(),
        boolean
    );
}

#[rstest::rstest]
#[case::boolean(Secret::Bool(false))]
#[case::object(Secret::from_json(serde_json::json!({"key": 1})))]
#[case::null(Secret::from_json(serde_json::Value::Null))]
#[tokio::test]
async fn non_string_manager_values_resolve_to_none(#[case] value: Secret) {
    let resolver = managed(Ok(Some(value)), Some("environment"));
    assert_eq!(resolver.get_secret("key", None).await.unwrap(), None);
    assert_eq!(resolver.get_secret_str("key", None).await.unwrap(), None);
    assert_eq!(resolver.get_secret_bool("key", None).await.unwrap(), None);
}

#[rstest::rstest]
#[case::capitalized_true(Some("True"), Some(Secret::Bool(true)))]
#[case::lowercase_true(Some("true"), Some(Secret::String(SecretValue::new("true"))))]
#[case::missing(None, None)]
#[tokio::test]
async fn manager_failures_fall_back_to_the_environment_like_literal_eval(
    #[case] environment: Option<&'static str>,
    #[case] expected: Option<Secret>,
) {
    assert_eq!(
        managed(Err(()), environment)
            .get_secret("key", Some(Secret::Bool(false)))
            .await
            .unwrap(),
        expected
    );
}

#[tokio::test]
async fn prefix_is_removed_once_and_resolved_from_environment() {
    let state = SecretManagerState::default();
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
    let resolver = resolver(Some("worker-value"));
    let result = tokio::spawn(async move { resolver.get_secret_str("KEY", None).await })
        .await
        .unwrap()
        .unwrap();
    assert_eq!(result.unwrap().expose(), "worker-value");
}

#[cfg(feature = "aws")]
mod aws {
    use super::*;
    use litellm_secrets::{AccessMode, aws::AwsSecretsManagerV2};
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
    #[case::missing(400, serde_json::json!({"__type":"ResourceNotFoundException"}))]
    #[case::denied(400, serde_json::json!({"__type":"AccessDeniedException"}))]
    #[case::malformed(200, serde_json::json!({}))]
    #[tokio::test]
    async fn read_results_follow_the_selected_failure_policy(
        #[case] status: u16,
        #[case] body: serde_json::Value,
        #[values(FailurePolicy::Propagate, FailurePolicy::EnvironmentFallback)]
        policy: FailurePolicy,
        #[values(None, Some("environment"))] environment: Option<&'static str>,
        #[values(None, Some("default"))] default: Option<&str>,
    ) {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(status).set_body_json(body.clone()))
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
        if body.get("__type").and_then(serde_json::Value::as_str)
            == Some("ResourceNotFoundException")
        {
            assert_eq!(result.unwrap(), None);
        } else if policy == FailurePolicy::EnvironmentFallback {
            assert_eq!(
                result.unwrap().as_ref().map(SecretValue::expose),
                environment
            );
        } else if let Some(default) = default {
            assert_eq!(result.unwrap().unwrap().expose(), default);
        } else {
            assert!(matches!(result, Err(Error::Aws(_))));
        }
    }

    #[rstest::rstest]
    #[case::boolean(serde_json::json!(false))]
    #[case::object(serde_json::json!({"key":1}))]
    #[case::null(serde_json::Value::Null)]
    #[case::string(serde_json::json!("true"))]
    #[tokio::test]
    async fn primary_secret_values_other_than_strings_resolve_to_none(
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
        let text = value.as_str();
        assert_eq!(
            resolver
                .get_secret("KEY", Some(Secret::Bool(true)))
                .await
                .unwrap(),
            text.map(|text| Secret::String(SecretValue::new(text)))
        );
        assert_eq!(
            resolver
                .get_secret_str("KEY", None)
                .await
                .unwrap()
                .as_ref()
                .map(SecretValue::expose),
            text
        );
        assert_eq!(
            resolver.get_secret_bool("KEY", None).await.unwrap(),
            text.map(|_| true)
        );
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
    use litellm_secrets::{
        FailurePolicy, KeyManagementSettings, SecretManager, google::GoogleSecretManager,
    };
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
        assert_eq!(result.unwrap(), None);
    } else {
        assert!(
            matches!(result, Err(Error::Google(litellm_secrets::google::Error::Status(actual))) if actual == status)
        );
    }
    let fallback = resolver
        .with_failure_policy(FailurePolicy::EnvironmentFallback)
        .get_secret_str("KEY", None)
        .await
        .unwrap();
    assert_eq!(
        fallback.as_ref().map(SecretValue::expose),
        (status != 404).then_some("environment")
    );
}
