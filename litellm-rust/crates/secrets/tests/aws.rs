#![cfg(feature = "aws")]

use std::sync::Arc;

use litellm_secrets::{
    AccessMode, Error, FailurePolicy, KeyManagementSettings, OidcResolver, Secret, SecretManager,
    SecretManagerState, SecretResolver, SecretValue, aws::AwsSecretsManagerV2,
    secret_manager_would_be_consulted,
};
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
#[case::missing(400, serde_json::json!({"__type":"ResourceNotFoundException"}), None)]
#[case::denied(400, serde_json::json!({"__type":"AccessDeniedException"}), None)]
#[case::malformed(200, serde_json::json!({}), None)]
#[case::invalid_primary(200, serde_json::json!({"SecretString":"not-json"}), Some("primary"))]
#[tokio::test]
async fn read_results_follow_the_selected_failure_policy(
    #[case] status: u16,
    #[case] body: serde_json::Value,
    #[case] primary_secret_name: Option<&str>,
    #[values(FailurePolicy::Propagate, FailurePolicy::EnvironmentFallback)] policy: FailurePolicy,
    #[values(None, Some("environment"))] environment: Option<&'static str>,
    #[values(None, Some("default"))] default: Option<&str>,
) {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(status).set_body_json(body.clone()))
        .expect(1)
        .mount(&server)
        .await;
    let resolver = SecretResolver::new_python_compatible(
        Arc::new(state(
            &server,
            KeyManagementSettings {
                primary_secret_name: primary_secret_name.map(str::to_owned),
                ..Default::default()
            },
        )),
        Arc::new(move |_: &str| environment.map(str::to_owned)),
        OidcResolver::new(litellm_http::Client::plain_for_test()),
    )
    .with_failure_policy(policy);
    let result = resolver
        .get_secret_str("KEY", default.map(SecretValue::new))
        .await;
    if primary_secret_name.is_none() {
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
    let resolver = SecretResolver::new_python_compatible(
        Arc::new(state(&server, settings)),
        Arc::new(|_: &str| Some("fallback".into())),
        OidcResolver::new(litellm_http::Client::plain_for_test()),
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
            ResponseTemplate::new(200).set_body_json(serde_json::json!({"SecretString":"remote"})),
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
    let resolver = SecretResolver::new_python_compatible(
        Arc::new(state),
        Arc::new(|_: &str| Some("environment".into())),
        OidcResolver::new(litellm_http::Client::plain_for_test()),
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

#[tokio::test]
async fn aws_handler_reads_ciphertext_decodes_trims_and_redacts() {
    use aws_sdk_kms::{
        Client,
        config::{BehaviorVersion, Credentials, Region},
    };
    use base64::{Engine, engine::general_purpose::STANDARD};
    use litellm_secrets::{
        Error, KeyManagementSettings, SecretManager, aws::AwsKms, get_secret_from_manager,
    };
    use wiremock::{Mock, MockServer, ResponseTemplate, matchers::body_json};

    let server = MockServer::start().await;
    Mock::given(body_json(
        serde_json::json!({"CiphertextBlob": STANDARD.encode("encrypted")}),
    ))
    .respond_with(
        ResponseTemplate::new(200)
            .set_body_json(serde_json::json!({"Plaintext":STANDARD.encode(" value\n")})),
    )
    .expect(1)
    .mount(&server)
    .await;
    let client = Client::from_conf(
        aws_sdk_kms::Config::builder()
            .behavior_version(BehaviorVersion::latest())
            .region(Region::new("us-east-1"))
            .credentials_provider(Credentials::new("test", "test", None, None, "test"))
            .endpoint_url(server.uri())
            .build(),
    );
    let manager = SecretManager::AwsKms(AwsKms::new(client));
    let settings = KeyManagementSettings::default();
    let value = get_secret_from_manager(&manager, "KEY", &settings, &|name: &str| {
        assert_eq!(name, "KEY");
        Some(format!(" {}\n", STANDARD.encode("encrypted")))
    })
    .await
    .unwrap()
    .unwrap();
    assert_eq!(value.as_str(), Some("value"));
    assert!(!format!("{value:?}").contains("value"));
    assert!(matches!(
        get_secret_from_manager(&manager, "KEY", &settings, &|_: &str| None).await,
        Err(Error::MissingCiphertext)
    ));
    assert!(matches!(
        get_secret_from_manager(&manager, "KEY", &settings, &|_: &str| Some("abc".into())).await,
        Err(Error::InvalidCiphertext)
    ));
}
