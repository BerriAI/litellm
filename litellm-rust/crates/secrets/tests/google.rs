#![cfg(feature = "google")]

use std::sync::Arc;

#[rstest::rstest]
#[case::missing(404)]
#[case::failure(503)]
#[tokio::test]
async fn google_resolver_distinguishes_absence_from_failure(#[case] status: u16) {
    use litellm_secrets::{
        Error, FailurePolicy, KeyManagementSettings, OidcResolver, SecretManager,
        SecretManagerState, SecretResolver, SecretValue, google::GoogleSecretManager,
    };
    use wiremock::{Mock, MockServer, ResponseTemplate, matchers::method};
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .respond_with(ResponseTemplate::new(status))
        .expect(1)
        .mount(&server)
        .await;
    let environment: Arc<dyn litellm_core_utils::settings::Lookup + Send + Sync> =
        Arc::new(|name: &str| match name {
            "VERTEX_AI_API_KEY" => Some("token".into()),
            "KEY" => Some("environment".into()),
            _ => None,
        });
    let manager = GoogleSecretManager::with_client(
        litellm_http::Client::plain_for_test(),
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
    let resolver = SecretResolver::new_python_compatible(
        Arc::new(state),
        environment,
        OidcResolver::new(litellm_http::Client::plain_for_test()),
    )
    .with_failure_policy(FailurePolicy::Propagate);
    let result = resolver.get_secret_str("KEY", None).await;
    if status == 404 {
        assert!(matches!(result, Err(Error::ManagedSecretMissing)));
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
        Some("environment")
    );
}

#[tokio::test]
async fn google_handler_requires_canonical_base64_and_preserves_plaintext_whitespace() {
    use base64::{Engine, engine::general_purpose::STANDARD};
    use google_cloud_kms_v1::client::KeyManagementService;
    use litellm_secrets::{
        Error, KeyManagementSettings, SecretManager, get_secret_from_manager, google::GoogleKms,
    };
    use wiremock::{
        Mock, MockServer, ResponseTemplate,
        matchers::{body_json, path},
    };

    let server = MockServer::start().await;
    let resource = "projects/project/locations/global/keyRings/ring/cryptoKeys/key";
    Mock::given(path(format!("/v1/{resource}:decrypt")))
        .and(body_json(
            serde_json::json!({"ciphertext":STANDARD.encode("encrypted")}),
        ))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(serde_json::json!({"plaintext":STANDARD.encode(" value\n")})),
        )
        .expect(1)
        .mount(&server)
        .await;
    let client = KeyManagementService::builder()
        .with_endpoint(server.uri())
        .with_credentials(google_cloud_auth::credentials::anonymous::Builder::new().build())
        .build()
        .await
        .unwrap();
    let manager = SecretManager::GoogleKms(GoogleKms::new(client, resource.into()));
    let settings = KeyManagementSettings::default();
    let value = get_secret_from_manager(&manager, "KEY", &settings, &|_: &str| {
        Some(STANDARD.encode("encrypted"))
    })
    .await
    .unwrap()
    .unwrap();
    assert_eq!(value.as_str(), Some(" value\n"));
    assert!(matches!(
        get_secret_from_manager(&manager, "KEY", &settings, &|_: &str| Some(format!(
            " {}",
            STANDARD.encode("encrypted")
        )))
        .await,
        Err(Error::InvalidCiphertext)
    ));
    assert!(matches!(
        get_secret_from_manager(&manager, "KEY", &settings, &|_: &str| None).await,
        Err(Error::MissingCiphertext)
    ));
}
