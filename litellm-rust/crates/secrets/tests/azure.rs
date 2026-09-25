#![cfg(feature = "azure")]

#[tokio::test]
async fn azure_handler_reads_missing_and_failed_secrets() {
    use litellm_secrets::{
        Error, KeyManagementSettings, KeyManagementSystem, SecretManager, azure::AzureKeyVault,
        get_secret_from_manager,
    };
    use wiremock::{
        Mock, MockServer, ResponseTemplate,
        matchers::{path, query_param},
    };

    let server = MockServer::start().await;
    Mock::given(path("/secrets/KEY"))
        .and(query_param("api-version", "7.4"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(serde_json::json!({"value": "value"})),
        )
        .expect(1)
        .mount(&server)
        .await;
    let manager = SecretManager::AzureKeyVault(
        AzureKeyVault::with_client(
            reqwest::Client::new(),
            server.uri().parse().unwrap(),
            std::sync::Arc::new(|name: &str| (name == "AZURE_AD_TOKEN").then(|| "fake".to_owned())),
        )
        .unwrap(),
    );
    assert_eq!(manager.system(), KeyManagementSystem::AzureKeyVault);
    let settings = KeyManagementSettings::default();
    let value = get_secret_from_manager(&manager, "KEY", &settings, &|_: &str| None)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(value.as_str(), Some("value"));

    let not_found = Mock::given(path("/secrets/MISSING"))
        .respond_with(ResponseTemplate::new(404))
        .expect(1)
        .mount_as_scoped(&server)
        .await;
    assert_eq!(
        get_secret_from_manager(&manager, "MISSING", &settings, &|_: &str| None)
            .await
            .unwrap(),
        None
    );
    drop(not_found);

    Mock::given(path("/secrets/FAILED"))
        .respond_with(ResponseTemplate::new(500))
        .expect(1)
        .mount(&server)
        .await;
    assert!(matches!(
        get_secret_from_manager(&manager, "FAILED", &settings, &|_: &str| None).await,
        Err(Error::Azure(_))
    ));
}

#[rstest::rstest]
#[case::null(serde_json::json!({"value":null}))]
#[case::absent(serde_json::json!({}))]
#[case::empty(serde_json::json!({"value":""}))]
#[tokio::test]
async fn successful_azure_responses_do_not_fall_back_when_the_value_is_empty_or_null(
    #[case] body: serde_json::Value,
) {
    use litellm_secrets::{
        OidcResolver, SecretManager, SecretManagerState, SecretResolver, SecretValue,
        azure::AzureKeyVault,
    };
    use std::sync::Arc;
    use wiremock::{Mock, MockServer, ResponseTemplate, matchers::any};
    let server = MockServer::start().await;
    Mock::given(any())
        .respond_with(ResponseTemplate::new(200).set_body_json(body.clone()))
        .expect(1)
        .mount(&server)
        .await;
    let manager = AzureKeyVault::with_client(
        reqwest::Client::new(),
        server.uri().parse().unwrap(),
        Arc::new(|name: &str| (name == "AZURE_AD_TOKEN").then(|| "token".into())),
    )
    .unwrap();
    let resolver = SecretResolver::new_python_compatible(
        Arc::new(SecretManagerState::new(
            SecretManager::AzureKeyVault(manager),
            Default::default(),
        )),
        Arc::new(|_: &str| Some("environment".into())),
        OidcResolver::new(litellm_http::Client::plain_for_test()),
    );
    assert_eq!(
        resolver
            .get_secret_str("KEY", Some(SecretValue::new("default")))
            .await
            .unwrap()
            .as_ref()
            .map(SecretValue::expose),
        body.get("value").and_then(serde_json::Value::as_str)
    );
}
