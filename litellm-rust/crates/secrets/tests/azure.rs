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
