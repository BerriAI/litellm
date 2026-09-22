#[cfg(feature = "aws")]
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

#[cfg(feature = "google")]
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
#[cfg(feature = "hashicorp")]
#[tokio::test]
async fn hashicorp_handler_resolves_found_missing_and_failed_values() {
    use std::sync::Arc;

    use litellm_core_utils::settings::Lookup;
    use litellm_secrets::{
        Error, FailurePolicy, KeyManagementSettings, SecretManager, SecretManagerState,
        SecretResolver, hashicorp::HashicorpVault, hashicorp::HashicorpVaultConfig,
    };
    use wiremock::{
        Mock, MockServer, ResponseTemplate,
        matchers::{method, path},
    };

    let found_server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/secret/data/KEY"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "data": {
                "data": {"key": "remote"},
                "metadata": {
                    "created_time": "",
                    "deletion_time": "",
                    "custom_metadata": null,
                    "destroyed": false,
                    "version": 1
                }
            },
            "lease_id": "",
            "lease_duration": 0,
            "renewable": false,
            "request_id": "",
            "warnings": null,
            "wrap_info": null
        })))
        .mount(&found_server)
        .await;
    let found_environment: Arc<dyn Lookup + Send + Sync> = Arc::new({
        let address = found_server.uri();
        move |name: &str| match name {
            "HCP_VAULT_ADDR" => Some(address.clone()),
            "HCP_VAULT_TOKEN" => Some("token".into()),
            _ => None,
        }
    });
    let found_config = HashicorpVaultConfig::from_environment(found_environment.as_ref()).unwrap();
    let found_manager = HashicorpVault::from_config(found_config, true).unwrap();
    let found_resolver = SecretResolver::new(
        Arc::new(SecretManagerState::new(
            SecretManager::HashicorpVault(found_manager),
            KeyManagementSettings {
                hosted_keys: Some(vec!["KEY".into()]),
                ..Default::default()
            },
        )),
        Arc::new(|_: &str| None),
        litellm_secrets::OidcResolver::default(),
    );
    assert_eq!(
        found_resolver
            .get_secret_str("KEY", None)
            .await
            .unwrap()
            .unwrap()
            .expose(),
        "remote"
    );

    let missing_server = MockServer::start().await;
    Mock::given(method("GET"))
        .respond_with(
            ResponseTemplate::new(404).set_body_json(serde_json::json!({"errors": ["missing"]})),
        )
        .mount(&missing_server)
        .await;
    let missing_environment: Arc<dyn Lookup + Send + Sync> = Arc::new({
        let address = missing_server.uri();
        move |name: &str| match name {
            "HCP_VAULT_ADDR" => Some(address.clone()),
            "HCP_VAULT_TOKEN" => Some("token".into()),
            _ => None,
        }
    });
    let missing_config =
        HashicorpVaultConfig::from_environment(missing_environment.as_ref()).unwrap();
    let missing_manager = HashicorpVault::from_config(missing_config, true).unwrap();
    let missing_state = SecretManagerState::new(
        SecretManager::HashicorpVault(missing_manager),
        KeyManagementSettings {
            hosted_keys: Some(vec!["KEY".into()]),
            ..Default::default()
        },
    );
    let missing = litellm_secrets::get_secret_from_manager(
        missing_state.backend().unwrap(),
        "KEY",
        missing_state.settings().unwrap(),
        &|_: &str| None,
    )
    .await
    .unwrap();
    assert!(missing.is_none());

    let failed_server = MockServer::start().await;
    Mock::given(method("GET"))
        .respond_with(
            ResponseTemplate::new(500).set_body_json(serde_json::json!({"errors": ["failed"]})),
        )
        .mount(&failed_server)
        .await;
    let failed_environment: Arc<dyn Lookup + Send + Sync> = Arc::new({
        let address = failed_server.uri();
        move |name: &str| match name {
            "HCP_VAULT_ADDR" => Some(address.clone()),
            "HCP_VAULT_TOKEN" => Some("token".into()),
            _ => None,
        }
    });
    let failed_config =
        HashicorpVaultConfig::from_environment(failed_environment.as_ref()).unwrap();
    let failed_manager = HashicorpVault::from_config(failed_config, true).unwrap();
    let failed_state = SecretManagerState::new(
        SecretManager::HashicorpVault(failed_manager),
        KeyManagementSettings {
            hosted_keys: Some(vec!["KEY".into()]),
            ..Default::default()
        },
    );
    let failed_resolver = SecretResolver::new(
        Arc::new(failed_state),
        Arc::new(|_: &str| None),
        litellm_secrets::OidcResolver::default(),
    )
    .with_failure_policy(FailurePolicy::Propagate);
    assert!(matches!(
        failed_resolver.get_secret_str("KEY", None).await,
        Err(Error::Hashicorp(
            litellm_secrets::hashicorp::Error::Status { status: 500 }
        ))
    ));
}

#[cfg(feature = "azure")]
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

#[cfg(feature = "cyberark")]
#[tokio::test]
async fn cyberark_handler_reads_values_and_surfaces_errors() {
    use std::time::Duration;

    use litellm_secrets::{
        Error, KeyManagementSettings, SecretManager, SecretValue, cyberark::CyberArkSecretManager,
        get_secret_from_manager,
    };
    use wiremock::{
        Mock, MockServer, ResponseTemplate,
        matchers::{body_string, path},
    };

    let server = MockServer::start().await;
    Mock::given(path("/authn/acct/admin/authenticate"))
        .and(body_string("k3y"))
        .respond_with(ResponseTemplate::new(200).set_body_string("token"))
        .mount(&server)
        .await;
    Mock::given(path("/secrets/acct/variable/KEY"))
        .respond_with(ResponseTemplate::new(200).set_body_string("value"))
        .mount(&server)
        .await;
    let manager = SecretManager::Cyberark(CyberArkSecretManager::with_client(
        reqwest::Client::new(),
        server.uri().parse().unwrap(),
        "acct".into(),
        "admin".into(),
        SecretValue::new("k3y"),
        Some(Duration::from_secs(60)),
    ));
    assert_eq!(
        manager.system(),
        litellm_secrets::KeyManagementSystem::Cyberark
    );
    let settings = KeyManagementSettings::default();
    let value = get_secret_from_manager(&manager, "KEY", &settings, &|_: &str| None)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(value.as_str(), Some("value"));

    Mock::given(path("/secrets/acct/variable/ERROR"))
        .respond_with(ResponseTemplate::new(500))
        .mount(&server)
        .await;
    assert!(matches!(
        get_secret_from_manager(&manager, "ERROR", &settings, &|_: &str| None).await,
        Err(Error::Cyberark(_))
    ));
}
