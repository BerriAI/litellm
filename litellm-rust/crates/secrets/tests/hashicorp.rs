#![cfg(feature = "hashicorp")]

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
    let found_resolver = SecretResolver::new_python_compatible(
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
    let failed_resolver = SecretResolver::new_python_compatible(
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
