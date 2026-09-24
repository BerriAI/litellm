#![cfg(all(
    feature = "aws",
    feature = "azure",
    feature = "google",
    feature = "hashicorp",
    feature = "cyberark"
))]
use std::sync::Arc;

use base64::{Engine, engine::general_purpose::STANDARD};
use litellm_core_utils::settings::Lookup;
use litellm_secrets::{
    KeyManagementSettings, SecretManager, SecretValue,
    aws::AwsSecretsManagerV2,
    azure::AzureKeyVault,
    cyberark::CyberArkSecretManager,
    get_secret_from_manager,
    google::GoogleSecretManager,
    hashicorp::{HashicorpVault, HashicorpVaultConfig},
};
use rstest::rstest;
use serde_json::json;
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{any, path},
};

#[derive(Clone, Copy, Debug)]
enum Provider {
    Aws,
    Azure,
    Google,
    Vault,
    Cyberark,
}

fn manager(provider: Provider, server: &MockServer) -> SecretManager {
    let environment: Arc<dyn Lookup + Send + Sync> = Arc::new({
        let address = server.uri();
        move |name: &str| match name {
            "HCP_VAULT_ADDR" | "AWS_BEDROCK_RUNTIME_ENDPOINT" => Some(address.clone()),
            "AWS_ACCESS_KEY_ID" | "AWS_SECRET_ACCESS_KEY" => Some("test".into()),
            "AZURE_AD_TOKEN" | "VERTEX_AI_API_KEY" | "HCP_VAULT_TOKEN" => Some("token".into()),
            _ => None,
        }
    });
    match provider {
        Provider::Aws => SecretManager::AwsSecretsManagerV2(
            AwsSecretsManagerV2::load_aws_secret_manager(
                Some(true),
                KeyManagementSettings {
                    aws_region_name: Some("us-east-1".into()),
                    ..Default::default()
                },
                environment,
            )
            .unwrap()
            .unwrap(),
        ),
        Provider::Azure => SecretManager::AzureKeyVault(
            AzureKeyVault::with_client(
                reqwest::Client::new(),
                server.uri().parse().unwrap(),
                environment,
            )
            .unwrap(),
        ),
        Provider::Google => SecretManager::GoogleSecretManager(
            GoogleSecretManager::with_client(
                reqwest::Client::new(),
                server.uri().parse().unwrap(),
                "project".into(),
                environment,
                None,
                false,
            )
            .unwrap(),
        ),
        Provider::Vault => SecretManager::HashicorpVault(
            HashicorpVault::from_config(
                HashicorpVaultConfig::from_environment(environment.as_ref()).unwrap(),
                true,
            )
            .unwrap(),
        ),
        Provider::Cyberark => SecretManager::Cyberark(CyberArkSecretManager::with_client(
            reqwest::Client::new(),
            server.uri().parse().unwrap(),
            "acct".into(),
            "admin".into(),
            SecretValue::new("key"),
            None,
        )),
    }
}

fn response(provider: Provider, value: &str) -> ResponseTemplate {
    match provider {
        Provider::Aws => ResponseTemplate::new(200).set_body_json(json!({"SecretString": value})),
        Provider::Azure => ResponseTemplate::new(200).set_body_json(json!({"value": value})),
        Provider::Google => ResponseTemplate::new(200)
            .set_body_json(json!({"payload": {"data": STANDARD.encode(value)}})),
        Provider::Vault => ResponseTemplate::new(200).set_body_json(json!({
            "data": {"data": {"key": value}, "metadata": {
                "created_time": "", "deletion_time": "", "custom_metadata": null,
                "destroyed": false, "version": 1
            }}, "lease_id": "", "lease_duration": 0, "renewable": false,
            "request_id": "", "warnings": null, "wrap_info": null
        })),
        Provider::Cyberark => ResponseTemplate::new(200).set_body_string(value),
    }
}

#[rstest]
#[case::aws(Provider::Aws)]
#[case::azure(Provider::Azure)]
#[case::google(Provider::Google)]
#[case::vault(Provider::Vault)]
#[case::cyberark(Provider::Cyberark)]
#[tokio::test]
async fn reads_preserve_values_and_distinguish_absence_from_failure(#[case] provider: Provider) {
    let server = MockServer::start().await;
    Mock::given(path("/authn/acct/admin/authenticate"))
        .respond_with(ResponseTemplate::new(200).set_body_string("token"))
        .with_priority(1)
        .mount(&server)
        .await;
    let manager = manager(provider, &server);
    let settings = KeyManagementSettings::default();
    for (name, value) in [
        ("TEXT", " value\n"),
        ("EMPTY", ""),
        ("BOOLEAN", "True"),
        ("JSON", "{\"key\":1}"),
    ] {
        let guard = Mock::given(any())
            .respond_with(response(provider, value))
            .with_priority(2)
            .mount_as_scoped(&server)
            .await;
        for _ in 0..2 {
            let result = get_secret_from_manager(&manager, name, &settings, &|_: &str| None)
                .await
                .unwrap()
                .unwrap();
            assert_eq!(result.as_str(), Some(value));
        }
        drop(guard);
    }
    let missing = match provider {
        Provider::Aws => ResponseTemplate::new(400)
            .set_body_json(json!({"__type": "ResourceNotFoundException", "Message": "missing"})),
        _ => ResponseTemplate::new(404).set_body_json(json!({"errors": ["missing"]})),
    };
    let guard = Mock::given(any())
        .respond_with(missing)
        .with_priority(2)
        .expect(2)
        .mount_as_scoped(&server)
        .await;
    for _ in 0..2 {
        assert!(
            get_secret_from_manager(&manager, "MISSING", &settings, &|_: &str| None)
                .await
                .unwrap()
                .is_none()
        );
    }
    drop(guard);
    let guard = Mock::given(any())
        .respond_with(ResponseTemplate::new(403).set_body_json(json!({"errors": ["forbidden"]})))
        .with_priority(2)
        .expect(2)
        .mount_as_scoped(&server)
        .await;
    for _ in 0..2 {
        assert!(
            get_secret_from_manager(&manager, "FAILED", &settings, &|_: &str| None)
                .await
                .is_err()
        );
    }
    drop(guard);
    let guard = Mock::given(any())
        .respond_with(response(provider, "recovered"))
        .with_priority(2)
        .expect(2)
        .mount_as_scoped(&server)
        .await;
    for name in ["MISSING", "FAILED"] {
        assert_eq!(
            get_secret_from_manager(&manager, name, &settings, &|_: &str| None)
                .await
                .unwrap()
                .unwrap()
                .as_str(),
            Some("recovered")
        );
    }
    drop(guard);
}

#[rstest]
#[case::aws(Provider::Aws)]
#[case::azure(Provider::Azure)]
#[case::google(Provider::Google)]
#[case::vault(Provider::Vault)]
#[case::cyberark(Provider::Cyberark)]
#[tokio::test]
async fn python_read_failures_preserve_provider_fallback_rules(
    #[case] provider: Provider,
    #[values(false, true)] missing: bool,
    #[values(None, Some("environment"), Some("True"), Some("true"))] environment_value: Option<
        &'static str,
    >,
) {
    use litellm_secrets::{OidcResolver, Secret, SecretManagerState, SecretResolver};
    let server = MockServer::start().await;
    Mock::given(path("/authn/acct/admin/authenticate"))
        .respond_with(ResponseTemplate::new(200).set_body_string("token"))
        .with_priority(1)
        .mount(&server)
        .await;
    let response = match (provider, missing) {
        (Provider::Aws, true) => {
            ResponseTemplate::new(400).set_body_json(json!({"__type":"ResourceNotFoundException"}))
        }
        (_, true) => ResponseTemplate::new(404).set_body_json(json!({"errors":["missing"]})),
        (_, false) => ResponseTemplate::new(403).set_body_json(json!({"errors":["forbidden"]})),
    };
    Mock::given(any())
        .respond_with(response)
        .with_priority(2)
        .expect(1)
        .mount(&server)
        .await;
    let resolver = SecretResolver::new_python_compatible(
        Arc::new(SecretManagerState::new(
            manager(provider, &server),
            KeyManagementSettings::default(),
        )),
        Arc::new(move |_: &str| environment_value.map(str::to_owned)),
        OidcResolver::default(),
    );
    let expected = if matches!(provider, Provider::Aws) {
        None
    } else {
        environment_value.map(|value| match value {
            "True" => Secret::Bool(true),
            value => Secret::String(SecretValue::new(value)),
        })
    };
    assert_eq!(
        resolver
            .get_secret("KEY", Some(Secret::String(SecretValue::new("default"))))
            .await
            .unwrap(),
        expected
    );
}
