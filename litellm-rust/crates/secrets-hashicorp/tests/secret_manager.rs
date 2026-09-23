use std::{collections::HashMap, sync::Arc, time::Duration};

use litellm_core_utils::settings::Lookup;
use litellm_secrets_hashicorp::{Error, HashicorpVault, HashicorpVaultConfig};
use litellm_secrets_types::{
    AwsOperationContext, BaseSecretManager, CyberarkOperationContext, HashicorpOperationContext,
    SecretOperationContext, SecretValue, SecretWriteContext,
};
use rstest::{fixture, rstest};
use serde::Deserialize;
use serde_json::json;
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{body_json, header, method, path},
};

fn config(server: &MockServer, values: &[(&str, &str)]) -> HashicorpVaultConfig {
    let mut environment_values: HashMap<String, String> = values
        .iter()
        .map(|(name, value)| ((*name).to_owned(), (*value).to_owned()))
        .collect();
    environment_values.insert("HCP_VAULT_ADDR".to_owned(), server.uri());
    let environment: Arc<dyn Lookup + Send + Sync> =
        Arc::new(move |name: &str| environment_values.get(name).cloned());
    HashicorpVaultConfig::from_environment(environment.as_ref()).unwrap()
}

fn manager(server: &MockServer, values: &[(&str, &str)]) -> HashicorpVault {
    HashicorpVault::from_config(config(server, values), true).unwrap()
}

fn auth_response(token: &str, lease_duration: u64) -> serde_json::Value {
    json!({
        "auth": {
            "client_token": token,
            "accessor": "",
            "policies": [],
            "token_policies": [],
            "metadata": null,
            "lease_duration": lease_duration,
            "renewable": false,
            "entity_id": "",
            "token_type": "service",
            "orphan": false
        },
        "lease_id": "",
        "lease_duration": lease_duration,
        "renewable": false,
        "request_id": "",
        "warnings": null,
        "wrap_info": null
    })
}

fn read_response(data: serde_json::Value) -> serde_json::Value {
    json!({
        "data": {
            "data": data,
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
    })
}

#[fixture]
fn token_values() -> Vec<(&'static str, &'static str)> {
    vec![("HCP_VAULT_TOKEN", "token")]
}

#[rstest]
#[tokio::test]
async fn token_reads_use_vault_headers_and_cache_values(token_values: Vec<(&str, &str)>) {
    let server: MockServer = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/secret/data/name"))
        .and(header("X-Vault-Token", "token"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(read_response(json!({"key": "value"}))),
        )
        .expect(1)
        .mount(&server)
        .await;
    let manager: HashicorpVault = manager(&server, &token_values);

    assert_eq!(
        manager
            .async_read_secret("name")
            .await
            .unwrap()
            .unwrap()
            .expose(),
        "value"
    );
    let requests = server.received_requests().await.unwrap();
    assert!(
        requests
            .iter()
            .all(|request| !request.headers.contains_key("X-Vault-Namespace"))
    );
    assert_eq!(
        manager
            .async_read_secret("name")
            .await
            .unwrap()
            .unwrap()
            .expose(),
        "value"
    );
}

#[rstest]
#[tokio::test]
async fn namespace_mount_and_prefix_are_sanitized_in_the_url() {
    let server: MockServer = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/kv-prod/data/virtual-keys/name"))
        .and(header("X-Vault-Namespace", "team-a"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(read_response(json!({"key": "value"}))),
        )
        .expect(1)
        .mount(&server)
        .await;
    let manager: HashicorpVault = manager(
        &server,
        &[
            ("HCP_VAULT_TOKEN", "token"),
            ("HCP_VAULT_SECRET_NAMESPACE", " /team-a/ "),
            ("HCP_VAULT_MOUNT_NAME", " /kv-prod/ "),
            ("HCP_VAULT_PATH_PREFIX", " /virtual-keys/ "),
        ],
    );

    let location = manager.secret_location("name").unwrap();
    assert_eq!(location.namespace.as_deref(), Some("team-a"));
    assert_eq!(location.mount, "kv-prod");
    assert_eq!(location.path, "virtual-keys/name");
    assert!(manager.async_read_secret("name").await.unwrap().is_some());
}

#[rstest]
fn trailing_address_slashes_are_removed() {
    let environment: Arc<dyn Lookup + Send + Sync> = Arc::new(|name: &str| match name {
        "HCP_VAULT_ADDR" => Some("http://vault.test:8200///".to_owned()),
        "HCP_VAULT_TOKEN" => Some("token".to_owned()),
        _ => None,
    });
    let config: HashicorpVaultConfig =
        HashicorpVaultConfig::from_environment(environment.as_ref()).unwrap();
    let manager: HashicorpVault = HashicorpVault::from_config(config, true).unwrap();

    assert_eq!(
        manager.secret_location("name").unwrap(),
        litellm_secrets_hashicorp::SecretLocation {
            namespace: None,
            mount: "secret".to_owned(),
            path: "name".to_owned(),
        }
    );
}

#[rstest]
#[case::negative("-1")]
#[case::not_a_number("not-a-number")]
fn invalid_refresh_intervals_are_rejected(#[case] value: &str) {
    let environment: Arc<dyn Lookup + Send + Sync> = Arc::new(move |name: &str| match name {
        "HCP_VAULT_REFRESH_INTERVAL" => Some(value.to_owned()),
        _ => None,
    });

    assert!(matches!(
        HashicorpVaultConfig::from_environment(environment.as_ref()),
        Err(Error::RefreshInterval)
    ));
}

#[rstest]
#[tokio::test]
async fn approle_login_uses_namespace_and_reuses_the_token() {
    let server: MockServer = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/auth/custom-approle/login"))
        .and(header("X-Vault-Namespace", "login-root"))
        .and(body_json(json!({"role_id": "role", "secret_id": "secret"})))
        .respond_with(ResponseTemplate::new(200).set_body_json(auth_response("login-token", 3600)))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/v1/secret/data/name"))
        .and(header("X-Vault-Token", "login-token"))
        .and(header("X-Vault-Namespace", "secret-root"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(read_response(json!({"key": "value"}))),
        )
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/v1/secret/data/name-2"))
        .respond_with(ResponseTemplate::new(404).set_body_json(json!({"errors": ["missing"]})))
        .expect(1)
        .mount(&server)
        .await;
    let manager: HashicorpVault = manager(
        &server,
        &[
            ("HCP_VAULT_APPROLE_ROLE_ID", "role"),
            ("HCP_VAULT_APPROLE_SECRET_ID", "secret"),
            ("HCP_VAULT_APPROLE_MOUNT_PATH", "custom-approle"),
            ("HCP_VAULT_NAMESPACE", "secret-root"),
            ("HCP_VAULT_LOGIN_NAMESPACE", "login-root"),
        ],
    );

    assert!(manager.async_read_secret("name").await.unwrap().is_some());
    assert!(manager.async_read_secret("name-2").await.unwrap().is_none());
}

#[rstest]
#[tokio::test]
async fn approle_tokens_expire_after_the_vault_lease() {
    let server: MockServer = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/auth/approle/login"))
        .respond_with(ResponseTemplate::new(200).set_body_json(auth_response("login-token", 1)))
        .expect(2)
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(read_response(json!({"key": "value"}))),
        )
        .expect(2)
        .mount(&server)
        .await;
    let manager: HashicorpVault = manager(
        &server,
        &[
            ("HCP_VAULT_APPROLE_ROLE_ID", "role"),
            ("HCP_VAULT_APPROLE_SECRET_ID", "secret"),
            ("HCP_VAULT_REFRESH_INTERVAL", "0"),
        ],
    );

    assert!(manager.async_read_secret("first").await.unwrap().is_some());
    tokio::time::sleep(Duration::from_secs(1) + Duration::from_millis(50)).await;
    assert!(manager.async_read_secret("second").await.unwrap().is_some());
}

#[rstest]
#[tokio::test]
async fn tls_login_posts_the_role_and_uses_the_client_identity() {
    let server: MockServer = MockServer::start().await;
    let directory: tempfile::TempDir = tempfile::tempdir().unwrap();
    let cert_path = directory.path().join("client.crt");
    let key_path = directory.path().join("client.key");
    std::fs::write(&cert_path, TEST_CERTIFICATE).unwrap();
    std::fs::write(&key_path, TEST_PRIVATE_KEY).unwrap();
    Mock::given(method("POST"))
        .and(path("/v1/auth/cert/login"))
        .and(header("X-Vault-Namespace", "login-ns"))
        .respond_with(ResponseTemplate::new(200).set_body_json(auth_response("cert-token", 0)))
        .expect(2)
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/v1/secret/data/name"))
        .and(header("X-Vault-Token", "cert-token"))
        .and(header("X-Vault-Namespace", "secret-ns"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(read_response(json!({"key": "value"}))),
        )
        .expect(2)
        .mount(&server)
        .await;
    let role_values: HashMap<String, String> = HashMap::from([
        ("HCP_VAULT_ADDR".to_owned(), server.uri()),
        (
            "HCP_VAULT_CLIENT_CERT".to_owned(),
            cert_path.to_str().unwrap().to_owned(),
        ),
        (
            "HCP_VAULT_CLIENT_KEY".to_owned(),
            key_path.to_str().unwrap().to_owned(),
        ),
        ("HCP_VAULT_CERT_ROLE".to_owned(), "vault-role".to_owned()),
        (
            "HCP_VAULT_LOGIN_NAMESPACE".to_owned(),
            "login-ns".to_owned(),
        ),
        (
            "HCP_VAULT_SECRET_NAMESPACE".to_owned(),
            "secret-ns".to_owned(),
        ),
    ]);
    let role_environment: Arc<dyn Lookup + Send + Sync> =
        Arc::new(move |name: &str| role_values.get(name).cloned());
    let role_manager: HashicorpVault = HashicorpVault::new(role_environment, true).unwrap();
    assert!(
        role_manager
            .async_read_secret("name")
            .await
            .unwrap()
            .is_some()
    );

    let no_role_values: HashMap<String, String> = HashMap::from([
        ("HCP_VAULT_ADDR".to_owned(), server.uri()),
        (
            "HCP_VAULT_CLIENT_CERT".to_owned(),
            cert_path.to_str().unwrap().to_owned(),
        ),
        (
            "HCP_VAULT_CLIENT_KEY".to_owned(),
            key_path.to_str().unwrap().to_owned(),
        ),
        (
            "HCP_VAULT_LOGIN_NAMESPACE".to_owned(),
            "login-ns".to_owned(),
        ),
        (
            "HCP_VAULT_SECRET_NAMESPACE".to_owned(),
            "secret-ns".to_owned(),
        ),
    ]);
    let no_role_environment: Arc<dyn Lookup + Send + Sync> =
        Arc::new(move |name: &str| no_role_values.get(name).cloned());
    let no_role_manager: HashicorpVault = HashicorpVault::new(no_role_environment, true).unwrap();
    assert!(
        no_role_manager
            .async_read_secret("name")
            .await
            .unwrap()
            .is_some()
    );
    let login_bodies: Vec<serde_json::Value> = server
        .received_requests()
        .await
        .unwrap()
        .iter()
        .filter(|request| request.method.as_str() == "POST")
        .map(|request| serde_json::from_slice(&request.body).unwrap())
        .collect();
    assert!(login_bodies.contains(&json!({"name": "vault-role"})));
    assert!(login_bodies.contains(&json!({})));
}

#[derive(Clone, Copy)]
enum ExpectedRead {
    Missing,
    Malformed,
    NonString,
}

#[rstest]
#[case::missing(404, json!({"errors": ["missing"]}), ExpectedRead::Missing)]
#[case::malformed(200, json!({"data": "invalid"}), ExpectedRead::Malformed)]
#[case::missing_key(200, json!({}), ExpectedRead::Missing)]
#[case::non_string(200, json!({"key": 1}), ExpectedRead::NonString)]
#[tokio::test]
async fn read_responses_distinguish_absence_and_malformed_payloads(
    token_values: Vec<(&str, &str)>,
    #[case] status: u16,
    #[case] body: serde_json::Value,
    #[case] expected: ExpectedRead,
) {
    let server: MockServer = MockServer::start().await;
    Mock::given(method("GET"))
        .respond_with(ResponseTemplate::new(status).set_body_json(
            if status == 200 && !matches!(expected, ExpectedRead::Malformed) {
                read_response(body)
            } else {
                body
            },
        ))
        .expect(1)
        .mount(&server)
        .await;
    let result: Result<Option<SecretValue>, Error> = manager(&server, &token_values)
        .async_read_secret("name")
        .await;
    match expected {
        ExpectedRead::Missing => assert!(result.unwrap().is_none()),
        ExpectedRead::Malformed => assert!(matches!(result, Err(Error::MalformedPayload))),
        ExpectedRead::NonString => assert!(matches!(result, Err(Error::NonStringValue))),
    }
}

#[rstest]
#[tokio::test]
async fn write_and_delete_invalidate_the_read_cache(token_values: Vec<(&str, &str)>) {
    let server: MockServer = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/secret/data/name"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(read_response(json!({"key": "value"}))),
        )
        .expect(2)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/secret/data/name"))
        .and(body_json(
            json!({"data": {"key": "updated", "description": "description"}}),
        ))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": {
                "created_time": "",
                "deletion_time": "",
                "custom_metadata": null,
                "destroyed": false,
                "version": 2
            },
            "lease_id": "",
            "lease_duration": 0,
            "renewable": false,
            "request_id": "",
            "warnings": null,
            "wrap_info": null
        })))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("DELETE"))
        .and(path("/v1/secret/data/name"))
        .respond_with(ResponseTemplate::new(204))
        .expect(1)
        .mount(&server)
        .await;
    let manager: HashicorpVault = manager(&server, &token_values);

    assert!(manager.async_read_secret("name").await.unwrap().is_some());
    assert!(
        manager
            .async_write_secret("name", SecretValue::new("updated"), Some("description"))
            .await
            .is_ok()
    );
    assert!(manager.async_read_secret("name").await.unwrap().is_some());
    manager.async_delete_secret("name").await.unwrap();
}

#[rstest]
#[tokio::test]
async fn base_manager_context_overrides_vault_location_and_data_key(
    token_values: Vec<(&str, &str)>,
) {
    let server: MockServer = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/alternate/data/managed/name"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(read_response(json!({"api_token": "value"}))),
        )
        .expect(2)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/alternate/data/managed/name"))
        .and(body_json(json!({
            "data": {"api_token": "updated", "description": "Managed key"}
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": {
                "created_time": "",
                "deletion_time": "",
                "custom_metadata": null,
                "destroyed": false,
                "version": 2
            },
            "lease_id": "",
            "lease_duration": 0,
            "renewable": false,
            "request_id": "",
            "warnings": null,
            "wrap_info": null
        })))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("DELETE"))
        .and(path("/v1/alternate/data/managed/name"))
        .respond_with(ResponseTemplate::new(204))
        .expect(1)
        .mount(&server)
        .await;
    let manager: HashicorpVault = manager(&server, &token_values);
    let operation = SecretOperationContext::Hashicorp(HashicorpOperationContext {
        mount: Some(" /alternate/ ".to_owned()),
        path_prefix: Some(" /managed/ ".to_owned()),
        data_key: Some("api_token".to_owned()),
        ..HashicorpOperationContext::default()
    });
    let write_context = SecretWriteContext {
        description: Some("Managed key".to_owned()),
        operation: operation.clone(),
        ..SecretWriteContext::default()
    };

    assert_eq!(
        BaseSecretManager::async_read_secret(&manager, "name", &operation)
            .await
            .unwrap()
            .unwrap()
            .expose(),
        "value"
    );
    BaseSecretManager::async_write_secret(
        &manager,
        "name",
        &SecretValue::new("updated"),
        &write_context,
    )
    .await
    .unwrap();
    assert!(
        BaseSecretManager::async_read_secret(&manager, "name", &operation)
            .await
            .unwrap()
            .is_some()
    );
    BaseSecretManager::async_delete_secret(&manager, "name", None, &operation)
        .await
        .unwrap();
}

#[rstest]
#[tokio::test]
async fn reads_cache_each_data_key_for_the_same_vault_path(token_values: Vec<(&str, &str)>) {
    let server: MockServer = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/secret/data/name"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(read_response(json!({
                "key": "primary",
                "alternate": "secondary"
            }))),
        )
        .expect(2)
        .mount(&server)
        .await;
    let manager: HashicorpVault = manager(&server, &token_values);
    let alternate = SecretOperationContext::Hashicorp(HashicorpOperationContext {
        data_key: Some("alternate".to_owned()),
        ..HashicorpOperationContext::default()
    });

    assert_eq!(
        manager
            .async_read_secret("name")
            .await
            .unwrap()
            .unwrap()
            .expose(),
        "primary"
    );
    assert_eq!(
        BaseSecretManager::async_read_secret(&manager, "name", &alternate)
            .await
            .unwrap()
            .unwrap()
            .expose(),
        "secondary"
    );
    assert_eq!(
        manager
            .async_read_secret("name")
            .await
            .unwrap()
            .unwrap()
            .expose(),
        "primary"
    );
}

#[rstest]
#[tokio::test]
async fn base_manager_context_timeout_limits_vault_io(token_values: Vec<(&str, &str)>) {
    let server: MockServer = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/secret/data/name"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_delay(Duration::from_millis(100))
                .set_body_json(read_response(json!({"key": "value"}))),
        )
        .expect(1)
        .mount(&server)
        .await;
    let manager: HashicorpVault = manager(&server, &token_values);
    let context = SecretOperationContext::Hashicorp(HashicorpOperationContext {
        timeout: Some(Duration::from_millis(10)),
        ..HashicorpOperationContext::default()
    });

    assert!(matches!(
        BaseSecretManager::async_read_secret(&manager, "name", &context).await,
        Err(Error::Timeout)
    ));
}

#[rstest]
#[case::aws(SecretOperationContext::Aws(AwsOperationContext::default()))]
#[case::cyberark(SecretOperationContext::Cyberark(CyberarkOperationContext::default()))]
#[tokio::test]
async fn foreign_contexts_cannot_access_vault_secrets(
    token_values: Vec<(&str, &str)>,
    #[case] context: SecretOperationContext,
    #[values(false, true)] cached: bool,
) {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/secret/data/name"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(read_response(json!({"key": "value"}))),
        )
        .expect(u64::from(cached))
        .mount(&server)
        .await;
    let manager = manager(&server, &token_values);
    if cached {
        assert!(manager.async_read_secret("name").await.unwrap().is_some());
    }
    assert!(matches!(
        BaseSecretManager::async_read_secret(&manager, "name", &context).await,
        Err(Error::InvalidOperationContext)
    ));
    assert!(matches!(
        BaseSecretManager::async_write_secret(
            &manager,
            "name",
            &SecretValue::new("replacement"),
            &SecretWriteContext {
                operation: context.clone(),
                ..SecretWriteContext::default()
            },
        )
        .await,
        Err(Error::InvalidOperationContext)
    ));
    assert!(matches!(
        BaseSecretManager::async_delete_secret(&manager, "name", None, &context).await,
        Err(Error::InvalidOperationContext)
    ));
    assert!(matches!(
        manager
            .async_rotate_secret_with_context(
                "name",
                "new",
                &SecretValue::new("replacement"),
                &context
            )
            .await,
        Err(Error::InvalidOperationContext)
    ));
    assert_eq!(
        server.received_requests().await.unwrap().len(),
        usize::from(cached)
    );
}

#[rstest]
#[tokio::test]
async fn rotation_applies_timeout_to_each_request(token_values: Vec<(&str, &str)>) {
    let server = MockServer::start().await;
    let timeout = Duration::from_secs(1);
    let delay = timeout / 2;
    Mock::given(method("GET"))
        .and(path("/v1/alternate/data/managed/current"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_delay(delay)
                .set_body_json(read_response(json!({"api_token": "original"}))),
        )
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/alternate/data/managed/new"))
        .and(body_json(json!({
            "data": {"api_token": "replacement", "description": "Rotated from current"}
        })))
        .respond_with(
            ResponseTemplate::new(200)
                .set_delay(delay)
                .set_body_json(json!({
                    "data": {
                        "created_time": "",
                        "deletion_time": "",
                        "custom_metadata": null,
                        "destroyed": false,
                        "version": 1
                    },
                    "lease_id": "",
                    "lease_duration": 0,
                    "renewable": false,
                    "request_id": "",
                    "warnings": null,
                    "wrap_info": null
                })),
        )
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/v1/alternate/data/managed/new"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_delay(delay)
                .set_body_json(read_response(json!({"api_token": "replacement"}))),
        )
        .mount(&server)
        .await;
    Mock::given(method("DELETE"))
        .and(path("/v1/alternate/data/managed/current"))
        .respond_with(ResponseTemplate::new(204).set_delay(delay))
        .mount(&server)
        .await;
    let manager = manager(&server, &token_values);
    let context = SecretOperationContext::Hashicorp(HashicorpOperationContext {
        timeout: Some(timeout),
        mount: Some("alternate".to_owned()),
        path_prefix: Some("managed".to_owned()),
        data_key: Some("api_token".to_owned()),
    });

    manager
        .async_rotate_secret_with_context(
            "current",
            "new",
            &SecretValue::new("replacement"),
            &context,
        )
        .await
        .unwrap();
    let requests = server.received_requests().await.unwrap();
    let operations: Vec<_> = requests
        .iter()
        .map(|request| (request.method.as_str(), request.url.path()))
        .collect();
    assert_eq!(
        operations,
        [
            ("GET", "/v1/alternate/data/managed/current"),
            ("POST", "/v1/alternate/data/managed/new"),
            ("GET", "/v1/alternate/data/managed/new"),
            ("DELETE", "/v1/alternate/data/managed/current"),
        ]
    );
}

#[rstest]
#[tokio::test]
async fn no_auth_and_invalid_names_fail_without_requests() {
    let server: MockServer = MockServer::start().await;
    let manager: HashicorpVault = manager(&server, &[]);

    assert!(matches!(
        manager.async_read_secret("name").await,
        Err(Error::NoAuthConfigured)
    ));
    assert!(matches!(
        manager.async_read_secret("../name").await,
        Err(Error::InvalidSecretName(_))
    ));
    assert!(server.received_requests().await.unwrap().is_empty());
}

#[rstest]
#[tokio::test]
async fn debug_output_redacts_authentication_values() {
    let server: MockServer = MockServer::start().await;
    let manager: HashicorpVault =
        HashicorpVault::from_config(config(&server, &[("HCP_VAULT_TOKEN", "token-value")]), true)
            .unwrap();
    let debug: String = format!("{manager:?}");
    assert!(!debug.contains("token-value"));
    assert!(!debug.contains("secret-id"));
}

#[derive(Deserialize)]
struct ParityCase {
    env: HashMap<String, String>,
    expected_secret_url: String,
    expected_login_url: Option<String>,
    expected_login_namespace: Option<String>,
    expected_secret_namespace: Option<String>,
    secret_name: String,
}

#[fixture]
fn parity_cases() -> Vec<ParityCase> {
    serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../tests/test_litellm/secret_managers/hashicorp_vault_parity.json"
    )))
    .unwrap()
}

#[rstest]
fn configuration_matches_python_parity_fixture(parity_cases: Vec<ParityCase>) {
    for case in parity_cases {
        let values: HashMap<String, String> = case.env.clone();
        let environment: Arc<dyn Lookup + Send + Sync> =
            Arc::new(move |name: &str| values.get(name).cloned());
        let config: HashicorpVaultConfig =
            HashicorpVaultConfig::from_environment(environment.as_ref()).unwrap();
        let manager: HashicorpVault = HashicorpVault::from_config(config.clone(), true).unwrap();
        let location = manager.secret_location(&case.secret_name).unwrap();
        let namespace = location
            .namespace
            .as_deref()
            .map(|namespace| format!("{namespace}/"))
            .unwrap_or_default();
        assert_eq!(
            format!(
                "{}/v1/{}{}/data/{}",
                config.address, namespace, location.mount, location.path
            ),
            case.expected_secret_url
        );
        let login_url = config.approle.as_ref().map_or_else(
            || {
                config
                    .tls_cert
                    .as_ref()
                    .map(|_| format!("{}/v1/auth/cert/login", config.address))
            },
            |approle| {
                Some(format!(
                    "{}/v1/auth/{}/login",
                    config.address, approle.mount_path
                ))
            },
        );
        assert_eq!(login_url, case.expected_login_url);
        assert_eq!(
            manager.config().login_namespace(),
            case.expected_login_namespace.as_deref()
        );
        assert_eq!(
            manager.config().secret_namespace(),
            case.expected_secret_namespace.as_deref()
        );
    }
}

#[rstest]
#[tokio::test]
#[ignore]
async fn live_vault_round_trip() {
    let environment: Arc<dyn Lookup + Send + Sync> =
        Arc::new(litellm_core_utils::settings::ProcessEnvironment);
    let manager: HashicorpVault = HashicorpVault::new(environment, true).unwrap();
    let name: String = std::env::var("LITELLM_VAULT_LIVE_SECRET_NAME").unwrap();
    let value: SecretValue = SecretValue::new("native-live-value");
    let location = manager.secret_location(&name).unwrap();
    println!(
        "native provenance: {} vaultrs {} {:?} {} {}",
        module_path!(),
        manager.config().address,
        location.namespace,
        location.mount,
        location.path
    );
    manager
        .async_write_secret(&name, value.clone(), None)
        .await
        .unwrap();
    assert_eq!(
        manager.async_read_secret(&name).await.unwrap().unwrap(),
        value
    );
    manager.async_delete_secret(&name).await.unwrap();
    assert!(manager.async_read_secret(&name).await.unwrap().is_none());
}

const TEST_CERTIFICATE: &str = "-----BEGIN CERTIFICATE-----
MIIDDzCCAfegAwIBAgIUeMzLFLM/mRbPGbNAew5N2UTscocwDQYJKoZIhvcNAQEL
BQAwFzEVMBMGA1UEAwwMbGl0ZWxsbS10ZXN0MB4XDTI2MDkyMTIwMjA1OVoXDTI2
MDkyMjIwMjA1OVowFzEVMBMGA1UEAwwMbGl0ZWxsbS10ZXN0MIIBIjANBgkqhkiG
9w0BAQEFAAOCAQ8AMIIBCgKCAQEAveYoSUJXybmkHmQsBfhBcv2Ob5Oy8ejZu+B3
vTnrPumW4ANi1XXKBSazRGB3fEtAgr+3KhKeHaSKEQeBwJkAEBfdmQv0tpXICwHs
1kFNtU0owy54HVW5/ia+LMszsFcPzVIoMnbUOuiKr9RaV7P+IEFzILPBVuV4DoYH
yocjD3+9QNqokWgNL8LK37JijmNEFVaKFz0X6SyL2VRDlfPWTEBK52Gp/pvDgA6G
eTSfyI+kCm9h5ECTYUAtmatk9WPVS8sWOqV1EXVanFyYBU+mDxoywAS1/6CHeIPh
bNmCOZjPoO9qWBJ7ZyGhOconBigXY8qnlXymev+44IPHrx4urwIDAQABo1MwUTAd
BgNVHQ4EFgQUvaZrZ6HKtbr3ekeZmgy4b5Pq95QwHwYDVR0jBBgwFoAUvaZrZ6HK
tbr3ekeZmgy4b5Pq95QwDwYDVR0TAQH/BAUwAwEB/zANBgkqhkiG9w0BAQsFAAOC
AQEAEejrD8d1qDxW55XxQ4IC31rufoEvDV955jyvh2kALPaN/i5oWsBGI+UAQZna
aaoQXwzlmHrtDUBWl0LztVTUamIleUep2+PLLauqqt43vxppxMX8Jn2mnPO20YE/
hIzGx0jN/LBG8PDyLSvHdlgjP9ofA4Vg4rTQugdXRgOvlCE/epnH/MADcg9KYJtJ
C1RObCIkL3LcdUbjStJRCY/U/FeWcgyncEPz95OFDkbrlNDajb6o6CkYfouqvhTc
8XlgjjAVKIbAbRgbVu3elsquuFM97x2DzWDjkrMNmDt1FJ9ubK36gL6B3o0UMaoQ
00R7x/eqvH+EkWa/2ekW9lpleQ==
-----END CERTIFICATE-----
";

const TEST_PRIVATE_KEY: &str = "-----BEGIN PRIVATE KEY-----
MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC95ihJQlfJuaQe
ZCwF+EFy/Y5vk7Lx6Nm74He9Oes+6ZbgA2LVdcoFJrNEYHd8S0CCv7cqEp4dpIoR
B4HAmQAQF92ZC/S2lcgLAezWQU21TSjDLngdVbn+Jr4syzOwVw/NUigydtQ66Iqv
1FpXs/4gQXMgs8FW5XgOhgfKhyMPf71A2qiRaA0vwsrfsmKOY0QVVooXPRfpLIvZ
VEOV89ZMQErnYan+m8OADoZ5NJ/Ij6QKb2HkQJNhQC2Zq2T1Y9VLyxY6pXURdVqc
XJgFT6YPGjLABLX/oId4g+Fs2YI5mM+g72pYEntnIaE5yicGKBdjyqeVfKZ6/7jg
g8evHi6vAgMBAAECggEAGdJjlP6b8Fa5bdaCM/ebcrbuuNZVJVbb0JPHxGfNSLs7
pE9hj5QaOdQW2Uviw3h6F61ZCzQH4xD+Iy2po5ZKb2XHYKnDB1bboj+LRGER337T
9aJqe9at2VTMVEv3Rdm40NsEk0QcPLxlK16NQFK90gYEUSSQPDAswJDSG2R/zHn+
vADI907mW/goEJHeLn8PWGlNlSiR6x+5JJtq+GXCzUzVvJYQSCLGxCSl2x2H+0g7
NhFI0zPpdzNmO/h+yhzaFb6Rp5U8+ZsnZ3qYjQ/03gw1myTDKJt1YaO9JvArnNYX
hcJQQ8Rt0bHhcrZA16bBOpqZlo5pKCicwI/netgN8QKBgQDcFz7AzdJ26sMSV32V
rwrMgIoggt8qDjO1ARwqW35A1TIge0FoW4M4KpsXQGGfT341uU1esXEcyZ/1L/5X
3ql2gX4DbOYLZLWYzZGR2hq33oi8HkhN98QrEwL9emSH8NqYX3Xxja3PrmCrSYJe
Zbnd9TIm2XkxyMoyXJu6M/QvnwKBgQDc4dzqTbxoGEGa5MuJoGmMwPnqgdG9UM5J
eExVnh7osxc2sOdsiPeRjjQTxs9v2kJwctC359OJoo9yGaaJeSghU4LEWJo1sqnA
fzSCLammYvtVAtniyNv5Mxk/6Uimi4NNDKaAKB+m4K2uSn3U9AmY7KPYMGaSbS9W
XSnobjxm8QKBgC8bPpAvvWs8ZhIn7bY659nLbUT2HeO3dHO6UBf0yzn/J6JyHxbB
93zvCZDZc8uQTRgcmCW7XtVlhjoJUqvl+Wlm39zF0xr/LCsPXKfWAb/2/lcdOCaP
8Emz4QD10EyUTYUtcWYJB/mafhBLRH8F0Nlj4J8WDu2L51MOJTqeYhZLAoGAWffN
icocAbJPlo22sdoa4+/+W5yBF8GAJMDRJtZ+9H1t6SLpQHYRkMIBSETkXUTjZvX9
Ocs9iIQkNW9pO/mTdO+VBfCo71JUfknR02xR+6m5gYjlws/ZeYlssXGN2/hbhNiw
QOcW7Vv6olFJK6Iy/oz0t6wPO3kpnN3Zogi0paECgYEAwo44M1DdYCtV0snhmYM9
5u0mPfYt5P2SVLXyUbr+vFTfrTL/WKnXIJgbsnj3Gvf+GIZv9tKcXhSNmEHQCYX4
X3w9iTPddCHuvZ1fpufi2TyArJh0OkoNtLXJHTKrHjf2N+61AQzFiv5WieJrdE+H
qr32PTUuVGPyO9LyTY4/RL0=
-----END PRIVATE KEY-----
";
