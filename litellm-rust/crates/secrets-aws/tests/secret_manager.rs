use std::{
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

use aws_sdk_secretsmanager::{
    Client,
    config::{BehaviorVersion, Credentials, Region, retry::RetryConfig},
};
use litellm_secrets_aws::{AwsSecretsManagerV2, Error, RotationResponse};
use litellm_secrets_types::{
    AwsOperationContext, BaseSecretManager, KeyManagementSettings, SecretOperationContext,
    SecretValue, SecretWriteContext,
};
use rstest::{fixture, rstest};
use serde_json::json;
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{body_partial_json, header},
};

fn manager(server: &MockServer, settings: KeyManagementSettings) -> AwsSecretsManagerV2 {
    let client = Client::from_conf(
        aws_sdk_secretsmanager::Config::builder()
            .behavior_version(BehaviorVersion::latest())
            .region(Region::new("us-east-1"))
            .credentials_provider(Credentials::new("test", "test", None, None, "test"))
            .endpoint_url(server.uri())
            .retry_config(RetryConfig::disabled())
            .build(),
    );
    AwsSecretsManagerV2::new(client, (&settings).into())
}

fn loaded_manager(server: &MockServer) -> AwsSecretsManagerV2 {
    let endpoint_url = server.uri();
    let environment: Arc<dyn litellm_core_utils::settings::Lookup + Send + Sync> =
        Arc::new(move |name: &str| match name {
            "AWS_BEDROCK_RUNTIME_ENDPOINT" => Some(endpoint_url.clone()),
            "AWS_ACCESS_KEY_ID" => Some("test".into()),
            "AWS_SECRET_ACCESS_KEY" => Some("test".into()),
            _ => None,
        });
    AwsSecretsManagerV2::load_aws_secret_manager(
        Some(true),
        KeyManagementSettings {
            aws_region_name: Some("us-east-1".into()),
            ..Default::default()
        },
        environment,
    )
    .unwrap()
    .unwrap()
}

#[fixture]
fn default_settings() -> KeyManagementSettings {
    KeyManagementSettings::default()
}

#[rstest]
#[case::string_value("KEY", Some("value"))]
#[case::missing_value("missing", None)]
#[case::non_string_value("BOOL", None)]
#[tokio::test]
async fn primary_lookup_preserves_read_semantics(
    default_settings: KeyManagementSettings,
    #[case] name: &str,
    #[case] expected: Option<&str>,
) {
    let server = MockServer::start().await;
    Mock::given(header("x-amz-target", "secretsmanager.GetSecretValue"))
        .and(body_partial_json(json!({"SecretId":"primary"})))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(
                json!({"SecretString":json!({"KEY":"value", "BOOL":true}).to_string()}),
            ),
        )
        .expect(1)
        .mount(&server)
        .await;
    let manager = manager(&server, default_settings);
    assert_eq!(
        manager
            .read_secret_for_resolver(name, Some("primary"), &|_: &str| None)
            .await
            .unwrap()
            .and_then(|v| v.as_str().map(str::to_owned))
            .as_deref(),
        expected
    );
}

#[rstest]
#[case::access_key("AWS_ACCESS_KEY_ID")]
#[case::secret_access_key("AWS_SECRET_ACCESS_KEY")]
#[case::region_name("AWS_REGION_NAME")]
#[case::region("AWS_REGION")]
#[case::bedrock_endpoint("AWS_BEDROCK_RUNTIME_ENDPOINT")]
#[tokio::test]
async fn bootstrap_keys_bypass_primary_lookup(
    default_settings: KeyManagementSettings,
    #[case] name: &str,
) {
    let server = MockServer::start().await;
    let manager = manager(&server, default_settings);
    assert_eq!(
        manager
            .read_secret_for_resolver(name, Some("primary"), &|_: &str| Some("bootstrap".into()))
            .await
            .unwrap()
            .unwrap()
            .as_str()
            .unwrap(),
        "bootstrap"
    );
}

#[rstest]
#[tokio::test]
async fn failed_read_returns_none_but_invalid_primary_json_is_an_error(
    default_settings: KeyManagementSettings,
) {
    let server = MockServer::start().await;
    Mock::given(body_partial_json(json!({"SecretId":"missing"})))
        .respond_with(
            ResponseTemplate::new(400).set_body_json(json!({"__type":"ResourceNotFoundException"})),
        )
        .mount(&server)
        .await;
    Mock::given(body_partial_json(json!({"SecretId":"invalid"})))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"SecretString":"not-json"})))
        .mount(&server)
        .await;
    Mock::given(body_partial_json(json!({"SecretId":"no-string"})))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"Name":"no-string"})))
        .mount(&server)
        .await;
    let manager = manager(&server, default_settings);
    assert!(
        manager
            .async_read_secret("missing")
            .await
            .unwrap()
            .is_none()
    );
    assert!(matches!(
        manager
            .read_secret_for_resolver("KEY", Some("invalid"), &|_: &str| None)
            .await,
        Err(Error::PrimarySecret)
    ));
    assert!(matches!(
        manager.async_read_secret("no-string").await,
        Err(Error::MissingString)
    ));
}

#[rstest]
#[tokio::test]
async fn same_name_rotation_uses_put_and_returns_its_response(
    default_settings: KeyManagementSettings,
) {
    let server = MockServer::start().await;
    Mock::given(header("x-amz-target", "secretsmanager.PutSecretValue"))
        .and(body_partial_json(
            json!({"SecretId":"key", "SecretString":"replacement"}),
        ))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(json!({"Name":"key", "VersionId":"version"})),
        )
        .expect(1)
        .mount(&server)
        .await;
    let response = manager(&server, default_settings)
        .async_rotate_secret("key", "key", &SecretValue::new("replacement"))
        .await
        .unwrap();
    match response {
        RotationResponse::Updated(output) => assert_eq!(output.version_id(), Some("version")),
        _ => panic!("rotation created a second secret"),
    }
    assert_eq!(server.received_requests().await.unwrap().len(), 1);
}

#[rstest]
#[tokio::test]
async fn renamed_rotation_reads_creates_verifies_then_deletes(
    default_settings: KeyManagementSettings,
) {
    let server = MockServer::start().await;
    let step = AtomicUsize::new(0);
    Mock::given(wiremock::matchers::method("POST"))
        .respond_with(move |request: &wiremock::Request| {
            let body: serde_json::Value = request.body_json().unwrap();
            let action = request
                .headers
                .get("x-amz-target")
                .unwrap()
                .to_str()
                .unwrap();
            match step.fetch_add(1, Ordering::SeqCst) {
                0 => {
                    assert_eq!(action, "secretsmanager.GetSecretValue");
                    assert_eq!(body["SecretId"], "old");
                    ResponseTemplate::new(200).set_body_json(json!({"SecretString":"old-value"}))
                }
                1 => {
                    assert_eq!(action, "secretsmanager.CreateSecret");
                    assert_eq!(body["Name"], "new");
                    assert_eq!(body["Description"], "Rotated from old");
                    assert_eq!(body["SecretString"], "replacement");
                    ResponseTemplate::new(200).set_body_json(json!({"Name":"new"}))
                }
                2 => {
                    assert_eq!(action, "secretsmanager.GetSecretValue");
                    assert_eq!(body["SecretId"], "new");
                    ResponseTemplate::new(200).set_body_json(json!({"SecretString":"replacement"}))
                }
                3 => {
                    assert_eq!(action, "secretsmanager.DeleteSecret");
                    assert_eq!(body["SecretId"], "old");
                    assert_eq!(body["RecoveryWindowInDays"], 7);
                    ResponseTemplate::new(200).set_body_json(json!({"Name":"old"}))
                }
                _ => panic!("unexpected request"),
            }
        })
        .expect(4)
        .mount(&server)
        .await;
    assert!(matches!(
        manager(&server, default_settings)
            .async_rotate_secret("old", "new", &SecretValue::new("replacement"))
            .await
            .unwrap(),
        RotationResponse::Created(_)
    ));
}

#[rstest]
#[tokio::test]
async fn creation_passes_tags_and_kms_and_survives_replication_failure() {
    let server = MockServer::start().await;
    Mock::given(header("x-amz-target", "secretsmanager.CreateSecret"))
        .and(body_partial_json(json!({"Name":"key", "SecretString":"value", "KmsKeyId":"kms-key", "Tags":[{"Key":"stage", "Value":"test"}]})))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"Name":"key"}))).expect(1).mount(&server).await;
    Mock::given(header(
        "x-amz-target",
        "secretsmanager.ReplicateSecretToRegions",
    ))
    .and(body_partial_json(
        json!({"SecretId":"key", "AddReplicaRegions":[{"Region":"replica-region"}]}),
    ))
    .respond_with(
        ResponseTemplate::new(400).set_body_json(json!({"__type":"InvalidRequestException"})),
    )
    .expect(1)
    .mount(&server)
    .await;
    let settings = KeyManagementSettings {
        kms_key_id: Some("kms-key".into()),
        tags: Some(std::collections::BTreeMap::from([(
            "stage".into(),
            "test".into(),
        )])),
        replica_regions: Some(vec!["replica-region".into()]),
        ..Default::default()
    };
    let manager = manager(&server, settings);
    assert_eq!(
        manager
            .async_write_secret("key", &SecretValue::new("value"), None)
            .await
            .unwrap()
            .name(),
        Some("key")
    );
    assert!(
        manager
            .async_replicate_secret("key", &[])
            .await
            .unwrap()
            .is_none()
    );
}

#[rstest]
#[tokio::test]
async fn trait_write_uses_typed_write_context(default_settings: KeyManagementSettings) {
    let server = MockServer::start().await;
    Mock::given(header("x-amz-target", "secretsmanager.CreateSecret"))
        .and(body_partial_json(json!({
            "Name": "key",
            "SecretString": "value",
            "Description": "created by caller",
            "Tags": [{"Key": "stage", "Value": "test"}],
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"Name": "key"})))
        .expect(1)
        .mount(&server)
        .await;
    let context = SecretWriteContext {
        description: Some("created by caller".into()),
        tags: std::collections::BTreeMap::from([("stage".into(), "test".into())]),
        ..Default::default()
    };
    let response = BaseSecretManager::async_write_secret(
        &manager(&server, default_settings),
        "key",
        &SecretValue::new("value"),
        &context,
    )
    .await
    .unwrap();
    assert_eq!(response.name(), Some("key"));
}

#[rstest]
#[tokio::test]
async fn trait_delete_accepts_an_unspecified_recovery_window(
    default_settings: KeyManagementSettings,
) {
    let server = MockServer::start().await;
    Mock::given(header("x-amz-target", "secretsmanager.DeleteSecret"))
        .and(body_partial_json(json!({"SecretId": "key"})))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"Name": "key"})))
        .expect(1)
        .mount(&server)
        .await;
    let response = BaseSecretManager::async_delete_secret(
        &manager(&server, default_settings),
        "key",
        None,
        &SecretOperationContext::default(),
    )
    .await
    .unwrap();
    assert_eq!(response.name(), Some("key"));
}

#[rstest]
#[tokio::test]
async fn trait_read_uses_the_aws_region_from_its_operation_context() {
    let server = MockServer::start().await;
    Mock::given(header("x-amz-target", "secretsmanager.GetSecretValue"))
        .respond_with(|request: &wiremock::Request| {
            let authorization = request
                .headers
                .get("authorization")
                .unwrap()
                .to_str()
                .unwrap();
            assert!(authorization.contains("/us-west-2/secretsmanager/aws4_request"));
            ResponseTemplate::new(200).set_body_json(json!({"SecretString": "value"}))
        })
        .expect(1)
        .mount(&server)
        .await;
    let context = SecretOperationContext::Aws(AwsOperationContext {
        region_name: Some("us-west-2".into()),
        ..Default::default()
    });
    let value = BaseSecretManager::async_read_secret(&loaded_manager(&server), "key", &context)
        .await
        .unwrap();
    assert_eq!(value.unwrap().expose(), "value");
}

#[rstest]
#[tokio::test]
async fn trait_read_applies_the_aws_operation_timeout() {
    let server = MockServer::start().await;
    Mock::given(header("x-amz-target", "secretsmanager.GetSecretValue"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_delay(Duration::from_secs(1))
                .set_body_json(json!({"SecretString": "late"})),
        )
        .expect(1)
        .mount(&server)
        .await;
    let context = SecretOperationContext::Aws(AwsOperationContext {
        timeout: Some(Duration::from_millis(30)),
        ..Default::default()
    });
    assert!(matches!(
        BaseSecretManager::async_read_secret(&loaded_manager(&server), "key", &context).await,
        Err(Error::Timeout)
    ));
}

#[rstest]
#[tokio::test]
async fn credential_failures_are_not_swallowed_as_missing_secrets() {
    use aws_credential_types::provider::{ProvideCredentials, error::CredentialsError, future};
    #[derive(Debug)]
    struct FailedCredentials;
    impl ProvideCredentials for FailedCredentials {
        fn provide_credentials<'a>(&'a self) -> future::ProvideCredentials<'a>
        where
            Self: 'a,
        {
            future::ProvideCredentials::ready(Err(CredentialsError::provider_error(
                "private-auth-detail",
            )))
        }
    }
    let server = MockServer::start().await;
    let config = aws_sdk_secretsmanager::Config::builder()
        .behavior_version(BehaviorVersion::latest())
        .region(Region::new("us-east-1"))
        .credentials_provider(FailedCredentials)
        .endpoint_url(server.uri())
        .retry_config(RetryConfig::disabled())
        .build();
    let manager = AwsSecretsManagerV2::new(Client::from_conf(config), Default::default());
    let error = manager.async_read_secret("key").await.unwrap_err();
    assert!(!format!("{error:?}").contains("private-auth-detail"));
    assert!(matches!(error, Error::Read(_)));
    assert!(server.received_requests().await.unwrap().is_empty());
}

#[rstest]
#[tokio::test]
async fn read_timeout_is_an_error_and_cannot_be_mistaken_for_missing() {
    let server = MockServer::start().await;
    Mock::given(wiremock::matchers::method("POST"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_delay(Duration::from_secs(1))
                .set_body_json(json!({"SecretString":"late"})),
        )
        .mount(&server)
        .await;
    let config = aws_sdk_secretsmanager::Config::builder()
        .behavior_version(BehaviorVersion::latest())
        .region(Region::new("us-east-1"))
        .credentials_provider(Credentials::new("test", "test", None, None, "test"))
        .endpoint_url(server.uri())
        .retry_config(RetryConfig::disabled())
        .timeout_config(
            aws_sdk_secretsmanager::config::timeout::TimeoutConfig::builder()
                .operation_timeout(Duration::from_millis(30))
                .build(),
        )
        .build();
    let manager = AwsSecretsManagerV2::new(Client::from_conf(config), Default::default());
    assert!(matches!(
        manager.async_read_secret("key").await,
        Err(Error::Timeout)
    ));
}

#[rstest]
#[case::denied(400, "AccessDeniedException")]
#[case::throttled(400, "ThrottlingException")]
#[case::unavailable(503, "ServiceUnavailableException")]
#[tokio::test]
async fn service_failures_remain_errors(
    default_settings: KeyManagementSettings,
    #[case] status: u16,
    #[case] code: &str,
) {
    let server = MockServer::start().await;
    Mock::given(header("x-amz-target", "secretsmanager.GetSecretValue"))
        .respond_with(ResponseTemplate::new(status).set_body_json(json!({"__type":code})))
        .expect(1)
        .mount(&server)
        .await;
    assert!(matches!(
        manager(&server, default_settings)
            .async_read_secret("key")
            .await,
        Err(Error::Read(_))
    ));
}
