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
    AwsOperationContext, BaseSecretManager, KeyManagementSettings, SecretDeleter, SecretValue,
    SecretWriteContext, SecretWriter,
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
    let response = SecretWriter::async_write_secret(
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
async fn trait_delete_uses_the_provider_recovery_policy(default_settings: KeyManagementSettings) {
    let server = MockServer::start().await;
    Mock::given(header("x-amz-target", "secretsmanager.DeleteSecret"))
        .and(body_partial_json(json!({"SecretId": "key"})))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"Name": "key"})))
        .expect(1)
        .mount(&server)
        .await;
    let response = SecretDeleter::async_delete_secret(
        &manager(&server, default_settings),
        "key",
        &AwsOperationContext::default(),
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
    let context = AwsOperationContext {
        region_name: Some("us-west-2".into()),
        ..Default::default()
    };
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
    let context = AwsOperationContext {
        timeout: Some(Duration::from_millis(30)),
        ..Default::default()
    };
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

async fn scripted_actions(
    server: &MockServer,
    actions: Vec<(&'static str, serde_json::Value, u16, serde_json::Value)>,
) {
    let count = actions.len() as u64;
    let step = AtomicUsize::new(0);
    Mock::given(wiremock::matchers::method("POST"))
        .respond_with(move |request: &wiremock::Request| {
            let (action, expected, status, response) =
                &actions[step.fetch_add(1, Ordering::SeqCst)];
            assert_eq!(
                request.headers["x-amz-target"],
                format!("secretsmanager.{action}")
            );
            let body: serde_json::Value = request.body_json().unwrap();
            let actual = serde_json::Value::Object(
                body.as_object()
                    .unwrap()
                    .iter()
                    .filter(|(key, _)| key.as_str() != "ClientRequestToken")
                    .map(|(key, value)| (key.clone(), value.clone()))
                    .collect(),
            );
            assert_eq!(&actual, expected);
            ResponseTemplate::new(*status).set_body_json(response)
        })
        .expect(count)
        .mount(server)
        .await;
}

#[rstest]
#[case::write(false)]
#[case::rotate_back(true)]
#[tokio::test]
async fn recovery_window_alias_is_restored_updated_and_tagged(#[case] rotate: bool) {
    let server = MockServer::start().await;
    let description = if rotate {
        "Rotated from old"
    } else {
        "description"
    };
    let write = json!({"Name":"key", "SecretString":"new", "Description":description,
        "KmsKeyId":"kms", "Tags":[{"Key":"stage", "Value":"test"}]});
    let actions = if rotate {
        vec![(
            "GetSecretValue",
            json!({"SecretId":"old"}),
            200,
            json!({"SecretString":"old"}),
        )]
    } else {
        vec![]
    };
    let recovery = vec![
        (
            "CreateSecret",
            write,
            400,
            json!({"__type":"ResourceExistsException"}),
        ),
        (
            "DescribeSecret",
            json!({"SecretId":"key"}),
            200,
            json!({"DeletedDate":1}),
        ),
        (
            "RestoreSecret",
            json!({"SecretId":"key"}),
            200,
            json!({"Name":"key"}),
        ),
        (
            "UpdateSecret",
            json!({"SecretId":"key", "SecretString":"new", "Description":description,
            "KmsKeyId":"kms"}),
            200,
            json!({"ARN":"restored-arn", "Name":"key", "VersionId":"new-version"}),
        ),
        (
            "TagResource",
            json!({"SecretId":"key", "Tags":[{"Key":"stage", "Value":"test"}]}),
            200,
            json!({}),
        ),
    ];
    let verification = if rotate {
        vec![
            (
                "GetSecretValue",
                json!({"SecretId":"key"}),
                200,
                json!({"SecretString":"new"}),
            ),
            (
                "DeleteSecret",
                json!({"SecretId":"old", "RecoveryWindowInDays":7}),
                200,
                json!({}),
            ),
        ]
    } else {
        vec![]
    };
    scripted_actions(
        &server,
        actions
            .into_iter()
            .chain(recovery)
            .chain(verification)
            .collect(),
    )
    .await;
    let manager = manager(
        &server,
        KeyManagementSettings {
            kms_key_id: Some("kms".into()),
            tags: Some(std::collections::BTreeMap::from([(
                "stage".into(),
                "test".into(),
            )])),
            ..Default::default()
        },
    );
    let output = if rotate {
        match manager
            .async_rotate_secret("old", "key", &SecretValue::new("new"))
            .await
            .unwrap()
        {
            RotationResponse::Created(output) => output,
            _ => panic!("expected restored alias"),
        }
    } else {
        manager
            .async_write_secret("key", &SecretValue::new("new"), Some(description))
            .await
            .unwrap()
    };
    assert_eq!(
        (output.arn(), output.name(), output.version_id()),
        (Some("restored-arn"), Some("key"), Some("new-version"))
    );
}

#[rstest]
#[case::live(200, json!({"Name":"key"}))]
#[case::missing(400, json!({"__type":"ResourceNotFoundException"}))]
#[case::denied(400, json!({"__type":"AccessDeniedException"}))]
#[tokio::test]
async fn create_failure_does_not_overwrite_an_alias_without_a_deletion_date(
    #[case] status: u16,
    #[case] described: serde_json::Value,
) {
    let server = MockServer::start().await;
    scripted_actions(
        &server,
        vec![
            (
                "CreateSecret",
                json!({"Name":"key", "SecretString":"new"}),
                400,
                json!({"__type":"ResourceExistsException"}),
            ),
            (
                "DescribeSecret",
                json!({"SecretId":"key"}),
                status,
                described,
            ),
        ],
    )
    .await;
    assert!(matches!(
        manager(&server, Default::default())
            .async_write_secret("key", &SecretValue::new("new"), None)
            .await,
        Err(Error::Create(_))
    ));
}

#[tokio::test]
async fn failed_update_reschedules_deletion_of_a_restored_alias() {
    let server = MockServer::start().await;
    scripted_actions(
        &server,
        vec![
            (
                "CreateSecret",
                json!({"Name":"key", "SecretString":"new"}),
                400,
                json!({"__type":"ResourceExistsException"}),
            ),
            (
                "DescribeSecret",
                json!({"SecretId":"key"}),
                200,
                json!({"DeletedDate":1}),
            ),
            ("RestoreSecret", json!({"SecretId":"key"}), 200, json!({})),
            (
                "UpdateSecret",
                json!({"SecretId":"key", "SecretString":"new"}),
                400,
                json!({"__type":"InvalidRequestException"}),
            ),
            (
                "DeleteSecret",
                json!({"SecretId":"key", "RecoveryWindowInDays":7}),
                200,
                json!({}),
            ),
        ],
    )
    .await;
    assert!(
        manager(&server, Default::default())
            .async_write_secret("key", &SecretValue::new("new"), None)
            .await
            .is_err()
    );
}

#[rstest]
#[case::unconfigured(None)]
#[case::empty(Some(vec![]))]
#[case::configured(Some(vec!["region-a".into(), "region-b".into()]))]
#[tokio::test]
async fn creation_replicates_only_to_configured_regions(#[case] regions: Option<Vec<String>>) {
    let server = MockServer::start().await;
    let create = vec![(
        "CreateSecret",
        json!({"Name":"key", "SecretString":"value", "KmsKeyId":"kms-key"}),
        200,
        json!({"Name":"key", "VersionId":"created"}),
    )];
    let replicate = regions
        .as_ref()
        .filter(|regions| !regions.is_empty())
        .map(|regions| {
            (
                "ReplicateSecretToRegions",
                json!({"SecretId":"key", "AddReplicaRegions":regions.iter()
            .map(|region| json!({"Region":region})).collect::<Vec<_>>()}),
                200,
                json!({"ARN":"replica-arn"}),
            )
        });
    scripted_actions(&server, create.into_iter().chain(replicate).collect()).await;
    let environment: Arc<dyn litellm_core_utils::settings::Lookup + Send + Sync> = {
        let endpoint = server.uri();
        Arc::new(move |name: &str| match name {
            "AWS_BEDROCK_RUNTIME_ENDPOINT" => Some(endpoint.clone()),
            "AWS_ACCESS_KEY_ID" | "AWS_SECRET_ACCESS_KEY" => Some("test".into()),
            _ => None,
        })
    };
    let manager = AwsSecretsManagerV2::load_aws_secret_manager(
        Some(true),
        KeyManagementSettings {
            aws_region_name: Some("us-east-1".into()),
            replica_regions: regions,
            kms_key_id: Some("kms-key".into()),
            ..Default::default()
        },
        environment,
    )
    .unwrap()
    .unwrap();
    assert_eq!(
        manager
            .async_write_secret("key", &SecretValue::new("value"), None)
            .await
            .unwrap()
            .version_id(),
        Some("created")
    );
}

#[rstest]
#[case::success(200)]
#[case::denied(403)]
#[tokio::test]
async fn direct_replication_returns_response_or_service_error(#[case] status: u16) {
    let server = MockServer::start().await;
    scripted_actions(&server, vec![("ReplicateSecretToRegions",
        json!({"SecretId":"key", "AddReplicaRegions":[{"Region":"region-a"}, {"Region":"region-b"}]}),
        status, if status == 200 { json!({"ARN":"replicated-arn"}) }
        else { json!({"__type":"AccessDeniedException"}) })]).await;
    let result = manager(&server, Default::default())
        .async_replicate_secret("key", &["region-a".into(), "region-b".into()])
        .await;
    if status == 200 {
        assert_eq!(result.unwrap().unwrap().arn(), Some("replicated-arn"));
    } else {
        assert!(matches!(result, Err(Error::Replicate(_))));
    }
}

#[rstest]
#[case::create(false)]
#[case::replicate(true)]
#[tokio::test]
async fn write_and_replication_timeouts_remain_errors(#[case] replicate: bool) {
    let server = MockServer::start().await;
    Mock::given(wiremock::matchers::method("POST"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_delay(Duration::from_secs(1))
                .set_body_json(json!({})),
        )
        .expect(if replicate { 1 } else { 2 })
        .mount(&server)
        .await;
    let client = Client::from_conf(
        aws_sdk_secretsmanager::Config::builder()
            .behavior_version(BehaviorVersion::latest())
            .region(Region::new("us-east-1"))
            .credentials_provider(Credentials::new("test", "test", None, None, "test"))
            .endpoint_url(server.uri())
            .retry_config(RetryConfig::disabled())
            .timeout_config(
                aws_sdk_secretsmanager::config::timeout::TimeoutConfig::builder()
                    .operation_timeout(Duration::from_millis(50))
                    .build(),
            )
            .build(),
    );
    let manager = AwsSecretsManagerV2::new(client, Default::default());
    if replicate {
        assert!(matches!(
            manager
                .async_replicate_secret("key", &["region".into()])
                .await,
            Err(Error::Replicate(_))
        ));
    } else {
        assert!(matches!(
            manager
                .async_write_secret("key", &SecretValue::new("value"), None)
                .await,
            Err(Error::Create(_))
        ));
    }
}

#[rstest]
#[case::environment(false)]
#[case::operation_override(true)]
#[tokio::test]
async fn endpoint_overrides_replace_the_service_and_override_the_region(
    #[case] override_context: bool,
) {
    let configured = MockServer::start().await;
    let explicit = MockServer::start().await;
    let target = if override_context {
        &explicit
    } else {
        &configured
    };
    Mock::given(wiremock::matchers::path_regex("^/secretsmanager/?$"))
        .and(header("x-amz-target", "secretsmanager.GetSecretValue"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"SecretString":"value"})))
        .expect(1)
        .mount(target)
        .await;
    let endpoint = format!("{}/bedrock-runtime", configured.uri());
    let manager = AwsSecretsManagerV2::load_aws_secret_manager(
        Some(true),
        KeyManagementSettings {
            aws_region_name: Some("cn-north-1".into()),
            ..Default::default()
        },
        Arc::new(move |name: &str| match name {
            "AWS_BEDROCK_RUNTIME_ENDPOINT" => Some(endpoint.clone()),
            "AWS_ACCESS_KEY_ID" | "AWS_SECRET_ACCESS_KEY" => Some("test".into()),
            _ => None,
        }),
    )
    .unwrap()
    .unwrap();
    let context = AwsOperationContext {
        bedrock_runtime_endpoint: override_context
            .then(|| format!("{}/bedrock-runtime", explicit.uri())),
        ..Default::default()
    };
    assert_eq!(
        BaseSecretManager::async_read_secret(&manager, "key", &context)
            .await
            .unwrap()
            .unwrap()
            .expose(),
        "value"
    );
    assert!(
        if override_context {
            configured
        } else {
            explicit
        }
        .received_requests()
        .await
        .unwrap()
        .is_empty()
    );
}

#[rstest]
#[case::text("value")]
#[case::json(r#"{"api_key":"test","metadata":{"team":"test"},"temperature":0.7}"#)]
#[case::empty("")]
#[case::unicode(" π\n ")]
#[tokio::test]
async fn write_read_delete_preserves_the_complete_secret_string(#[case] value: &str) {
    let server = MockServer::start().await;
    scripted_actions(
        &server,
        vec![
            (
                "CreateSecret",
                json!({"Name":"key", "SecretString":value, "Description":"description"}),
                200,
                json!({"Name":"key"}),
            ),
            (
                "GetSecretValue",
                json!({"SecretId":"key"}),
                200,
                json!({"SecretString":value}),
            ),
            (
                "DeleteSecret",
                json!({"SecretId":"key", "RecoveryWindowInDays":7}),
                200,
                json!({"Name":"key"}),
            ),
        ],
    )
    .await;
    let manager = manager(&server, Default::default());
    assert_eq!(
        manager
            .async_write_secret("key", &SecretValue::new(value), Some("description"))
            .await
            .unwrap()
            .name(),
        Some("key")
    );
    assert_eq!(
        manager
            .async_read_secret("key")
            .await
            .unwrap()
            .unwrap()
            .expose(),
        value
    );
    assert_eq!(
        manager
            .async_delete_secret("key", Some(7))
            .await
            .unwrap()
            .name(),
        Some("key")
    );
}

#[rstest]
#[case::unset(None)]
#[case::disabled(Some(false))]
fn disabled_secret_manager_loader_does_not_require_environment(#[case] enabled: Option<bool>) {
    assert!(
        AwsSecretsManagerV2::load_aws_secret_manager(
            enabled,
            Default::default(),
            Arc::new(|_: &str| panic!("disabled loader consulted the environment"))
        )
        .unwrap()
        .is_none()
    );
}

#[rstest]
#[case::role(None, None)]
#[case::cross_account(Some("external-id"), None)]
#[case::web_identity(None, Some("identity-token"))]
#[tokio::test]
async fn configured_sts_credentials_sign_the_secret_request(
    #[case] external_id: Option<&str>,
    #[case] identity: Option<&str>,
) {
    use aws_sdk_secretsmanager::primitives::{DateTime, DateTimeFormat};
    let server = MockServer::start().await;
    let expiry = DateTime::from(std::time::SystemTime::now() + Duration::from_secs(3600))
        .fmt(DateTimeFormat::DateTime)
        .unwrap();
    let action = if identity.is_some() {
        "AssumeRoleWithWebIdentity"
    } else {
        "AssumeRole"
    };
    let expected_external = external_id.map(str::to_owned);
    let expected_identity = identity.map(str::to_owned);
    Mock::given(wiremock::matchers::body_string_contains(format!("Action={action}")))
        .respond_with(move |request: &wiremock::Request| {
            let body = std::str::from_utf8(&request.body).unwrap();
            assert!(body.contains("RoleArn=test-role"), "{body}");
            assert!(body.contains("RoleSessionName=parity-session"), "{body}");
            if let Some(value) = &expected_external { assert!(body.contains(&format!("ExternalId={value}"))); }
            if let Some(value) = &expected_identity { assert!(body.contains(&format!("WebIdentityToken={value}"))); }
            ResponseTemplate::new(200).set_body_string(format!(
                "<{action}Response><{action}Result><Credentials><AccessKeyId>assumed-key</AccessKeyId>\
                 <SecretAccessKey>assumed-secret</SecretAccessKey><SessionToken>session-token</SessionToken>\
                 <Expiration>{expiry}</Expiration></Credentials></{action}Result></{action}Response>"))
        }).expect(1).mount(&server).await;
    Mock::given(header("x-amz-target", "secretsmanager.GetSecretValue"))
        .and(header("x-amz-security-token", "session-token"))
        .respond_with(|request: &wiremock::Request| {
            assert!(
                request.headers["authorization"]
                    .to_str()
                    .unwrap()
                    .contains("Credential=assumed-key/")
            );
            ResponseTemplate::new(200).set_body_json(json!({"SecretString":"value"}))
        })
        .expect(1)
        .mount(&server)
        .await;
    let endpoint = server.uri();
    let manager = AwsSecretsManagerV2::load_aws_secret_manager(
        Some(true),
        KeyManagementSettings {
            aws_region_name: Some("us-east-1".into()),
            aws_role_name: Some("test-role".into()),
            aws_session_name: Some("parity-session".into()),
            aws_external_id: external_id.map(SecretValue::new),
            aws_web_identity_token: identity.map(SecretValue::new),
            aws_sts_endpoint: Some(endpoint.clone()),
            ..Default::default()
        },
        Arc::new(move |name: &str| match name {
            "AWS_BEDROCK_RUNTIME_ENDPOINT" => Some(endpoint.clone()),
            "AWS_ACCESS_KEY_ID" | "AWS_SECRET_ACCESS_KEY" => Some("source-key".into()),
            _ => None,
        }),
    )
    .unwrap()
    .unwrap();
    assert_eq!(
        manager
            .async_read_secret("key")
            .await
            .unwrap()
            .unwrap()
            .expose(),
        "value"
    );
}

#[tokio::test]
async fn configured_profile_credentials_override_static_environment_credentials() {
    const CHILD_ENDPOINT: &str = "LITELLM_SECRETS_PROFILE_TEST_ENDPOINT";
    if let Ok(endpoint) = std::env::var(CHILD_ENDPOINT) {
        let manager = AwsSecretsManagerV2::load_aws_secret_manager(
            Some(true),
            KeyManagementSettings {
                aws_region_name: Some("us-east-1".into()),
                aws_profile_name: Some("parity".into()),
                ..Default::default()
            },
            Arc::new(move |name: &str| match name {
                "AWS_BEDROCK_RUNTIME_ENDPOINT" => Some(endpoint.clone()),
                "AWS_ACCESS_KEY_ID" | "AWS_SECRET_ACCESS_KEY" => Some("wrong-static-key".into()),
                _ => None,
            }),
        )
        .unwrap()
        .unwrap();
        assert_eq!(
            manager
                .async_read_secret("key")
                .await
                .unwrap()
                .unwrap()
                .expose(),
            "profile-value"
        );
        return;
    }
    let server = MockServer::start().await;
    Mock::given(header("x-amz-target", "secretsmanager.GetSecretValue"))
        .and(header("x-amz-security-token", "profile-session"))
        .respond_with(|request: &wiremock::Request| {
            assert!(
                request.headers["authorization"]
                    .to_str()
                    .unwrap()
                    .contains("Credential=profile-key/")
            );
            ResponseTemplate::new(200).set_body_json(json!({"SecretString":"profile-value"}))
        })
        .expect(1)
        .mount(&server)
        .await;
    let directory = tempfile::tempdir().unwrap();
    let credentials = directory.path().join("credentials");
    let config = directory.path().join("config");
    std::fs::write(&credentials, "[parity]\naws_access_key_id=profile-key\naws_secret_access_key=profile-secret\naws_session_token=profile-session\n").unwrap();
    std::fs::write(&config, "").unwrap();
    let endpoint = server.uri();
    let result = tokio::task::spawn_blocking(move || {
        std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "configured_profile_credentials_override_static_environment_credentials",
                "--nocapture",
            ])
            .env(CHILD_ENDPOINT, endpoint)
            .env("AWS_SHARED_CREDENTIALS_FILE", credentials)
            .env("AWS_CONFIG_FILE", config)
            .output()
            .unwrap()
    })
    .await
    .unwrap();
    assert!(
        result.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );
}
