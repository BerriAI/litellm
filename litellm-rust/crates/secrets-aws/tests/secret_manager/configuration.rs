use super::*;

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
    let config = client_builder(&server)
        .credentials_provider(FailedCredentials)
        .build();
    let manager = AwsSecretsManagerV2::new(Client::from_conf(config), Default::default());
    let error = manager.async_read_secret("key").await.unwrap_err();
    assert!(!format!("{error:?}").contains("private-auth-detail"));
    assert!(matches!(error, Error::Read(_)));
    assert!(server.received_requests().await.unwrap().is_empty());
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
                "configuration::configured_profile_credentials_override_static_environment_credentials",
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
