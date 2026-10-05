use super::*;

#[rstest]
#[case::string_value("KEY", Some(Secret::String(SecretValue::new("value"))))]
#[case::missing_value("missing", None)]
#[case::non_string_value("BOOL", Some(Secret::Bool(true)))]
#[tokio::test]
async fn primary_lookup_preserves_read_semantics(
    default_settings: KeyManagementSettings,
    #[case] name: &str,
    #[case] expected: Option<Secret>,
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
            .unwrap(),
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
    let config = client_builder(&server)
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
