use aws_sdk_kms::{
    Client,
    config::{BehaviorVersion, Credentials, Region, retry::RetryConfig},
};
use base64::{Engine, engine::general_purpose::STANDARD};
use litellm_secrets_aws::{AwsKms, Error, load_aws_kms};
use litellm_secrets_types::KeyManagementSettings;
use rstest::rstest;
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{body_json, header},
};

#[rstest]
#[tokio::test]
async fn kms_decrypt_calls_the_sdk_without_applying_lookup_policy() {
    let server = MockServer::start().await;
    let plaintext = "  private-value\n";
    Mock::given(header("x-amz-target", "TrentService.Decrypt"))
        .and(body_json(
            serde_json::json!({"CiphertextBlob": STANDARD.encode("encrypted")}),
        ))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(serde_json::json!({"Plaintext": STANDARD.encode(plaintext)})),
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
            .retry_config(RetryConfig::disabled())
            .build(),
    );
    let manager = AwsKms::new(client);
    assert_eq!(
        manager.decrypt(b"encrypted".to_vec()).await.unwrap(),
        plaintext.as_bytes()
    );
}

#[rstest]
#[case::unset(None)]
#[case::disabled(Some(false))]
fn disabled_kms_loader_does_not_require_environment_configuration(#[case] enabled: Option<bool>) {
    use std::sync::Arc;
    assert!(
        load_aws_kms(
            enabled,
            &KeyManagementSettings::default(),
            Arc::new(|_: &str| None)
        )
        .unwrap()
        .is_none()
    );
}

#[rstest]
#[case::settings(Some("configured-region"), None, None)]
#[case::region_name(None, Some("AWS_REGION_NAME"), Some("environment-region"))]
#[case::region(None, Some("AWS_REGION"), Some("environment-region"))]
#[case::default_region(None, Some("AWS_DEFAULT_REGION"), Some("environment-region"))]
fn enabled_kms_loader_accepts_supported_region_sources(
    #[case] configured_region: Option<&'static str>,
    #[case] environment_region_name: Option<&'static str>,
    #[case] environment_region: Option<&'static str>,
) {
    use std::sync::Arc;
    let settings = KeyManagementSettings {
        aws_region_name: configured_region.map(str::to_owned),
        ..KeyManagementSettings::default()
    };
    let environment = Arc::new(move |name: &str| {
        (Some(name) == environment_region_name)
            .then(|| environment_region.map(str::to_owned))
            .flatten()
    });

    assert!(
        load_aws_kms(Some(true), &settings, environment)
            .unwrap()
            .is_some()
    );
}

#[rstest]
fn enabled_kms_loader_rejects_missing_region() {
    use std::sync::Arc;
    assert!(matches!(
        load_aws_kms(
            Some(true),
            &KeyManagementSettings::default(),
            Arc::new(|_: &str| None),
        ),
        Err(Error::MissingRegion)
    ));
}
