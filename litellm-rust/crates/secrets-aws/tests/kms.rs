use aws_sdk_kms::{
    Client,
    config::{BehaviorVersion, Credentials, Region, retry::RetryConfig},
};
use base64::{Engine, engine::general_purpose::STANDARD};
use litellm_secrets_aws::AwsKms;
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{body_json, header},
};

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

#[test]
fn disabled_kms_loader_does_not_require_environment_configuration() {
    use litellm_secrets_aws::load_aws_kms;
    use litellm_secrets_types::KeyManagementSettings;
    use std::sync::Arc;
    for enabled in [None, Some(false)] {
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
}
