use base64::{Engine, engine::general_purpose::STANDARD};
use google_cloud_kms_v1::client::KeyManagementService;
use litellm_secrets_google::GoogleKms;
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{body_json, path},
};

#[tokio::test]
async fn google_kms_decrypts_using_the_configured_resource() {
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
        .with_retry_policy(google_cloud_gax::retry_policy::NeverRetry)
        .build()
        .await
        .unwrap();
    let manager = GoogleKms::new(client, resource.into());
    assert_eq!(
        manager.decrypt(b"encrypted".to_vec()).await.unwrap(),
        b" value\n"
    );
}

#[tokio::test]
async fn disabled_google_kms_loader_does_not_require_environment_configuration() {
    use std::sync::Arc;
    for enabled in [None, Some(false)] {
        assert!(
            litellm_secrets_google::load_google_kms(enabled, Arc::new(|_: &str| None))
                .await
                .unwrap()
                .is_none()
        );
    }
}
