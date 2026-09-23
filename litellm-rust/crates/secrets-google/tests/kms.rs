use base64::{Engine, engine::general_purpose::STANDARD};
use google_cloud_kms_v1::client::KeyManagementService;
use litellm_core_utils::settings::Lookup;
use litellm_secrets_google::{Error, GoogleKms, kms::validate_environment};
use rstest::rstest;
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{body_json, path},
};

#[rstest]
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

#[rstest]
#[case::unset(None)]
#[case::disabled(Some(false))]
#[tokio::test]
async fn disabled_google_kms_loader_does_not_require_environment_configuration(
    #[case] enabled: Option<bool>,
) {
    use std::sync::Arc;
    assert!(
        litellm_secrets_google::load_google_kms(enabled, Arc::new(|_: &str| None))
            .await
            .unwrap()
            .is_none()
    );
}

#[rstest]
#[case::credentials_missing(None, None, "GOOGLE_APPLICATION_CREDENTIALS")]
#[case::resource_missing(Some("credentials"), None, "GOOGLE_KMS_RESOURCE_NAME")]
fn enabled_google_kms_requires_all_environment_values(
    #[case] credentials: Option<&str>,
    #[case] resource: Option<&str>,
    #[case] missing: &'static str,
) {
    let environment = move |name: &str| match name {
        "GOOGLE_APPLICATION_CREDENTIALS" => credentials.map(str::to_owned),
        "GOOGLE_KMS_RESOURCE_NAME" => resource.map(str::to_owned),
        _ => None,
    };

    assert!(matches!(
        validate_environment(&environment as &dyn Lookup),
        Err(Error::MissingEnvironment(name)) if name == missing
    ));
}

#[rstest]
fn complete_google_kms_environment_is_valid() {
    let environment = |name: &str| match name {
        "GOOGLE_APPLICATION_CREDENTIALS" => Some("credentials".to_owned()),
        "GOOGLE_KMS_RESOURCE_NAME" => Some("resource".to_owned()),
        _ => None,
    };

    assert!(validate_environment(&environment as &dyn Lookup).is_ok());
}
