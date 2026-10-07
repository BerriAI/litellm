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
async fn disabled_google_kms_loader_does_not_read_environment_configuration(
    #[case] enabled: Option<bool>,
) {
    use std::sync::Arc;
    assert!(
        litellm_secrets_google::load_google_kms(
            enabled,
            Arc::new(|name: &str| panic!("disabled Google KMS read {name}")),
        )
        .await
        .unwrap()
        .is_none()
    );
}

#[rstest]
#[tokio::test]
async fn enabled_google_kms_loader_accepts_application_default_credentials() {
    use std::sync::Arc;
    let environment = Arc::new(|name: &str| {
        (name == "GOOGLE_KMS_RESOURCE_NAME")
            .then(|| "projects/project/locations/global/keyRings/ring/cryptoKeys/key".to_owned())
    });

    assert!(
        litellm_secrets_google::load_google_kms(Some(true), environment)
            .await
            .unwrap()
            .is_some()
    );
}

#[rstest]
#[case::resource_missing(None, None, "GOOGLE_KMS_RESOURCE_NAME")]
fn enabled_google_kms_requires_resource_name(
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
#[case::service_account_file(Some("credentials"))]
#[case::application_default_credentials(None)]
fn google_kms_environment_is_valid_without_required_credential_file(
    #[case] credentials: Option<&str>,
) {
    let environment = |name: &str| match name {
        "GOOGLE_APPLICATION_CREDENTIALS" => credentials.map(str::to_owned),
        "GOOGLE_KMS_RESOURCE_NAME" => Some("resource".to_owned()),
        _ => None,
    };

    assert!(validate_environment(&environment as &dyn Lookup).is_ok());
}
