#[cfg(feature = "aws")]
#[tokio::test]
async fn aws_handler_reads_ciphertext_decodes_trims_and_redacts() {
    use aws_sdk_kms::{
        Client,
        config::{BehaviorVersion, Credentials, Region},
    };
    use base64::{Engine, engine::general_purpose::STANDARD};
    use litellm_secrets::{
        Error, KeyManagementSettings, SecretManager, aws::AwsKms, get_secret_from_manager,
    };
    use wiremock::{Mock, MockServer, ResponseTemplate, matchers::body_json};

    let server = MockServer::start().await;
    Mock::given(body_json(
        serde_json::json!({"CiphertextBlob": STANDARD.encode("encrypted")}),
    ))
    .respond_with(
        ResponseTemplate::new(200)
            .set_body_json(serde_json::json!({"Plaintext":STANDARD.encode(" value\n")})),
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
            .build(),
    );
    let manager = SecretManager::AwsKms(AwsKms::new(client));
    let settings = KeyManagementSettings::default();
    let value = get_secret_from_manager(&manager, "KEY", &settings, &|name: &str| {
        assert_eq!(name, "KEY");
        Some(format!(" {}\n", STANDARD.encode("encrypted")))
    })
    .await
    .unwrap()
    .unwrap();
    assert_eq!(value.as_str(), Some("value"));
    assert!(!format!("{value:?}").contains("value"));
    assert!(matches!(
        get_secret_from_manager(&manager, "KEY", &settings, &|_: &str| None).await,
        Err(Error::MissingCiphertext)
    ));
    assert!(matches!(
        get_secret_from_manager(&manager, "KEY", &settings, &|_: &str| Some("abc".into())).await,
        Err(Error::InvalidCiphertext)
    ));
}

#[cfg(feature = "google")]
#[tokio::test]
async fn google_handler_requires_canonical_base64_and_preserves_plaintext_whitespace() {
    use base64::{Engine, engine::general_purpose::STANDARD};
    use google_cloud_kms_v1::client::KeyManagementService;
    use litellm_secrets::{
        Error, KeyManagementSettings, SecretManager, get_secret_from_manager, google::GoogleKms,
    };
    use wiremock::{
        Mock, MockServer, ResponseTemplate,
        matchers::{body_json, path},
    };

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
        .build()
        .await
        .unwrap();
    let manager = SecretManager::GoogleKms(GoogleKms::new(client, resource.into()));
    let settings = KeyManagementSettings::default();
    let value = get_secret_from_manager(&manager, "KEY", &settings, &|_: &str| {
        Some(STANDARD.encode("encrypted"))
    })
    .await
    .unwrap()
    .unwrap();
    assert_eq!(value.as_str(), Some(" value\n"));
    assert!(matches!(
        get_secret_from_manager(&manager, "KEY", &settings, &|_: &str| Some(format!(
            " {}",
            STANDARD.encode("encrypted")
        )))
        .await,
        Err(Error::InvalidCiphertext)
    ));
    assert!(matches!(
        get_secret_from_manager(&manager, "KEY", &settings, &|_: &str| None).await,
        Err(Error::MissingCiphertext)
    ));
}
