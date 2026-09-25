#![cfg(feature = "cyberark")]

#[tokio::test]
async fn cyberark_handler_reads_values_and_surfaces_errors() {
    use std::time::Duration;

    use litellm_secrets::{
        Error, KeyManagementSettings, SecretManager, SecretValue, cyberark::CyberArkSecretManager,
        get_secret_from_manager,
    };
    use wiremock::{
        Mock, MockServer, ResponseTemplate,
        matchers::{body_string, path},
    };

    let server = MockServer::start().await;
    Mock::given(path("/authn/acct/admin/authenticate"))
        .and(body_string("k3y"))
        .respond_with(ResponseTemplate::new(200).set_body_string("token"))
        .mount(&server)
        .await;
    Mock::given(path("/secrets/acct/variable/KEY"))
        .respond_with(ResponseTemplate::new(200).set_body_string("value"))
        .mount(&server)
        .await;
    let manager = SecretManager::Cyberark(CyberArkSecretManager::with_client(
        reqwest::Client::new(),
        server.uri().parse().unwrap(),
        "acct".into(),
        "admin".into(),
        SecretValue::new("k3y"),
        Some(Duration::from_secs(60)),
    ));
    assert_eq!(
        manager.system(),
        litellm_secrets::KeyManagementSystem::Cyberark
    );
    let settings = KeyManagementSettings::default();
    let value = get_secret_from_manager(&manager, "KEY", &settings, &|_: &str| None)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(value.as_str(), Some("value"));

    Mock::given(path("/secrets/acct/variable/ERROR"))
        .respond_with(ResponseTemplate::new(500))
        .mount(&server)
        .await;
    assert!(matches!(
        get_secret_from_manager(&manager, "ERROR", &settings, &|_: &str| None).await,
        Err(Error::Cyberark(_))
    ));
}
