use super::*;

#[rstest]
#[tokio::test]
async fn successful_reads_cache_auth_secret_and_redact_values() {
    let server = MockServer::start().await;
    mount_auth(&server, 1).await;
    let token = STANDARD.encode(TOKEN_JSON);
    Mock::given(path("/secrets/acct/variable/OPENAI_API_KEY"))
        .and(header("authorization", format!("Token token=\"{token}\"")))
        .respond_with(ResponseTemplate::new(200).set_body_string("sk-live"))
        .expect(1)
        .mount(&server)
        .await;
    let manager = manager(&server, Duration::from_secs(60));

    for _ in 0..2 {
        let value = manager
            .async_read_secret("OPENAI_API_KEY")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(value.expose(), "sk-live");
        assert!(!format!("{value:?}").contains("sk-live"));
    }
}

#[rstest]
#[tokio::test]
async fn concurrent_reads_share_authentication_and_secret_requests() {
    let server = MockServer::start().await;
    Mock::given(path("/authn/acct/admin/authenticate"))
        .and(body_string("k3y"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(TOKEN_JSON)
                .set_delay(Duration::from_millis(20)),
        )
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(path("/secrets/acct/variable/key"))
        .and(header(
            "authorization",
            format!("Token token=\"{}\"", STANDARD.encode(TOKEN_JSON)),
        ))
        .respond_with(ResponseTemplate::new(200).set_body_string("value"))
        .expect(1)
        .mount(&server)
        .await;
    let manager = manager(&server, Duration::from_secs(60));

    let (first, second) = tokio::join!(
        manager.async_read_secret("key"),
        manager.async_read_secret("key")
    );

    assert_eq!(first.unwrap().unwrap().expose(), "value");
    assert_eq!(second.unwrap().unwrap().expose(), "value");
}

#[rstest]
#[case::host("host/team/app", "/authn/acct/host%2Fteam%2Fapp/authenticate")]
#[case::user("alice@devops", "/authn/acct/alice%40devops/authenticate")]
#[tokio::test]
async fn authentication_encodes_login(#[case] username: &str, #[case] expected_path: &str) {
    let server = MockServer::start().await;
    Mock::given(RawPath(expected_path.to_owned()))
        .and(method("POST"))
        .and(body_string("k3y"))
        .respond_with(ResponseTemplate::new(200).set_body_string(TOKEN_JSON))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(path("/secrets/acct/variable/key"))
        .respond_with(ResponseTemplate::new(200).set_body_string("value"))
        .mount(&server)
        .await;
    let manager = CyberArkSecretManager::with_client(
        reqwest::Client::new(),
        server.uri().parse().unwrap(),
        "acct".into(),
        username.into(),
        SecretValue::new("k3y"),
        None,
    );

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

#[rstest]
#[tokio::test]
async fn rejected_cached_token_is_reauthenticated_once() {
    let server = MockServer::start().await;
    mount_auth(&server, 2).await;
    Mock::given(path("/secrets/acct/variable/first"))
        .respond_with(ResponseTemplate::new(200).set_body_string("first-value"))
        .expect(1)
        .mount(&server)
        .await;
    let attempts = Arc::new(AtomicUsize::new(0));
    let attempts_for_response = Arc::clone(&attempts);
    Mock::given(path("/secrets/acct/variable/second"))
        .respond_with(move |_: &Request| {
            if attempts_for_response.fetch_add(1, Ordering::SeqCst) == 0 {
                ResponseTemplate::new(401)
            } else {
                ResponseTemplate::new(200).set_body_string("second-value")
            }
        })
        .expect(2)
        .mount(&server)
        .await;
    let manager = manager(&server, Duration::from_secs(600));

    assert_eq!(
        manager
            .async_read_secret("first")
            .await
            .unwrap()
            .unwrap()
            .expose(),
        "first-value"
    );
    assert_eq!(
        manager
            .async_read_secret("second")
            .await
            .unwrap()
            .unwrap()
            .expose(),
        "second-value"
    );
    assert_eq!(attempts.load(Ordering::SeqCst), 2);
}

#[rstest]
#[tokio::test]
async fn failed_authentication_is_not_cached_and_does_not_read_secret() {
    let server = MockServer::start().await;
    let failing = Mock::given(path("/authn/acct/admin/authenticate"))
        .respond_with(ResponseTemplate::new(401))
        .expect(1)
        .mount_as_scoped(&server)
        .await;
    let unused_secret = Mock::given(path("/secrets/acct/variable/key"))
        .respond_with(ResponseTemplate::new(200).set_body_string("value"))
        .expect(0)
        .mount_as_scoped(&server)
        .await;
    let manager = manager(&server, Duration::from_secs(60));
    assert!(matches!(
        manager.async_read_secret("key").await,
        Err(Error::AuthStatus(401))
    ));
    drop(unused_secret);
    drop(failing);
    mount_auth(&server, 1).await;
    Mock::given(path("/secrets/acct/variable/key"))
        .respond_with(ResponseTemplate::new(200).set_body_string("value"))
        .expect(1)
        .mount(&server)
        .await;
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

#[rstest]
#[tokio::test]
async fn trait_read_applies_cyberark_operation_timeout_to_authentication() {
    let server = MockServer::start().await;
    Mock::given(path("/authn/acct/admin/authenticate"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(TOKEN_JSON)
                .set_delay(Duration::from_millis(50)),
        )
        .mount(&server)
        .await;
    let manager = manager(&server, Duration::from_secs(60));
    let context = CyberarkOperationContext {
        timeout: Some(Duration::from_millis(10)),
    };

    let result = BaseSecretManager::async_read_secret(&manager, "key", &context).await;

    match result {
        Err(Error::Timeout) => {}
        Err(Error::Http(error)) => assert!(error.is_timeout()),
        other => panic!("expected timeout, got {other:?}"),
    }
}

#[rstest]
fn new_validates_credentials_before_license_and_configuration() {
    let empty: Arc<dyn litellm_core_utils::settings::Lookup + Send + Sync> =
        Arc::new(|_: &str| None);
    assert!(matches!(
        CyberArkSecretManager::new(empty, true),
        Err(Error::MissingCredentials)
    ));
    assert!(matches!(
        CyberArkSecretManager::new(
            Arc::new(|name: &str| (name == "CYBERARK_API_KEY").then(|| "k3y".into())),
            false
        ),
        Err(Error::EnterpriseRequired)
    ));
    assert!(matches!(
        CyberArkSecretManager::new(
            Arc::new(|name: &str| (name == "CYBERARK_CLIENT_CERT").then(|| "cert".into())),
            true
        ),
        Err(Error::MissingCredentials)
    ));
    assert!(matches!(
        CyberArkSecretManager::new(
            Arc::new(|name: &str| match name {
                "CYBERARK_API_KEY" => Some("k3y".into()),
                "CYBERARK_REFRESH_INTERVAL" => Some("abc".into()),
                _ => None,
            }),
            true
        ),
        Err(Error::RefreshInterval)
    ));
    assert!(matches!(
        CyberArkSecretManager::new(
            Arc::new(|name: &str| match name {
                "CYBERARK_API_KEY" => Some("k3y".into()),
                "CYBERARK_API_BASE" => Some("not a url".into()),
                _ => None,
            }),
            true
        ),
        Err(Error::Endpoint)
    ));
}

#[rstest]
fn certificate_only_credentials_are_validated_as_a_client_identity() {
    let result = CyberArkSecretManager::new(
        Arc::new(|name: &str| match name {
            "CYBERARK_CLIENT_CERT" => Some("/missing/cert".into()),
            "CYBERARK_CLIENT_KEY" => Some("/missing/key".into()),
            _ => None,
        }),
        true,
    );

    assert!(matches!(result, Err(Error::ClientCertificate)));
}

#[rstest]
#[case::certificate_only("")]
#[case::certificate_and_api_key("k3y")]
#[tokio::test]
async fn configured_client_identity_preserves_auth_request_and_read_result(
    client_identity_directory: tempfile::TempDir,
    #[case] api_key: &'static str,
) {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/authn/default/admin/authenticate"))
        .and(body_string(api_key))
        .respond_with(ResponseTemplate::new(200).set_body_string(TOKEN_JSON))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/secrets/default/variable/key"))
        .and(header(
            "authorization",
            format!("Token token=\"{}\"", STANDARD.encode(TOKEN_JSON)),
        ))
        .respond_with(ResponseTemplate::new(200).set_body_string(" value\n"))
        .expect(1)
        .mount(&server)
        .await;
    let endpoint = server.uri();
    let certificate = client_identity_directory.path().join("client.crt");
    let key = client_identity_directory.path().join("client.key");
    let manager = CyberArkSecretManager::new(
        Arc::new(move |name: &str| match name {
            "CYBERARK_API_BASE" => Some(endpoint.clone()),
            "CYBERARK_API_KEY" => Some(api_key.into()),
            "CYBERARK_CLIENT_CERT" => Some(certificate.to_str().unwrap().into()),
            "CYBERARK_CLIENT_KEY" => Some(key.to_str().unwrap().into()),
            _ => None,
        }),
        true,
    )
    .unwrap();

    assert!(server.received_requests().await.unwrap().is_empty());
    assert_eq!(
        manager
            .async_read_secret("key")
            .await
            .unwrap()
            .unwrap()
            .expose(),
        " value\n"
    );
}

#[rstest]
#[case::certificate_only("", "client.crt")]
#[case::key_only("", "client.key")]
#[case::certificate_with_api_key("k3y", "client.crt")]
#[case::key_with_api_key("k3y", "client.key")]
fn invalid_client_identity_is_not_ignored(
    client_identity_directory: tempfile::TempDir,
    #[case] api_key: &'static str,
    #[case] invalid_file: &str,
) {
    std::fs::write(
        client_identity_directory.path().join(invalid_file),
        "not PEM",
    )
    .unwrap();
    let certificate = client_identity_directory.path().join("client.crt");
    let key = client_identity_directory.path().join("client.key");

    let result = CyberArkSecretManager::new(
        Arc::new(move |name: &str| match name {
            "CYBERARK_API_KEY" => Some(api_key.into()),
            "CYBERARK_CLIENT_CERT" => Some(certificate.to_str().unwrap().into()),
            "CYBERARK_CLIENT_KEY" => Some(key.to_str().unwrap().into()),
            _ => None,
        }),
        true,
    );

    assert!(matches!(result, Err(Error::ClientCertificate)));
}

#[rstest]
#[case::certificate_only("")]
#[case::certificate_and_api_key("k3y")]
fn client_identity_does_not_bypass_the_enterprise_requirement(#[case] api_key: &'static str) {
    let result = CyberArkSecretManager::new(
        Arc::new(move |name: &str| match name {
            "CYBERARK_API_KEY" => Some(api_key.into()),
            "CYBERARK_CLIENT_CERT" => Some("/missing/cert".into()),
            "CYBERARK_CLIENT_KEY" => Some("/missing/key".into()),
            _ => None,
        }),
        false,
    );

    assert!(matches!(result, Err(Error::EnterpriseRequired)));
}

#[rstest]
#[tokio::test]
async fn new_reads_environment_defaults_end_to_end() {
    let server = MockServer::start().await;
    Mock::given(path("/authn/default/admin/authenticate"))
        .and(body_string("k3y"))
        .respond_with(ResponseTemplate::new(200).set_body_string(TOKEN_JSON))
        .mount(&server)
        .await;
    Mock::given(path("/secrets/default/variable/key"))
        .respond_with(ResponseTemplate::new(200).set_body_string("value"))
        .mount(&server)
        .await;
    let endpoint = server.uri();
    let manager = CyberArkSecretManager::new(
        Arc::new(move |name: &str| match name {
            "CYBERARK_API_BASE" => Some(endpoint.clone()),
            "CYBERARK_API_KEY" => Some("k3y".into()),
            _ => None,
        }),
        true,
    )
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

#[rstest]
fn new_reports_missing_client_certificate_files() {
    assert!(matches!(
        CyberArkSecretManager::new(
            Arc::new(|name: &str| match name {
                "CYBERARK_API_KEY" => Some("k3y".into()),
                "CYBERARK_CLIENT_CERT" => Some("/missing/cert".into()),
                "CYBERARK_CLIENT_KEY" => Some("/missing/key".into()),
                _ => None,
            }),
            true
        ),
        Err(Error::ClientCertificate)
    ));
}

#[rstest]
#[tokio::test]
async fn trailing_slash_endpoint_preserves_base_path() {
    let server = MockServer::start().await;
    Mock::given(path("/prefix/authn/acct/admin/authenticate"))
        .and(body_string("k3y"))
        .respond_with(ResponseTemplate::new(200).set_body_string(TOKEN_JSON))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(path("/prefix/secrets/acct/variable/key"))
        .respond_with(ResponseTemplate::new(200).set_body_string("value"))
        .mount(&server)
        .await;
    let endpoint = format!("{}/prefix/", server.uri()).parse().unwrap();
    let manager = CyberArkSecretManager::with_client(
        reqwest::Client::new(),
        endpoint,
        "acct".into(),
        "admin".into(),
        SecretValue::new("k3y"),
        Some(Duration::from_secs(60)),
    );
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
