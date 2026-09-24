use super::*;

#[rstest]
fn trailing_address_slashes_are_removed() {
    let environment: Arc<dyn Lookup + Send + Sync> = Arc::new(|name: &str| match name {
        "HCP_VAULT_ADDR" => Some("http://vault.test:8200///".to_owned()),
        "HCP_VAULT_TOKEN" => Some("token".to_owned()),
        _ => None,
    });
    let config: HashicorpVaultConfig =
        HashicorpVaultConfig::from_environment(environment.as_ref()).unwrap();
    let manager: HashicorpVault = HashicorpVault::from_config(config, true).unwrap();

    assert_eq!(
        manager.secret_location("name").unwrap(),
        litellm_secrets_hashicorp::SecretLocation {
            namespace: None,
            mount: "secret".to_owned(),
            path: "name".to_owned(),
        }
    );
}

#[rstest]
#[case::negative("-1")]
#[case::not_a_number("not-a-number")]
fn invalid_refresh_intervals_are_rejected(#[case] value: &str) {
    let environment: Arc<dyn Lookup + Send + Sync> = Arc::new(move |name: &str| match name {
        "HCP_VAULT_REFRESH_INTERVAL" => Some(value.to_owned()),
        _ => None,
    });

    assert!(matches!(
        HashicorpVaultConfig::from_environment(environment.as_ref()),
        Err(Error::RefreshInterval)
    ));
}

#[rstest]
#[tokio::test]
async fn approle_login_uses_namespace_and_reuses_the_token() {
    let server: MockServer = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/auth/custom-approle/login"))
        .and(header("X-Vault-Namespace", "login-root"))
        .and(body_json(json!({"role_id": "role", "secret_id": "secret"})))
        .respond_with(ResponseTemplate::new(200).set_body_json(auth_response("login-token", 3600)))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/v1/secret/data/name"))
        .and(header("X-Vault-Token", "login-token"))
        .and(header("X-Vault-Namespace", "secret-root"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(read_response(json!({"key": "value"}))),
        )
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/v1/secret/data/name-2"))
        .respond_with(ResponseTemplate::new(404).set_body_json(json!({"errors": ["missing"]})))
        .expect(1)
        .mount(&server)
        .await;
    let manager: HashicorpVault = manager(
        &server,
        &[
            ("HCP_VAULT_APPROLE_ROLE_ID", "role"),
            ("HCP_VAULT_APPROLE_SECRET_ID", "secret"),
            ("HCP_VAULT_APPROLE_MOUNT_PATH", "custom-approle"),
            ("HCP_VAULT_NAMESPACE", "secret-root"),
            ("HCP_VAULT_LOGIN_NAMESPACE", "login-root"),
        ],
    );

    assert!(manager.async_read_secret("name").await.unwrap().is_some());
    assert!(manager.async_read_secret("name-2").await.unwrap().is_none());
}

#[rstest]
#[tokio::test]
async fn tls_login_posts_the_role_and_uses_the_client_identity() {
    let server: MockServer = MockServer::start().await;
    let directory: tempfile::TempDir = tempfile::tempdir().unwrap();
    let cert_path = directory.path().join("client.crt");
    let key_path = directory.path().join("client.key");
    std::fs::write(&cert_path, TEST_CERTIFICATE).unwrap();
    std::fs::write(&key_path, TEST_PRIVATE_KEY).unwrap();
    Mock::given(method("POST"))
        .and(path("/v1/auth/cert/login"))
        .and(header("X-Vault-Namespace", "login-ns"))
        .respond_with(ResponseTemplate::new(200).set_body_json(auth_response("cert-token", 0)))
        .expect(2)
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/v1/secret/data/name"))
        .and(header("X-Vault-Token", "cert-token"))
        .and(header("X-Vault-Namespace", "secret-ns"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(read_response(json!({"key": "value"}))),
        )
        .expect(2)
        .mount(&server)
        .await;
    let role_values: HashMap<String, String> = HashMap::from([
        ("HCP_VAULT_ADDR".to_owned(), server.uri()),
        (
            "HCP_VAULT_CLIENT_CERT".to_owned(),
            cert_path.to_str().unwrap().to_owned(),
        ),
        (
            "HCP_VAULT_CLIENT_KEY".to_owned(),
            key_path.to_str().unwrap().to_owned(),
        ),
        ("HCP_VAULT_CERT_ROLE".to_owned(), "vault-role".to_owned()),
        (
            "HCP_VAULT_LOGIN_NAMESPACE".to_owned(),
            "login-ns".to_owned(),
        ),
        (
            "HCP_VAULT_SECRET_NAMESPACE".to_owned(),
            "secret-ns".to_owned(),
        ),
    ]);
    let role_environment: Arc<dyn Lookup + Send + Sync> =
        Arc::new(move |name: &str| role_values.get(name).cloned());
    let role_manager: HashicorpVault = HashicorpVault::new(role_environment, true).unwrap();
    assert!(
        role_manager
            .async_read_secret("name")
            .await
            .unwrap()
            .is_some()
    );

    let no_role_values: HashMap<String, String> = HashMap::from([
        ("HCP_VAULT_ADDR".to_owned(), server.uri()),
        (
            "HCP_VAULT_CLIENT_CERT".to_owned(),
            cert_path.to_str().unwrap().to_owned(),
        ),
        (
            "HCP_VAULT_CLIENT_KEY".to_owned(),
            key_path.to_str().unwrap().to_owned(),
        ),
        (
            "HCP_VAULT_LOGIN_NAMESPACE".to_owned(),
            "login-ns".to_owned(),
        ),
        (
            "HCP_VAULT_SECRET_NAMESPACE".to_owned(),
            "secret-ns".to_owned(),
        ),
    ]);
    let no_role_environment: Arc<dyn Lookup + Send + Sync> =
        Arc::new(move |name: &str| no_role_values.get(name).cloned());
    let no_role_manager: HashicorpVault = HashicorpVault::new(no_role_environment, true).unwrap();
    assert!(
        no_role_manager
            .async_read_secret("name")
            .await
            .unwrap()
            .is_some()
    );
    let login_bodies: Vec<serde_json::Value> = server
        .received_requests()
        .await
        .unwrap()
        .iter()
        .filter(|request| request.method.as_str() == "POST")
        .map(|request| serde_json::from_slice(&request.body).unwrap())
        .collect();
    assert!(login_bodies.contains(&json!({"name": "vault-role"})));
    assert!(login_bodies.contains(&json!({})));
}

#[rstest]
#[tokio::test]
async fn no_auth_and_invalid_names_fail_without_requests() {
    let server: MockServer = MockServer::start().await;
    let manager: HashicorpVault = manager(&server, &[]);

    assert!(matches!(
        manager.async_read_secret("name").await,
        Err(Error::NoAuthConfigured)
    ));
    assert!(matches!(
        manager.async_read_secret("../name").await,
        Err(Error::InvalidSecretName(_))
    ));
    assert!(server.received_requests().await.unwrap().is_empty());
}

#[rstest]
#[tokio::test]
async fn debug_output_redacts_authentication_values() {
    let server: MockServer = MockServer::start().await;
    let manager: HashicorpVault =
        HashicorpVault::from_config(config(&server, &[("HCP_VAULT_TOKEN", "token-value")]), true)
            .unwrap();
    let debug: String = format!("{manager:?}");
    assert!(!debug.contains("token-value"));
    assert!(!debug.contains("secret-id"));
}

#[rstest]
fn configuration_matches_python_parity_fixture(parity_cases: Vec<ParityCase>) {
    for case in parity_cases {
        let values: HashMap<String, String> = case.env.clone();
        let environment: Arc<dyn Lookup + Send + Sync> =
            Arc::new(move |name: &str| values.get(name).cloned());
        let config: HashicorpVaultConfig =
            HashicorpVaultConfig::from_environment(environment.as_ref()).unwrap();
        let manager: HashicorpVault = HashicorpVault::from_config(config.clone(), true).unwrap();
        let location = manager.secret_location(&case.secret_name).unwrap();
        let namespace = location
            .namespace
            .as_deref()
            .map(|namespace| format!("{namespace}/"))
            .unwrap_or_default();
        assert_eq!(
            format!(
                "{}/v1/{}{}/data/{}",
                config.address, namespace, location.mount, location.path
            ),
            case.expected_secret_url
        );
        let login_url = config.approle.as_ref().map_or_else(
            || {
                config
                    .tls_cert
                    .as_ref()
                    .map(|_| format!("{}/v1/auth/cert/login", config.address))
            },
            |approle| {
                Some(format!(
                    "{}/v1/auth/{}/login",
                    config.address, approle.mount_path
                ))
            },
        );
        assert_eq!(login_url, case.expected_login_url);
        assert_eq!(
            manager.config().login_namespace(),
            case.expected_login_namespace.as_deref()
        );
        assert_eq!(
            manager.config().secret_namespace(),
            case.expected_secret_namespace.as_deref()
        );
    }
}

#[rstest]
#[case::separate(
    Some("legacy"),
    Some("root"),
    Some("teams/team-a"),
    Some("root"),
    Some("teams/team-a")
)]
#[case::legacy(Some("admin"), None, None, Some("admin"), Some("admin"))]
#[case::login_override(Some("admin"), Some("root"), None, Some("root"), Some("admin"))]
#[case::secret_override(
    Some("admin"),
    None,
    Some("teams/team-a"),
    Some("admin"),
    Some("teams/team-a")
)]
#[case::no_namespace(None, None, None, None, None)]
#[tokio::test]
async fn login_and_secret_namespaces_follow_python_precedence(
    #[case] legacy: Option<&str>,
    #[case] login: Option<&str>,
    #[case] secret: Option<&str>,
    #[case] expected_login: Option<&'static str>,
    #[case] expected_secret: Option<&'static str>,
) {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/auth/approle/login"))
        .and(body_json(json!({"role_id":"role", "secret_id":"secret"})))
        .respond_with(move |request: &wiremock::Request| {
            assert_eq!(
                request
                    .headers
                    .get("X-Vault-Namespace")
                    .map(|value| value.to_str().unwrap()),
                expected_login
            );
            ResponseTemplate::new(200).set_body_json(auth_response("login-token", 3600))
        })
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/v1/secret/data/key"))
        .and(header("X-Vault-Token", "login-token"))
        .respond_with(move |request: &wiremock::Request| {
            assert_eq!(
                request
                    .headers
                    .get("X-Vault-Namespace")
                    .map(|value| value.to_str().unwrap()),
                expected_secret
            );
            ResponseTemplate::new(200).set_body_json(read_response(json!({"key":"value"})))
        })
        .expect(1)
        .mount(&server)
        .await;
    let values: Vec<_> = [
        ("HCP_VAULT_NAMESPACE", legacy),
        ("HCP_VAULT_LOGIN_NAMESPACE", login),
        ("HCP_VAULT_SECRET_NAMESPACE", secret),
        ("HCP_VAULT_APPROLE_ROLE_ID", Some("role")),
        ("HCP_VAULT_APPROLE_SECRET_ID", Some("secret")),
    ]
    .into_iter()
    .filter_map(|(key, value)| value.map(|value| (key, value)))
    .collect();
    assert_eq!(
        manager(&server, &values)
            .async_read_secret("key")
            .await
            .unwrap()
            .unwrap()
            .expose(),
        "value"
    );
}
