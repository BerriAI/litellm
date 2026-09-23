use std::{
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

use base64::{Engine, engine::general_purpose::STANDARD};
use litellm_secrets_cyberark::{CyberArkSecretManager, DeleteOutcome, Error};
use litellm_secrets_types::{BaseSecretManager, CyberarkOperationContext, SecretValue};
use rstest::{fixture, rstest};
use serde::Deserialize;
use wiremock::{
    Match, Mock, MockServer, Request, ResponseTemplate,
    matchers::{body_string, header, method, path},
};

const TOKEN_JSON: &str = r#"{"protected":"p","payload":"q","signature":"s"}"#;

#[derive(Deserialize)]
struct ParityFixture {
    account: String,
    username: String,
    api_key: String,
    authenticate_path: String,
    token_json: String,
    authorization_header: String,
    policy_path: String,
    secrets: Vec<ParitySecret>,
}

#[derive(Deserialize)]
struct ParitySecret {
    name: String,
    path: String,
    policy_body: String,
}

#[derive(Debug)]
struct RawPath(String);

impl Match for RawPath {
    fn matches(&self, request: &Request) -> bool {
        request.url.path() == self.0
    }
}

#[fixture]
fn parity_fixture() -> ParityFixture {
    serde_json::from_str(include_str!("fixtures/parity.json")).unwrap()
}

#[fixture]
fn client_identity_directory() -> tempfile::TempDir {
    let identity = rcgen::generate_simple_self_signed(vec!["localhost".into()]).unwrap();
    let directory = tempfile::tempdir().unwrap();
    std::fs::write(directory.path().join("client.crt"), identity.cert.pem()).unwrap();
    std::fs::write(
        directory.path().join("client.key"),
        identity.signing_key.serialize_pem(),
    )
    .unwrap();
    directory
}

fn manager(server: &MockServer, ttl: Duration) -> CyberArkSecretManager {
    CyberArkSecretManager::with_client(
        reqwest::Client::new(),
        server.uri().parse().unwrap(),
        "acct".into(),
        "admin".into(),
        SecretValue::new("k3y"),
        Some(ttl),
    )
}

async fn mount_auth(server: &MockServer, expected: u64) {
    Mock::given(method("POST"))
        .and(path("/authn/acct/admin/authenticate"))
        .and(body_string("k3y"))
        .respond_with(ResponseTemplate::new(200).set_body_string(TOKEN_JSON))
        .expect(expected)
        .mount(server)
        .await;
}

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
async fn a_rejected_refreshed_token_surfaces_the_error_without_another_retry() {
    let server = MockServer::start().await;
    mount_auth(&server, 2).await;
    Mock::given(path("/secrets/acct/variable/first"))
        .respond_with(ResponseTemplate::new(200).set_body_string("first-value"))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(path("/secrets/acct/variable/second"))
        .respond_with(ResponseTemplate::new(401))
        .expect(2)
        .mount(&server)
        .await;
    let manager = manager(&server, Duration::from_secs(60));
    assert_eq!(
        manager
            .async_read_secret("first")
            .await
            .unwrap()
            .unwrap()
            .expose(),
        "first-value"
    );

    let result = tokio::time::timeout(Duration::from_secs(5), manager.async_read_secret("second"))
        .await
        .expect("authentication retries must terminate");

    assert!(matches!(result, Err(Error::Status(401))));
}

#[rstest]
#[tokio::test]
async fn rejected_write_token_is_reauthenticated_once() {
    let server = MockServer::start().await;
    mount_auth(&server, 2).await;
    Mock::given(path("/policies/acct/policy/root"))
        .respond_with(ResponseTemplate::new(409))
        .expect(1)
        .mount(&server)
        .await;
    let attempts = Arc::new(AtomicUsize::new(0));
    let attempts_for_response = Arc::clone(&attempts);
    Mock::given(method("POST"))
        .and(path("/secrets/acct/variable/key"))
        .and(body_string("value"))
        .respond_with(move |_: &Request| {
            if attempts_for_response.fetch_add(1, Ordering::SeqCst) == 0 {
                ResponseTemplate::new(401)
            } else {
                ResponseTemplate::new(200)
            }
        })
        .expect(2)
        .mount(&server)
        .await;
    let manager = manager(&server, Duration::from_secs(600));

    manager
        .async_write_secret("key", &SecretValue::new("value"), None)
        .await
        .unwrap();
    assert_eq!(attempts.load(Ordering::SeqCst), 2);
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
#[case::not_found(404)]
#[case::unauthorized(401)]
#[case::forbidden(403)]
#[case::server_error(500)]
#[tokio::test]
async fn failed_reads_are_not_cached(#[case] status: u16) {
    let server = MockServer::start().await;
    mount_auth(&server, 1).await;
    let failing = Mock::given(path("/secrets/acct/variable/key"))
        .respond_with(ResponseTemplate::new(status))
        .expect(1)
        .mount_as_scoped(&server)
        .await;
    let manager = manager(&server, Duration::from_secs(60));
    let result = manager.async_read_secret("key").await;
    if status == 404 {
        assert_eq!(result.unwrap(), None);
    } else {
        assert!(matches!(result, Err(Error::Status(actual)) if actual == status));
    }
    drop(failing);
    Mock::given(path("/secrets/acct/variable/key"))
        .respond_with(ResponseTemplate::new(200).set_body_string("recovered"))
        .expect(1)
        .mount(&server)
        .await;
    for _ in 0..2 {
        assert_eq!(
            manager
                .async_read_secret("key")
                .await
                .unwrap()
                .unwrap()
                .expose(),
            "recovered"
        );
    }
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
#[tokio::test]
async fn expired_tokens_and_secrets_are_fetched_again() {
    let server = MockServer::start().await;
    mount_auth(&server, 2).await;
    Mock::given(path("/secrets/acct/variable/key"))
        .respond_with(ResponseTemplate::new(200).set_body_string("value"))
        .expect(2)
        .mount(&server)
        .await;
    let manager = manager(&server, Duration::from_millis(1));
    for _ in 0..2 {
        assert!(manager.async_read_secret("key").await.unwrap().is_some());
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

#[rstest]
#[case::plain("OPENAI_API_KEY")]
#[case::path("team/app/key")]
#[case::punctuation("a b+c.d-e_f~g")]
#[case::quote("needs \"quote\"")]
#[tokio::test]
async fn secret_names_use_python_quote_encoding(parity_fixture: ParityFixture, #[case] name: &str) {
    let secret = parity_fixture
        .secrets
        .iter()
        .find(|secret| secret.name == name)
        .unwrap();
    let server = MockServer::start().await;
    mount_auth(&server, 1).await;
    Mock::given(RawPath(secret.path.clone()))
        .respond_with(ResponseTemplate::new(200).set_body_string("value"))
        .expect(1)
        .mount(&server)
        .await;
    assert_eq!(
        manager(&server, Duration::from_secs(60))
            .async_read_secret(name)
            .await
            .unwrap()
            .unwrap()
            .expose(),
        "value"
    );
}

#[rstest]
#[case::created(201)]
#[case::already_exists(409)]
#[case::unprocessable(422)]
#[case::server_error(500)]
#[tokio::test]
async fn writes_tolerate_policy_status_and_cache_value(#[case] policy_status: u16) {
    let server = MockServer::start().await;
    mount_auth(&server, 1).await;
    Mock::given(path("/policies/acct/policy/root"))
        .and(header("content-type", "application/x-yaml"))
        .and(body_string("- !variable \"team/app\"\n"))
        .respond_with(ResponseTemplate::new(policy_status))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(path("/secrets/acct/variable/team%2Fapp"))
        .and(body_string("v"))
        .respond_with(ResponseTemplate::new(200))
        .expect(1)
        .mount(&server)
        .await;
    let manager = manager(&server, Duration::from_secs(60));
    manager
        .async_write_secret("team/app", &SecretValue::new("v"), None)
        .await
        .unwrap();
    assert_eq!(
        manager
            .async_read_secret("team/app")
            .await
            .unwrap()
            .unwrap()
            .expose(),
        "v"
    );
}

#[rstest]
#[tokio::test]
async fn failed_value_write_is_not_cached() {
    let server = MockServer::start().await;
    mount_auth(&server, 1).await;
    Mock::given(path("/policies/acct/policy/root"))
        .respond_with(ResponseTemplate::new(409))
        .mount(&server)
        .await;
    Mock::given(path("/secrets/acct/variable/key"))
        .and(body_string("v"))
        .respond_with(ResponseTemplate::new(403))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(path("/secrets/acct/variable/key"))
        .respond_with(ResponseTemplate::new(200).set_body_string("recovered"))
        .expect(1)
        .mount(&server)
        .await;
    let manager = manager(&server, Duration::from_secs(60));
    assert!(matches!(
        manager
            .async_write_secret("key", &SecretValue::new("v"), None)
            .await,
        Err(Error::Status(403))
    ));
    assert_eq!(
        manager
            .async_read_secret("key")
            .await
            .unwrap()
            .unwrap()
            .expose(),
        "recovered"
    );
}

#[rstest]
#[case::parent("../etc")]
#[case::embedded_parent("team/../etc")]
#[case::control("key\n")]
#[tokio::test]
async fn unsafe_names_fail_before_http_calls(#[case] name: &str) {
    let server = MockServer::start().await;
    let manager = manager(&server, Duration::from_secs(60));
    assert!(matches!(
        manager.async_read_secret(name).await,
        Err(Error::Operation(
            litellm_secrets_types::Error::UnsafeSecretName
        ))
    ));
    assert!(matches!(
        manager
            .async_write_secret(name, &SecretValue::new("v"), None)
            .await,
        Err(Error::Operation(
            litellm_secrets_types::Error::UnsafeSecretName
        ))
    ));
}

#[rstest]
#[tokio::test]
async fn delete_invalidates_cache_and_reports_not_supported() {
    let server = MockServer::start().await;
    mount_auth(&server, 1).await;
    Mock::given(path("/secrets/acct/variable/key"))
        .respond_with(ResponseTemplate::new(200).set_body_string("v"))
        .expect(2)
        .mount(&server)
        .await;
    let manager = manager(&server, Duration::from_secs(60));
    assert_eq!(
        manager
            .async_read_secret("key")
            .await
            .unwrap()
            .unwrap()
            .expose(),
        "v"
    );
    assert_eq!(
        manager.async_delete_secret("key", Some(7)).await.unwrap(),
        DeleteOutcome::NotSupported
    );
    assert_eq!(
        manager
            .async_read_secret("key")
            .await
            .unwrap()
            .unwrap()
            .expose(),
        "v"
    );
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

#[rstest]
#[tokio::test]
async fn writes_match_python_parity_fixture(parity_fixture: ParityFixture) {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(&parity_fixture.authenticate_path))
        .and(body_string(&parity_fixture.api_key))
        .respond_with(ResponseTemplate::new(200).set_body_string(&parity_fixture.token_json))
        .expect(1)
        .mount(&server)
        .await;
    let manager = CyberArkSecretManager::with_client(
        reqwest::Client::new(),
        server.uri().parse().unwrap(),
        parity_fixture.account,
        parity_fixture.username,
        SecretValue::new(parity_fixture.api_key),
        Some(Duration::from_secs(60)),
    );
    for secret in parity_fixture.secrets {
        Mock::given(method("POST"))
            .and(path(&parity_fixture.policy_path))
            .and(header(
                "authorization",
                &parity_fixture.authorization_header,
            ))
            .and(header("content-type", "application/x-yaml"))
            .and(body_string(&secret.policy_body))
            .respond_with(ResponseTemplate::new(201))
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(RawPath(secret.path))
            .and(header(
                "authorization",
                &parity_fixture.authorization_header,
            ))
            .and(body_string("value"))
            .respond_with(ResponseTemplate::new(201))
            .expect(1)
            .mount(&server)
            .await;
        manager
            .async_write_secret(&secret.name, &SecretValue::new("value"), None)
            .await
            .unwrap();
        assert_eq!(
            manager
                .async_read_secret(&secret.name)
                .await
                .unwrap()
                .unwrap()
                .expose(),
            "value"
        );
    }
}

#[rstest]
#[tokio::test]
#[ignore]
async fn live_conjur_round_trip() {
    let endpoint: reqwest::Url = std::env::var("CYBERARK_API_BASE").unwrap().parse().unwrap();
    let account = std::env::var("CYBERARK_ACCOUNT").unwrap();
    let username = std::env::var("CYBERARK_USERNAME").unwrap();
    let api_key = SecretValue::new(std::env::var("CYBERARK_API_KEY").unwrap());
    let name = format!(
        "{}-{}",
        std::env::var("LITELLM_CONJUR_LIVE_SECRET_NAME").unwrap(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );
    let manager = CyberArkSecretManager::with_client(
        reqwest::Client::new(),
        endpoint.clone(),
        account.clone(),
        username.clone(),
        api_key.clone(),
        Some(Duration::from_secs(60)),
    );

    assert!(manager.async_read_secret(&name).await.unwrap().is_none());
    for expected in ["first-π\n", " second-π "] {
        manager
            .async_write_secret(&name, &SecretValue::new(expected), None)
            .await
            .unwrap();
        let verifier = CyberArkSecretManager::with_client(
            reqwest::Client::new(),
            endpoint.clone(),
            account.clone(),
            username.clone(),
            api_key.clone(),
            Some(Duration::from_secs(60)),
        );
        assert_eq!(
            verifier
                .async_read_secret(&name)
                .await
                .unwrap()
                .unwrap()
                .expose(),
            expected
        );
    }
    manager
        .async_rotate_secret(&name, &name, &SecretValue::new("rotated-value"))
        .await
        .unwrap();
    let alias = format!("{name}-rotated");
    manager
        .async_rotate_secret(&name, &alias, &SecretValue::new("new-alias-value"))
        .await
        .unwrap();
    assert_eq!(
        manager
            .async_read_secret(&name)
            .await
            .unwrap()
            .unwrap()
            .expose(),
        "rotated-value"
    );
    assert_eq!(
        manager
            .async_read_secret(&alias)
            .await
            .unwrap()
            .unwrap()
            .expose(),
        "new-alias-value"
    );
}

#[rstest]
#[case::same_alias("old")]
#[case::new_alias("new")]
#[tokio::test]
async fn rotation_stores_the_replacement_and_retains_other_aliases(#[case] new_name: &'static str) {
    use std::sync::atomic::{AtomicBool, Ordering};
    let server = MockServer::start().await;
    mount_auth(&server, 1).await;
    let written = Arc::new(AtomicBool::new(false));
    let read_state = written.clone();
    Mock::given(method("GET"))
        .respond_with(move |request: &wiremock::Request| {
            let name = request.url.path().rsplit('/').next().unwrap();
            let value = if name == new_name && read_state.load(Ordering::SeqCst) {
                "new-value"
            } else {
                "old-value"
            };
            ResponseTemplate::new(200).set_body_string(value)
        })
        .expect(if new_name == "old" { 2 } else { 3 })
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/policies/acct/policy/root"))
        .and(body_string(format!("- !variable \"{new_name}\"\n")))
        .respond_with(ResponseTemplate::new(201))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path(format!("/secrets/acct/variable/{new_name}")))
        .and(body_string("new-value"))
        .respond_with(move |_: &wiremock::Request| {
            written.store(true, Ordering::SeqCst);
            ResponseTemplate::new(201)
        })
        .expect(1)
        .mount(&server)
        .await;
    let manager = manager(&server, Duration::from_secs(60));
    manager
        .async_rotate_secret("old", new_name, &SecretValue::new("new-value"))
        .await
        .unwrap();
    assert_eq!(
        manager
            .async_read_secret(new_name)
            .await
            .unwrap()
            .unwrap()
            .expose(),
        "new-value"
    );
    assert_eq!(
        manager
            .async_read_secret("old")
            .await
            .unwrap()
            .unwrap()
            .expose(),
        if new_name == "old" {
            "new-value"
        } else {
            "old-value"
        }
    );
    assert!(
        server
            .received_requests()
            .await
            .unwrap()
            .iter()
            .all(|request| request.method != "DELETE")
    );
}

#[tokio::test]
async fn rotation_verifies_the_remote_value_instead_of_the_write_cache() {
    let server = MockServer::start().await;
    mount_auth(&server, 1).await;
    Mock::given(method("GET"))
        .respond_with(ResponseTemplate::new(200).set_body_string("unchanged"))
        .expect(2)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/policies/acct/policy/root"))
        .respond_with(ResponseTemplate::new(201))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/secrets/acct/variable/new"))
        .respond_with(ResponseTemplate::new(201))
        .expect(1)
        .mount(&server)
        .await;
    let result = manager(&server, Duration::from_secs(60))
        .async_rotate_secret("old", "new", &SecretValue::new("replacement"))
        .await;
    assert!(matches!(
        result,
        Err(litellm_secrets_types::RotationError::Verification {
            source: Error::Operation(litellm_secrets_types::Error::NewSecretMismatch),
            ..
        })
    ));
}

#[rstest]
#[case::colon("foo: bar")]
#[case::comment("foo # bar")]
#[case::plain("plain-alias")]
#[case::email("team/user@example.com")]
#[case::quote("needs \"quote\"")]
#[case::backslash("a\\b")]
#[tokio::test]
async fn policy_writes_preserve_yaml_metacharacters_as_one_variable(#[case] name: &'static str) {
    let server = MockServer::start().await;
    mount_auth(&server, 1).await;
    Mock::given(method("POST"))
        .and(path("/policies/acct/policy/root"))
        .respond_with(move |request: &wiremock::Request| {
            let body = std::str::from_utf8(&request.body).unwrap();
            let scalar = body
                .strip_prefix("- !variable ")
                .unwrap()
                .strip_suffix('\n')
                .unwrap();
            assert_eq!(serde_json::from_str::<String>(scalar).unwrap(), name);
            ResponseTemplate::new(201)
        })
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(body_string("value"))
        .respond_with(ResponseTemplate::new(201))
        .expect(1)
        .mount(&server)
        .await;
    let manager = manager(&server, Duration::from_secs(60));
    manager
        .async_write_secret(name, &SecretValue::new("value"), None)
        .await
        .unwrap();
    assert_eq!(
        manager
            .async_read_secret(name)
            .await
            .unwrap()
            .unwrap()
            .expose(),
        "value"
    );
}
