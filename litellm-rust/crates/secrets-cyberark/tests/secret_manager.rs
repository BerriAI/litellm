use std::{sync::Arc, time::Duration};

use base64::{Engine, engine::general_purpose::STANDARD};
use litellm_secrets_cyberark::{CyberArkSecretManager, DeleteOutcome, Error};
use litellm_secrets_types::SecretValue;
use serde::Deserialize;
use wiremock::{
    Match, Mock, MockServer, Request, ResponseTemplate,
    matchers::{body_string, header, method, path},
};

const TOKEN_JSON: &str = r#"{"protected":"p","payload":"q","signature":"s"}"#;

#[derive(Deserialize)]
struct ParityFixture {
    endpoint: String,
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

fn fixture() -> ParityFixture {
    serde_json::from_str(include_str!("fixtures/parity.json")).unwrap()
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

#[tokio::test]
async fn concurrent_reads_share_authentication_request() {
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
        .expect(2)
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

#[rstest::rstest]
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

#[rstest::rstest]
#[tokio::test]
async fn secret_names_use_python_quote_encoding(
    #[values("OPENAI_API_KEY", "team/app/key", "a b+c.d-e_f~g", "needs \"quote\"")] name: &str,
) {
    let fixture = fixture();
    let secret = fixture
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

#[rstest::rstest]
#[case(201)]
#[case(409)]
#[case(422)]
#[case(500)]
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

#[tokio::test]
async fn unsafe_names_fail_before_http_calls() {
    let server = MockServer::start().await;
    let manager = manager(&server, Duration::from_secs(60));
    assert!(matches!(
        manager
            .async_write_secret("../etc", &SecretValue::new("v"), None)
            .await,
        Err(Error::Operation(
            litellm_secrets_types::Error::UnsafeSecretName
        ))
    ));
}

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
        manager.async_delete_secret("key", 7).await.unwrap(),
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

#[test]
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

#[test]
fn new_reports_missing_client_certificate_files() {
    assert!(matches!(
        CyberArkSecretManager::new(
            Arc::new(|name: &str| match name {
                "CYBERARK_CLIENT_CERT" => Some("/missing/cert".into()),
                "CYBERARK_CLIENT_KEY" => Some("/missing/key".into()),
                _ => None,
            }),
            true
        ),
        Err(Error::ClientCertificate)
    ));
}

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

#[test]
fn parity_fixture_matches_authentication_contract() {
    let fixture = fixture();
    assert_eq!(fixture.endpoint, "http://conjur.test:8080");
    assert_eq!(fixture.account, "acct");
    assert_eq!(fixture.username, "admin");
    assert_eq!(fixture.api_key, "k3y");
    assert_eq!(fixture.authenticate_path, "/authn/acct/admin/authenticate");
    assert_eq!(fixture.token_json, TOKEN_JSON);
    assert_eq!(
        fixture.authorization_header,
        format!("Token token=\"{}\"", STANDARD.encode(TOKEN_JSON))
    );
    assert_eq!(fixture.policy_path, "/policies/acct/policy/root");
    assert_eq!(fixture.secrets.len(), 4);
    assert_eq!(
        fixture.secrets[1].policy_body,
        "- !variable \"team/app/key\"\n"
    );
}
