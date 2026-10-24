use std::sync::Arc;

use litellm_secrets_azure::{AzureKeyVault, Error};
use litellm_secrets_types::{Secret, SecretValue};
use rstest::{fixture, rstest};
use serde::Deserialize;
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{header, path, query_param},
};

fn manager(server: &MockServer) -> AzureKeyVault {
    AzureKeyVault::with_client(
        litellm_http::Client::plain_for_test(),
        server.uri().parse().unwrap(),
        Arc::new(|name: &str| (name == "AZURE_AD_TOKEN").then(|| "fake".to_owned())),
    )
    .unwrap()
}

#[rstest]
#[tokio::test]
async fn reads_secret_with_bearer_token_and_api_version() {
    let server = MockServer::start().await;
    Mock::given(path("/secrets/OPENAI-API-KEY"))
        .and(query_param("api-version", "7.4"))
        .and(header("authorization", "Bearer fake"))
        .and(header("accept", "application/json"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(serde_json::json!({"value": "s3cret", "id": "secret-id"})),
        )
        .expect(1)
        .mount(&server)
        .await;

    let secret = manager(&server)
        .get_secret("OPENAI-API-KEY")
        .await
        .unwrap()
        .unwrap();

    assert_eq!(secret, Secret::String(SecretValue::new("s3cret")));
}

#[rstest]
#[tokio::test]
async fn preserves_secret_contents_and_redacts_debug_output() {
    let server = MockServer::start().await;
    let value = " \tvalue-π\n";
    Mock::given(path("/secrets/NAME"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"value": value})))
        .expect(1)
        .mount(&server)
        .await;

    let secret = manager(&server).get_secret("NAME").await.unwrap().unwrap();

    assert_eq!(secret.as_str(), Some(value));
    assert!(!format!("{secret:?}").contains(value));
}

#[rstest]
#[tokio::test]
async fn percent_encodes_secret_name_path_segment() {
    let server = MockServer::start().await;
    Mock::given(path("/secrets/name%2Fwith%20spaces"))
        .and(query_param("api-version", "7.4"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(serde_json::json!({"value": "value"})),
        )
        .expect(1)
        .mount(&server)
        .await;

    let secret = manager(&server)
        .get_secret("name/with spaces")
        .await
        .unwrap()
        .unwrap();

    assert_eq!(secret.as_str(), Some("value"));
}

#[rstest]
#[case::not_found(404, None)]
#[case::unauthorized(401, Some(401))]
#[case::forbidden(403, Some(403))]
#[case::throttled(429, Some(429))]
#[case::server_error(500, Some(500))]
#[tokio::test]
async fn handles_statuses(#[case] status: u16, #[case] expected_status: Option<u16>) {
    let server = MockServer::start().await;
    Mock::given(path("/secrets/NAME"))
        .respond_with(ResponseTemplate::new(status))
        .expect(1)
        .mount(&server)
        .await;

    let result = manager(&server).get_secret("NAME").await;

    match expected_status {
        None => assert_eq!(result.unwrap(), None),
        Some(status) => assert!(matches!(result, Err(Error::Status(actual)) if actual == status)),
    }
}

#[rstest]
#[tokio::test]
async fn missing_value_is_an_error() {
    let server = MockServer::start().await;
    Mock::given(path("/secrets/NAME"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({})))
        .expect(1)
        .mount(&server)
        .await;

    assert!(matches!(
        manager(&server).get_secret("NAME").await,
        Err(Error::MissingValue)
    ));
}

#[rstest]
#[case::missing(None, true)]
#[case::http(Some("http://vault.example"), false)]
#[case::relative(Some("vault.example"), false)]
#[case::malformed(Some("://"), false)]
fn new_validates_vault_environment(
    #[case] uri: Option<&'static str>,
    #[case] missing_environment: bool,
) {
    let result = AzureKeyVault::new(
        litellm_http::Client::plain_for_test(),
        Arc::new(move |name: &str| {
            (name == "AZURE_KEY_VAULT_URI")
                .then(|| uri.map(str::to_owned))
                .flatten()
        }),
    );

    if missing_environment {
        assert!(matches!(
            result,
            Err(Error::MissingEnvironment("AZURE_KEY_VAULT_URI"))
        ));
    } else {
        assert!(matches!(result, Err(Error::VaultUri)));
    }
}

#[rstest]
#[case::public_cloud("https://myvault.vault.azure.net", "https://vault.azure.net/.default")]
#[case::government_cloud(
    "https://v.vault.usgovcloudapi.net/",
    "https://vault.usgovcloudapi.net/.default"
)]
#[case::local("http://localhost:8080", "https://localhost/.default")]
fn derives_scope_from_vault_host(#[case] uri: &str, #[case] expected: &str) {
    let manager = AzureKeyVault::with_client(
        litellm_http::Client::plain_for_test(),
        uri.parse().unwrap(),
        Arc::new(|_: &str| None),
    )
    .unwrap();

    assert_eq!(manager.scope(), expected);
}

#[rstest]
#[tokio::test]
async fn missing_credentials_do_not_request_vault() {
    let server = MockServer::start().await;
    Mock::given(path("/secrets/NAME"))
        .respond_with(ResponseTemplate::new(200))
        .expect(0)
        .mount(&server)
        .await;

    assert!(
        manager_without_credentials(&server)
            .get_secret("NAME")
            .await
            .is_err()
    );
}

fn manager_without_credentials(server: &MockServer) -> AzureKeyVault {
    AzureKeyVault::with_client(
        litellm_http::Client::plain_for_test(),
        server.uri().parse().unwrap(),
        Arc::new(|name: &str| {
            (name == "AZURE_CREDENTIAL").then(|| "ClientSecretCredential".to_owned())
        }),
    )
    .unwrap()
}

#[derive(Deserialize)]
struct Fixture {
    cases: Vec<FixtureCase>,
}

#[derive(Deserialize)]
struct FixtureCase {
    secret_name: String,
    response: FixtureResponse,
    expected: FixtureExpected,
}

#[derive(Deserialize)]
struct FixtureResponse {
    status: u16,
    body: serde_json::Value,
}

#[derive(Deserialize)]
struct FixtureExpected {
    value: Option<String>,
    missing: Option<bool>,
    error: Option<bool>,
}

#[fixture]
fn parity_fixture() -> Fixture {
    serde_json::from_str(include_str!("fixtures/key_vault_parity.json")).unwrap()
}

#[rstest]
#[tokio::test]
async fn parity_fixture_matches_python_backend_contract(parity_fixture: Fixture) {
    for case in parity_fixture.cases {
        let server = MockServer::start().await;
        Mock::given(path(format!("/secrets/{}", case.secret_name)))
            .respond_with(
                ResponseTemplate::new(case.response.status).set_body_json(case.response.body),
            )
            .expect(1)
            .mount(&server)
            .await;
        let result = manager(&server).get_secret(&case.secret_name).await;
        if case.expected.missing == Some(true) {
            assert_eq!(result.unwrap(), None);
        } else if case.expected.error == Some(true) {
            assert!(result.is_err());
        } else {
            assert_eq!(
                result.unwrap().unwrap().as_str(),
                case.expected.value.as_deref()
            );
        }
    }
}

#[tokio::test]
async fn trait_read_limits_the_operation_duration() {
    use litellm_secrets_types::{AzureOperationContext, BaseSecretManager};
    use std::time::Duration;
    let server = MockServer::start().await;
    Mock::given(wiremock::matchers::method("GET"))
        .respond_with(ResponseTemplate::new(200).set_delay(Duration::from_secs(1)))
        .mount(&server)
        .await;
    let manager = manager(&server);
    let context = AzureOperationContext {
        timeout: Some(Duration::from_millis(30)),
    };
    assert!(matches!(
        BaseSecretManager::async_read_secret(&manager, "key", &context).await,
        Err(Error::Timeout)
    ));
}
