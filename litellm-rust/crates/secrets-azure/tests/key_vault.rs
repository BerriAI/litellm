use std::sync::Arc;

use litellm_secrets_azure::{AzureKeyVault, Error};
use litellm_secrets_types::{Secret, SecretValue};
use serde::Deserialize;
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{header, path, query_param},
};

fn manager(server: &MockServer) -> AzureKeyVault {
    AzureKeyVault::with_client(
        reqwest::Client::new(),
        server.uri().parse().unwrap(),
        Arc::new(|name: &str| (name == "AZURE_AD_TOKEN").then(|| "fake".to_owned())),
    )
    .unwrap()
}

#[tokio::test]
async fn reads_secret_with_bearer_token_and_api_version() {
    let server = MockServer::start().await;
    Mock::given(path("/secrets/OPENAI-API-KEY"))
        .and(query_param("api-version", "7.4"))
        .and(header("authorization", "Bearer fake"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(serde_json::json!({"value": "s3cret", "id": "secret-id"})),
        )
        .expect(1)
        .mount(&server)
        .await;

    let secret = manager(&server)
        .get_secret_from_azure_key_vault("OPENAI-API-KEY")
        .await
        .unwrap()
        .unwrap();

    assert_eq!(secret, Secret::String(SecretValue::new("s3cret")));
}

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
        .get_secret_from_azure_key_vault("name/with spaces")
        .await
        .unwrap()
        .unwrap();

    assert_eq!(secret.as_str(), Some("value"));
}

#[rstest::rstest]
#[case::not_found(404, None)]
#[case::forbidden(403, Some(403))]
#[tokio::test]
async fn handles_statuses(#[case] status: u16, #[case] expected_status: Option<u16>) {
    let server = MockServer::start().await;
    Mock::given(path("/secrets/NAME"))
        .respond_with(ResponseTemplate::new(status))
        .expect(1)
        .mount(&server)
        .await;

    let result = manager(&server)
        .get_secret_from_azure_key_vault("NAME")
        .await;

    match expected_status {
        None => assert_eq!(result.unwrap(), None),
        Some(status) => assert!(matches!(result, Err(Error::Status(actual)) if actual == status)),
    }
}

#[tokio::test]
async fn missing_value_is_an_error() {
    let server = MockServer::start().await;
    Mock::given(path("/secrets/NAME"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({})))
        .expect(1)
        .mount(&server)
        .await;

    assert!(matches!(
        manager(&server)
            .get_secret_from_azure_key_vault("NAME")
            .await,
        Err(Error::MissingValue)
    ));
}

#[test]
fn new_validates_vault_environment() {
    assert!(matches!(
        AzureKeyVault::new(Arc::new(|_: &str| None)),
        Err(Error::MissingEnvironment("AZURE_KEY_VAULT_URI"))
    ));
    assert!(matches!(
        AzureKeyVault::new(Arc::new(|name: &str| {
            (name == "AZURE_KEY_VAULT_URI").then(|| "http://vault.example".to_owned())
        })),
        Err(Error::VaultUri)
    ));
    assert!(matches!(
        AzureKeyVault::new(Arc::new(|name: &str| {
            (name == "AZURE_KEY_VAULT_URI").then(|| "vault.example".to_owned())
        })),
        Err(Error::VaultUri)
    ));
}

#[rstest::rstest]
#[case("https://myvault.vault.azure.net", "https://vault.azure.net/.default")]
#[case(
    "https://v.vault.usgovcloudapi.net/",
    "https://vault.usgovcloudapi.net/.default"
)]
#[case("http://localhost:8080", "https://localhost/.default")]
#[test]
fn derives_scope_from_vault_host(#[case] uri: &str, #[case] expected: &str) {
    let manager = AzureKeyVault::with_client(
        reqwest::Client::new(),
        uri.parse().unwrap(),
        Arc::new(|_: &str| None),
    )
    .unwrap();

    assert_eq!(manager.scope(), expected);
}

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
            .get_secret_from_azure_key_vault("NAME")
            .await
            .is_err()
    );
}

fn manager_without_credentials(server: &MockServer) -> AzureKeyVault {
    AzureKeyVault::with_client(
        reqwest::Client::new(),
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

#[tokio::test]
async fn parity_fixture_matches_python_backend_contract() {
    let fixture: Fixture =
        serde_json::from_str(include_str!("fixtures/key_vault_parity.json")).unwrap();
    for case in fixture.cases {
        let server = MockServer::start().await;
        Mock::given(path(format!("/secrets/{}", case.secret_name)))
            .respond_with(
                ResponseTemplate::new(case.response.status).set_body_json(case.response.body),
            )
            .expect(1)
            .mount(&server)
            .await;
        let result = manager(&server)
            .get_secret_from_azure_key_vault(&case.secret_name)
            .await;
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
