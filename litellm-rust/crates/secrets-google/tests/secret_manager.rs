use std::{sync::Arc, time::Duration};

use base64::{Engine, engine::general_purpose::STANDARD};
use litellm_secrets_google::{Error, GoogleSecretManager};
use litellm_secrets_types::Secret;
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{header, path},
};

fn manager(server: &MockServer, always_read: bool, ttl: Duration) -> GoogleSecretManager {
    GoogleSecretManager::with_client(
        reqwest::Client::new(),
        server.uri().parse().unwrap(),
        "project".into(),
        Arc::new(|name: &str| (name == "VERTEX_AI_API_KEY").then(|| "token".into())),
        Some(ttl),
        always_read,
    )
    .unwrap()
}

#[rstest::rstest]
#[case::nonempty("private-value")]
#[case::empty("")]
#[tokio::test]
async fn successful_reads_use_auth_latest_version_and_cache_including_empty_values(
    #[case] value: &str,
) {
    let server = MockServer::start().await;
    Mock::given(path(
        "/v1/projects/project/secrets/key/versions/latest:access",
    ))
    .and(header("authorization", "Bearer token"))
    .respond_with(
        ResponseTemplate::new(200)
            .set_body_json(serde_json::json!({"payload":{"data":STANDARD.encode(value)}})),
    )
    .expect(1)
    .mount(&server)
    .await;
    let manager = manager(&server, false, Duration::from_secs(60));
    for _ in 0..2 {
        assert_eq!(
            manager
                .get_secret_from_google_secret_manager("key")
                .await
                .unwrap()
                .unwrap()
                .as_str()
                .unwrap(),
            value
        );
    }
}

#[rstest::rstest]
#[case::not_found(ResponseTemplate::new(404))]
#[case::missing_payload(
    ResponseTemplate::new(200).set_body_json(serde_json::json!({"payload":{}}))
)]
#[tokio::test]
async fn negative_cache_returns_none_after_initial_error(#[case] response: ResponseTemplate) {
    let server = MockServer::start().await;
    Mock::given(path(
        "/v1/projects/project/secrets/key/versions/latest:access",
    ))
    .respond_with(response)
    .expect(1)
    .mount(&server)
    .await;
    let manager = manager(&server, false, Duration::from_secs(60));
    assert!(matches!(
        manager.get_secret_from_google_secret_manager("key").await,
        Err(Error::Status(404) | Error::MissingPayload)
    ));
    assert!(
        manager
            .get_secret_from_google_secret_manager("key")
            .await
            .unwrap()
            .is_none()
    );
}

#[rstest::rstest]
#[case::always_read(true, Duration::from_secs(60))]
#[case::expired_cache(false, Duration::from_millis(1))]
#[tokio::test]
async fn always_read_and_expired_cache_fetch_again(
    #[case] always_read: bool,
    #[case] ttl: Duration,
) {
    let server = MockServer::start().await;
    Mock::given(path(
        "/v1/projects/project/secrets/key/versions/latest:access",
    ))
    .respond_with(
        ResponseTemplate::new(200)
            .set_body_json(serde_json::json!({"payload":{"data":STANDARD.encode("value")}})),
    )
    .expect(2)
    .mount(&server)
    .await;
    let manager = manager(&server, always_read, ttl);
    for _ in 0..2 {
        tokio::time::sleep(Duration::from_millis(5)).await;
        assert!(
            manager
                .get_secret_from_google_secret_manager("key")
                .await
                .unwrap()
                .is_some()
        );
    }
}

#[test]
fn google_manager_requires_host_license_and_project_configuration() {
    assert!(matches!(
        GoogleSecretManager::new(Arc::new(|_: &str| None), false),
        Err(Error::EnterpriseRequired)
    ));
    assert!(matches!(
        GoogleSecretManager::new(Arc::new(|_: &str| None), true),
        Err(Error::MissingEnvironment(
            "GOOGLE_SECRET_MANAGER_PROJECT_ID"
        ))
    ));
}

#[rstest::rstest]
#[case::boolean("true", Some(Secret::Bool(true)))]
#[case::null("null", None)]
#[case::string(
    "\"text\"",
    Some(Secret::String(litellm_secrets_types::SecretValue::new("text")))
)]
#[case::object(
    "{\"key\":1}",
    Secret::from_json(serde_json::json!({"key":1}))
)]
#[tokio::test]
async fn cached_values_preserve_python_json_conversion(
    #[case] raw: &str,
    #[case] expected: Option<Secret>,
) {
    let server = MockServer::start().await;
    Mock::given(path(
        "/v1/projects/project/secrets/key/versions/latest:access",
    ))
    .respond_with(
        ResponseTemplate::new(200)
            .set_body_json(serde_json::json!({"payload":{"data":STANDARD.encode(raw)}})),
    )
    .expect(1)
    .mount(&server)
    .await;
    let manager = manager(&server, false, Duration::from_secs(60));
    assert_eq!(
        manager
            .get_secret_from_google_secret_manager("key")
            .await
            .unwrap()
            .unwrap()
            .as_str(),
        Some(raw)
    );
    assert_eq!(
        manager
            .get_secret_from_google_secret_manager("key")
            .await
            .unwrap(),
        expected
    );
}
