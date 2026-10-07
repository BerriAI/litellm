use std::{sync::Arc, time::Duration};

use base64::{Engine, engine::general_purpose::STANDARD};
use litellm_secrets_google::{Error, GoogleSecretManager};

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
#[case::not_found(404, serde_json::json!({}))]
#[case::unauthorized(401, serde_json::json!({}))]
#[case::forbidden(403, serde_json::json!({}))]
#[case::throttled(429, serde_json::json!({}))]
#[case::unavailable(503, serde_json::json!({}))]
#[case::missing_payload(200, serde_json::json!({"payload":{}}))]
#[case::invalid_base64(200, serde_json::json!({"payload":{"data":"%%%"}}))]
#[tokio::test]
async fn failed_or_missing_reads_are_not_cached(
    #[case] status: u16,
    #[case] body: serde_json::Value,
) {
    let server = MockServer::start().await;
    let manager = manager(&server, false, Duration::from_secs(60));
    let failing = Mock::given(path(
        "/v1/projects/project/secrets/key/versions/latest:access",
    ))
    .respond_with(ResponseTemplate::new(status).set_body_json(body))
    .expect(1)
    .mount_as_scoped(&server)
    .await;
    let result = manager.get_secret_from_google_secret_manager("key").await;
    match status {
        404 => assert_eq!(result.unwrap(), None),
        200 => assert!(matches!(
            result,
            Err(Error::MissingPayload | Error::Base64(_))
        )),
        status => assert!(matches!(result, Err(Error::Status(actual)) if actual == status)),
    }
    drop(failing);
    Mock::given(path(
        "/v1/projects/project/secrets/key/versions/latest:access",
    ))
    .respond_with(
        ResponseTemplate::new(200)
            .set_body_json(serde_json::json!({"payload":{"data":STANDARD.encode("recovered")}})),
    )
    .expect(1)
    .mount(&server)
    .await;
    for _ in 0..2 {
        assert_eq!(
            manager
                .get_secret_from_google_secret_manager("key")
                .await
                .unwrap()
                .unwrap()
                .as_str(),
            Some("recovered")
        );
    }
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
#[case("true")]
#[case("null")]
#[case("\"text\"")]
#[case("{\"key\":1}")]
#[tokio::test]
async fn cache_preserves_raw_values(#[case] raw: &str) {
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
    for _ in 0..2 {
        assert_eq!(
            manager
                .get_secret_from_google_secret_manager("key")
                .await
                .unwrap()
                .unwrap()
                .as_str(),
            Some(raw)
        );
    }
}
