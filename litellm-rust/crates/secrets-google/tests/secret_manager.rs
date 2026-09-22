use std::{sync::Arc, time::Duration};

use base64::{Engine, engine::general_purpose::STANDARD};
use litellm_secrets_google::{Error, GoogleSecretManager};
use rstest::{fixture, rstest};

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

#[fixture]
fn default_ttl() -> Duration {
    Duration::from_secs(60)
}

#[rstest]
#[case::nonempty("private-value")]
#[case::empty("")]
#[tokio::test]
async fn successful_reads_use_auth_latest_version_and_cache_including_empty_values(
    default_ttl: Duration,
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
    let manager = manager(&server, false, default_ttl);
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

enum ExpectedReadFailure {
    Missing,
    Status(u16),
    MissingPayload,
    Base64,
    Utf8,
}

#[rstest]
#[case::not_found(404, serde_json::json!({}), ExpectedReadFailure::Missing)]
#[case::unauthorized(401, serde_json::json!({}), ExpectedReadFailure::Status(401))]
#[case::forbidden(403, serde_json::json!({}), ExpectedReadFailure::Status(403))]
#[case::throttled(429, serde_json::json!({}), ExpectedReadFailure::Status(429))]
#[case::unavailable(503, serde_json::json!({}), ExpectedReadFailure::Status(503))]
#[case::missing_payload(
    200,
    serde_json::json!({"payload":{}}),
    ExpectedReadFailure::MissingPayload
)]
#[case::invalid_base64(
    200,
    serde_json::json!({"payload":{"data":"%%%"}}),
    ExpectedReadFailure::Base64
)]
#[case::invalid_utf8(
    200,
    serde_json::json!({"payload":{"data":STANDARD.encode([0xff])}}),
    ExpectedReadFailure::Utf8
)]
#[tokio::test]
async fn failed_or_missing_reads_are_not_cached(
    default_ttl: Duration,
    #[case] status: u16,
    #[case] body: serde_json::Value,
    #[case] expected: ExpectedReadFailure,
) {
    let server = MockServer::start().await;
    let manager = manager(&server, false, default_ttl);
    let failing = Mock::given(path(
        "/v1/projects/project/secrets/key/versions/latest:access",
    ))
    .respond_with(ResponseTemplate::new(status).set_body_json(body))
    .expect(1)
    .mount_as_scoped(&server)
    .await;
    let result = manager.get_secret_from_google_secret_manager("key").await;
    match expected {
        ExpectedReadFailure::Missing => assert_eq!(result.unwrap(), None),
        ExpectedReadFailure::Status(expected) => {
            assert!(matches!(result, Err(Error::Status(actual)) if actual == expected));
        }
        ExpectedReadFailure::MissingPayload => {
            assert!(matches!(result, Err(Error::MissingPayload)));
        }
        ExpectedReadFailure::Base64 => assert!(matches!(result, Err(Error::Base64(_)))),
        ExpectedReadFailure::Utf8 => assert!(matches!(result, Err(Error::Utf8))),
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

#[rstest]
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

#[rstest]
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

#[rstest]
#[case::provider_specific("GOOGLE_SECRET_MANAGER_REFRESH_INTERVAL")]
#[case::shared("SECRET_MANAGER_REFRESH_INTERVAL")]
fn google_manager_rejects_invalid_refresh_intervals(#[case] variable: &'static str) {
    let environment = Arc::new(move |name: &str| match name {
        "GOOGLE_SECRET_MANAGER_PROJECT_ID" => Some("project".to_owned()),
        name if name == variable => Some("not-a-number".to_owned()),
        _ => None,
    });

    assert!(matches!(
        GoogleSecretManager::new(environment, true),
        Err(Error::RefreshInterval)
    ));
}

#[rstest]
#[case::boolean("true")]
#[case::null("null")]
#[case::string("\"text\"")]
#[case::object("{\"key\":1}")]
#[tokio::test]
async fn cache_preserves_raw_values(default_ttl: Duration, #[case] raw: &str) {
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
    let manager = manager(&server, false, default_ttl);
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
