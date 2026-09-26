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
        litellm_http::Client::plain_for_test(),
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

#[tokio::test]
async fn matching_checksum_is_accepted_and_cached() {
    let server = MockServer::start().await;
    let value = "private-value";
    Mock::given(path(
        "/v1/projects/project/secrets/key/versions/latest:access",
    ))
    .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
        "payload": {
            "data": STANDARD.encode(value),
            "dataCrc32c": crc32c::crc32c(value.as_bytes()).to_string()
        }
    })))
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
            Some(value)
        );
    }
}

enum ExpectedReadFailure {
    Missing,
    Status(u16),
    MissingPayload,
    Base64,
    Utf8,
    Checksum,
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
#[case::checksum_mismatch(
    200,
    serde_json::json!({"payload":{"data":STANDARD.encode("corrupt"),"dataCrc32c":"0"}}),
    ExpectedReadFailure::Checksum
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
        ExpectedReadFailure::Checksum => assert!(matches!(result, Err(Error::Checksum))),
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
        GoogleSecretManager::new(
            litellm_http::Client::plain_for_test(),
            Arc::new(|_: &str| None),
            false
        ),
        Err(Error::EnterpriseRequired)
    ));
    assert!(matches!(
        GoogleSecretManager::new(
            litellm_http::Client::plain_for_test(),
            Arc::new(|_: &str| None),
            true
        ),
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
        GoogleSecretManager::new(litellm_http::Client::plain_for_test(), environment, true),
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

#[tokio::test]
async fn trait_read_limits_the_operation_duration() {
    use litellm_secrets_types::{BaseSecretManager, GoogleOperationContext};
    use std::time::Duration;
    let server = MockServer::start().await;
    Mock::given(wiremock::matchers::method("GET"))
        .respond_with(ResponseTemplate::new(200).set_delay(Duration::from_secs(1)))
        .mount(&server)
        .await;
    let manager = manager(&server, false, Duration::from_secs(60));
    let context = GoogleOperationContext {
        timeout: Some(Duration::from_millis(30)),
    };
    assert!(matches!(
        BaseSecretManager::async_read_secret(&manager, "key", &context).await,
        Err(Error::Timeout)
    ));
}

#[tokio::test]
async fn concurrent_reads_share_one_secret_request() {
    let server = MockServer::start().await;
    Mock::given(wiremock::matchers::method("GET"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(serde_json::json!({"payload": {"data": STANDARD.encode("value")}}))
                .set_delay(Duration::from_millis(20)),
        )
        .expect(1)
        .mount(&server)
        .await;
    let manager = manager(&server, false, Duration::from_secs(60));
    let (first, second) = tokio::join!(
        manager.get_secret_from_google_secret_manager("key"),
        manager.get_secret_from_google_secret_manager("key")
    );
    assert_eq!(first.unwrap().unwrap().as_str(), Some("value"));
    assert_eq!(second.unwrap().unwrap().as_str(), Some("value"));
}

#[rstest]
#[case::missing(404, serde_json::json!({}))]
#[case::failure(403, serde_json::json!({}))]
#[case::no_payload(200, serde_json::json!({"payload":{}}))]
#[tokio::test]
async fn python_reads_reuse_cached_absence_until_expiry(
    #[case] status: u16,
    #[case] body: serde_json::Value,
    #[values(false, true)] always_read: bool,
) {
    let server = MockServer::start().await;
    let manager = manager(&server, always_read, Duration::from_secs(60));
    let failing = Mock::given(path(
        "/v1/projects/project/secrets/key/versions/latest:access",
    ))
    .respond_with(ResponseTemplate::new(status).set_body_json(body))
    .expect(1)
    .mount_as_scoped(&server)
    .await;
    let result = manager.get_secret_for_python("key").await;
    match status {
        404 => assert!(matches!(result, Err(Error::Status(404)))),
        403 => assert!(matches!(result, Err(Error::Status(403)))),
        200 => assert!(matches!(result, Err(Error::MissingPayload))),
        _ => unreachable!(),
    }
    drop(failing);
    Mock::given(path(
        "/v1/projects/project/secrets/key/versions/latest:access",
    ))
    .respond_with(
        ResponseTemplate::new(200)
            .set_body_json(serde_json::json!({"payload":{"data":STANDARD.encode("recovered")}})),
    )
    .expect(u64::from(always_read))
    .mount(&server)
    .await;
    assert_eq!(
        manager
            .get_secret_for_python("key")
            .await
            .unwrap()
            .as_ref()
            .and_then(|value| value.as_str()),
        always_read.then_some("recovered")
    );
}

#[tokio::test]
async fn python_cached_absence_expires_and_allows_recovery() {
    let server = MockServer::start().await;
    let manager = manager(&server, false, Duration::from_millis(20));
    let missing = Mock::given(path(
        "/v1/projects/project/secrets/key/versions/latest:access",
    ))
    .respond_with(ResponseTemplate::new(404))
    .expect(1)
    .mount_as_scoped(&server)
    .await;
    assert!(matches!(
        manager.get_secret_for_python("key").await,
        Err(Error::Status(404))
    ));
    drop(missing);
    tokio::time::sleep(Duration::from_millis(40)).await;
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
    assert_eq!(
        manager
            .get_secret_for_python("key")
            .await
            .unwrap()
            .unwrap()
            .as_str(),
        Some("recovered")
    );
}
