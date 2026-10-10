use std::{
    collections::VecDeque,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, AtomicUsize, Ordering},
    },
};

use litellm_auth_copilot::{
    CopilotAuthService, CopilotSession, CopilotSessionSource, DEFAULT_API_BASE,
    PersistedSessionSource, SessionFuture,
};
use litellm_auth_types::{Error, SecretValue};
use rstest::rstest;
use serde_json::json;
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{header, method},
};

#[rstest]
#[case::default(None, DEFAULT_API_BASE)]
#[case::tenant(
    Some("https://api.enterprise.githubcopilot.com/"),
    "https://api.enterprise.githubcopilot.com"
)]
#[case::uppercase(Some("https://API.GITHUBCOPILOT.COM"), DEFAULT_API_BASE)]
#[case::apex(Some("https://githubcopilot.com"), "https://githubcopilot.com")]
#[case::attacker(Some("https://attacker.example"), DEFAULT_API_BASE)]
#[case::suffix_attack(Some("https://githubcopilot.com.attacker.example"), DEFAULT_API_BASE)]
#[case::prefix_attack(Some("https://attacker-githubcopilot.com"), DEFAULT_API_BASE)]
#[case::cleartext(Some("http://api.githubcopilot.com"), DEFAULT_API_BASE)]
#[case::username(
    Some("https://user@api.enterprise.githubcopilot.com"),
    DEFAULT_API_BASE
)]
#[case::port(
    Some("https://api.enterprise.githubcopilot.com:8443"),
    DEFAULT_API_BASE
)]
#[case::query(
    Some("https://api.enterprise.githubcopilot.com?redirect=attacker"),
    DEFAULT_API_BASE
)]
#[case::fragment(
    Some("https://api.enterprise.githubcopilot.com#attacker"),
    DEFAULT_API_BASE
)]
#[case::invalid(Some("not a URL"), DEFAULT_API_BASE)]
fn session_endpoints_are_restricted_to_trusted_https_hosts(
    #[case] endpoint: Option<&str>,
    #[case] expected: &str,
) {
    let session = CopilotSession::new(SecretValue::new("private-token"), endpoint, 200).unwrap();
    assert_eq!(session.api_base(), expected);
    assert!(!format!("{session:?}").contains("private-token"));
}

struct Sessions {
    pending: Mutex<VecDeque<Result<CopilotSession, Error>>>,
    acquisitions: AtomicUsize,
}

impl CopilotSessionSource for Sessions {
    fn acquire(&self, _: u64) -> SessionFuture<'_> {
        self.acquisitions.fetch_add(1, Ordering::SeqCst);
        let result = self
            .pending
            .lock()
            .unwrap()
            .pop_front()
            .expect("scripted session");
        Box::pin(async move {
            tokio::task::yield_now().await;
            result
        })
    }
}

fn session(token: &str, host: &str, expires: u64) -> CopilotSession {
    CopilotSession::new(SecretValue::new(token), Some(host), expires).unwrap()
}

#[tokio::test]
async fn concurrent_calls_share_a_session_and_refresh_token_and_endpoint_together() {
    let source = Arc::new(Sessions {
        pending: Mutex::new(VecDeque::from([
            Ok(session(
                "first-token",
                "https://first.githubcopilot.com",
                200,
            )),
            Ok(session(
                "second-token",
                "https://second.githubcopilot.com",
                300,
            )),
        ])),
        acquisitions: AtomicUsize::new(0),
    });
    let now = Arc::new(AtomicU64::new(100));
    let clock = now.clone();
    let service = CopilotAuthService::new(
        source.clone(),
        Arc::new(move || clock.load(Ordering::SeqCst)),
    );
    let (first, second, third) =
        tokio::join!(service.session(), service.session(), service.session());
    assert_eq!(first.unwrap().token().expose(), "first-token");
    assert_eq!(
        second.unwrap().api_base(),
        "https://first.githubcopilot.com"
    );
    assert_eq!(third.unwrap().token().expose(), "first-token");
    assert_eq!(source.acquisitions.load(Ordering::SeqCst), 1);
    now.store(200, Ordering::SeqCst);
    let refreshed = service.session().await.unwrap();
    assert_eq!(refreshed.token().expose(), "second-token");
    assert_eq!(refreshed.api_base(), "https://second.githubcopilot.com");
    assert_eq!(source.acquisitions.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn failed_refresh_never_returns_an_expired_token_and_can_be_retried() {
    let source = Arc::new(Sessions {
        pending: Mutex::new(VecDeque::from([
            Ok(session("expired", DEFAULT_API_BASE, 100)),
            Err(Error::ProviderAuthentication("refresh rejected".into())),
            Ok(session("fresh", DEFAULT_API_BASE, 200)),
        ])),
        acquisitions: AtomicUsize::new(0),
    });
    let service = CopilotAuthService::new(source.clone(), Arc::new(|| 100));
    assert!(service.session().await.is_err());
    assert_eq!(
        service.session().await.unwrap_err(),
        Error::ProviderAuthentication("refresh rejected".into())
    );
    assert_eq!(service.session().await.unwrap().token().expose(), "fresh");
    assert_eq!(source.acquisitions.load(Ordering::SeqCst), 3);
}

#[tokio::test]
async fn unexpired_persisted_session_needs_no_login_or_token_request() {
    let directory = tempfile::tempdir().unwrap();
    let cached = directory.path().join("session.json");
    std::fs::write(
        &cached,
        serde_json::to_vec(&json!({
            "token": "cached", "expires_at": 200,
            "endpoints": {"api": "https://tenant.githubcopilot.com"}
        }))
        .unwrap(),
    )
    .unwrap();
    let source = PersistedSessionSource::new(
        litellm_http::Client::no_redirect_for_test(),
        directory.path().join("missing-access-token"),
        cached,
        "http://127.0.0.1:1".into(),
    );
    let loaded = source.acquire(100).await.unwrap();
    assert_eq!(loaded.token().expose(), "cached");
    assert_eq!(loaded.api_base(), "https://tenant.githubcopilot.com");
}

#[rstest]
#[case::expired(json!({"token": "stale", "expires_at": 100}))]
#[case::malformed(json!({"token": "stale"}))]
#[tokio::test]
async fn unusable_persisted_sessions_refresh_with_the_login_token(
    #[case] cached_document: serde_json::Value,
) {
    let directory = tempfile::tempdir().unwrap();
    let cached = directory.path().join("session.json");
    let access = directory.path().join("access-token");
    std::fs::write(&cached, serde_json::to_vec(&cached_document).unwrap()).unwrap();
    std::fs::write(&access, "login-token\n").unwrap();
    let upstream = MockServer::start().await;
    Mock::given(method("GET"))
        .and(header("authorization", "token login-token"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "token": "refreshed", "expires_at": 200,
            "endpoints": {"api": "https://fresh.githubcopilot.com"}
        })))
        .expect(1)
        .mount(&upstream)
        .await;
    let source = PersistedSessionSource::new(
        litellm_http::Client::no_redirect_for_test(),
        access,
        cached,
        upstream.uri(),
    );
    let loaded = source.acquire(100).await.unwrap();
    assert_eq!(loaded.token().expose(), "refreshed");
    assert_eq!(loaded.api_base(), "https://fresh.githubcopilot.com");
}

#[tokio::test]
async fn token_endpoint_redirects_cannot_forward_login_credentials() {
    let directory = tempfile::tempdir().unwrap();
    let access = directory.path().join("access-token");
    std::fs::write(&access, "login-token").unwrap();
    let attacker = MockServer::start().await;
    let upstream = MockServer::start().await;
    Mock::given(method("GET"))
        .respond_with(ResponseTemplate::new(302).insert_header("location", attacker.uri()))
        .mount(&upstream)
        .await;
    let source = PersistedSessionSource::new(
        litellm_http::Client::no_redirect_for_test(),
        access,
        directory.path().join("missing-session"),
        upstream.uri(),
    );
    let error = source.acquire(100).await.unwrap_err();
    assert_eq!(
        error,
        Error::ProviderAuthentication("Copilot token request was rejected".into())
    );
    assert!(attacker.received_requests().await.unwrap().is_empty());
    assert!(!error.to_string().contains("login-token"));
}

#[rstest]
#[case::empty("")]
#[case::whitespace(" \n\t")]
fn empty_session_tokens_are_authentication_failures(#[case] token: &str) {
    assert_eq!(
        CopilotSession::new(SecretValue::new(token), None, 200).unwrap_err(),
        Error::ProviderAuthentication("Copilot session token is empty".into()),
    );
}
