use std::collections::BTreeMap;
use std::convert::Infallible;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use reqwest::header::{AUTHORIZATION, HeaderName, HeaderValue};
use reqwest::{Method, Request, Url};
use serde_json::json;
use sha2::{Digest, Sha256};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

use super::*;
use crate::provider_callbacks::handler::{
    AuthenticatedProviderRequest, ProviderAttemptContext, ProviderRequest, ProviderRequestBody,
    send_authenticated_provider_request,
};
use crate::provider_callbacks::{
    CallbackDecision, ProviderAttemptObserver, ProviderError, ProviderPostCall, ProviderPreCall,
};

fn runtime() -> Arc<AuthRuntime> {
    Arc::new(AuthRuntime {
        http: AuthHttpClient::new(
            reqwest::Client::builder(),
            Duration::from_secs(1),
            Duration::from_secs(2),
        )
        .unwrap(),
        clock: Arc::new(SystemClock),
        credentials: Arc::new(TestAdapter),
    })
}

async fn server(response: String) -> (Url, tokio::task::JoinHandle<String>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = Url::parse(&format!(
        "http://{}/operation?version=one",
        listener.local_addr().unwrap()
    ))
    .unwrap();
    let task = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut wire = Vec::new();
        let mut buffer = [0; 4096];
        loop {
            let count = socket.read(&mut buffer).await.unwrap();
            assert_ne!(count, 0);
            wire.extend_from_slice(&buffer[..count]);
            let text = String::from_utf8_lossy(&wire);
            if let Some((headers, body)) = text.split_once("\r\n\r\n") {
                let length = headers
                    .lines()
                    .find_map(|line| {
                        line.to_lowercase()
                            .strip_prefix("content-length: ")
                            .map(|value| value.parse::<usize>().unwrap())
                    })
                    .unwrap_or(0);
                if body.len() >= length {
                    break;
                }
            }
        }
        socket.write_all(response.as_bytes()).await.unwrap();
        String::from_utf8(wire).unwrap()
    });
    (url, task)
}

fn ok_response() -> String {
    "HTTP/1.1 200 OK\r\ncontent-length: 2\r\nconnection: close\r\n\r\n{}".into()
}

struct Signer(Arc<AtomicUsize>);

impl RequestAuthorizer for Signer {
    fn authorize(&self, mut request: Request) -> AuthFuture<'_, Request> {
        Box::pin(async move {
            self.0.fetch_add(1, Ordering::SeqCst);
            let body = request.body().and_then(|body| body.as_bytes()).unwrap();
            let signed = format!(
                "{} {} {}",
                request.method(),
                request.url(),
                String::from_utf8_lossy(body)
            );
            let value = format!("{:x}", Sha256::digest(signed.as_bytes()));
            request.headers_mut().insert(
                HeaderName::from_static("x-signature"),
                value.parse().unwrap(),
            );
            Ok(request)
        })
    }
}

struct TestAdapter;
enum TestConfig {
    Header,
    Signer(Arc<AtomicUsize>),
}

impl AuthAdapter for TestAdapter {
    type Config = TestConfig;

    fn build(
        &self,
        config: Self::Config,
        _runtime: Arc<AuthRuntime>,
    ) -> AuthFuture<'_, AuthBinding> {
        Box::pin(async move {
            let authorizer: Arc<dyn RequestAuthorizer> = match config {
                TestConfig::Header => Arc::new(StaticHeaderAuthorizer::new(
                    HeaderName::from_static("x-api-key"),
                    SecretString::new("test-key"),
                    vec![AUTHORIZATION],
                )),
                TestConfig::Signer(calls) => Arc::new(Signer(calls)),
            };
            Ok(AuthBinding::new(authorizer))
        })
    }
}

struct Observer {
    reject: bool,
    events: Vec<&'static str>,
    headers: BTreeMap<String, String>,
}

impl ProviderAttemptObserver for Observer {
    type Error = Infallible;
    async fn pre_call(&mut self, event: &ProviderPreCall) -> Result<CallbackDecision, Infallible> {
        self.events.push("pre");
        self.headers = event.headers.clone();
        Ok(if self.reject {
            CallbackDecision::Reject {
                message: "blocked".into(),
                status_code: None,
            }
        } else {
            CallbackDecision::Replace {
                payload: json!({"masked": true}),
            }
        })
    }
    async fn post_call(&mut self, _: &ProviderPostCall) -> Result<CallbackDecision, Infallible> {
        self.events.push("post");
        Ok(CallbackDecision::Unchanged)
    }
    async fn error(&mut self, _: &ProviderError) -> Result<(), Infallible> {
        self.events.push("error");
        Ok(())
    }
}

#[tokio::test]
async fn adapters_authorize_the_final_callback_body_through_the_shared_transport() {
    let resources = runtime();
    let calls = Arc::new(AtomicUsize::new(0));
    for config in [TestConfig::Header, TestConfig::Signer(calls.clone())] {
        let signed = matches!(config, TestConfig::Signer(_));
        let (url, server) = server(ok_response()).await;
        let session = TestAdapter
            .build(config, resources.clone())
            .await
            .unwrap()
            .bind(&url, ReplayPolicy::Never)
            .unwrap();
        let mut observer = Observer {
            reject: false,
            events: Vec::new(),
            headers: BTreeMap::new(),
        };
        let result = send_authenticated_provider_request(
            AuthenticatedProviderRequest {
                client: &resources.http,
                session: &session,
                request: resources
                    .http
                    .request(Method::POST, url.clone())
                    .header(AUTHORIZATION, "old-key")
                    .build()
                    .unwrap(),
                operation: AuthOperation::Initial,
                deadline: Instant::now() + Duration::from_secs(2),
                body: ProviderRequestBody::Json,
            },
            ProviderRequest {
                provider: "test".into(),
                model: "model".into(),
                body: BTreeMap::from([("private".into(), json!(true))]),
                api_base: url.to_string(),
                headers: BTreeMap::new(),
            },
            ProviderAttemptContext {
                call_id: "call".into(),
                trace_id: None,
                attempt: 1,
            },
            &mut observer,
        )
        .await
        .unwrap();
        let wire = server.await.unwrap();
        assert_eq!(result.status, 200);
        assert_eq!(observer.events, ["pre", "post"]);
        assert!(wire.starts_with("POST /operation?version=one HTTP/1.1\r\n"));
        assert!(wire.ends_with("{\"masked\":true}"));
        assert!(!wire.contains("private"));
        if signed {
            let expected = format!(
                "{:x}",
                Sha256::digest(format!("POST {url} {{\"masked\":true}}").as_bytes())
            );
            assert!(wire.contains(&format!("x-signature: {expected}\r\n")));
        } else {
            assert!(wire.contains("x-api-key: test-key\r\n"));
            assert!(!wire.contains("old-key"));
        }
    }
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn rejected_hooks_and_foreign_destinations_do_not_invoke_the_authorizer() {
    let resources = runtime();
    let calls = Arc::new(AtomicUsize::new(0));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = Url::parse(&format!("http://{}/", listener.local_addr().unwrap())).unwrap();
    let session =
        AuthSession::new(Arc::new(Signer(calls.clone())), &url, ReplayPolicy::Never).unwrap();
    let mut observer = Observer {
        reject: true,
        events: Vec::new(),
        headers: BTreeMap::new(),
    };
    let result = send_authenticated_provider_request(
        AuthenticatedProviderRequest {
            client: &resources.http,
            session: &session,
            request: resources
                .http
                .request(Method::POST, url.clone())
                .build()
                .unwrap(),
            operation: AuthOperation::Initial,
            deadline: Instant::now() + Duration::from_secs(1),
            body: ProviderRequestBody::Json,
        },
        ProviderRequest {
            provider: "test".into(),
            model: "model".into(),
            body: BTreeMap::new(),
            api_base: url.to_string(),
            headers: BTreeMap::new(),
        },
        ProviderAttemptContext {
            call_id: "call".into(),
            trace_id: None,
            attempt: 1,
        },
        &mut observer,
    )
    .await;
    assert!(matches!(result, Err(crate::Error::InvalidRequest(message)) if message == "blocked"));
    for (target, operation) in [
        ("https://example.com/", AuthOperation::Initial),
        ("https://example.com/", AuthOperation::FollowUp),
        (url.as_str(), AuthOperation::FollowUp),
    ] {
        let request = resources
            .http
            .request(Method::GET, Url::parse(target).unwrap())
            .build()
            .unwrap();
        assert!(matches!(
            resources
                .http
                .send(
                    &session,
                    request,
                    operation,
                    Instant::now() + Duration::from_secs(1)
                )
                .await,
            Err(crate::Error::InvalidResponse(_))
        ));
    }
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert!(
        tokio::time::timeout(Duration::from_millis(30), listener.accept())
            .await
            .is_err()
    );
}

struct RotatingToken(AtomicUsize);
impl TokenProvider for RotatingToken {
    fn token(&self) -> AuthFuture<'_, TokenCredential> {
        Box::pin(async move {
            Ok(TokenCredential::NoStore(SecretString::new(format!(
                "token-{}",
                self.0.fetch_add(1, Ordering::SeqCst)
            ))))
        })
    }
}

#[tokio::test]
async fn no_store_tokens_are_requested_for_every_attempt() {
    let resources = runtime();
    let provider = Arc::new(RotatingToken(AtomicUsize::new(0)));
    let authorizer = Arc::new(BearerTokenAuthorizer::new(
        provider.clone(),
        resources.clock.clone(),
        Vec::new(),
    ));
    for index in 0..2 {
        let (url, server) = server(ok_response()).await;
        let session = AuthSession::new(authorizer.clone(), &url, ReplayPolicy::SameOrigin).unwrap();
        let request = resources.http.request(Method::GET, url).build().unwrap();
        resources
            .http
            .send(
                &session,
                request,
                AuthOperation::FollowUp,
                Instant::now() + Duration::from_secs(1),
            )
            .await
            .unwrap();
        assert!(
            server
                .await
                .unwrap()
                .contains(&format!("authorization: Bearer token-{index}\r\n"))
        );
    }
    assert_eq!(provider.0.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn redirects_cannot_bypass_destination_checks() {
    let resources = runtime();
    let target = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let response = format!(
        "HTTP/1.1 302 Found\r\nlocation: http://{}/stolen\r\ncontent-length: 0\r\nconnection: close\r\n\r\n",
        target.local_addr().unwrap()
    );
    let (url, server) = server(response).await;
    let session = TestAdapter
        .build(TestConfig::Header, resources.clone())
        .await
        .unwrap()
        .bind(&url, ReplayPolicy::SameOrigin)
        .unwrap();
    let request = resources.http.request(Method::GET, url).build().unwrap();
    assert_eq!(
        resources
            .http
            .send(
                &session,
                request,
                AuthOperation::Initial,
                Instant::now() + Duration::from_secs(1)
            )
            .await
            .unwrap()
            .status(),
        302
    );
    server.await.unwrap();
    assert!(
        tokio::time::timeout(Duration::from_millis(30), target.accept())
            .await
            .is_err()
    );
}

struct PendingToken;
impl TokenProvider for PendingToken {
    fn token(&self) -> AuthFuture<'_, TokenCredential> {
        Box::pin(std::future::pending())
    }
}

#[tokio::test]
async fn deadline_bounds_credential_acquisition_before_network_io() {
    let resources = runtime();
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = Url::parse(&format!("http://{}/", listener.local_addr().unwrap())).unwrap();
    let authorizer = Arc::new(BearerTokenAuthorizer::new(
        Arc::new(PendingToken),
        resources.clock.clone(),
        Vec::new(),
    ));
    let session = AuthSession::new(authorizer, &url, ReplayPolicy::Never).unwrap();
    let request = resources.http.request(Method::GET, url).build().unwrap();
    assert!(
        matches!(resources.http.send(&session, request, AuthOperation::Initial, Instant::now() + Duration::from_millis(10)).await, Err(crate::Error::Network(message)) if message == "Request timed out")
    );
}

#[tokio::test]
async fn secrets_are_redacted_and_header_injection_is_rejected() {
    let secret = SecretString::new("private-token");
    assert_eq!(format!("{secret:?}"), "[redacted]");
    let authorizer = StaticHeaderAuthorizer::new(AUTHORIZATION, secret, Vec::new());
    let request = reqwest::Client::new()
        .get("https://example.com/")
        .build()
        .unwrap();
    let request = authorizer.authorize(request).await.unwrap();
    assert!(request.headers()[AUTHORIZATION].is_sensitive());
    assert_eq!(
        request.headers()[AUTHORIZATION],
        HeaderValue::from_static("private-token")
    );
    let invalid = StaticHeaderAuthorizer::new(
        AUTHORIZATION,
        SecretString::new("token\r\nx-secret: bad"),
        Vec::new(),
    );
    assert!(matches!(
        invalid.authorize(request).await,
        Err(AuthError {
            kind: AuthErrorKind::InvalidHeader,
            ..
        })
    ));
}

struct ExpiredToken;
impl TokenProvider for ExpiredToken {
    fn token(&self) -> AuthFuture<'_, TokenCredential> {
        Box::pin(async {
            Ok(TokenCredential::Cached(TokenLease {
                token: SecretString::new("expired-secret"),
                expires_at: Instant::now() - Duration::from_secs(1),
            }))
        })
    }
}

#[tokio::test]
async fn expired_credentials_fail_before_the_provider_request() {
    let resources = runtime();
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = Url::parse(&format!("http://{}/", listener.local_addr().unwrap())).unwrap();
    let session = AuthSession::new(
        Arc::new(BearerTokenAuthorizer::new(
            Arc::new(ExpiredToken),
            resources.clock.clone(),
            Vec::new(),
        )),
        &url,
        ReplayPolicy::Never,
    )
    .unwrap();
    let request = resources.http.request(Method::POST, url).build().unwrap();
    let result = resources
        .http
        .send(
            &session,
            request,
            AuthOperation::Initial,
            Instant::now() + Duration::from_secs(1),
        )
        .await;
    assert!(
        matches!(result, Err(crate::Error::Auth(message)) if message == "credential provider returned an expired token")
    );
    assert!(
        tokio::time::timeout(Duration::from_millis(30), listener.accept())
            .await
            .is_err()
    );
}

#[tokio::test]
async fn callbacks_and_transport_share_one_credential_acquisition_per_attempt() {
    let resources = runtime();
    let provider = Arc::new(RotatingToken(AtomicUsize::new(0)));
    let (url, server) = server(ok_response()).await;
    let session = AuthSession::new(
        Arc::new(BearerTokenAuthorizer::new(
            provider.clone(),
            resources.clock.clone(),
            Vec::new(),
        )),
        &url,
        ReplayPolicy::Never,
    )
    .unwrap();
    let mut observer = Observer {
        reject: false,
        events: Vec::new(),
        headers: BTreeMap::new(),
    };
    send_authenticated_provider_request(
        AuthenticatedProviderRequest {
            client: &resources.http,
            session: &session,
            request: resources
                .http
                .request(Method::POST, url.clone())
                .build()
                .unwrap(),
            operation: AuthOperation::Initial,
            deadline: Instant::now() + Duration::from_secs(1),
            body: ProviderRequestBody::Json,
        },
        ProviderRequest {
            provider: "test".into(),
            model: "model".into(),
            body: BTreeMap::new(),
            api_base: url.to_string(),
            headers: BTreeMap::new(),
        },
        ProviderAttemptContext {
            call_id: "call".into(),
            trace_id: None,
            attempt: 1,
        },
        &mut observer,
    )
    .await
    .unwrap();
    assert_eq!(observer.headers["authorization"], "Bearer token-0");
    assert!(
        server
            .await
            .unwrap()
            .contains("authorization: Bearer token-0\r\n")
    );
    assert_eq!(provider.0.load(Ordering::SeqCst), 1);
}

impl CredentialResolver for TestAdapter {
    fn resolve(
        &self,
        credential: CredentialSpec,
        _runtime: Arc<AuthRuntime>,
    ) -> AuthFuture<'_, AuthBinding> {
        Box::pin(async move {
            match credential {
                CredentialSpec::Header { name, value } => Ok(AuthBinding::new(Arc::new(
                    StaticHeaderAuthorizer::new(name, value, Vec::new()),
                ))),
            }
        })
    }
}

#[tokio::test]
async fn expired_deadline_does_not_invoke_the_authorizer() {
    let calls = Arc::new(AtomicUsize::new(0));
    let runtime = runtime();
    let url = Url::parse("http://127.0.0.1:1/request").unwrap();
    let session =
        AuthSession::new(Arc::new(Signer(calls.clone())), &url, ReplayPolicy::Never).unwrap();
    let error = runtime
        .http
        .send(
            &session,
            runtime.http.request(Method::POST, url).build().unwrap(),
            AuthOperation::Initial,
            Instant::now(),
        )
        .await
        .unwrap_err();
    assert!(matches!(error, crate::Error::Network(_)));
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}
