use std::{
    fmt,
    sync::Arc,
    time::{Duration, Instant},
};

use litellm_auth_types::{Error, ErrorDetail, SecretValue};
use litellm_http::CredentialClient;
use moka::{
    Expiry,
    future::Cache,
    ops::compute::{CompResult, Op},
};
use sha2::{Digest, Sha256};

use crate::{config::WatsonxIamConfig, exchange::acquire_token};

const ACQUISITION_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Clone)]
pub struct WatsonxAuthService {
    tokens: Cache<TokenCacheKey, Arc<TokenEntry>>,
}

impl Default for WatsonxAuthService {
    fn default() -> Self {
        Self {
            tokens: Cache::builder()
                .max_capacity(64)
                .expire_after(TokenExpiry)
                .build(),
        }
    }
}

#[derive(Clone)]
pub struct WatsonxToken {
    key: TokenCacheKey,
    entry: Arc<TokenEntry>,
}

impl fmt::Debug for WatsonxToken {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("WatsonxToken")
            .field("key", &"[REDACTED]")
            .field("entry", &self.entry)
            .finish()
    }
}

impl WatsonxToken {
    pub fn secret(&self) -> &SecretValue {
        &self.entry.secret
    }
}

#[derive(Clone)]
pub(crate) struct TokenEntry {
    pub(crate) secret: SecretValue,
    pub(crate) usable_until: Instant,
}

impl fmt::Debug for TokenEntry {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TokenEntry")
            .field("secret", &"[REDACTED]")
            .field("usable_until", &self.usable_until)
            .finish()
    }
}

struct TokenExpiry;

impl Expiry<TokenCacheKey, Arc<TokenEntry>> for TokenExpiry {
    fn expire_after_create(
        &self,
        _key: &TokenCacheKey,
        entry: &Arc<TokenEntry>,
        _created_at: Instant,
    ) -> Option<Duration> {
        Some(entry.usable_until.saturating_duration_since(Instant::now()))
    }

    fn expire_after_update(
        &self,
        _key: &TokenCacheKey,
        entry: &Arc<TokenEntry>,
        _created_at: Instant,
        _duration: Option<Duration>,
    ) -> Option<Duration> {
        Some(entry.usable_until.saturating_duration_since(Instant::now()))
    }
}

#[derive(Clone, Hash, PartialEq, Eq)]
struct TokenCacheKey {
    api_key_digest: [u8; 32],
    endpoint: String,
}

impl fmt::Debug for TokenCacheKey {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TokenCacheKey")
            .field("api_key_digest", &"[REDACTED]")
            .field("endpoint", &"[REDACTED]")
            .finish()
    }
}

impl TokenCacheKey {
    fn new(config: &WatsonxIamConfig) -> Self {
        Self {
            api_key_digest: Sha256::digest(config.api_key().expose()).into(),
            endpoint: config.endpoint().to_string(),
        }
    }
}

impl WatsonxAuthService {
    pub async fn access_token(
        &self,
        config: &WatsonxIamConfig,
        http: &CredentialClient,
    ) -> Result<WatsonxToken, Error> {
        if config.api_key().expose().trim().is_empty() {
            return Err(Error::InvalidConfiguration(ErrorDetail::Empty {
                subject: "Watsonx API key".into(),
            }));
        }
        let key = TokenCacheKey::new(config);
        let result = tokio::time::timeout(
            ACQUISITION_TIMEOUT,
            self.tokens
                .try_get_with(key.clone(), acquire_token(config, http)),
        )
        .await
        .map_err(|error| Error::CredentialRequest {
            status: None,
            source: ErrorDetail::failed("Watsonx IAM token acquisition", error),
        })?
        .map_err(|error| (*error).clone())?;
        if result.usable_until <= Instant::now() {
            return Err(Error::InvalidCredentialResponse(ErrorDetail::InvalidType {
                field: "Watsonx IAM token response".into(),
                expected: "a token usable beyond the refresh buffer",
            }));
        }
        Ok(WatsonxToken {
            key,
            entry: Arc::clone(&result),
        })
    }

    pub async fn invalidate(&self, receipt: &WatsonxToken) -> bool {
        let result = self
            .tokens
            .entry(receipt.key.clone())
            .and_compute_with(|entry| async move {
                if entry.is_some_and(|entry| Arc::ptr_eq(entry.value(), &receipt.entry)) {
                    Op::Remove
                } else {
                    Op::Nop
                }
            })
            .await;
        matches!(result, CompResult::Removed(_))
    }
}

#[cfg(test)]
mod tests {
    use litellm_http::CredentialClient as Client;
    use rstest::{fixture, rstest};
    use serde_json::{Value, json};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use wiremock::{
        Mock, MockServer, ResponseTemplate,
        matchers::{body_string_contains, header, method, path},
    };

    use super::*;

    #[fixture]
    fn http() -> Client {
        Client::no_redirect_for_test()
    }

    #[fixture]
    fn service() -> WatsonxAuthService {
        WatsonxAuthService::default()
    }

    #[fixture]
    async fn server() -> MockServer {
        MockServer::start().await
    }

    #[fixture]
    async fn destination() -> MockServer {
        MockServer::start().await
    }

    fn config(server: &MockServer, api_key: &str) -> WatsonxIamConfig {
        WatsonxIamConfig::with_endpoint(
            SecretValue::new(api_key),
            format!("{}/identity/token", server.uri()),
        )
    }

    #[rstest]
    #[tokio::test]
    async fn shares_one_exchange_across_concurrent_calls_and_reuses_the_token(
        service: WatsonxAuthService,
        http: Client,
        #[future(awt)] server: MockServer,
    ) {
        Mock::given(method("POST"))
            .and(path("/identity/token"))
            .and(header("content-type", "application/x-www-form-urlencoded"))
            .and(body_string_contains("apikey=key%26value"))
            .and(body_string_contains(
                "grant_type=urn%3Aibm%3Aparams%3Aoauth%3Agrant-type%3Aapikey",
            ))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "access_token": "access-token",
                "expires_in": 3600
            })))
            .expect(1)
            .mount(&server)
            .await;
        let config = config(&server, "key&value");
        let (first, concurrent) = tokio::join!(
            service.access_token(&config, &http),
            service.access_token(&config, &http)
        );
        assert_eq!(first.unwrap().secret().expose(), "access-token");
        assert_eq!(concurrent.unwrap().secret().expose(), "access-token");
        assert_eq!(
            service
                .access_token(&config, &http)
                .await
                .unwrap()
                .secret()
                .expose(),
            "access-token"
        );
    }

    #[rstest]
    #[tokio::test]
    async fn refreshes_a_token_inside_its_expiration_buffer(
        service: WatsonxAuthService,
        http: Client,
        #[future(awt)] server: MockServer,
    ) {
        let config = config(&server, "key");
        let initial = Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "access_token": "short-lived",
                "expires_in": crate::exchange::REFRESH_BUFFER.as_secs()
            })))
            .expect(1)
            .mount_as_scoped(&server)
            .await;
        assert!(matches!(
            service.access_token(&config, &http).await,
            Err(Error::InvalidCredentialResponse(_))
        ));
        drop(initial);
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "access_token": "refreshed",
                "expires_in": 3600
            })))
            .expect(1)
            .mount(&server)
            .await;
        assert_eq!(
            service
                .access_token(&config, &http)
                .await
                .unwrap()
                .secret()
                .expose(),
            "refreshed"
        );
    }

    #[rstest]
    #[case::missing_token(json!({"expires_in": 3600}))]
    #[case::empty_token(json!({"access_token": " ", "expires_in": 3600}))]
    #[case::missing_expiration(json!({"access_token": "sensitive-value"}))]
    #[case::invalid_expiration(json!({"access_token": "token", "expires_in": "sensitive-value"}))]
    #[case::expired(json!({"access_token": "token", "expires_in": 0}))]
    #[tokio::test]
    async fn rejects_invalid_token_responses_without_exposing_them(
        service: WatsonxAuthService,
        http: Client,
        #[future(awt)] server: MockServer,
        #[case] body: Value,
    ) {
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(body))
            .mount(&server)
            .await;
        let error = service
            .access_token(&config(&server, "api-key"), &http)
            .await
            .unwrap_err();
        assert!(matches!(error, Error::InvalidCredentialResponse(_)));
        assert!(!error.to_string().contains("sensitive-value"));
        assert!(!error.to_string().contains("api-key"));
    }

    #[rstest]
    #[case::unauthorized(401)]
    #[case::rate_limit(429)]
    #[case::unavailable(503)]
    #[tokio::test]
    async fn failed_exchanges_are_not_cached_or_exposed(
        service: WatsonxAuthService,
        http: Client,
        #[future(awt)] server: MockServer,
        #[case] status: u16,
    ) {
        let config = WatsonxIamConfig::with_endpoint(
            SecretValue::new("secret-api-key"),
            format!("{}/identity/token?credential=secret-url", server.uri()),
        );
        let failure = Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(status).set_body_string("secret-response"))
            .expect(1)
            .mount_as_scoped(&server)
            .await;
        let error = service.access_token(&config, &http).await.unwrap_err();
        assert!(matches!(
            error,
            Error::CredentialRequest {
                status: Some(actual),
                ..
            } if actual == status
        ));
        assert!(!format!("{error:?}").contains("secret-"));
        drop(failure);
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "access_token": "recovered",
                "expires_in": 3600
            })))
            .expect(1)
            .mount(&server)
            .await;
        assert_eq!(
            service
                .access_token(&config, &http)
                .await
                .unwrap()
                .secret()
                .expose(),
            "recovered"
        );
    }

    #[rstest]
    #[tokio::test]
    async fn concurrent_failures_share_one_exchange_and_recover(
        service: WatsonxAuthService,
        http: Client,
        #[future(awt)] server: MockServer,
    ) {
        let failure = Mock::given(method("POST"))
            .respond_with(
                ResponseTemplate::new(503)
                    .set_delay(Duration::from_millis(20))
                    .set_body_string("secret-body"),
            )
            .expect(1)
            .mount_as_scoped(&server)
            .await;
        let config = config(&server, "key");
        let (first, second) = tokio::join!(
            service.access_token(&config, &http),
            service.access_token(&config, &http)
        );
        assert!(matches!(
            first,
            Err(Error::CredentialRequest {
                status: Some(503),
                ..
            })
        ));
        assert!(matches!(
            second,
            Err(Error::CredentialRequest {
                status: Some(503),
                ..
            })
        ));
        drop(failure);
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "access_token": "recovered",
                "expires_in": 3600
            })))
            .expect(1)
            .mount(&server)
            .await;
        assert_eq!(
            service
                .access_token(&config, &http)
                .await
                .unwrap()
                .secret()
                .expose(),
            "recovered"
        );
    }

    #[rstest]
    #[tokio::test]
    async fn canceled_initializer_releases_waiter(
        service: WatsonxAuthService,
        http: Client,
        #[future(awt)] server: MockServer,
    ) {
        Mock::given(method("POST"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_delay(Duration::from_millis(50))
                    .set_body_json(json!({
                        "access_token": "recovered",
                        "expires_in": 3600
                    })),
            )
            .mount(&server)
            .await;
        let config = config(&server, "key");
        let first = tokio::spawn({
            let service = service.clone();
            let http = http.clone();
            let config = config.clone();
            async move { service.access_token(&config, &http).await }
        });
        tokio::time::sleep(Duration::from_millis(10)).await;
        let second = tokio::spawn({
            let service = service.clone();
            let http = http.clone();
            async move { service.access_token(&config, &http).await }
        });
        first.abort();
        let result = tokio::time::timeout(Duration::from_secs(1), second)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(result.secret().expose(), "recovered");
    }

    #[rstest]
    #[tokio::test(start_paused = true)]
    async fn waiter_keeps_its_original_acquisition_deadline(
        service: WatsonxAuthService,
        http: Client,
        #[future(awt)] server: MockServer,
    ) {
        Mock::given(method("POST"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_delay(Duration::from_secs(60))
                    .set_body_json(json!({
                        "access_token": "too-late",
                        "expires_in": 3600
                    })),
            )
            .mount(&server)
            .await;
        let config = config(&server, "key");
        let first = tokio::spawn({
            let service = service.clone();
            let http = http.clone();
            let config = config.clone();
            async move { service.access_token(&config, &http).await }
        });
        tokio::time::advance(Duration::from_millis(1)).await;
        tokio::task::yield_now().await;
        let second = tokio::spawn({
            let service = service.clone();
            let http = http.clone();
            async move {
                let started = tokio::time::Instant::now();
                let result = service.access_token(&config, &http).await;
                (started.elapsed(), result)
            }
        });
        tokio::time::advance(Duration::from_secs(29)).await;
        assert!(matches!(
            first.await.unwrap(),
            Err(Error::CredentialRequest {
                status: None,
                source: ErrorDetail::Failed { .. },
            })
        ));
        let (elapsed, result) = second.await.unwrap();
        assert!(elapsed <= Duration::from_secs(31), "{elapsed:?}");
        assert!(matches!(
            result,
            Err(Error::CredentialRequest {
                status: None,
                source: ErrorDetail::Failed { .. },
            })
        ));
    }

    #[rstest]
    #[case::temporary_redirect(307)]
    #[case::permanent_redirect(308)]
    #[tokio::test]
    async fn credential_redirects_are_rejected_without_following(
        service: WatsonxAuthService,
        http: Client,
        #[future(awt)] server: MockServer,
        #[future(awt)] destination: MockServer,
        #[case] status: u16,
    ) {
        Mock::given(method("POST"))
            .and(path("/identity/token"))
            .respond_with(
                ResponseTemplate::new(status)
                    .insert_header("location", format!("{}/second", destination.uri()))
                    .set_body_string("secret-body"),
            )
            .expect(1)
            .mount(&server)
            .await;
        let error = service
            .access_token(&config(&server, "secret-key"), &http)
            .await
            .unwrap_err();
        assert!(matches!(
            error,
            Error::CredentialRequest {
                status: Some(actual),
                ..
            } if actual == status
        ));
        assert_eq!(server.received_requests().await.unwrap().len(), 1);
        assert_eq!(destination.received_requests().await.unwrap().len(), 0);
        assert!(!format!("{error:?}").contains("secret-body"));
        assert!(!format!("{error:?}").contains(&server.uri()));
    }

    #[rstest]
    #[tokio::test]
    async fn truncated_iam_response_has_no_http_status(service: WatsonxAuthService, http: Client) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut request = [0; 1024];
            assert_ne!(stream.read(&mut request).await.unwrap(), 0);
            stream
                .write_all(
                    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 64\r\nConnection: close\r\n\r\n{\"access_token\":\"partial\"",
                )
                .await
                .unwrap();
        });
        let config = WatsonxIamConfig::with_endpoint(
            SecretValue::new("secret-key"),
            format!("http://{address}/identity/token"),
        );
        let error = service.access_token(&config, &http).await.unwrap_err();
        server.await.unwrap();
        assert!(matches!(
            error,
            Error::CredentialRequest {
                status: None,
                source: ErrorDetail::Failed {
                    operation: "Watsonx IAM token response",
                    ..
                },
            }
        ));
        assert!(!format!("{error:?}").contains("partial"));
    }

    #[rstest]
    #[tokio::test]
    async fn credentials_do_not_share_cached_tokens(
        service: WatsonxAuthService,
        http: Client,
        #[future(awt)] server: MockServer,
    ) {
        Mock::given(body_string_contains("apikey=first-key"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "access_token": "first-token",
                "expires_in": 3600
            })))
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(body_string_contains("apikey=second-key"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "access_token": "second-token",
                "expires_in": 3600
            })))
            .expect(1)
            .mount(&server)
            .await;
        assert_eq!(
            service
                .access_token(&config(&server, "first-key"), &http)
                .await
                .unwrap()
                .secret()
                .expose(),
            "first-token"
        );
        assert_eq!(
            service
                .access_token(&config(&server, "second-key"), &http)
                .await
                .unwrap()
                .secret()
                .expose(),
            "second-token"
        );
    }

    #[rstest]
    #[tokio::test]
    async fn token_receipts_redact_tokens_keys_and_endpoints(
        service: WatsonxAuthService,
        http: Client,
        #[future(awt)] server: MockServer,
    ) {
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "access_token": "private-token",
                "expires_in": 3600
            })))
            .mount(&server)
            .await;
        let config = WatsonxIamConfig::with_endpoint(
            SecretValue::new("private-api-key"),
            format!("{}/identity/token?private-url", server.uri()),
        );
        let receipt = service.access_token(&config, &http).await.unwrap();
        let debug = format!("{receipt:?}");
        assert!(!debug.contains("private-token"));
        assert!(!debug.contains("private-api-key"));
        assert!(!debug.contains(&server.uri()));
    }

    #[rstest]
    #[tokio::test]
    async fn endpoints_do_not_share_cached_tokens(
        service: WatsonxAuthService,
        http: Client,
        #[future(awt)] server: MockServer,
    ) {
        Mock::given(path("/first"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "access_token": "first-token", "expires_in": 3600
            })))
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(path("/second"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "access_token": "second-token", "expires_in": 3600
            })))
            .expect(1)
            .mount(&server)
            .await;
        let first = WatsonxIamConfig::with_endpoint(
            SecretValue::new("key"),
            format!("{}/first", server.uri()),
        );
        let second = WatsonxIamConfig::with_endpoint(
            SecretValue::new("key"),
            format!("{}/second", server.uri()),
        );
        assert_eq!(
            service
                .access_token(&first, &http)
                .await
                .unwrap()
                .secret()
                .expose(),
            "first-token"
        );
        assert_eq!(
            service
                .access_token(&second, &http)
                .await
                .unwrap()
                .secret()
                .expose(),
            "second-token"
        );
        assert_eq!(
            service
                .access_token(&first, &http)
                .await
                .unwrap()
                .secret()
                .expose(),
            "first-token"
        );
    }

    #[rstest]
    #[tokio::test]
    async fn stale_receipts_cannot_invalidate_a_later_acquisition(
        service: WatsonxAuthService,
        http: Client,
        #[future(awt)] server: MockServer,
    ) {
        let config = config(&server, "key");
        let initial = Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "access_token": "rejected", "expires_in": 3600
            })))
            .expect(1)
            .mount_as_scoped(&server)
            .await;
        let rejected = service.access_token(&config, &http).await.unwrap();
        assert!(service.invalidate(&rejected).await);
        drop(initial);
        Mock::given(method("POST"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_delay(Duration::from_millis(20))
                    .set_body_json(json!({
                        "access_token": "rejected", "expires_in": 3600
                    })),
            )
            .expect(1)
            .mount(&server)
            .await;
        let refreshed = service.access_token(&config, &http).await.unwrap();
        assert_eq!(refreshed.secret().expose(), "rejected");
        assert!(!service.invalidate(&rejected).await);
        assert_eq!(
            service
                .access_token(&config, &http)
                .await
                .unwrap()
                .secret()
                .expose(),
            "rejected"
        );
    }

    #[rstest]
    #[tokio::test]
    async fn network_errors_have_no_http_status(service: WatsonxAuthService, http: Client) {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        drop(listener);
        let config = WatsonxIamConfig::with_endpoint(
            SecretValue::new("secret-key"),
            format!("http://{address}/?secret-query"),
        );
        let error = service.access_token(&config, &http).await.unwrap_err();
        assert!(matches!(
            error,
            Error::CredentialRequest { status: None, .. }
        ));
        assert!(!format!("{error:?}").contains("secret-"));
    }

    #[rstest]
    #[tokio::test]
    async fn uses_the_configured_http_client(
        service: WatsonxAuthService,
        #[future(awt)] server: MockServer,
    ) {
        let settings = litellm_http::HttpSettings {
            user_agent: Some("configured".into()),
            ..Default::default()
        };
        let client_config = litellm_http::Resolution::from(&settings).config;
        let pool = litellm_http::HttpClientPool::new(std::sync::Arc::new(
            litellm_http::media::PublicDnsResolver,
        ));
        let http = pool.credential_client(&client_config).unwrap();
        Mock::given(method("POST"))
            .and(header("user-agent", "configured"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "access_token": "token", "expires_in": 3600
            })))
            .expect(1)
            .mount(&server)
            .await;
        assert_eq!(
            service
                .access_token(&config(&server, "key"), &http)
                .await
                .unwrap()
                .secret()
                .expose(),
            "token"
        );
    }
}
