#![forbid(unsafe_code)]

use std::time::{Duration, Instant};

use litellm_auth_types::{Error, ErrorDetail, SecretValue};
use litellm_http::Client;
use moka::{
    Expiry,
    future::Cache,
    ops::compute::{CompResult, Op},
};
use serde::Deserialize;
use sha2::{Digest, Sha256};

const IAM_ENDPOINT: &str = "https://iam.cloud.ibm.com/identity/token";
const REFRESH_BUFFER: Duration = Duration::from_secs(10);

#[derive(Clone, Debug)]
pub struct WatsonxIamConfig {
    api_key: SecretValue,
    endpoint: String,
}

impl WatsonxIamConfig {
    pub fn new(api_key: SecretValue) -> Self {
        Self::with_endpoint(api_key, IAM_ENDPOINT.into())
    }

    pub fn with_endpoint(api_key: SecretValue, endpoint: String) -> Self {
        Self { api_key, endpoint }
    }
}

#[derive(Clone)]
pub struct WatsonxAuthService {
    tokens: Cache<TokenCacheKey, CachedToken>,
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

impl WatsonxAuthService {
    pub async fn access_token(
        &self,
        config: &WatsonxIamConfig,
        http: &Client,
    ) -> Result<SecretValue, Error> {
        if config.api_key.expose().trim().is_empty() {
            return Err(Error::InvalidConfiguration(ErrorDetail::Empty {
                subject: "Watsonx API key".into(),
            }));
        }
        self.tokens
            .entry(TokenCacheKey::new(config))
            .and_try_compute_with(|entry| async move {
                if entry.is_some() {
                    return Ok(Op::Nop);
                }
                acquire_token(config, http).await.map(Op::Put)
            })
            .await
            .map(|result| {
                result
                    .into_entry()
                    .expect("token acquisition retains an entry")
                    .into_value()
                    .token
            })
    }

    pub async fn invalidate_if_matches(
        &self,
        config: &WatsonxIamConfig,
        rejected: &SecretValue,
    ) -> bool {
        let result = self
            .tokens
            .entry(TokenCacheKey::new(config))
            .and_compute_with(|entry| async move {
                if entry.is_some_and(|entry| entry.value().token == *rejected) {
                    Op::Remove
                } else {
                    Op::Nop
                }
            })
            .await;
        matches!(result, CompResult::Removed(_))
    }
}

#[derive(Clone, Hash, PartialEq, Eq)]
struct TokenCacheKey {
    api_key_digest: [u8; 32],
    endpoint: String,
}

impl TokenCacheKey {
    fn new(config: &WatsonxIamConfig) -> Self {
        Self {
            api_key_digest: Sha256::digest(config.api_key.expose()).into(),
            endpoint: config.endpoint.clone(),
        }
    }
}

#[derive(Clone)]
struct CachedToken {
    token: SecretValue,
    ttl: Duration,
}

struct TokenExpiry;

impl Expiry<TokenCacheKey, CachedToken> for TokenExpiry {
    fn expire_after_create(
        &self,
        _key: &TokenCacheKey,
        token: &CachedToken,
        _created_at: Instant,
    ) -> Option<Duration> {
        Some(token.ttl)
    }
}

#[derive(Deserialize)]
struct IamTokenResponse {
    access_token: SecretValue,
    expires_in: u64,
}

async fn acquire_token(config: &WatsonxIamConfig, http: &Client) -> Result<CachedToken, Error> {
    let response = http
        .post(&config.endpoint)
        .header("Accept", "application/json")
        .form(&[
            ("grant_type", "urn:ibm:params:oauth:grant-type:apikey"),
            ("apikey", config.api_key.expose()),
        ])
        .timeout(Duration::from_secs(30))
        .send()
        .await
        .and_then(reqwest::Response::error_for_status)
        .map_err(|error| Error::CredentialRequest {
            status: error.status().map(|status| status.as_u16()),
            source: ErrorDetail::failed("Watsonx IAM token request", error.without_url()),
        })?
        .json::<IamTokenResponse>()
        .await
        .map_err(|_| invalid_token_response())?;
    if response.access_token.expose().trim().is_empty() || response.expires_in == 0 {
        return Err(invalid_token_response());
    }
    Ok(CachedToken {
        token: response.access_token,
        ttl: Duration::from_secs(response.expires_in).saturating_sub(REFRESH_BUFFER),
    })
}

fn invalid_token_response() -> Error {
    Error::InvalidCredentialResponse(ErrorDetail::InvalidType {
        field: "Watsonx IAM token response".into(),
        expected: "a nonempty access_token and positive expires_in",
    })
}

#[cfg(test)]
mod tests {
    use rstest::{fixture, rstest};
    use serde_json::{Value, json};
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
        assert_eq!(first.unwrap().expose(), "access-token");
        assert_eq!(concurrent.unwrap().expose(), "access-token");
        assert_eq!(
            service.access_token(&config, &http).await.unwrap().expose(),
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
                "expires_in": REFRESH_BUFFER.as_secs()
            })))
            .expect(1)
            .mount_as_scoped(&server)
            .await;
        assert_eq!(
            service.access_token(&config, &http).await.unwrap().expose(),
            "short-lived"
        );
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
            service.access_token(&config, &http).await.unwrap().expose(),
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
        assert!(error.to_string().contains(&status.to_string()));
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
            service.access_token(&config, &http).await.unwrap().expose(),
            "recovered"
        );
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
                .expose(),
            "first-token"
        );
        assert_eq!(
            service
                .access_token(&config(&server, "second-key"), &http)
                .await
                .unwrap()
                .expose(),
            "second-token"
        );
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
            service.access_token(&first, &http).await.unwrap().expose(),
            "first-token"
        );
        assert_eq!(
            service.access_token(&second, &http).await.unwrap().expose(),
            "second-token"
        );
        assert_eq!(
            service.access_token(&first, &http).await.unwrap().expose(),
            "first-token"
        );
    }

    #[rstest]
    #[tokio::test]
    async fn invalidates_only_the_rejected_credential_without_replaying(
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
        assert!(
            !service
                .invalidate_if_matches(&config, &SecretValue::new("other"))
                .await
        );
        assert_eq!(
            service.access_token(&config, &http).await.unwrap(),
            rejected
        );
        assert!(service.invalidate_if_matches(&config, &rejected).await);
        drop(initial);
        Mock::given(method("POST"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_delay(Duration::from_millis(20))
                    .set_body_json(json!({
                        "access_token": "refreshed", "expires_in": 3600
                    })),
            )
            .expect(1)
            .mount(&server)
            .await;
        let (refreshed, invalidated) = tokio::join!(
            service.access_token(&config, &http),
            service.invalidate_if_matches(&config, &rejected)
        );
        assert_eq!(refreshed.unwrap().expose(), "refreshed");
        assert!(!invalidated);
        assert!(!service.invalidate_if_matches(&config, &rejected).await);
        assert_eq!(
            service.access_token(&config, &http).await.unwrap().expose(),
            "refreshed"
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
        let http = pool
            .client(&client_config, litellm_http::ClientVariant::Provider)
            .unwrap();
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
                .expose(),
            "token"
        );
    }
}
