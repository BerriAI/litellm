#![forbid(unsafe_code)]

use std::time::{Duration, Instant};

use litellm_auth_types::{Error, ErrorDetail, SecretValue};
use litellm_http::Client;
use moka::{Expiry, future::Cache};
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
        let key = TokenCacheKey {
            api_key_digest: Sha256::digest(config.api_key.expose()).into(),
            endpoint: config.endpoint.clone(),
        };
        self.tokens
            .try_get_with(key, acquire_token(config, http))
            .await
            .map(|cached| cached.token)
            .map_err(|error| (*error).clone())
    }
}

#[derive(Clone, Hash, PartialEq, Eq)]
struct TokenCacheKey {
    api_key_digest: [u8; 32],
    endpoint: String,
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
        .map_err(|error| {
            Error::CredentialAcquisition(ErrorDetail::failed(
                "Watsonx IAM token request",
                error.without_url(),
            ))
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
    Error::CredentialAcquisition(ErrorDetail::InvalidType {
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
        assert!(matches!(error, Error::CredentialAcquisition(_)));
        assert!(!error.to_string().contains("sensitive-value"));
        assert!(!error.to_string().contains("api-key"));
    }

    #[rstest]
    #[tokio::test]
    async fn failed_exchanges_are_not_cached_or_exposed(
        service: WatsonxAuthService,
        http: Client,
        #[future(awt)] server: MockServer,
    ) {
        let config = WatsonxIamConfig::with_endpoint(
            SecretValue::new("secret-api-key"),
            format!("{}/identity/token?credential=secret-url", server.uri()),
        );
        let failure = Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(401).set_body_string("secret-response"))
            .expect(1)
            .mount_as_scoped(&server)
            .await;
        let error = service.access_token(&config, &http).await.unwrap_err();
        assert!(matches!(error, Error::CredentialAcquisition(_)));
        assert!(error.to_string().contains("401"));
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
}
