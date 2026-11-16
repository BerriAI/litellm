//! google-auth's `service_account.Credentials` refresh: a self-signed assertion exchanged at the
//! key's `token_uri` through `_client.jwt_grant`.

use std::{sync::Arc, time::Duration};

use jsonwebtoken::{Algorithm, EncodingKey, Header};
use litellm_http::{
    ClientVariant, HttpClientPool, HttpSettings, Resolution, media::PublicDnsResolver,
};
use litellm_python_compat::{
    Value as PyValue,
    json::{dumps, from_json},
    repr::{repr, to_str},
};
use litellm_secret_value::SecretValue;
use serde::Serialize;
use serde_json::{Map, Value};
use time::OffsetDateTime;

use super::{
    builders::token_failure,
    credentials::{Credentials, Expiry, Loaded, Minted, RefreshFuture},
    failure::{GoogleLoginFailure, PythonException},
};
use crate::constants::{CLOUD_PLATFORM_SCOPE, GOOGLE_OAUTH_TOKEN_ENDPOINT};

const JWT_BEARER_GRANT: &str = "urn:ietf:params:oauth:grant-type:jwt-bearer";
/// `service_account._DEFAULT_TOKEN_LIFETIME_SECS`.
const ASSERTION_LIFETIME: Duration = Duration::from_secs(3600);
/// `_exponential_backoff`'s three attempts.
const TOKEN_REQUEST_ATTEMPTS: u32 = 3;
/// `_exponential_backoff`'s first wait, doubled before each later attempt.
const FIRST_RETRY_WAIT: Duration = Duration::from_secs(1);
/// `transport.DEFAULT_RETRYABLE_STATUS_CODES`.
const RETRYABLE_STATUSES: [u16; 5] = [500, 503, 504, 408, 429];
const RETRYABLE_ERRORS: [&str; 3] = [
    "internal_failure",
    "server_error",
    "temporarily_unavailable",
];

#[derive(Serialize)]
struct Assertion<'a> {
    iat: i64,
    exp: i64,
    iss: &'a str,
    aud: &'a str,
    scope: &'a str,
}

pub(crate) struct ServiceAccountGrant {
    key: EncodingKey,
    key_id: Option<String>,
    email: String,
    token_uri: String,
    first_retry_wait: Duration,
    pool: HttpClientPool,
}

/// `service_account.Credentials.from_service_account_info` with the key already checked for the
/// fields google-auth requires.
pub(crate) fn service_account_grant(
    info: &Map<String, Value>,
) -> Result<Loaded, GoogleLoginFailure> {
    Ok(Arc::new(grant(info, FIRST_RETRY_WAIT)?))
}

fn text(info: &Map<String, Value>, field: &str) -> String {
    info.get(field)
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned()
}

fn grant(
    info: &Map<String, Value>,
    first_retry_wait: Duration,
) -> Result<ServiceAccountGrant, GoogleLoginFailure> {
    let key = EncodingKey::from_rsa_pem(text(info, "private_key").as_bytes())
        .map_err(|error| GoogleLoginFailure::new(PythonException::Value, error.to_string()))?;
    Ok(ServiceAccountGrant {
        key,
        key_id: info
            .get("private_key_id")
            .and_then(Value::as_str)
            .map(str::to_owned),
        email: text(info, "client_email"),
        token_uri: text(info, "token_uri"),
        first_retry_wait,
        pool: HttpClientPool::new(Arc::new(PublicDnsResolver)),
    })
}

/// `_client._handle_error_response`: a body that is not JSON is the text itself, else
/// `"{error}: {error_description}"` beside the response data.
fn refused(body: &str) -> GoogleLoginFailure {
    let data = match serde_json::from_str::<Value>(body) {
        Ok(Value::String(text)) => return GoogleLoginFailure::new(PythonException::Refresh, text),
        Ok(data) => data,
        Err(_) => return GoogleLoginFailure::new(PythonException::Refresh, body),
    };
    let details = match &data {
        Value::Object(fields) if fields.contains_key("error") => format!(
            "{}: {}",
            to_str(&from_json(fields["error"].clone())),
            to_str(&from_json(
                fields
                    .get("error_description")
                    .cloned()
                    .unwrap_or(Value::Null)
            ))
        ),
        other => dumps(&from_json(other.clone())).unwrap_or_else(|_| other.to_string()),
    };
    GoogleLoginFailure::new(
        PythonException::Refresh,
        format!(
            "({}, {})",
            repr(&PyValue::Str(details)),
            repr(&from_json(data))
        ),
    )
}

/// `_client._can_retry`.
fn retryable(status: u16, body: &str) -> bool {
    if RETRYABLE_STATUSES.contains(&status) {
        return true;
    }
    let Ok(Value::Object(fields)) = serde_json::from_str::<Value>(body) else {
        return false;
    };
    let field = |name: &str| match fields.get(name) {
        None | Some(Value::Null) => Some(""),
        Some(Value::String(text)) => Some(text.as_str()),
        Some(_) => None,
    };
    match (field("error"), field("error_description")) {
        (Some(code), Some(description)) => RETRYABLE_ERRORS
            .iter()
            .any(|retryable| *retryable == code || *retryable == description),
        _ => false,
    }
}

/// `_client._parse_expiry`.
fn expiry(granted: &Map<String, Value>, now: OffsetDateTime) -> Result<Expiry, GoogleLoginFailure> {
    let seconds = match granted.get("expires_in") {
        None | Some(Value::Null) => return Ok(Expiry::Never),
        Some(Value::Number(number)) => number.as_f64(),
        Some(Value::String(text)) => text
            .trim()
            .parse::<i64>()
            .ok()
            .map(|seconds| seconds as f64),
        Some(_) => None,
    };
    let seconds = seconds.ok_or_else(|| {
        GoogleLoginFailure::new(
            PythonException::Type,
            "unsupported type for timedelta seconds component",
        )
    })?;
    Ok(Expiry::At(now + time::Duration::seconds_f64(seconds)))
}

impl ServiceAccountGrant {
    fn assertion(&self, now: OffsetDateTime) -> Result<String, GoogleLoginFailure> {
        let issued = now.unix_timestamp();
        let header = Header {
            kid: self.key_id.clone(),
            ..Header::new(Algorithm::RS256)
        };
        jsonwebtoken::encode(
            &header,
            &Assertion {
                iat: issued,
                exp: issued + ASSERTION_LIFETIME.as_secs() as i64,
                iss: &self.email,
                aud: GOOGLE_OAUTH_TOKEN_ENDPOINT,
                scope: CLOUD_PLATFORM_SCOPE,
            },
            &self.key,
        )
        .map_err(|error| GoogleLoginFailure::new(PythonException::Value, error.to_string()))
    }

    async fn post(&self, assertion: &str) -> Result<(u16, String), GoogleLoginFailure> {
        let client = self
            .pool
            .client(
                &Resolution::from(&HttpSettings::default()).config,
                ClientVariant::Provider,
            )
            .map_err(|error| token_failure(&error))?;
        let response = client
            .post(&self.token_uri)
            .form(&[("grant_type", JWT_BEARER_GRANT), ("assertion", assertion)])
            .send()
            .await
            .map_err(|error| token_failure(&error))?;
        let status = response.status().as_u16();
        let body = response
            .text()
            .await
            .map_err(|error| token_failure(&error))?;
        Ok((status, body))
    }

    async fn mint(&self) -> Result<Minted, GoogleLoginFailure> {
        let now = OffsetDateTime::now_utc();
        let assertion = self.assertion(now)?;
        let mut wait = self.first_retry_wait;
        let mut answer = self.post(&assertion).await?;
        for _ in 1..TOKEN_REQUEST_ATTEMPTS {
            if answer.0 == 200 || !retryable(answer.0, &answer.1) {
                break;
            }
            tokio::time::sleep(wait).await;
            wait *= 2;
            answer = self.post(&assertion).await?;
        }
        let (status, body) = answer;
        if status != 200 {
            return Err(refused(&body));
        }
        let granted = match serde_json::from_str::<Value>(&body) {
            Ok(Value::Object(granted)) => granted,
            _ => return Err(refused(&body)),
        };
        let Some(token) = granted.get("access_token") else {
            return Err(GoogleLoginFailure::new(
                PythonException::Refresh,
                format!(
                    "('No access token in response.', {})",
                    repr(&from_json(Value::Object(granted)))
                ),
            ));
        };
        Ok(Minted {
            value: SecretValue::new(to_str(&from_json(token.clone()))),
            expiry: expiry(&granted, OffsetDateTime::now_utc())?,
        })
    }
}

impl Credentials for ServiceAccountGrant {
    fn refresh(&self) -> RefreshFuture<'_> {
        Box::pin(self.mint())
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::sync::{
        LazyLock, Mutex,
        atomic::{AtomicUsize, Ordering},
    };

    use axum::{Router, routing::post};
    use rsa::{RsaPrivateKey, pkcs8::EncodePrivateKey};
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    pub(crate) static PRIVATE_KEY: LazyLock<String> = LazyLock::new(|| {
        RsaPrivateKey::new(&mut rand::rngs::OsRng, 2048)
            .unwrap()
            .to_pkcs8_pem(rsa::pkcs8::LineEnding::LF)
            .unwrap()
            .to_string()
    });

    pub(crate) fn key(token_uri: &str, extra: Value) -> Map<String, Value> {
        json!({
            "type": "service_account", "client_email": "sa@p.iam.gserviceaccount.com",
            "private_key_id": "key-1", "private_key": *PRIVATE_KEY, "token_uri": token_uri,
        })
        .as_object()
        .unwrap()
        .clone()
        .into_iter()
        .chain(extra.as_object().unwrap().clone())
        .collect()
    }

    pub(crate) struct Endpoint {
        pub(crate) url: String,
        pub(crate) forms: Arc<Mutex<Vec<String>>>,
        server: tokio::task::JoinHandle<()>,
    }

    impl Drop for Endpoint {
        fn drop(&mut self) {
            self.server.abort();
        }
    }

    /// A token endpoint that answers with each scripted status and body in turn.
    pub(crate) async fn endpoint(answers: Vec<(u16, Value)>) -> Endpoint {
        let forms = Arc::new(Mutex::new(Vec::new()));
        let recorded = forms.clone();
        let next = Arc::new(AtomicUsize::new(0));
        let answers = Arc::new(answers);
        let app = Router::new().route(
            "/token",
            post(move |form: String| {
                let recorded = recorded.clone();
                let answers = answers.clone();
                let index = next.fetch_add(1, Ordering::SeqCst);
                async move {
                    recorded.lock().unwrap().push(form);
                    let (status, body) = answers[index.min(answers.len() - 1)].clone();
                    (
                        axum::http::StatusCode::from_u16(status).unwrap(),
                        body.to_string(),
                    )
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}/token", listener.local_addr().unwrap());
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        Endpoint { url, forms, server }
    }

    fn quick(url: &str) -> ServiceAccountGrant {
        grant(&key(url, json!({})), Duration::from_millis(1)).unwrap()
    }

    #[tokio::test]
    async fn the_assertion_is_google_auth_s_jwt_bearer_grant() {
        let endpoint = endpoint(vec![(
            200,
            json!({"access_token": "sa-token", "expires_in": 3599}),
        )])
        .await;
        let before = OffsetDateTime::now_utc();

        let minted = quick(&endpoint.url).mint().await.unwrap();

        assert_eq!(minted.value.expose(), "sa-token");
        let Expiry::At(expiry) = minted.expiry else {
            panic!("a granted token expires");
        };
        assert!(expiry >= before + time::Duration::seconds(3599));
        let form = endpoint.forms.lock().unwrap()[0].clone();
        let assertion = form
            .strip_prefix(
                "grant_type=urn%3Aietf%3Aparams%3Aoauth%3Agrant-type%3Ajwt-bearer&assertion=",
            )
            .unwrap();
        let header = jsonwebtoken::decode_header(assertion).unwrap();
        assert_eq!(
            (header.alg, header.kid.as_deref()),
            (Algorithm::RS256, Some("key-1"))
        );
        let claims: Value = serde_json::from_slice(
            &base64::Engine::decode(
                &base64::engine::general_purpose::URL_SAFE_NO_PAD,
                assertion.split('.').nth(1).unwrap(),
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(
            (
                &claims["iss"],
                &claims["aud"],
                &claims["scope"],
                claims["exp"].as_i64().unwrap() - claims["iat"].as_i64().unwrap()
            ),
            (
                &json!("sa@p.iam.gserviceaccount.com"),
                &json!(GOOGLE_OAUTH_TOKEN_ENDPOINT),
                &json!(CLOUD_PLATFORM_SCOPE),
                3600
            )
        );
    }

    #[rstest]
    #[case::invalid_grant(
        400,
        json!({"error": "invalid_grant", "error_description": "Invalid JWT Signature."}),
        "('invalid_grant: Invalid JWT Signature.', {'error': 'invalid_grant', 'error_description': 'Invalid JWT Signature.'})"
    )]
    #[case::no_description(
        400,
        json!({"error": "invalid_client"}),
        "('invalid_client: None', {'error': 'invalid_client'})"
    )]
    #[case::no_error_field(401, json!({"message": "no"}), "('{\"message\": \"no\"}', {'message': 'no'})")]
    #[case::json_text(403, json!("plain text"), "plain text")]
    #[tokio::test]
    async fn a_refusal_raises_google_auth_s_refresh_error(
        #[case] status: u16,
        #[case] answer: Value,
        #[case] expected: &str,
    ) {
        let endpoint = endpoint(vec![(status, answer)]).await;
        assert_eq!(
            quick(&endpoint.url).mint().await.unwrap_err(),
            GoogleLoginFailure::new(PythonException::Refresh, expected)
        );
        assert_eq!(endpoint.forms.lock().unwrap().len(), 1);
    }

    #[rstest]
    #[case::server_error(vec![(500, json!({})), (500, json!({})), (500, json!({}))], 3, false)]
    #[case::recovers(vec![(503, json!({})), (200, json!({"access_token": "t"}))], 2, true)]
    #[case::retryable_error_code(
        vec![(400, json!({"error": "temporarily_unavailable"})), (200, json!({"access_token": "t"}))],
        2,
        true
    )]
    #[tokio::test]
    async fn a_retryable_answer_is_tried_three_times_at_most(
        #[case] answers: Vec<(u16, Value)>,
        #[case] attempts: usize,
        #[case] granted: bool,
    ) {
        let endpoint = endpoint(answers).await;
        assert_eq!(quick(&endpoint.url).mint().await.is_ok(), granted);
        assert_eq!(endpoint.forms.lock().unwrap().len(), attempts);
    }

    #[rstest]
    #[case::missing(json!({"access_token": "t"}), Expiry::Never)]
    #[case::string_seconds(json!({"access_token": "t", "expires_in": "60"}), Expiry::At(NOW + time::Duration::seconds(60)))]
    #[case::number(json!({"access_token": "t", "expires_in": 60}), Expiry::At(NOW + time::Duration::seconds(60)))]
    fn the_expiry_is_parsed_as_google_auth_parses_it(
        #[case] granted: Value,
        #[case] expected: Expiry,
    ) {
        assert_eq!(expiry(granted.as_object().unwrap(), NOW).unwrap(), expected);
    }

    const NOW: OffsetDateTime = time::macros::datetime!(2026-11-16 12:00 UTC);

    #[tokio::test]
    async fn an_answer_without_a_token_is_refused() {
        let endpoint = endpoint(vec![(200, json!({"expires_in": 60}))]).await;
        assert_eq!(
            quick(&endpoint.url).mint().await.unwrap_err(),
            GoogleLoginFailure::new(
                PythonException::Refresh,
                "('No access token in response.', {'expires_in': 60})"
            )
        );
    }
}
