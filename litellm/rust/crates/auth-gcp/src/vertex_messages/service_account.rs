//! google-auth's `service_account.Credentials` refresh: a self-signed assertion exchanged at the
//! key's `token_uri` through `_client.jwt_grant`.

use std::{sync::Arc, time::Duration};

use base64::{Engine, engine::general_purpose::URL_SAFE_NO_PAD};
use jsonwebtoken::{Algorithm, EncodingKey};
use serde::Deserialize;
use time::OffsetDateTime;

use super::{
    builders::token_failure,
    credentials::{
        Credentials, Expiry, Lifetime, Loaded, Minted, RefreshFuture, TokenRefusal, TokenReply,
    },
    failure::{GoogleLoginFailure, PythonException},
    transport::Requests,
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

/// A service account key, as `from_service_account_info` reads it.
#[derive(Clone, Default, Deserialize, PartialEq, veil::Redact)]
pub(crate) struct ServiceAccountInfo {
    pub(crate) client_email: Option<String>,
    pub(crate) token_uri: Option<String>,
    #[redact(with = "[REDACTED]")]
    pub(crate) private_key: Option<String>,
    pub(crate) private_key_id: Option<String>,
    pub(crate) project_id: Option<String>,
}

/// A service account's token grant: the key it signs its assertion with and the endpoint it
/// exchanges the assertion at.
pub(crate) struct ServiceAccountGrant {
    key: EncodingKey,
    key_id: Option<String>,
    email: String,
    token_uri: String,
    first_retry_wait: Duration,
    requests: Requests,
}

/// `service_account.Credentials.from_service_account_info` with the key already checked for the
/// fields google-auth requires.
pub(crate) fn service_account_grant(
    info: &ServiceAccountInfo,
) -> Result<Loaded, GoogleLoginFailure> {
    Ok(Arc::new(grant(
        info,
        FIRST_RETRY_WAIT,
        Requests::process(),
    )?))
}

fn grant(
    info: &ServiceAccountInfo,
    first_retry_wait: Duration,
    requests: Requests,
) -> Result<ServiceAccountGrant, GoogleLoginFailure> {
    let key = EncodingKey::from_rsa_pem(info.private_key.as_deref().unwrap_or_default().as_bytes())
        .map_err(|error| GoogleLoginFailure::new(PythonException::Value, error.to_string()))?;
    Ok(ServiceAccountGrant {
        key,
        key_id: info.private_key_id.clone(),
        email: info.client_email.clone().unwrap_or_default(),
        token_uri: info.token_uri.clone().unwrap_or_default(),
        first_retry_wait,
        requests,
    })
}

/// A refused token request: the endpoint's `error` and `error_description`, or its text.
pub(crate) fn refused(body: &str) -> GoogleLoginFailure {
    let rejected = TokenRefusal::parse(body).is_some_and(|refusal| refusal.rejects_the_grant());
    let message = match (
        serde_json::from_str::<String>(body),
        TokenRefusal::parse(body),
    ) {
        (Ok(text), _) => text,
        (
            _,
            Some(TokenRefusal {
                error: Some(error),
                error_description: Some(description),
                ..
            }),
        ) => format!("{error}: {description}"),
        (
            _,
            Some(TokenRefusal {
                error: Some(error), ..
            }),
        ) => error,
        _ => body.to_owned(),
    };
    let failure = GoogleLoginFailure::new(PythonException::Refresh, message);
    match rejected {
        true => failure.rejecting_the_grant(),
        false => failure,
    }
}

/// `_client._can_retry`.
fn retryable(status: u16, body: &str) -> bool {
    if RETRYABLE_STATUSES.contains(&status) {
        return true;
    }
    let Some(refusal) = TokenRefusal::parse(body) else {
        return false;
    };
    let code = refusal.error.unwrap_or_default();
    let description = refusal.error_description.unwrap_or_default();
    RETRYABLE_ERRORS
        .iter()
        .any(|retryable| *retryable == code || *retryable == description)
}

/// `_client._parse_expiry`.
pub(crate) fn expiry(
    reply: &TokenReply,
    now: OffsetDateTime,
) -> Result<Expiry, GoogleLoginFailure> {
    let seconds = match &reply.expires_in {
        None => return Ok(Expiry::Never),
        Some(Lifetime::Seconds(seconds)) => Some(*seconds),
        Some(Lifetime::Text(text)) => text
            .trim()
            .parse::<i64>()
            .ok()
            .map(|seconds| seconds as f64),
    };
    let seconds = seconds.ok_or_else(invalid_lifetime)?;
    Ok(Expiry::At(now + time::Duration::seconds_f64(seconds)))
}

#[derive(serde::Serialize)]
struct AssertionHeader<'a> {
    typ: &'static str,
    alg: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    kid: Option<&'a str>,
}

#[derive(serde::Serialize)]
struct AssertionClaims<'a> {
    iat: i64,
    exp: i64,
    iss: &'a str,
    aud: &'static str,
    scope: &'static str,
}

impl ServiceAccountGrant {
    /// A self-signed RS256 JWT bearer assertion for the cloud-platform scope, as google-auth's
    /// `_make_authorization_grant_assertion` asks for.
    fn assertion(&self, now: OffsetDateTime) -> Result<String, GoogleLoginFailure> {
        let issued = now.unix_timestamp();
        let header = AssertionHeader {
            typ: "JWT",
            alg: "RS256",
            kid: self.key_id.as_deref(),
        };
        let claims = AssertionClaims {
            iat: issued,
            exp: issued + ASSERTION_LIFETIME.as_secs() as i64,
            iss: &self.email,
            aud: GOOGLE_OAUTH_TOKEN_ENDPOINT,
            scope: CLOUD_PLATFORM_SCOPE,
        };
        let segment = |json: serde_json::Result<Vec<u8>>| {
            json.map(|json| URL_SAFE_NO_PAD.encode(json))
                .map_err(|error| GoogleLoginFailure::new(PythonException::Value, error.to_string()))
        };
        let signing_input = format!(
            "{}.{}",
            segment(serde_json::to_vec(&header))?,
            segment(serde_json::to_vec(&claims))?
        );
        let signature =
            jsonwebtoken::crypto::sign(signing_input.as_bytes(), &self.key, Algorithm::RS256)
                .map_err(|error| {
                    GoogleLoginFailure::new(PythonException::Value, error.to_string())
                })?;
        Ok(format!("{signing_input}.{signature}"))
    }

    async fn mint(&self) -> Result<Minted, GoogleLoginFailure> {
        let assertion = self.assertion(OffsetDateTime::now_utc())?;
        let form = [
            ("assertion", assertion.as_str()),
            ("grant_type", JWT_BEARER_GRANT),
        ];
        let (status, body) = token_endpoint_request(
            &self.requests,
            &self.token_uri,
            &form,
            self.first_retry_wait,
        )
        .await?;
        if status != 200 {
            return Err(refused(&body));
        }
        let (token, reply) = granted(&body)?;
        Ok(Minted::from_reply(
            token,
            expiry(&reply, OffsetDateTime::now_utc())?,
        ))
    }
}

/// `_client._token_endpoint_request_no_throw`: a form post with `accept: */*`, tried again after
/// a retryable answer up to three times, waiting twice as long before each later attempt.
pub(crate) async fn token_endpoint_request(
    requests: &Requests,
    token_uri: &str,
    form: &[(&str, &str)],
    first_retry_wait: Duration,
) -> Result<(u16, String), GoogleLoginFailure> {
    let post = || async {
        let response = requests
            .client(token_uri)?
            .post(token_uri)
            .header(http::header::ACCEPT, "*/*")
            .form(form)
            .send()
            .await
            .map_err(|error| token_failure(&error))?;
        let status = response.status().as_u16();
        let body = response
            .text()
            .await
            .map_err(|error| token_failure(&error))?;
        Ok::<_, GoogleLoginFailure>((status, body))
    };
    let mut wait = first_retry_wait;
    let mut answer = post().await?;
    for _ in 1..TOKEN_REQUEST_ATTEMPTS {
        if answer.0 == 200 || !retryable(answer.0, &answer.1) {
            break;
        }
        tokio::time::sleep(wait).await;
        wait *= 2;
        answer = post().await?;
    }
    Ok(answer)
}

/// A token reply whose `expires_in` is not a number of seconds.
pub(crate) fn invalid_lifetime() -> GoogleLoginFailure {
    GoogleLoginFailure::new(
        PythonException::Refresh,
        "The token endpoint's reply has an expires_in that is not a number of seconds.",
    )
}

/// The granted token reply, which must hold `access_token`, and the token it holds.
pub(crate) fn granted(body: &str) -> Result<(Option<String>, TokenReply), GoogleLoginFailure> {
    let reply = TokenReply::parse(body)?;
    let Some(token) = reply.access_token.clone() else {
        return Err(GoogleLoginFailure::new(
            PythonException::Refresh,
            "The token endpoint's reply has no access_token.",
        ));
    };
    Ok((token, reply))
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
    use rsa::{
        RsaPrivateKey,
        pkcs8::{DecodePrivateKey, EncodePrivateKey, EncodePublicKey},
    };
    use rstest::rstest;
    use serde_json::{Value, json};

    use super::*;

    pub(crate) static PRIVATE_KEY: LazyLock<String> = LazyLock::new(|| {
        RsaPrivateKey::new(&mut rand::rngs::OsRng, 2048)
            .unwrap()
            .to_pkcs8_pem(rsa::pkcs8::LineEnding::LF)
            .unwrap()
            .to_string()
    });

    /// A service account key file's JSON, with `extra` fields added or replaced.
    pub(crate) fn key_json(token_uri: &str, extra: Value) -> Value {
        let info: serde_json::Map<String, Value> = json!({
            "type": "service_account", "client_email": "sa@p.iam.gserviceaccount.com",
            "private_key_id": "key-1", "private_key": *PRIVATE_KEY, "token_uri": token_uri,
        })
        .as_object()
        .unwrap()
        .clone()
        .into_iter()
        .chain(extra.as_object().unwrap().clone())
        .collect();
        Value::Object(info)
    }

    pub(crate) fn key(token_uri: &str, extra: Value) -> ServiceAccountInfo {
        serde_json::from_value(key_json(token_uri, extra)).unwrap()
    }

    pub(crate) struct Endpoint {
        pub(crate) url: String,
        pub(crate) forms: Arc<Mutex<Vec<String>>>,
        /// Each request's `accept` header.
        pub(crate) accepts: Arc<Mutex<Vec<Option<String>>>>,
        /// Each request's `content-type` header.
        pub(crate) content_types: Arc<Mutex<Vec<Option<String>>>>,
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
        let accepts = Arc::new(Mutex::new(Vec::new()));
        let accepted = accepts.clone();
        let content_types = Arc::new(Mutex::new(Vec::new()));
        let typed = content_types.clone();
        let next = Arc::new(AtomicUsize::new(0));
        let answers = Arc::new(answers);
        let app = Router::new().route(
            "/token",
            post(move |headers: axum::http::HeaderMap, form: String| {
                let recorded = recorded.clone();
                accepted.lock().unwrap().push(
                    headers
                        .get(axum::http::header::ACCEPT)
                        .map(|accept| accept.to_str().unwrap().to_owned()),
                );
                typed.lock().unwrap().push(
                    headers
                        .get(axum::http::header::CONTENT_TYPE)
                        .map(|kind| kind.to_str().unwrap().to_owned()),
                );
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
        Endpoint {
            url,
            forms,
            accepts,
            content_types,
            server,
        }
    }

    fn quick(url: &str) -> ServiceAccountGrant {
        grant(
            &key(url, json!({})),
            Duration::from_millis(1),
            Requests::with_environment(&[]),
        )
        .unwrap()
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

        assert_eq!(minted.token.unwrap().expose(), "sa-token");
        let Expiry::At(expiry) = minted.expiry else {
            panic!("a granted token expires");
        };
        assert!(expiry >= before + time::Duration::seconds(3599));
        let form = endpoint.forms.lock().unwrap()[0].clone();
        let assertion = form
            .strip_prefix("assertion=")
            .and_then(|form| {
                form.strip_suffix(
                    "&grant_type=urn%3Aietf%3Aparams%3Aoauth%3Agrant-type%3Ajwt-bearer",
                )
            })
            .unwrap();
        let segments: Vec<&str> = assertion.split('.').collect();
        let decoded = |segment: &str| {
            serde_json::from_slice::<Value>(
                &base64::Engine::decode(&base64::engine::general_purpose::URL_SAFE_NO_PAD, segment)
                    .unwrap(),
            )
            .unwrap()
        };
        let claims = decoded(segments[1]);
        let issued = claims["iat"].as_i64().unwrap();
        assert!((before.unix_timestamp()..=before.unix_timestamp() + 5).contains(&issued));
        assert_eq!(
            (decoded(segments[0]), claims),
            (
                json!({"typ": "JWT", "alg": "RS256", "kid": "key-1"}),
                json!({
                    "iat": issued, "exp": issued + 3600, "iss": "sa@p.iam.gserviceaccount.com",
                    "aud": GOOGLE_OAUTH_TOKEN_ENDPOINT, "scope": CLOUD_PLATFORM_SCOPE,
                })
            )
        );
        let public = RsaPrivateKey::from_pkcs8_pem(&PRIVATE_KEY)
            .unwrap()
            .to_public_key()
            .to_public_key_pem(rsa::pkcs8::LineEnding::LF)
            .unwrap();
        assert!(
            jsonwebtoken::crypto::verify(
                segments[2],
                format!("{}.{}", segments[0], segments[1]).as_bytes(),
                &jsonwebtoken::DecodingKey::from_rsa_pem(public.as_bytes()).unwrap(),
                Algorithm::RS256,
            )
            .unwrap()
        );
        assert_eq!(endpoint.accepts.lock().unwrap()[0].as_deref(), Some("*/*"));
    }

    #[rstest]
    #[case::invalid_grant(
        400,
        json!({"error": "invalid_grant", "error_description": "Invalid JWT Signature."}),
        "invalid_grant: Invalid JWT Signature."
    )]
    #[case::no_description(
        400,
        json!({"error": "invalid_client"}),
        "invalid_client"
    )]
    #[case::no_error_field(401, json!({"message": "no"}), r#"{"message":"no"}"#)]
    #[case::json_text(403, json!("plain text"), "plain text")]
    #[tokio::test]
    async fn a_refusal_is_a_refresh_error_naming_the_endpoint_s_error(
        #[case] status: u16,
        #[case] answer: Value,
        #[case] expected: &str,
    ) {
        let endpoint = endpoint(vec![(status, answer)]).await;
        let failure = quick(&endpoint.url).mint().await.unwrap_err();
        assert_eq!(
            (failure.exception, failure.message.as_str()),
            (PythonException::Refresh, expected)
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
        let reply = TokenReply::parse(&granted.to_string()).unwrap();
        assert_eq!(expiry(&reply, NOW).unwrap(), expected);
    }

    const NOW: OffsetDateTime = time::macros::datetime!(2026-11-16 12:00 UTC);

    #[tokio::test]
    async fn an_answer_without_a_token_is_refused() {
        let endpoint = endpoint(vec![(200, json!({"expires_in": 60}))]).await;
        assert_eq!(
            quick(&endpoint.url).mint().await.unwrap_err(),
            GoogleLoginFailure::new(
                PythonException::Refresh,
                "The token endpoint's reply has no access_token."
            )
        );
    }

    /// An endpoint answering 200 with a body that is not JSON text.
    async fn raw_endpoint(body: &'static str) -> (String, tokio::task::JoinHandle<()>) {
        let app = Router::new().route("/token", post(move || async move { body }));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}/token", listener.local_addr().unwrap());
        (
            url,
            tokio::spawn(async move { axum::serve(listener, app).await.unwrap() }),
        )
    }

    #[rstest]
    #[case::html("<html>token service unavailable</html>")]
    #[case::json_text("\"granted\"")]
    #[case::list("[1]")]
    #[case::number("7")]
    #[case::null("null")]
    #[tokio::test]
    async fn a_granted_body_that_is_not_an_object_is_a_refresh_error(#[case] body: &'static str) {
        let (url, server) = raw_endpoint(body).await;
        assert_eq!(
            quick(&url).mint().await.unwrap_err(),
            GoogleLoginFailure::new(
                PythonException::Refresh,
                "The token endpoint's reply is not a JSON object."
            )
        );
        server.abort();
    }

    #[rstest]
    #[case::named(json!({}), json!({"typ": "JWT", "alg": "RS256", "kid": "key-1"}))]
    #[case::null(json!({"private_key_id": null}), json!({"typ": "JWT", "alg": "RS256"}))]
    fn the_assertion_header_names_the_signing_key(#[case] extra: Value, #[case] expected: Value) {
        let assertion = grant(
            &key("http://127.0.0.1:9/token", extra),
            FIRST_RETRY_WAIT,
            Requests::with_environment(&[]),
        )
        .unwrap()
        .assertion(OffsetDateTime::now_utc())
        .unwrap();
        let header = base64::Engine::decode(
            &base64::engine::general_purpose::URL_SAFE_NO_PAD,
            assertion.split('.').next().unwrap(),
        )
        .unwrap();
        assert_eq!(serde_json::from_slice::<Value>(&header).unwrap(), expected);
    }
}
