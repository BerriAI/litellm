//! google-auth's `service_account.Credentials` refresh: a self-signed assertion exchanged at the
//! key's `token_uri` through `_client.jwt_grant`.

use std::{sync::Arc, time::Duration};

use base64::{
    Engine,
    engine::general_purpose::{STANDARD, URL_SAFE_NO_PAD},
};
use ring::{
    rand::SystemRandom,
    signature::{RSA_PKCS1_SHA256, RsaKeyPair},
};
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
/// The PEM labels a service account's `private_key` may carry: PKCS#8, as Google issues keys,
/// and the PKCS#1 `RSA PRIVATE KEY` form.
const PKCS8_LABEL: &str = "PRIVATE KEY";
const PKCS1_LABEL: &str = "RSA PRIVATE KEY";
const UNLOADABLE_KEY: &str = "The service account private_key is not a PEM RSA private key";
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
    key: RsaKeyPair,
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
    let key = rsa_key_pair(info.private_key.as_deref().unwrap_or_default())?;
    Ok(ServiceAccountGrant {
        key,
        key_id: info.private_key_id.clone(),
        email: info.client_email.clone().unwrap_or_default(),
        token_uri: info.token_uri.clone().unwrap_or_default(),
        first_retry_wait,
        requests,
    })
}

/// The RSA key a PEM `private_key` holds, as PKCS#8 `PRIVATE KEY` or PKCS#1 `RSA PRIVATE KEY`.
fn rsa_key_pair(pem: &str) -> Result<RsaKeyPair, GoogleLoginFailure> {
    let unloadable = |detail: &str| {
        GoogleLoginFailure::new(
            PythonException::Value,
            format!("{UNLOADABLE_KEY}: {detail}."),
        )
    };
    let loaded = match (pem_body(pem, PKCS8_LABEL), pem_body(pem, PKCS1_LABEL)) {
        (Some(der), _) => RsaKeyPair::from_pkcs8(&der),
        (None, Some(der)) => RsaKeyPair::from_der(&der),
        (None, None) => return Err(unloadable("no readable PEM private key block")),
    };
    loaded.map_err(|rejected| unloadable(&rejected.to_string()))
}

/// The DER bytes of the first PEM block labelled `label`, or `None` when there is no such block
/// or its body is not base64.
fn pem_body(pem: &str, label: &str) -> Option<Vec<u8>> {
    let begin = format!("-----BEGIN {label}-----");
    let end = format!("-----END {label}-----");
    let (_, rest) = pem.split_once(&begin)?;
    let (body, _) = rest.split_once(&end)?;
    let body: String = body.chars().filter(|c| !c.is_ascii_whitespace()).collect();
    STANDARD.decode(body).ok()
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
        // PKCS#1 v1.5 padding is deterministic, so the random source is never drawn from.
        let mut signature = vec![0; self.key.public().modulus_len()];
        self.key
            .sign(
                &RSA_PKCS1_SHA256,
                &SystemRandom::new(),
                signing_input.as_bytes(),
                &mut signature,
            )
            .map_err(|_| {
                GoogleLoginFailure::new(
                    PythonException::Value,
                    "The service account private_key could not sign the assertion.",
                )
            })?;
        Ok(format!(
            "{signing_input}.{}",
            URL_SAFE_NO_PAD.encode(signature)
        ))
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
        Mutex,
        atomic::{AtomicUsize, Ordering},
    };

    use axum::{Router, routing::post};
    use ring::signature::{KeyPair, RSA_PKCS1_2048_8192_SHA256, UnparsedPublicKey};
    use rstest::rstest;
    use serde_json::{Value, json};

    use super::*;

    /// A fixed 2048-bit test key in PKCS#8 form, as Google issues service account keys.
    pub(crate) const PRIVATE_KEY: &str = "-----BEGIN PRIVATE KEY-----\nMIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQDGRPkmZkzek9X0\nGGxUpS00yBkTR9RHh1yXLK0SIoIT9SgyZaP5+gETM7Mo3oT4/q1EUnx+9WgvqZPn\nkgMP+/P2vNc07tgYn+DZqkn+M4ys2sGZnK+VQTpcw2AE+4zaSlJJztpGxyMkDAcg\ntHJeKHc0aWxw52d1K+SLzLeIz9Niy3uL5t7iZlFpUDA4LAfTCXlzYW+f+zcXXbof\nTyHRlfl6kvYoOa7ss+9YVXVhp4GOfhLOT4tsYfO3gbNSVrkxzdF6iLG2Wqh9GHMD\n7TdI3aSUXHj7qpwRx6PBY8eH74oHlUlSpi6BHgDGeTK/HdxbrI7WK6hT+9FIP55h\n1Tbk4twjAgMBAAECggEADPgIaNyalplXW2B/MlVp//o+uaYiWo0EhdbxEy7/dzK7\nO5Phn/S2CuPmcJWHLFcfgK9fxiYwET4N/XQgwydMRycFcXu8b6mCWa24qh03f6I0\nTuOS3WtA2ArnkvipUMLS+YT2sVga3yOL3wKVBvFHd6QvT0tNPYeh3j5SNJ7ASV8m\nH2RgRpz9LmOr6CCgOmr/52ldidQiBW3GtqEo8WilF7z+q9tQ6KDFZmcXU+4T2Xzs\nR5ZPIIkZni4X5vDhw7F6NSrbNMbUREDtVStdcwRqmMLoMs9DI6SxEKRnjsV0HxRm\nL0SIs+AiptJPIbVVzWHkMI+L/8J9qiTjl4nozWlZAQKBgQDn0fshqf6yD2SSUf7h\nmpJbkcyw54szZPfJRLftyo0KvKCIGjys+I0+cBsCHPafbOHB70aV9/7pzil3a1uw\nU0C7WOYeTdw9lxm5TVnaAYPJl/iqCfTm+eZkTPnDZRDOeJimoheLN2UB52yKx+78\nXl1AYn/VsnlWd+Kt3sfcng6sQQKBgQDa8yAAHgeDJ1LvfHebzZeJU1QAYw9mO/QJ\nrCMvMTJY2azGJdo61SohDzsWYaFMT/QdlCMYu8W7V2keDZVaJbV2PTulp/XQyOB2\nCOW2899644FMtCkcIv9gDYoVt/xY64TcX5uofq2wdr96pnzlarU8MueUR1qM+mU+\ngJLi1UJ/YwKBgAWihAEweaFoYGByH8jqEe6p35jZN/f6108MHZBpIP6G0TwdryXh\nMMW4wd9ORuxfICmvoD6a9ZZHTZi6glMJ4wPqyEUqTVUt7BEmAS4cZVCsIf44GbzX\njDMZ7+RA+6SZJ9S9EedvZEQklO3WqEd4H7ol6seN/AcPCPKC/MavDNpBAoGBANVc\nRrrWiIuOLouhxwzU8+6mfR1DayY9Fjq2vDKoxl8+/9a33EmooD1HElVlHkG3SDPq\nv6jCkpVVQOJAamJGjOaEFTQA0NJ8bf+X/AKk9ApU2RNeHRGGKUX4667uc8JTAE2f\n/ZfcmnVzwnRK+Tq3GZVoJbOhNORcvXyC3qCdIQfxAoGBAMipsn9l6sqMgNddwtzp\n2n7CuvycTgNQgS+wzy2e3JgQaJYlXTe8J5IVwRbq3MbICm+8gsfMc4cBCvRtmWOZ\nA9P2N1liNx1TYJ2D31nC1FgAt5z+X8ODDqgZjCXvCSVS+AbA0ivkQtzXm7zwLVD7\nKR6kOMrEoPsqjSaI14o9qNFx\n-----END PRIVATE KEY-----\n";

    /// [`PRIVATE_KEY`] in PKCS#1 `RSA PRIVATE KEY` form.
    const PKCS1_PRIVATE_KEY: &str = "-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIBAAKCAQEAxkT5JmZM3pPV9BhsVKUtNMgZE0fUR4dclyytEiKCE/UoMmWj\n+foBEzOzKN6E+P6tRFJ8fvVoL6mT55IDD/vz9rzXNO7YGJ/g2apJ/jOMrNrBmZyv\nlUE6XMNgBPuM2kpSSc7aRscjJAwHILRyXih3NGlscOdndSvki8y3iM/TYst7i+be\n4mZRaVAwOCwH0wl5c2Fvn/s3F126H08h0ZX5epL2KDmu7LPvWFV1YaeBjn4Szk+L\nbGHzt4GzUla5Mc3ReoixtlqofRhzA+03SN2klFx4+6qcEcejwWPHh++KB5VJUqYu\ngR4Axnkyvx3cW6yO1iuoU/vRSD+eYdU25OLcIwIDAQABAoIBAAz4CGjcmpaZV1tg\nfzJVaf/6PrmmIlqNBIXW8RMu/3cyuzuT4Z/0tgrj5nCVhyxXH4CvX8YmMBE+Df10\nIMMnTEcnBXF7vG+pglmtuKodN3+iNE7jkt1rQNgK55L4qVDC0vmE9rFYGt8ji98C\nlQbxR3ekL09LTT2Hod4+UjSewElfJh9kYEac/S5jq+ggoDpq/+dpXYnUIgVtxrah\nKPFopRe8/qvbUOigxWZnF1PuE9l87EeWTyCJGZ4uF+bw4cOxejUq2zTG1ERA7VUr\nXXMEapjC6DLPQyOksRCkZ47FdB8UZi9EiLPgIqbSTyG1Vc1h5DCPi//Cfaok45eJ\n6M1pWQECgYEA59H7Ian+sg9kklH+4ZqSW5HMsOeLM2T3yUS37cqNCrygiBo8rPiN\nPnAbAhz2n2zhwe9Glff+6c4pd2tbsFNAu1jmHk3cPZcZuU1Z2gGDyZf4qgn05vnm\nZEz5w2UQzniYpqIXizdlAedsisfu/F5dQGJ/1bJ5Vnfird7H3J4OrEECgYEA2vMg\nAB4HgydS73x3m82XiVNUAGMPZjv0CawjLzEyWNmsxiXaOtUqIQ87FmGhTE/0HZQj\nGLvFu1dpHg2VWiW1dj07paf10MjgdgjltvPfeuOBTLQpHCL/YA2KFbf8WOuE3F+b\nqH6tsHa/eqZ85Wq1PDLnlEdajPplPoCS4tVCf2MCgYAFooQBMHmhaGBgch/I6hHu\nqd+Y2Tf3+tdPDB2QaSD+htE8Ha8l4TDFuMHfTkbsXyApr6A+mvWWR02YuoJTCeMD\n6shFKk1VLewRJgEuHGVQrCH+OBm814wzGe/kQPukmSfUvRHnb2REJJTt1qhHeB+6\nJerHjfwHDwjygvzGrwzaQQKBgQDVXEa61oiLji6LoccM1PPupn0dQ2smPRY6trwy\nqMZfPv/Wt9xJqKA9RxJVZR5Bt0gz6r+owpKVVUDiQGpiRozmhBU0ANDSfG3/l/wC\npPQKVNkTXh0RhilF+Ouu7nPCUwBNn/2X3Jp1c8J0Svk6txmVaCWzoTTkXL18gt6g\nnSEH8QKBgQDIqbJ/ZerKjIDXXcLc6dp+wrr8nE4DUIEvsM8tntyYEGiWJV03vCeS\nFcEW6tzGyApvvILHzHOHAQr0bZljmQPT9jdZYjcdU2Cdg99ZwtRYALec/l/Dgw6o\nGYwl7wklUvgGwNIr5ELc15u88C1Q+ykepDjKxKD7Ko0miNeKPajRcQ==\n-----END RSA PRIVATE KEY-----\n";

    /// The assertion `jsonwebtoken`'s RS256 signer made from [`PRIVATE_KEY`] at [`NOW`], kept to
    /// show the `ring` signer makes the same bytes.
    const EXPECTED_ASSERTION: &str = "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiIsImtpZCI6ImtleS0xIn0.eyJpYXQiOjE3OTQ4MzA0MDAsImV4cCI6MTc5NDgzNDAwMCwiaXNzIjoic2FAcC5pYW0uZ3NlcnZpY2VhY2NvdW50LmNvbSIsImF1ZCI6Imh0dHBzOi8vb2F1dGgyLmdvb2dsZWFwaXMuY29tL3Rva2VuIiwic2NvcGUiOiJodHRwczovL3d3dy5nb29nbGVhcGlzLmNvbS9hdXRoL2Nsb3VkLXBsYXRmb3JtIn0.A70cgGdEj8lZ14LuFsxtqNBpMnaXajiZpmcCkLcjX2W3eCDtb_48AIoRTGgkFa4lHl6ftEaN8u05dV9rXMTITx7cV8QnyLwjo8-AgDxz4nTcFM9WgdjQaIk22H971166jrPJFQnfZ1LiM3Dwllr5SjWZ7MeUFDEw0cSEdIDSHAQfp9YVLua4435Qwgl_0XUDwoQUy_F__QrNDKtXHya1CXVWcqn5pxiHkPHibpp3qne_b1jOuFTG5XPz1m5XcJe65lbhmzxrJ8VopdApGjcKZKv_44Tib6GUItizrVlijjzIjb9IxujcsFT2tf41tvG1-_s5BhcQVM04q0Uw5RTkew";

    /// A service account key file's JSON, with `extra` fields added or replaced.
    pub(crate) fn key_json(token_uri: &str, extra: Value) -> Value {
        let info: serde_json::Map<String, Value> = json!({
            "type": "service_account", "client_email": "sa@p.iam.gserviceaccount.com",
            "private_key_id": "key-1", "private_key": PRIVATE_KEY, "token_uri": token_uri,
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
        assert!(verifies(
            &format!("{}.{}", segments[0], segments[1]),
            segments[2]
        ));
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

    /// Whether `signature` is [`PRIVATE_KEY`]'s RS256 signature of `signing_input`.
    fn verifies(signing_input: &str, signature: &str) -> bool {
        let der = pem_body(PRIVATE_KEY, PKCS8_LABEL).unwrap();
        let public = RsaKeyPair::from_pkcs8(&der)
            .unwrap()
            .public_key()
            .as_ref()
            .to_vec();
        UnparsedPublicKey::new(&RSA_PKCS1_2048_8192_SHA256, public)
            .verify(
                signing_input.as_bytes(),
                &URL_SAFE_NO_PAD.decode(signature).unwrap(),
            )
            .is_ok()
    }

    #[rstest]
    #[case::pkcs8(PRIVATE_KEY)]
    #[case::pkcs1(PKCS1_PRIVATE_KEY)]
    fn the_assertion_is_byte_identical_to_jsonwebtoken_s(#[case] pem: &str) {
        let assertion = grant(
            &key("http://127.0.0.1:9/token", json!({"private_key": pem})),
            FIRST_RETRY_WAIT,
            Requests::with_environment(&[]),
        )
        .unwrap()
        .assertion(NOW)
        .unwrap();
        assert_eq!(assertion, EXPECTED_ASSERTION);
    }

    #[rstest]
    fn the_assertion_signature_verifies_against_the_key() {
        let (signing_input, signature) = EXPECTED_ASSERTION.rsplit_once('.').unwrap();
        assert!(verifies(signing_input, signature));
    }

    #[rstest]
    #[case::not_pem("nope", "no readable PEM private key block")]
    #[case::bad_base64(
        "-----BEGIN PRIVATE KEY-----\n!!!\n-----END PRIVATE KEY-----\n",
        "no readable PEM private key block"
    )]
    #[case::not_a_key(
        "-----BEGIN PRIVATE KEY-----\nAAAA\n-----END PRIVATE KEY-----\n",
        "InvalidEncoding"
    )]
    fn an_unloadable_key_is_a_value_error(#[case] pem: &str, #[case] detail: &str) {
        let failure = grant(
            &key("http://127.0.0.1:9/token", json!({"private_key": pem})),
            FIRST_RETRY_WAIT,
            Requests::with_environment(&[]),
        )
        .err()
        .unwrap();
        assert_eq!(
            failure,
            GoogleLoginFailure::new(
                PythonException::Value,
                format!("{UNLOADABLE_KEY}: {detail}.")
            )
        );
    }
}
