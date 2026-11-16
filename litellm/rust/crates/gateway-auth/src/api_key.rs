use axum::{
    extract::FromRequestParts,
    http::{HeaderMap, header::AUTHORIZATION, request::Parts},
};
use litellm_secret_value::SecretValue;

use crate::{Credential, Error};

/// The caller's key, read from the headers in the order `get_api_key` in `user_api_key_auth.py`
/// reads them and with its scheme handling. A request without one is rejected with
/// [`Error::InvalidToken`].
pub struct ApiKey(pub SecretValue);

const LITELLM_API_KEY: &str = "x-litellm-api-key";
const PROVIDER_KEY_HEADERS: [&str; 4] = [
    "api-key",
    "x-api-key",
    "x-goog-api-key",
    "ocp-apim-subscription-key",
];
const SCHEMES: [&str; 3] = ["Bearer ", "Basic ", "bearer "];
const AWS_SIGNATURE: &str = "AWS4-HMAC-SHA256";

impl ApiKey {
    pub fn from_headers(headers: &HeaderMap) -> Result<Self, Error> {
        read(headers)
            .filter(|key| !key.is_empty())
            .map(|key| Self(SecretValue::new(key)))
            .ok_or(Error::InvalidToken)
    }
}

impl From<ApiKey> for Credential {
    fn from(ApiKey(key): ApiKey) -> Self {
        Self::Token(key)
    }
}

impl<S: Send + Sync> FromRequestParts<S> for ApiKey {
    type Rejection = Error;

    async fn from_request_parts(parts: &mut Parts, _: &S) -> Result<Self, Error> {
        Self::from_headers(&parts.headers)
    }
}

/// The first header present wins even when it yields no key, as in Python. A value that is not
/// ASCII also yields none, since Python decodes it as Latin-1 and the key would differ.
fn read(headers: &HeaderMap) -> Option<String> {
    let present = |name| headers.get(name).filter(|value| !value.is_empty());
    if let Some(value) = present(LITELLM_API_KEY) {
        let value = value.to_str().ok()?;
        return Some(without_scheme(value).unwrap_or_else(|| value.into()));
    }
    if let Some(value) = present(AUTHORIZATION.as_str()) {
        return without_scheme(value.to_str().ok()?);
    }
    PROVIDER_KEY_HEADERS
        .into_iter()
        .find_map(present)
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned)
}

/// The key under a scheme Python recognizes. Like `str.replace`, stripping removes every
/// occurrence of the scheme, not only the prefix.
fn without_scheme(value: &str) -> Option<String> {
    if let Some(scheme) = SCHEMES.into_iter().find(|scheme| value.starts_with(scheme)) {
        return Some(value.replace(scheme, ""));
    }
    value
        .starts_with(AWS_SIGNATURE)
        .then(|| {
            signed_credential(value, "Credential=Bearer", true)
                .or_else(|| signed_credential(value, "Credential=", false))
        })
        .flatten()
        .map(str::to_owned)
}

/// The first match of `{marker}\s+([^/\s,]+)`, or `{marker}([^/\s,]+)` without `spaced`.
fn signed_credential<'a>(value: &'a str, marker: &str, spaced: bool) -> Option<&'a str> {
    value.match_indices(marker).find_map(|(at, _)| {
        let rest = &value[at + marker.len()..];
        let credential = rest.trim_start_matches(|c: char| c.is_ascii_whitespace());
        if spaced == (credential.len() == rest.len()) {
            return None;
        }
        let end = credential
            .find(|c: char| c == '/' || c == ',' || c.is_ascii_whitespace())
            .unwrap_or(credential.len());
        (end > 0).then(|| &credential[..end])
    })
}

#[cfg(test)]
mod tests {
    use axum::http::{HeaderName, HeaderValue};
    use rstest::rstest;

    use super::*;

    fn key(headers: &[(&'static str, &[u8])]) -> Option<String> {
        let headers = headers
            .iter()
            .map(|(name, value)| {
                (
                    HeaderName::from_static(name),
                    HeaderValue::from_bytes(value).unwrap(),
                )
            })
            .collect();
        ApiKey::from_headers(&headers)
            .ok()
            .map(|ApiKey(key)| key.expose().to_owned())
    }

    const AWS: &str = "AWS4-HMAC-SHA256 Credential=Bearer sk-1/20240101/us-east-1/bedrock/aws4_request, SignedHeaders=host, Signature=abc";

    #[rstest]
    #[case::bearer(&[("authorization", "Bearer sk-1")], Some("sk-1"))]
    #[case::every_occurrence_stripped(&[("authorization", "Bearer sk-Bearer 1")], Some("sk-1"))]
    #[case::first_of_duplicates(&[("authorization", "Bearer sk-1"), ("authorization", "Bearer sk-2")], Some("sk-1"))]
    #[case::lowercase_scheme(&[("authorization", "bearer sk-1")], Some("sk-1"))]
    #[case::basic(&[("authorization", "Basic sk-1")], Some("sk-1"))]
    #[case::aws_signature(&[("authorization", AWS)], Some("sk-1"))]
    #[case::aws_signature_without_bearer(&[("authorization", "AWS4-HMAC-SHA256 Credential=sk-1/2024, Signature=abc")], Some("sk-1"))]
    #[case::aws_signature_tab(&[("authorization", "AWS4-HMAC-SHA256 Credential=Bearer\tsk-1/2024")], Some("sk-1"))]
    #[case::aws_signature_bearer_without_key(&[("authorization", "AWS4-HMAC-SHA256 Credential=Bearer /2024")], Some("Bearer"))]
    #[case::aws_signature_without_credential(&[("authorization", "AWS4-HMAC-SHA256 Signature=abc")], None)]
    #[case::no_scheme(&[("authorization", "sk-1")], None)]
    #[case::uppercase_scheme(&[("authorization", "BEARER sk-1")], None)]
    #[case::empty_token(&[("authorization", "Bearer ")], None)]
    #[case::litellm_header_raw(&[("x-litellm-api-key", "sk-1")], Some("sk-1"))]
    #[case::litellm_header_bearer(&[("x-litellm-api-key", "Bearer sk-1")], Some("sk-1"))]
    #[case::litellm_header_aws_without_credential(&[("x-litellm-api-key", "AWS4-HMAC-SHA256 x")], Some("AWS4-HMAC-SHA256 x"))]
    #[case::litellm_header_wins(&[("authorization", "Bearer sk-1"), ("x-litellm-api-key", "sk-2")], Some("sk-2"))]
    #[case::empty_litellm_header_is_absent(&[("authorization", "Bearer sk-1"), ("x-litellm-api-key", "")], Some("sk-1"))]
    #[case::unrecognized_authorization_wins_over_provider_header(&[("authorization", "sk-1"), ("x-api-key", "sk-2")], None)]
    #[case::azure(&[("api-key", "sk-1")], Some("sk-1"))]
    #[case::anthropic(&[("x-api-key", "sk-1")], Some("sk-1"))]
    #[case::google(&[("x-goog-api-key", "sk-1")], Some("sk-1"))]
    #[case::azure_apim(&[("ocp-apim-subscription-key", "sk-1")], Some("sk-1"))]
    #[case::provider_headers_keep_their_scheme(&[("x-api-key", "Bearer sk-1")], Some("Bearer sk-1"))]
    #[case::provider_header_order(&[("ocp-apim-subscription-key", "sk-4"), ("x-goog-api-key", "sk-3"), ("x-api-key", "sk-2"), ("api-key", "sk-1")], Some("sk-1"))]
    #[case::missing(&[], None)]
    fn reads_the_key_python_reads(
        #[case] headers: &[(&'static str, &'static str)],
        #[case] expected: Option<&str>,
    ) {
        let headers: Vec<_> = headers
            .iter()
            .map(|(name, value)| (*name, value.as_bytes()))
            .collect();
        assert_eq!(key(&headers).as_deref(), expected);
    }

    #[rstest]
    #[case::authorization("authorization")]
    #[case::litellm_header("x-litellm-api-key")]
    fn a_value_that_is_not_ascii_is_no_key_and_no_fallback(#[case] name: &'static str) {
        assert_eq!(
            key(&[(name, b"Bearer sk-\xe9"), ("x-api-key", b"sk-2")]),
            None
        );
    }
}
