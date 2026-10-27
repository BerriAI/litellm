use std::iter;

use axum::{
    extract::{FromRef, FromRequestParts, Query},
    http::{HeaderMap, HeaderName, HeaderValue, Uri, header::AUTHORIZATION, request::Parts},
};
use axum_extra::{
    TypedHeader,
    headers::{
        self, Header, HeaderMapExt,
        authorization::{Authorization, Credentials},
    },
};
use litellm_auth_types::SecretValue;
use serde::Deserialize;

use crate::Error;

#[derive(Clone, Default)]
pub struct ApiKeySources {
    custom_header: Option<HeaderName>,
    query_key: bool,
}

impl ApiKeySources {
    pub fn new(custom_header: Option<HeaderName>) -> Self {
        Self {
            custom_header,
            query_key: false,
        }
    }

    pub fn with_query_key(self) -> Self {
        Self {
            query_key: true,
            ..self
        }
    }
}

pub struct ApiKey(pub SecretValue);

impl From<ApiKey> for SecretValue {
    fn from(value: ApiKey) -> Self {
        value.0
    }
}

impl<S> FromRequestParts<S> for ApiKey
where
    ApiKeySources: FromRef<S>,
    S: Send + Sync,
{
    type Rejection = Error;

    async fn from_request_parts(parts: &mut Parts, state: &S) -> Result<Self, Error> {
        let sources = ApiKeySources::from_ref(state);
        let custom = sources
            .custom_header
            .as_ref()
            .map(|name| custom_key(&parts.headers, name));
        let standard = || {
            STANDARD_SOURCES
                .iter()
                .find_map(|source| source(&parts.headers).transpose())
                .map(|key| key.map_err(|_| Error::InvalidToken))
        };
        let query = || sources.query_key.then(|| query_key(&parts.uri));
        custom
            .flatten()
            .or_else(standard)
            .or_else(query)
            .unwrap_or(Err(Error::InvalidToken))
            .map(Self)
    }
}

type Source = fn(&HeaderMap) -> Result<Option<SecretValue>, headers::Error>;

const STANDARD_SOURCES: [Source; 6] = [
    key::<LitellmApiKey>,
    key::<KeyAuthorization>,
    key::<ApiKeyHeader>,
    key::<XApiKey>,
    key::<GoogleApiKey>,
    key::<ApimSubscriptionKey>,
];

fn key<H: Header + Into<Option<SecretValue>>>(
    headers: &HeaderMap,
) -> Result<Option<SecretValue>, headers::Error> {
    Ok(headers.typed_try_get::<H>()?.and_then(Into::into))
}

fn custom_key(headers: &HeaderMap, name: &HeaderName) -> Option<Result<SecretValue, Error>> {
    headers.contains_key(name).then(|| {
        KeyAuthorization::decode(&mut headers.get_all(name).iter())
            .ok()
            .and_then(Into::into)
            .ok_or(Error::InvalidToken)
    })
}

#[derive(Deserialize)]
struct KeyParam {
    key: Option<String>,
}

fn query_key(uri: &Uri) -> Result<SecretValue, Error> {
    Query::<KeyParam>::try_from_uri(uri)
        .ok()
        .and_then(|Query(param)| param.key)
        .filter(|key| !key.is_empty())
        .map(SecretValue::new)
        .ok_or(Error::InvalidToken)
}

macro_rules! key_header {
    ($name:ident, $header:expr, $parse:expr) => {
        struct $name(Option<SecretValue>);

        impl Header for $name {
            fn name() -> &'static HeaderName {
                static NAME: HeaderName = $header;
                &NAME
            }

            fn decode<'i, I: Iterator<Item = &'i HeaderValue>>(
                values: &mut I,
            ) -> Result<Self, headers::Error> {
                let value = single(values)?
                    .to_str()
                    .map_err(|_| headers::Error::invalid())?;
                if value.is_empty() {
                    return Ok(Self(None));
                }
                let parse: fn(&str) -> Option<&str> = $parse;
                parse(value)
                    .map(|key| Self(Some(SecretValue::new(key))))
                    .ok_or_else(headers::Error::invalid)
            }

            fn encode<E: Extend<HeaderValue>>(&self, values: &mut E) {
                values.extend(
                    self.0
                        .as_ref()
                        .and_then(|key| HeaderValue::from_str(key.expose()).ok()),
                );
            }
        }

        impl From<$name> for Option<SecretValue> {
            fn from(value: $name) -> Self {
                value.0
            }
        }
    };
}

key_header!(
    LitellmApiKey,
    HeaderName::from_static("x-litellm-api-key"),
    |value| scheme_credential(value).or(Some(value))
);
key_header!(KeyAuthorization, AUTHORIZATION, scheme_credential);
key_header!(ApiKeyHeader, HeaderName::from_static("api-key"), raw);
key_header!(XApiKey, HeaderName::from_static("x-api-key"), raw);
key_header!(GoogleApiKey, HeaderName::from_static("x-goog-api-key"), raw);
key_header!(
    ApimSubscriptionKey,
    HeaderName::from_static("ocp-apim-subscription-key"),
    raw
);

fn raw(value: &str) -> Option<&str> {
    Some(value)
}

fn scheme_credential(value: &str) -> Option<&str> {
    let (scheme, credentials) = value.split_once(' ')?;
    let token = if scheme.eq_ignore_ascii_case("bearer") || scheme.eq_ignore_ascii_case("basic") {
        credentials.trim()
    } else if scheme.eq_ignore_ascii_case("aws4-hmac-sha256") {
        aws4_access_key(credentials)?
    } else {
        return None;
    };
    is_token(token).then_some(token)
}

fn aws4_access_key(parameters: &str) -> Option<&str> {
    let credential = parameters
        .split(',')
        .find_map(|parameter| parameter.trim().strip_prefix("Credential="))?;
    let credential = credential.strip_prefix("Bearer ").unwrap_or(credential);
    credential.split('/').next()
}

pub struct Bearer(pub SecretValue);

impl From<Bearer> for SecretValue {
    fn from(value: Bearer) -> Self {
        value.0
    }
}

impl<S: Send + Sync> FromRequestParts<S> for Bearer {
    type Rejection = Error;

    async fn from_request_parts(parts: &mut Parts, state: &S) -> Result<Self, Error> {
        let TypedHeader(Single(Authorization(BearerToken(token)))) =
            TypedHeader::from_request_parts(parts, state)
                .await
                .map_err(|_| Error::InvalidToken)?;
        Ok(Self(token))
    }
}

struct BearerToken(SecretValue);

impl Credentials for BearerToken {
    const SCHEME: &'static str = "Bearer";

    fn decode(value: &HeaderValue) -> Option<Self> {
        let token = value.to_str().ok()?.get(Self::SCHEME.len()..)?.trim();
        is_token(token).then(|| Self(SecretValue::new(token)))
    }

    fn encode(&self) -> HeaderValue {
        HeaderValue::from_str(&format!("{} {}", Self::SCHEME, self.0.expose()))
            .expect("token was decoded from a valid header value")
    }
}

struct Single<H>(H);

impl<H: Header> Header for Single<H> {
    fn name() -> &'static HeaderName {
        H::name()
    }

    fn decode<'i, I: Iterator<Item = &'i HeaderValue>>(
        values: &mut I,
    ) -> Result<Self, headers::Error> {
        H::decode(&mut iter::once(single(values)?)).map(Self)
    }

    fn encode<E: Extend<HeaderValue>>(&self, values: &mut E) {
        self.0.encode(values);
    }
}

fn single<'i, I: Iterator<Item = &'i HeaderValue>>(
    values: &mut I,
) -> Result<&'i HeaderValue, headers::Error> {
    match (values.next(), values.next()) {
        (Some(value), None) => Ok(value),
        _ => Err(headers::Error::invalid()),
    }
}

fn is_token(token: &str) -> bool {
    !token.is_empty() && !token.contains(char::is_whitespace)
}

#[cfg(test)]
mod tests {
    use axum::http::Request;
    use rstest::rstest;

    use super::*;

    const AWS4_BEARER: &str = "AWS4-HMAC-SHA256 Credential=Bearer token/date/region/service/aws4_request, \
                               SignedHeaders=host, Signature=sig";
    const AWS4_RAW: &str = "AWS4-HMAC-SHA256 Credential=token/date/region/service/aws4_request, \
                            SignedHeaders=host, Signature=sig";

    fn parts(uri: &str, headers: &[(&str, &str)]) -> Parts {
        let request = headers
            .iter()
            .fold(Request::builder().uri(uri), |request, (name, value)| {
                request.header(*name, *value)
            })
            .body(())
            .unwrap();
        request.into_parts().0
    }

    async fn bearer(headers: &[(&str, &str)]) -> Option<String> {
        let mut parts = parts("/", headers);
        Bearer::from_request_parts(&mut parts, &())
            .await
            .ok()
            .map(|Bearer(token)| token.expose().to_owned())
    }

    async fn api_key(
        sources: ApiKeySources,
        uri: &str,
        headers: &[(&str, &str)],
    ) -> Option<String> {
        let mut parts = parts(uri, headers);
        ApiKey::from_request_parts(&mut parts, &sources)
            .await
            .ok()
            .map(|ApiKey(token)| token.expose().to_owned())
    }

    #[rstest]
    #[case::standard(&[("authorization", "Bearer token")], Some("token"))]
    #[case::scheme_is_case_insensitive(&[("authorization", "bearer token")], Some("token"))]
    #[case::surrounding_space_is_trimmed(&[("authorization", "Bearer  token ")], Some("token"))]
    #[case::empty(&[("authorization", "Bearer ")], None)]
    #[case::inner_whitespace(&[("authorization", "Bearer a b")], None)]
    #[case::other_scheme(&[("authorization", "Basic token")], None)]
    #[case::duplicate(&[("authorization", "Bearer token"), ("authorization", "Bearer token")], None)]
    #[case::other_header(&[("x-api-key", "token")], None)]
    #[tokio::test]
    async fn bearer_accepts_one_rfc_6750_token(
        #[case] headers: &[(&str, &str)],
        #[case] expected: Option<&str>,
    ) {
        assert_eq!(bearer(headers).await.as_deref(), expected);
    }

    const PRIORITY: [(&str, &str); 6] = [
        ("x-litellm-api-key", "Bearer {}"),
        ("authorization", "Bearer {}"),
        ("api-key", "{}"),
        ("x-api-key", "{}"),
        ("x-goog-api-key", "{}"),
        ("ocp-apim-subscription-key", "{}"),
    ];

    #[rstest]
    #[tokio::test]
    async fn higher_priority_source_wins_over_every_lower_one() {
        for (rank, (higher, format)) in PRIORITY.iter().enumerate() {
            let winner = format.replace("{}", higher);
            for (lower, lower_format) in &PRIORITY[rank + 1..] {
                let loser = lower_format.replace("{}", "loser");
                let headers = [(*lower, loser.as_str()), (*higher, winner.as_str())];
                let selected = api_key(ApiKeySources::default(), "/", &headers).await;
                assert_eq!(selected.as_deref(), Some(*higher), "{higher} over {lower}");
            }
            let selected = api_key(
                ApiKeySources::default().with_query_key(),
                "/?key=loser",
                &[(*higher, winner.as_str())],
            )
            .await;
            assert_eq!(selected.as_deref(), Some(*higher), "{higher} over query");
        }
    }

    #[rstest]
    #[case::bearer("Bearer token", Some("token"))]
    #[case::lowercase_bearer("bearer token", Some("token"))]
    #[case::basic_payload_is_not_decoded("Basic dXNlcjpwYXNz", Some("dXNlcjpwYXNz"))]
    #[case::aws4_bearer(AWS4_BEARER, Some("token"))]
    #[case::aws4_access_key(AWS4_RAW, Some("token"))]
    #[case::aws4_without_credential("AWS4-HMAC-SHA256 SignedHeaders=host", None)]
    #[case::raw_token("token", None)]
    #[case::empty_bearer("Bearer ", None)]
    #[case::inner_whitespace("Bearer token suffix", None)]
    #[case::unknown_scheme("Token token", None)]
    #[tokio::test]
    async fn authorization_accepts_bearer_basic_and_aws4(
        #[case] value: &str,
        #[case] expected: Option<&str>,
    ) {
        let headers = [("authorization", value), ("x-api-key", "lower")];
        assert_eq!(
            api_key(ApiKeySources::default(), "/", &headers)
                .await
                .as_deref(),
            expected
        );
    }

    #[rstest]
    #[case::raw("token", "token")]
    #[case::bearer("Bearer token", "token")]
    #[case::aws4(AWS4_BEARER, "token")]
    #[case::aws4_without_credential("AWS4-HMAC-SHA256 malformed", "AWS4-HMAC-SHA256 malformed")]
    #[tokio::test]
    async fn litellm_header_strips_a_known_scheme_or_keeps_the_raw_value(
        #[case] value: &str,
        #[case] expected: &str,
    ) {
        let headers = [("x-litellm-api-key", value)];
        assert_eq!(
            api_key(ApiKeySources::default(), "/", &headers)
                .await
                .as_deref(),
            Some(expected)
        );
    }

    #[rstest]
    #[case::litellm("x-litellm-api-key")]
    #[case::authorization("authorization")]
    #[case::api_key("api-key")]
    #[case::x_api_key("x-api-key")]
    #[case::google("x-goog-api-key")]
    #[tokio::test]
    async fn empty_header_falls_through_to_the_next_source(#[case] name: &str) {
        let headers = [(name, ""), ("ocp-apim-subscription-key", "lower")];
        assert_eq!(
            api_key(ApiKeySources::default(), "/", &headers)
                .await
                .as_deref(),
            Some("lower")
        );
    }

    #[rstest]
    #[case::duplicate_selected(&[("x-api-key", "first"), ("x-api-key", "second")], None)]
    #[case::malformed_selected(&[("authorization", "token"), ("x-api-key", "lower")], None)]
    #[case::lower_duplicate_is_ignored(
        &[("x-litellm-api-key", "token"), ("x-api-key", "first"), ("x-api-key", "second")],
        Some("token")
    )]
    #[case::lower_malformed_is_ignored(&[("x-litellm-api-key", "token"), ("authorization", "Token x")], Some("token"))]
    #[case::nothing(&[], None)]
    #[tokio::test]
    async fn only_the_selected_source_must_be_well_formed(
        #[case] headers: &[(&str, &str)],
        #[case] expected: Option<&str>,
    ) {
        assert_eq!(
            api_key(ApiKeySources::default(), "/", headers)
                .await
                .as_deref(),
            expected
        );
    }

    #[rstest]
    #[case::wins_over_every_standard_header("Bearer custom", Some("custom"))]
    #[case::uses_the_authorization_schemes(AWS4_BEARER, Some("token"))]
    #[case::empty_fails_closed("", None)]
    #[case::raw_fails_closed("custom", None)]
    #[tokio::test]
    async fn custom_header_overrides_standard_sources(
        #[case] value: &str,
        #[case] expected: Option<&str>,
    ) {
        let headers: Vec<_> = PRIORITY
            .iter()
            .map(|(name, _)| (*name, "Bearer lower"))
            .chain([("x-custom-key", value)])
            .collect();
        let sources =
            ApiKeySources::new(Some(HeaderName::from_static("x-custom-key"))).with_query_key();
        assert_eq!(
            api_key(sources, "/?key=lower", &headers).await.as_deref(),
            expected
        );
    }

    #[rstest]
    #[case::absent(&[("x-api-key", "lower")], Some("lower"))]
    #[case::duplicate(&[("x-custom-key", "Bearer a"), ("x-custom-key", "Bearer b"), ("x-api-key", "lower")], None)]
    #[tokio::test]
    async fn custom_header_is_selected_only_when_present_once(
        #[case] headers: &[(&str, &str)],
        #[case] expected: Option<&str>,
    ) {
        let sources = ApiKeySources::new(Some(HeaderName::from_static("x-custom-key")));
        assert_eq!(api_key(sources, "/", headers).await.as_deref(), expected);
    }

    #[rstest]
    #[case::form_decoded(
        true,
        "/v1beta/models/g:generateContent?key=query+token",
        Some("query token")
    )]
    #[case::other_parameters(true, "/?alt=sse&key=token", Some("token"))]
    #[case::disabled(false, "/?key=token", None)]
    #[case::empty(true, "/?key=", None)]
    #[case::duplicate(true, "/?key=first&key=second", None)]
    #[case::missing(true, "/?alt=sse", None)]
    #[tokio::test]
    async fn query_key_is_read_only_when_the_route_enables_it(
        #[case] enabled: bool,
        #[case] uri: &str,
        #[case] expected: Option<&str>,
    ) {
        let sources = if enabled {
            ApiKeySources::default().with_query_key()
        } else {
            ApiKeySources::default()
        };
        assert_eq!(api_key(sources, uri, &[]).await.as_deref(), expected);
    }
}
