use std::iter;

use axum::{
    extract::FromRequestParts,
    http::{HeaderName, HeaderValue, request::Parts},
};
use axum_extra::{
    TypedHeader,
    headers::{
        self, Header,
        authorization::{Authorization, Credentials},
    },
};
use litellm_auth_types::SecretValue;

use crate::Error;

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
        match (values.next(), values.next()) {
            (Some(value), None) => H::decode(&mut iter::once(value)).map(Self),
            _ => Err(headers::Error::invalid()),
        }
    }

    fn encode<E: Extend<HeaderValue>>(&self, values: &mut E) {
        self.0.encode(values);
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

    #[rstest]
    #[case::standard(&["Bearer token"], Some("token"))]
    #[case::scheme_is_case_insensitive(&["bearer token"], Some("token"))]
    #[case::surrounding_space_is_trimmed(&["Bearer  token "], Some("token"))]
    #[case::empty(&["Bearer "], None)]
    #[case::inner_whitespace(&["Bearer a b"], None)]
    #[case::other_scheme(&["Basic token"], None)]
    #[case::duplicate(&["Bearer token", "Bearer token"], None)]
    #[tokio::test]
    async fn bearer_accepts_one_rfc_6750_token(
        #[case] headers: &[&str],
        #[case] expected: Option<&str>,
    ) {
        let request = headers
            .iter()
            .fold(Request::builder(), |request, value| {
                request.header("authorization", *value)
            })
            .body(())
            .unwrap();
        let (mut parts, ()) = request.into_parts();
        let token = Bearer::from_request_parts(&mut parts, &())
            .await
            .ok()
            .map(|Bearer(token)| token.expose().to_owned());
        assert_eq!(token.as_deref(), expected);
    }
}
