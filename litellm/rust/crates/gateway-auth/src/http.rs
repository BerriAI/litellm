use axum::{
    extract::{FromRequestParts, MatchedPath, Request, State},
    http::{
        header::{AUTHORIZATION, AsHeaderName},
        request::Parts,
    },
    middleware::Next,
    response::Response,
};
use litellm_secret_value::SecretValue;

use crate::{AccessRequest, Auth, AuthenticatedRequest, Credential, Error};

pub trait CredentialExtractor: Send + Sync {
    fn extract(&self, parts: &Parts) -> Result<Credential, Error>;

    fn reject(&self, _: &Parts, error: Error) -> Error {
        error
    }
}

pub struct Bearer;

impl CredentialExtractor for Bearer {
    fn extract(&self, parts: &Parts) -> Result<Credential, Error> {
        bearer(parts).map(Credential::Token)
    }
}

pub struct BearerOrXApiKey;

impl CredentialExtractor for BearerOrXApiKey {
    fn extract(&self, parts: &Parts) -> Result<Credential, Error> {
        let Some(header) = key_header(parts) else {
            return Bearer.extract(parts);
        };
        single(parts, header)
            .and_then(token)
            .map(Credential::Token)
            .ok_or(Error::InvalidApiKey)
    }

    fn reject(&self, parts: &Parts, error: Error) -> Error {
        match error {
            Error::InvalidToken if key_header(parts).is_some() => Error::InvalidApiKey,
            error => error,
        }
    }
}

const API_KEY: &str = "api-key";
const X_API_KEY: &str = "x-api-key";

fn key_header(parts: &Parts) -> Option<&'static str> {
    if !blank(parts, AUTHORIZATION) {
        None
    } else if !blank(parts, API_KEY) {
        Some(API_KEY)
    } else {
        parts.headers.contains_key(X_API_KEY).then_some(X_API_KEY)
    }
}

fn blank(parts: &Parts, name: impl AsHeaderName) -> bool {
    parts
        .headers
        .get_all(name)
        .iter()
        .all(|value| value.is_empty())
}

fn bearer(parts: &Parts) -> Result<SecretValue, Error> {
    single(parts, AUTHORIZATION)
        .and_then(|value| value.strip_prefix("Bearer "))
        .and_then(token)
        .ok_or(Error::InvalidToken)
}

fn single(parts: &Parts, name: impl AsHeaderName) -> Option<&str> {
    let mut values = parts.headers.get_all(name).iter();
    match (values.next(), values.next()) {
        (Some(value), None) => value.to_str().ok(),
        _ => None,
    }
}

fn token(value: &str) -> Option<SecretValue> {
    Some(value)
        .filter(|value| !value.is_empty() && !value.bytes().any(|byte| byte.is_ascii_whitespace()))
        .map(SecretValue::new)
}

pub async fn authenticate(
    State(auth): State<Auth>,
    request: Request,
    next: Next,
) -> Result<Response, Error> {
    let (mut parts, body) = request.into_parts();
    let identity = auth.authenticate_parts(&parts).await?;
    let route = AccessRequest::Route {
        method: parts.method.to_string(),
        path: parts
            .extensions
            .get::<MatchedPath>()
            .map_or(parts.uri.path(), MatchedPath::as_str)
            .into(),
    };
    identity
        .authorize(route.clone())
        .await?
        .consume(identity.caller(), &route)?;
    parts.extensions.insert(identity);
    Ok(next.run(Request::from_parts(parts, body)).await)
}

impl<S: Send + Sync> FromRequestParts<S> for AuthenticatedRequest {
    type Rejection = Error;

    async fn from_request_parts(parts: &mut Parts, _: &S) -> Result<Self, Error> {
        parts
            .extensions
            .get::<Self>()
            .cloned()
            .ok_or(Error::MissingIdentity)
    }
}

pub struct RequireMasterKey;

impl FromRequestParts<Auth> for RequireMasterKey {
    type Rejection = Error;

    async fn from_request_parts(parts: &mut Parts, state: &Auth) -> Result<Self, Error> {
        state.validate().await?;
        let identity = state
            .authenticate_request(&Credential::Token(bearer(parts)?), parts)
            .await?;
        if identity.caller().authentication().method != crate::AuthenticationMethod::MasterKey {
            return Err(Error::InvalidToken);
        }
        parts.extensions.insert(identity);
        Ok(Self)
    }
}

#[cfg(test)]
mod tests {
    use axum::http::Request;
    use rstest::rstest;

    use super::*;

    fn parts(headers: &[(&str, &str)]) -> Parts {
        headers
            .iter()
            .fold(Request::builder(), |request, (name, value)| {
                request.header(*name, *value)
            })
            .body(())
            .unwrap()
            .into_parts()
            .0
    }

    #[rstest]
    #[case::x_api_key(&[("x-api-key", "key")], Some("key"))]
    #[case::api_key(&[("api-key", "key")], Some("key"))]
    #[case::api_key_wins(&[("api-key", "api"), ("x-api-key", "x")], Some("api"))]
    #[case::invalid_api_key_does_not_fall_back(&[("api-key", "a b"), ("x-api-key", "x")], None)]
    #[case::empty_api_key_falls_back(&[("api-key", ""), ("x-api-key", "key")], Some("key"))]
    #[case::api_key_is_taken_as_sent(&[("api-key", "Bearer key")], None)]
    #[case::bearer(&[("authorization", "Bearer key")], Some("key"))]
    #[case::bearer_wins(&[("authorization", "Bearer bearer"), ("x-api-key", "x")], Some("bearer"))]
    #[case::bearer_wins_over_both(&[("authorization", "Bearer bearer"), ("api-key", "api"), ("x-api-key", "x")], Some("bearer"))]
    #[case::bearer_extra_space(&[("authorization", "Bearer  key")], None)]
    #[case::invalid_bearer_does_not_fall_back(&[("authorization", "Basic key"), ("x-api-key", "x")], None)]
    #[case::invalid_bearer_does_not_fall_back_to_api_key(&[("authorization", "Basic key"), ("api-key", "api")], None)]
    #[case::empty_authorization_falls_back(&[("authorization", ""), ("x-api-key", "key")], Some("key"))]
    #[case::empty_authorization_falls_back_to_api_key(&[("authorization", ""), ("api-key", "key")], Some("key"))]
    #[case::empty(&[("x-api-key", "")], None)]
    #[case::inner_whitespace(&[("x-api-key", "a b")], None)]
    #[case::duplicate(&[("x-api-key", "key"), ("x-api-key", "key")], None)]
    #[case::duplicate_api_key(&[("api-key", "key"), ("api-key", "key"), ("x-api-key", "x")], None)]
    #[case::missing(&[], None)]
    fn bearer_or_x_api_key_reads_key_headers_only_without_authorization(
        #[case] headers: &[(&str, &str)],
        #[case] expected: Option<&str>,
    ) {
        let token = match BearerOrXApiKey.extract(&parts(headers)) {
            Ok(Credential::Token(token)) => Some(token.expose().to_owned()),
            _ => None,
        };
        assert_eq!(token.as_deref(), expected);
    }

    #[rstest]
    #[case::malformed_x_api_key(&[("x-api-key", "a b")])]
    #[case::malformed_api_key(&[("api-key", "a b")])]
    #[case::malformed_api_key_beside_x_api_key(&[("api-key", "a b"), ("x-api-key", "key")])]
    fn bearer_or_x_api_key_fails_a_malformed_key_header_as_invalid_api_key(
        #[case] headers: &[(&str, &str)],
    ) {
        assert!(matches!(
            BearerOrXApiKey.extract(&parts(headers)),
            Err(Error::InvalidApiKey)
        ));
    }

    #[rstest]
    #[case::x_api_key(&[("x-api-key", "bad")], true)]
    #[case::api_key(&[("api-key", "bad")], true)]
    #[case::api_key_beside_x_api_key(&[("api-key", "bad"), ("x-api-key", "key")], true)]
    #[case::bearer(&[("authorization", "Bearer bad"), ("api-key", "key")], false)]
    fn bearer_or_x_api_key_names_the_key_header_it_rejected(
        #[case] headers: &[(&str, &str)],
        #[case] api_key: bool,
    ) {
        let error = BearerOrXApiKey.reject(&parts(headers), Error::InvalidToken);
        assert_eq!(matches!(error, Error::InvalidApiKey), api_key);
    }
}
