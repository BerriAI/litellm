use std::{ops::Deref, sync::Arc};

use axum::{
    extract::{FromRequestParts, Request, State},
    http::{
        header::{AUTHORIZATION, AsHeaderName},
        request::Parts,
    },
    middleware::Next,
    response::Response,
};
use litellm_gateway_auth_core::{self as auth_core, Caller, Rows};
use litellm_secret_value::SecretValue;

use crate::{Auth, Credential, Error};

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

/// A caller the server already authenticated for this request with the same [`Auth`] the route
/// layer holds, which [`authenticate`] takes instead of reading the credential again. Only server
/// code can put it on a request, since clients never set extensions.
pub struct Preauthenticated<R: Rows>(pub Authenticated<R>);

impl<R: Rows> Clone for Preauthenticated<R> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

pub async fn authenticate<R: Rows>(
    State(auth): State<Auth<R>>,
    request: Request,
    next: Next,
) -> Result<Response, Error> {
    let (mut parts, body) = request.into_parts();
    let caller = match parts.extensions.remove::<Preauthenticated<R>>() {
        Some(Preauthenticated(caller)) => caller,
        None => auth.authenticate_parts(&parts).await?,
    };
    parts.extensions.insert(caller);
    Ok(next.run(Request::from_parts(parts, body)).await)
}

/// The caller a request was authenticated as, which a handler extracts and reads as the core's
/// [`Caller`].
pub struct Authenticated<R: Rows>(Arc<Caller<R>>);

impl<R: Rows> Clone for Authenticated<R> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<R: Rows> Deref for Authenticated<R> {
    type Target = Caller<R>;

    fn deref(&self) -> &Caller<R> {
        &self.0
    }
}

impl<R: Rows> From<Caller<R>> for Authenticated<R> {
    fn from(caller: Caller<R>) -> Self {
        Self(Arc::new(caller))
    }
}

impl<R: Rows, S: Send + Sync> FromRequestParts<S> for Authenticated<R> {
    type Rejection = Error;

    async fn from_request_parts(parts: &mut Parts, _: &S) -> Result<Self, Error> {
        parts
            .extensions
            .get::<Self>()
            .cloned()
            .ok_or(Error::MissingIdentity)
    }
}

/// The bearer token of `parts`, or no token at all. The authenticator answers that too, so a
/// server with no master key says so before it calls a token missing.
fn presented(parts: &Parts) -> Credential {
    bearer(parts).map_or(Credential::Transport, Credential::Token)
}

pub struct RequireMasterKey;

impl<R: Rows> FromRequestParts<Auth<R>> for RequireMasterKey {
    type Rejection = Error;

    async fn from_request_parts(parts: &mut Parts, state: &Auth<R>) -> Result<Self, Error> {
        let caller = state
            .authenticate_credential(&presented(parts), parts)
            .await?;
        match caller.credential {
            auth_core::Credential::MasterKey => {
                parts.extensions.insert(caller);
                Ok(Self)
            }
            auth_core::Credential::Key { .. } => Err(Error::InvalidToken),
        }
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
