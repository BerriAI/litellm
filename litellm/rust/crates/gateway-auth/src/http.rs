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
        if parts.headers.contains_key(AUTHORIZATION) {
            return Bearer.extract(parts);
        }
        single(parts, "x-api-key")
            .and_then(token)
            .map(Credential::Token)
    }
}

fn bearer(parts: &Parts) -> Result<SecretValue, Error> {
    single(parts, AUTHORIZATION)?
        .strip_prefix("Bearer ")
        .ok_or(Error::InvalidToken)
        .and_then(token)
}

fn single(parts: &Parts, name: impl AsHeaderName) -> Result<&str, Error> {
    let mut values = parts.headers.get_all(name).iter();
    let value = values.next().ok_or(Error::InvalidToken)?;
    if values.next().is_some() {
        return Err(Error::InvalidToken);
    }
    value.to_str().map_err(|_| Error::InvalidToken)
}

fn token(value: &str) -> Result<SecretValue, Error> {
    Some(value.trim())
        .filter(|value| !value.is_empty() && !value.bytes().any(|byte| byte.is_ascii_whitespace()))
        .map(SecretValue::new)
        .ok_or(Error::InvalidToken)
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

    #[rstest]
    #[case::x_api_key(&[("x-api-key", "key")], Some("key"))]
    #[case::bearer(&[("authorization", "Bearer key")], Some("key"))]
    #[case::bearer_wins(&[("authorization", "Bearer bearer"), ("x-api-key", "x")], Some("bearer"))]
    #[case::invalid_bearer_does_not_fall_back(&[("authorization", "Basic key"), ("x-api-key", "x")], None)]
    #[case::empty(&[("x-api-key", "")], None)]
    #[case::inner_whitespace(&[("x-api-key", "a b")], None)]
    #[case::duplicate(&[("x-api-key", "key"), ("x-api-key", "key")], None)]
    #[case::missing(&[], None)]
    fn bearer_or_x_api_key_reads_x_api_key_only_without_authorization(
        #[case] headers: &[(&str, &str)],
        #[case] expected: Option<&str>,
    ) {
        let (parts, ()) = headers
            .iter()
            .fold(Request::builder(), |request, (name, value)| {
                request.header(*name, *value)
            })
            .body(())
            .unwrap()
            .into_parts();
        let token = match BearerOrXApiKey.extract(&parts) {
            Ok(Credential::Token(token)) => Some(token.expose().to_owned()),
            _ => None,
        };
        assert_eq!(token.as_deref(), expected);
    }
}
