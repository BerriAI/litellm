use axum::{
    extract::{FromRequestParts, MatchedPath, Request, State},
    http::{header::AUTHORIZATION, request::Parts},
    middleware::Next,
    response::Response,
};
use litellm_gateway_auth_core::Token;

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

/// The `Authorization: Bearer` key exactly as `get_api_key` in `user_api_key_auth.py` reads it, and
/// no key at all for a request Python would read another way, so its caller hands that request to
/// Python instead of authenticating a key Python would not have picked.
pub struct PythonBearer;

const LITELLM_API_KEY: &str = "x-litellm-api-key";
const BEARER: &str = "Bearer ";

impl CredentialExtractor for PythonBearer {
    fn extract(&self, parts: &Parts) -> Result<Credential, Error> {
        // TODO: `x-litellm-api-key`, the other schemes and the provider key headers, which Python
        // reads in `get_api_key` and this leaves to Python
        if parts.headers.contains_key(LITELLM_API_KEY) {
            return Err(Error::InvalidToken);
        }
        parts
            .headers
            .get(AUTHORIZATION)
            .and_then(|value| value.to_str().ok())
            .filter(|value| value.starts_with(BEARER))
            // Python strips with `str.replace`, so every occurrence of the scheme goes, not only the prefix.
            .map(|value| value.replace(BEARER, ""))
            .filter(|token| !token.is_empty())
            .map(|token| Credential::Token(Token::new(token)))
            .ok_or(Error::InvalidToken)
    }
}

fn bearer(parts: &Parts) -> Result<Token, Error> {
    let mut values = parts.headers.get_all(AUTHORIZATION).iter();
    let value = values.next().ok_or(Error::InvalidToken)?;
    if values.next().is_some() {
        return Err(Error::InvalidToken);
    }
    let token = value
        .to_str()
        .ok()
        .and_then(|value| value.strip_prefix("Bearer "))
        .map(str::trim)
        .filter(|value| !value.is_empty() && !value.bytes().any(|byte| byte.is_ascii_whitespace()))
        .ok_or(Error::InvalidToken)?;
    Ok(Token::new(token))
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
    use axum::http::{HeaderName, HeaderValue, Request};
    use rstest::rstest;

    use super::*;

    fn parts(headers: &[(&'static str, &[u8])]) -> Parts {
        let (mut parts, ()) = Request::new(()).into_parts();
        headers.iter().for_each(|(name, value)| {
            parts.headers.append(
                HeaderName::from_static(name),
                HeaderValue::from_bytes(value).unwrap(),
            );
        });
        parts
    }

    fn token(parts: &Parts) -> Option<Token> {
        match PythonBearer.extract(parts) {
            Ok(Credential::Token(token)) => Some(token),
            Ok(Credential::Transport) | Err(_) => None,
        }
    }

    #[rstest]
    #[case::bearer(&[("authorization", "Bearer sk-1")], Some("sk-1"))]
    #[case::every_occurrence_stripped(&[("authorization", "Bearer sk-Bearer 1")], Some("sk-1"))]
    #[case::first_of_duplicates(&[("authorization", "Bearer sk-1"), ("authorization", "Bearer sk-2")], Some("sk-1"))]
    #[case::lowercase_scheme(&[("authorization", "bearer sk-1")], None)]
    #[case::basic(&[("authorization", "Basic sk-1")], None)]
    #[case::no_scheme(&[("authorization", "sk-1")], None)]
    #[case::empty_token(&[("authorization", "Bearer ")], None)]
    #[case::litellm_header_wins_in_python(&[("authorization", "Bearer sk-1"), ("x-litellm-api-key", "sk-2")], None)]
    #[case::provider_header_only(&[("x-api-key", "sk-1")], None)]
    #[case::missing(&[], None)]
    fn reads_only_the_bearer_key_python_would_read(
        #[case] headers: &[(&'static str, &'static str)],
        #[case] expected: Option<&str>,
    ) {
        let headers: Vec<_> = headers
            .iter()
            .map(|(name, value)| (*name, value.as_bytes()))
            .collect();
        assert_eq!(
            token(&parts(&headers)).as_ref().map(Token::expose),
            expected
        );
    }

    #[rstest]
    fn a_value_that_is_not_ascii_is_no_key() {
        assert_eq!(token(&parts(&[("authorization", b"Bearer sk-\xe9")])), None);
    }
}
