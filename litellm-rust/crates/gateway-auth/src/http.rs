use axum::{
    extract::{FromRequestParts, MatchedPath, Request, State},
    http::{HeaderName, header::AUTHORIZATION, request::Parts},
    middleware::Next,
    response::Response,
};
use litellm_auth_types::SecretValue;

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

/// The headers the proxy reads a key from: `Authorization: Bearer`, then the Anthropic SDK's
/// `x-api-key`, then Azure's `api-key`. The first header present must hold a valid key.
pub struct ApiKeyHeaders;

const X_API_KEY: HeaderName = HeaderName::from_static("x-api-key");
const API_KEY: HeaderName = HeaderName::from_static("api-key");

impl CredentialExtractor for ApiKeyHeaders {
    fn extract(&self, parts: &Parts) -> Result<Credential, Error> {
        if parts.headers.contains_key(AUTHORIZATION) {
            return bearer(parts).map(Credential::Token);
        }
        [X_API_KEY, API_KEY]
            .iter()
            .find(|name| parts.headers.contains_key(*name))
            .ok_or(Error::MissingToken)
            .and_then(|name| header_token(parts, name, ""))
            .map(Credential::Token)
    }
}

fn bearer(parts: &Parts) -> Result<SecretValue, Error> {
    header_token(parts, &AUTHORIZATION, "Bearer ")
}

fn header_token(parts: &Parts, name: &HeaderName, scheme: &str) -> Result<SecretValue, Error> {
    let mut values = parts.headers.get_all(name).iter();
    let value = values.next().ok_or(Error::MissingToken)?;
    if values.next().is_some() {
        return Err(Error::InvalidToken);
    }
    let token = value
        .to_str()
        .ok()
        .and_then(|value| value.strip_prefix(scheme))
        .map(str::trim)
        .filter(|value| !value.is_empty() && !value.bytes().any(|byte| byte.is_ascii_whitespace()))
        .ok_or(Error::InvalidToken)?;
    Ok(SecretValue::new(token))
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

    fn extract(headers: &[(&str, &str)]) -> Result<String, Error> {
        let mut request = Request::get("/");
        for (name, value) in headers {
            request = request.header(*name, *value);
        }
        match ApiKeyHeaders.extract(&request.body(()).unwrap().into_parts().0)? {
            Credential::Token(token) => Ok(token.expose().to_owned()),
            Credential::Transport => unreachable!("headers carry tokens"),
        }
    }

    #[rstest]
    #[case::bearer(&[("authorization", "Bearer sk-a")], "sk-a")]
    #[case::anthropic(&[("x-api-key", "sk-a")], "sk-a")]
    #[case::azure(&[("api-key", "sk-a")], "sk-a")]
    #[case::trimmed(&[("x-api-key", " sk-a ")], "sk-a")]
    #[case::bearer_wins(&[("x-api-key", "sk-b"), ("authorization", "Bearer sk-a")], "sk-a")]
    #[case::anthropic_before_azure(&[("api-key", "sk-b"), ("x-api-key", "sk-a")], "sk-a")]
    fn reads_the_first_present_key_header(
        #[case] headers: &[(&str, &str)],
        #[case] expected: &str,
    ) {
        assert_eq!(extract(headers).unwrap(), expected);
    }

    #[rstest]
    #[case::none(&[], 401, "No api key")]
    #[case::wrong_scheme(&[("authorization", "Basic sk-a"), ("x-api-key", "sk-b")], 401, "Invalid")]
    #[case::empty(&[("x-api-key", "")], 401, "Invalid")]
    #[case::inner_space(&[("x-api-key", "sk a")], 401, "Invalid")]
    #[case::duplicate(&[("x-api-key", "sk-a"), ("x-api-key", "sk-a")], 401, "Invalid")]
    fn rejects_missing_or_malformed_keys(
        #[case] headers: &[(&str, &str)],
        #[case] status: u16,
        #[case] message: &str,
    ) {
        let error = extract(headers).unwrap_err();
        assert_eq!(error.status().as_u16(), status);
        assert!(error.to_string().contains(message));
    }
}
