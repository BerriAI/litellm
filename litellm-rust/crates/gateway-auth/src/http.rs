use axum::{
    extract::{FromRequestParts, MatchedPath, Request, State},
    http::request::Parts,
    middleware::Next,
    response::Response,
};
use litellm_secret_value::SecretValue;

use crate::{AccessRequest, ApiKey, Auth, AuthenticatedRequest, Credential, Error};

pub trait CredentialExtractor: Send + Sync {
    fn extract(&self, parts: &Parts) -> Result<Credential, Error>;
}

/// Reads the key from the headers Python reads, in Python's order.
pub struct ApiKeyHeaders;

impl CredentialExtractor for ApiKeyHeaders {
    fn extract(&self, parts: &Parts) -> Result<Credential, Error> {
        match ApiKey::from_headers(&parts.headers) {
            ApiKey::Token(token) if !token.is_empty() => {
                Ok(Credential::Token(SecretValue::new(token)))
            }
            ApiKey::Token(_) | ApiKey::Missing | ApiKey::Unreadable => Err(Error::InvalidToken),
        }
    }
}

pub async fn authenticate(
    State(auth): State<Auth>,
    request: Request,
    next: Next,
) -> Result<Response, Error> {
    let (mut parts, body) = request.into_parts();
    let identity = auth.admit(&parts).await?;
    parts.extensions.insert(identity);
    Ok(next.run(Request::from_parts(parts, body)).await)
}

/// Whether `identity` may call the method and matched route of `parts`.
pub(crate) async fn authorize_route(
    identity: &AuthenticatedRequest,
    parts: &Parts,
) -> Result<(), Error> {
    let path = parts
        .extensions
        .get::<MatchedPath>()
        .ok_or(Error::MissingRoute)?;
    let route = AccessRequest::Route {
        method: parts.method.to_string(),
        path: path.as_str().into(),
    };
    identity
        .authorize(route.clone())
        .await?
        .consume(identity.caller(), &route)
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
