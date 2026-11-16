use axum::{
    extract::{FromRequestParts, MatchedPath, Request, State},
    http::request::Parts,
    middleware::Next,
    response::Response,
};

use crate::{AccessRequest, ApiKey, Auth, AuthenticatedRequest, Error};

pub async fn authenticate(
    State(auth): State<Auth>,
    key: ApiKey,
    request: Request,
    next: Next,
) -> Result<Response, Error> {
    let (mut parts, body) = request.into_parts();
    let identity = auth.authenticate_request(&key.into(), &parts).await?;
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
            .authenticate_request(&ApiKey::from_headers(&parts.headers)?.into(), parts)
            .await?;
        if identity.caller().authentication().method != crate::AuthenticationMethod::MasterKey {
            return Err(Error::InvalidToken);
        }
        parts.extensions.insert(identity);
        Ok(Self)
    }
}
