use axum::extract::FromRequestParts;
use axum::http::StatusCode;
use axum::http::header::AUTHORIZATION;
use axum::http::request::Parts;
use subtle::ConstantTimeEq;

use crate::MasterKeyState;

pub struct RequireMasterKey;

impl<S> FromRequestParts<S> for RequireMasterKey
where
    S: MasterKeyState + Send + Sync,
{
    type Rejection = (StatusCode, String);

    async fn from_request_parts(parts: &mut Parts, state: &S) -> Result<Self, Self::Rejection> {
        let Some(expected) = state.master_key() else {
            return Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                "gateway auth not configured (set LITELLM_MASTER_KEY)".to_string(),
            ));
        };
        let provided = parts
            .headers
            .get(AUTHORIZATION)
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.strip_prefix("Bearer "))
            .map(str::trim);
        match provided {
            Some(token) if bool::from(token.as_bytes().ct_eq(expected.as_bytes())) => Ok(Self),
            _ => Err((
                StatusCode::UNAUTHORIZED,
                "missing or invalid bearer token".to_string(),
            )),
        }
    }
}
