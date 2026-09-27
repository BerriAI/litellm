mod error;

use std::sync::Arc;

use axum::{
    extract::FromRequestParts,
    http::{header::AUTHORIZATION, request::Parts},
};
use litellm_auth_types::SecretValue;
use litellm_config::Config;
use litellm_secrets::source::SecretSource;
use sha2::{Digest, Sha256};
use subtle::ConstantTimeEq;

pub use error::Error;

#[derive(Clone)]
pub struct Auth {
    master_key: Option<SecretValue>,
    secrets: Arc<dyn SecretSource>,
}

impl Auth {
    pub fn from_config(config: &Config, secrets: Arc<dyn SecretSource>) -> Self {
        Self {
            master_key: config.general_settings.master_key.clone(),
            secrets,
        }
    }

    async fn master_key(&self) -> Result<SecretValue, Error> {
        let configured = self.master_key.as_ref().ok_or(Error::Unconfigured)?;
        let resolved = match configured.expose().strip_prefix("os.environ/") {
            Some(name) if !name.is_empty() => self
                .secrets
                .get_secret_str(name)
                .await?
                .ok_or(Error::Unconfigured)?,
            Some(_) => return Err(Error::Unconfigured),
            None => configured.clone(),
        };
        if resolved.expose().trim().is_empty() {
            return Err(Error::Unconfigured);
        }
        Ok(resolved)
    }
}

pub fn hash_token(token: &str) -> String {
    format!("{:x}", Sha256::digest(token.as_bytes()))
}

pub struct RequireMasterKey;

impl FromRequestParts<Auth> for RequireMasterKey {
    type Rejection = Error;

    async fn from_request_parts(parts: &mut Parts, state: &Auth) -> Result<Self, Error> {
        let expected = state.master_key().await?;
        let provided = parts
            .headers
            .get(AUTHORIZATION)
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.strip_prefix("Bearer "))
            .map(str::trim)
            .ok_or(Error::InvalidToken)?;
        let actual_hash = Sha256::digest(provided.as_bytes());
        let expected_hash = Sha256::digest(expected.expose().as_bytes());
        match bool::from(actual_hash.ct_eq(&expected_hash)) {
            true => Ok(Self),
            false => Err(Error::InvalidToken),
        }
    }
}
