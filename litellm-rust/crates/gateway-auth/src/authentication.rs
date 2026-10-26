use std::sync::Arc;

use litellm_auth_types::SecretValue;
use litellm_config::Config;
use litellm_secrets::source::SecretSource;
use sha2::{Digest, Sha256};
use subtle::ConstantTimeEq;

use crate::{Caller, Error};

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

    pub async fn validate(&self) -> Result<(), Error> {
        self.master_key().await.map(drop)
    }

    pub async fn authenticate(&self, token: &SecretValue) -> Result<Caller, Error> {
        let expected = self.master_key().await?;
        if bool::from(Sha256::digest(token.expose()).ct_eq(&Sha256::digest(expected.expose()))) {
            Ok(Caller::Master)
        } else {
            Err(Error::InvalidToken)
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
