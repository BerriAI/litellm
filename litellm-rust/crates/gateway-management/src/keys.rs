use std::{future::Future, pin::Pin, sync::Arc, time::SystemTime};

use base64::{Engine, engine::general_purpose::URL_SAFE_NO_PAD};
use litellm_auth_types::SecretValue;
use litellm_gateway_auth::keys::{KeyHash, KeyStatus};
use rand::{RngCore, rngs::OsRng};

use crate::Error;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NewKey {
    pub hash: KeyHash,
    pub expires_at: Option<SystemTime>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KeyRecord {
    pub hash: KeyHash,
    pub status: KeyStatus,
}

#[derive(Debug)]
pub struct GeneratedKey {
    pub token: SecretValue,
    pub record: KeyRecord,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Revocation {
    Revoked,
    NotFound,
}

pub type StoreFuture<'a, T> = Pin<Box<dyn Future<Output = Result<T, Error>> + Send + 'a>>;

pub trait KeyStore: Send + Sync {
    fn create(&self, key: NewKey) -> StoreFuture<'_, ()>;
    fn get<'a>(&'a self, hash: &'a KeyHash) -> StoreFuture<'a, Option<KeyRecord>>;
    fn revoke<'a>(&'a self, hash: &'a KeyHash) -> StoreFuture<'a, Revocation>;
}

pub struct Keys {
    store: Arc<dyn KeyStore>,
}

impl Keys {
    pub fn new(store: Arc<dyn KeyStore>) -> Self {
        Self { store }
    }

    pub async fn generate(
        &self,
        expires_at: Option<SystemTime>,
        now: SystemTime,
    ) -> Result<GeneratedKey, Error> {
        if expires_at.is_some_and(|expiry| expiry <= now) {
            return Err(Error::InvalidExpiration);
        }
        let mut entropy = [0_u8; 32];
        OsRng.try_fill_bytes(&mut entropy).map_err(Error::Entropy)?;
        let token = SecretValue::new(format!("sk-{}", URL_SAFE_NO_PAD.encode(entropy)));
        let hash = KeyHash::from_token(&token);
        self.store
            .create(NewKey {
                hash: hash.clone(),
                expires_at,
            })
            .await?;
        Ok(GeneratedKey {
            token,
            record: KeyRecord {
                hash,
                status: KeyStatus::Active { expires_at },
            },
        })
    }

    pub async fn get(&self, hash: &KeyHash) -> Result<Option<KeyRecord>, Error> {
        self.store.get(hash).await
    }

    pub async fn revoke(&self, hash: &KeyHash) -> Result<Revocation, Error> {
        self.store.revoke(hash).await
    }
}
