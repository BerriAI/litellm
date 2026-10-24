use std::{future::Future, pin::Pin, time::SystemTime};

use litellm_auth_types::SecretValue;

use crate::{KeyError, hash_token};

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct KeyHash(String);

impl KeyHash {
    pub fn from_token(token: &SecretValue) -> Self {
        Self(hash_token(token.expose()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum KeyStatus {
    Active { expires_at: Option<SystemTime> },
    Revoked,
}

pub type LookupFuture<'a> =
    Pin<Box<dyn Future<Output = Result<Option<KeyStatus>, KeyError>> + Send + 'a>>;

pub trait KeyLookup: Send + Sync {
    fn lookup<'a>(&'a self, hash: &'a KeyHash) -> LookupFuture<'a>;
}

pub async fn verify_key(
    lookup: &dyn KeyLookup,
    token: &SecretValue,
    now: impl FnOnce() -> SystemTime,
) -> Result<KeyHash, KeyError> {
    let hash = KeyHash::from_token(token);
    match lookup.lookup(&hash).await? {
        Some(KeyStatus::Active { expires_at }) => {
            if expires_at.is_some_and(|expiry| expiry <= now()) {
                return Err(KeyError::Expired);
            }
            Ok(hash)
        }
        Some(KeyStatus::Revoked) | None => Err(KeyError::Invalid),
    }
}
