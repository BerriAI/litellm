use std::{future::Future, pin::Pin, time::SystemTime};

use litellm_gateway_auth_core::credential::key::{KeyHash, Token};

use crate::Error;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum KeyStatus {
    Active { expires_at: Option<SystemTime> },
    Revoked,
}

pub type LookupFuture<'a> =
    Pin<Box<dyn Future<Output = Result<Option<KeyStatus>, Error>> + Send + 'a>>;

pub trait KeyLookup: Send + Sync {
    fn lookup<'a>(&'a self, hash: &'a KeyHash) -> LookupFuture<'a>;
}

pub async fn verify_key(
    lookup: &dyn KeyLookup,
    token: &Token,
    now: impl FnOnce() -> SystemTime,
) -> Result<KeyHash, Error> {
    let hash = token.hash();
    match lookup.lookup(&hash).await? {
        Some(KeyStatus::Active { expires_at }) => {
            if expires_at.is_some_and(|expiry| expiry <= now()) {
                return Err(Error::Expired);
            }
            Ok(hash)
        }
        Some(KeyStatus::Revoked) | None => Err(Error::InvalidToken),
    }
}
