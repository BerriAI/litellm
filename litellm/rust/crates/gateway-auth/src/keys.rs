use std::{future::Future, pin::Pin, time::SystemTime};

use litellm_secret_value::SecretValue;

use crate::{Error, hash_token};

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
    Pin<Box<dyn Future<Output = Result<Option<KeyStatus>, Error>> + Send + 'a>>;

/// The virtual keys this server authenticates. A key it holds but cannot honor is an error, never
/// a status.
pub trait KeyLookup: Send + Sync {
    fn lookup<'a>(&'a self, hash: &'a KeyHash) -> LookupFuture<'a>;
}
