use std::{sync::Arc, time::SystemTime};

use litellm_secret_value::SecretValue;

use crate::hash_token;

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

/// What this server reads off the row of a virtual key its host authenticated.
pub trait HeldKey {
    fn expires_at(&self) -> Option<SystemTime>;

    /// The key's model list, where none reaches every model.
    fn models(&self) -> Arc<[String]>;
}
