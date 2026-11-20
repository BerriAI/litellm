use std::{collections::BTreeMap, sync::Arc, time::SystemTime};

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

/// A limit for each model a key names one for, by the name a caller asks for or its model group.
pub type ModelLimits = Arc<BTreeMap<String, i64>>;

/// The limits a virtual key holds, which the server enforces after authentication.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct KeyLimits {
    pub rpm: Option<i64>,
    pub tpm: Option<i64>,
    pub max_parallel_requests: Option<i64>,
    pub model_rpm: ModelLimits,
    pub model_tpm: ModelLimits,
}

impl KeyLimits {
    /// Whether any limit counts tokens, which a call reserves from its body before it runs.
    pub fn counts_tokens(&self) -> bool {
        self.tpm.is_some() || !self.model_tpm.is_empty()
    }

    /// Whether any limit is told from the request body: its tokens or the model it names.
    pub fn reads_body(&self) -> bool {
        self.counts_tokens() || !self.model_rpm.is_empty()
    }
}

/// What this server reads off the row of a virtual key its host authenticated.
pub trait HeldKey {
    fn expires_at(&self) -> Option<SystemTime>;

    fn limits(&self) -> KeyLimits;

    /// The priority class the key's `metadata.priority` names, which reserves it a share of a
    /// model group's capacity.
    fn priority(&self) -> Option<Arc<str>>;

    /// The key's model list, where none reaches every model.
    fn models(&self) -> Arc<[String]>;
}
