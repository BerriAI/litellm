use serde::{Deserialize, Serialize};
use std::hash::{Hash, Hasher};

use crate::Redacted;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SecretValue(Redacted<String>);

impl SecretValue {
    pub fn new(value: impl Into<String>) -> Self {
        Self(Redacted::new(value.into()))
    }

    pub fn expose(&self) -> &str {
        self.0.expose()
    }
}

impl From<String> for SecretValue {
    fn from(value: String) -> Self {
        Self::new(value)
    }
}

impl From<&str> for SecretValue {
    fn from(value: &str) -> Self {
        Self::new(value)
    }
}

impl PartialEq for SecretValue {
    fn eq(&self, other: &Self) -> bool {
        subtle::ConstantTimeEq::ct_eq(self.expose().as_bytes(), other.expose().as_bytes()).into()
    }
}

impl Eq for SecretValue {}

impl Hash for SecretValue {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.expose().hash(state);
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::SecretValue;

    #[rstest]
    fn debug_redacts_plaintext() {
        let debug = format!("{:?}", SecretValue::new("credential-value"));

        assert!(!debug.contains("credential-value"));
        assert!(debug.contains("REDACTED"));
    }

    #[rstest]
    fn equality_compares_plaintext_values() {
        assert_eq!(SecretValue::new("same"), SecretValue::new("same"));
        assert_ne!(SecretValue::new("same"), SecretValue::new("different"));
    }
}
