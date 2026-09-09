use std::fmt;
use std::time::SystemTime;

#[derive(Clone, PartialEq, Eq)]
pub struct SecretString(String);

impl SecretString {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    pub fn expose(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for SecretString {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("SecretString([REDACTED])")
    }
}

impl fmt::Display for SecretString {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("[REDACTED]")
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SuppliedSecret {
    pub source: String,
    pub value: Option<SecretString>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolvedCredential {
    pub value: SecretString,
    pub expires_at: Option<SystemTime>,
}
