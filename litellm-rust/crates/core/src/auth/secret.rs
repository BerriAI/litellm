use veil::Redact;

#[derive(Redact, Clone)]
pub struct SecretValue(#[redact(with = "[REDACTED]")] String);

impl SecretValue {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    pub fn expose(&self) -> &str {
        &self.0
    }
}

impl PartialEq for SecretValue {
    fn eq(&self, other: &Self) -> bool {
        subtle::ConstantTimeEq::ct_eq(self.0.as_bytes(), other.0.as_bytes()).into()
    }
}

impl Eq for SecretValue {}

#[cfg(test)]
mod tests {
    use super::SecretValue;

    #[test]
    fn debug_redacts_plaintext() {
        let debug = format!("{:?}", SecretValue::new("credential-value"));

        assert!(!debug.contains("credential-value"));
        assert!(debug.contains("REDACTED"));
    }

    #[test]
    fn equality_compares_plaintext_values() {
        assert_eq!(SecretValue::new("same"), SecretValue::new("same"));
        assert_ne!(SecretValue::new("same"), SecretValue::new("different"));
    }
}
