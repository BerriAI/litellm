use litellm_secret_value::SecretValue;

use crate::authn::jwt::is_jwt;

/// The credential a request presents.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Token(SecretValue);

impl Token {
    pub fn new(value: impl Into<SecretValue>) -> Self {
        Self(value.into())
    }

    pub fn expose(&self) -> &str {
        self.0.expose()
    }

    pub fn is_jwt(&self) -> bool {
        is_jwt(self.expose())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn debug_redacts_plaintext() {
        let debug = format!("{:?}", Token::new("credential-value"));
        assert!(!debug.contains("credential-value"), "{debug}");
        assert!(debug.contains("[REDACTED]"), "{debug}");
    }
}
