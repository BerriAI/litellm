use std::fmt;

use litellm_secret_value::SecretValue;
use sha2::{Digest, Sha256};
use subtle::ConstantTimeEq;

use crate::credential::jwt::is_jwt;

const VIRTUAL_KEY_PREFIX: &str = "sk-";

/// The credential a request presents.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Token(SecretValue);

impl Token {
    pub fn new(value: impl Into<SecretValue>) -> Self {
        Self(value.into())
    }

    /// The token main reports for a key a JWT mapped to, which is the hash the key stores.
    pub(super) fn from_stored(hash: &KeyHash) -> Self {
        Self::new(hash.as_str())
    }

    pub fn expose(&self) -> &str {
        self.0.expose()
    }

    pub fn is_empty(&self) -> bool {
        self.expose().is_empty()
    }

    pub fn is_virtual_key(&self) -> bool {
        self.expose().starts_with(VIRTUAL_KEY_PREFIX)
    }

    pub fn is_jwt(&self) -> bool {
        is_jwt(self.expose())
    }

    pub fn hash(&self) -> KeyHash {
        KeyHash(format!("{:x}", Sha256::digest(self.expose().as_bytes())))
    }

    pub(super) fn masked(&self) -> String {
        let chars: Vec<char> = self.expose().chars().collect();
        if chars.len() <= 8 {
            return "****".to_owned();
        }
        let first: String = chars[..4].iter().collect();
        let last: String = chars[chars.len() - 4..].iter().collect();
        format!("{first}****{last}")
    }
}

#[derive(Clone, Debug)]
pub struct MasterKey(SecretValue);

impl MasterKey {
    pub fn new(value: impl Into<SecretValue>) -> Self {
        Self(value.into())
    }

    /// `secrets.compare_digest`, which raises on non-ASCII text, and the proxy reads that as no match.
    pub fn matches(&self, token: &Token) -> bool {
        let (token, master_key) = (token.expose(), self.0.expose());
        token.is_ascii()
            && master_key.is_ascii()
            && bool::from(token.as_bytes().ct_eq(master_key.as_bytes()))
    }
}

/// The SHA-256 a key is stored and cached under.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct KeyHash(String);

impl KeyHash {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for KeyHash {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

/// The token the proxy names in an error: the presented one, or its hash once the key was read
/// from the database.
#[derive(Clone, Copy)]
pub(super) enum Shown<'a> {
    Presented(&'a Token),
    Stored(&'a KeyHash),
}

impl Shown<'_> {
    /// `abbreviate_api_key`
    pub(super) fn abbreviated(self, min_custom_key_length: usize) -> String {
        let text = match self {
            Self::Presented(token) => token.expose(),
            Self::Stored(hash) => hash.as_str(),
        };
        let chars: Vec<char> = text.chars().collect();
        if chars.len() < min_custom_key_length {
            return "sk-...".to_owned();
        }
        let last: String = chars[chars.len().saturating_sub(4)..].iter().collect();
        format!("sk-...{last}")
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    const MASTER: &str = "sk-master";

    #[test]
    fn hashes_with_sha256() {
        // FIPS 180-2 test vector for "abc"
        assert_eq!(
            Token::new("abc").hash().as_str(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[rstest]
    #[case::equal(MASTER, MASTER, true)]
    #[case::different("sk-other", MASTER, false)]
    #[case::prefix("sk-mast", MASTER, false)]
    #[case::non_ascii_never_matches("sk-ü", "sk-ü", false)]
    fn master_key_comparison(#[case] token: &str, #[case] master: &str, #[case] expected: bool) {
        assert_eq!(MasterKey::new(master).matches(&Token::new(token)), expected);
    }

    #[rstest]
    #[case::token(format!("{:?}", Token::new("credential-value")))]
    #[case::master_key(format!("{:?}", MasterKey::new("credential-value")))]
    fn debug_redacts_plaintext(#[case] debug: String) {
        assert!(!debug.contains("credential-value"), "{debug}");
        assert!(debug.contains("[REDACTED]"), "{debug}");
    }

    #[rstest]
    #[case::short_key_is_hidden("sk-1", 16, "sk-...")]
    #[case::long_key_shows_its_last_four("sk-1234567890abcdef", 16, "sk-...cdef")]
    fn expired_key_names_only_the_tail(
        #[case] token: &str,
        #[case] min_custom_key_length: usize,
        #[case] expected: &str,
    ) {
        assert_eq!(
            Shown::Presented(&Token::new(token)).abbreviated(min_custom_key_length),
            expected
        );
    }
}
