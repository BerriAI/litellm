//! The salt key's derived cipher key and the two stored formats it reads and writes

use std::borrow::Cow;

use base64::Engine;
use base64::engine::general_purpose::{STANDARD, URL_SAFE};
use litellm_secret_value::SecretValue;
use sha2::{Digest, Sha256};
use veil::Redact;

use crate::{DecryptError, EncryptError, gcm, secretbox};

const V2_GCM_PREFIX: &str = "v2:gcm:";

/// The cipher a new value is written with. Reads detect the format from the value itself
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Algorithm {
    /// NaCl SecretBox, URL-safe base64 with no prefix. What Python writes unless
    /// `general_settings.encryption_algorithm` is `aes-256-gcm`
    XSalsa20Poly1305,
    /// AES-256-GCM, URL-safe base64 behind `v2:gcm:`
    Aes256Gcm,
}

/// The 32-byte key both formats use, SHA-256 of the salt key as Python's `_derive_key` computes it. The caller
/// picks the salt key, `LITELLM_SALT_KEY` when it is set and the master key otherwise
#[derive(Clone, Redact)]
pub struct AtRestKey(#[redact(with = "[REDACTED]")] [u8; 32]);

/// A stored value read the way Python's `decrypt_value_helper(..., return_original_value=True)` reads it, where
/// DB params mix ciphertexts with plain strings. `Original` keeps why decryption failed so the caller can report it
#[derive(Redact, PartialEq)]
pub enum Plaintext<'a> {
    Decrypted(#[redact(with = "[REDACTED]")] String),
    Original {
        #[redact(with = "[REDACTED]")]
        value: &'a str,
        error: DecryptError,
    },
}

impl<'a> Plaintext<'a> {
    pub fn into_value(self) -> Cow<'a, str> {
        match self {
            Self::Decrypted(plaintext) => Cow::Owned(plaintext),
            Self::Original { value, .. } => Cow::Borrowed(value),
        }
    }
}

impl AtRestKey {
    pub fn from_salt_key(salt_key: &SecretValue) -> Self {
        Self(Sha256::digest(salt_key.expose().as_bytes()).into())
    }

    pub fn encrypt(&self, plaintext: &str, algorithm: Algorithm) -> Result<String, EncryptError> {
        match algorithm {
            Algorithm::XSalsa20Poly1305 => {
                Ok(URL_SAFE.encode(secretbox::seal(&self.0, plaintext.as_bytes())?))
            }
            Algorithm::Aes256Gcm => Ok(format!(
                "{V2_GCM_PREFIX}{}",
                URL_SAFE.encode(gcm::seal(&self.0, plaintext.as_bytes())?)
            )),
        }
    }

    pub fn decrypt(&self, stored: &str) -> Result<String, DecryptError> {
        let opened = match stored.strip_prefix(V2_GCM_PREFIX) {
            Some(payload) => gcm::open(&self.0, &decode(payload)?)?,
            None => secretbox::open(&self.0, &decode(stored)?)?,
        };
        String::from_utf8(opened).map_err(|error| DecryptError::NotUtf8(error.utf8_error()))
    }

    pub fn decrypt_or_original<'a>(&self, stored: &'a str) -> Plaintext<'a> {
        match self.decrypt(stored) {
            Ok(plaintext) => Plaintext::Decrypted(plaintext),
            Err(error) => Plaintext::Original {
                value: stored,
                error,
            },
        }
    }
}

/// Python reads with `urlsafe_b64decode`, which also takes the `+` and `/` of the standard alphabet that values
/// written before the URL-safe switch carry
fn decode(text: &str) -> Result<Vec<u8>, DecryptError> {
    URL_SAFE
        .decode(text)
        .or_else(|_| STANDARD.decode(text))
        .map_err(DecryptError::NotBase64)
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use base64::Engine;
    use base64::engine::general_purpose::URL_SAFE;
    use litellm_secret_value::SecretValue;
    use rstest::{fixture, rstest};
    use serde::Deserialize;

    use super::{Algorithm, AtRestKey, Plaintext, V2_GCM_PREFIX};
    use crate::{DecryptError, gcm};

    #[derive(Deserialize)]
    struct PythonFixture {
        salt_key: String,
        cases: Vec<PythonCase>,
    }

    #[derive(Deserialize)]
    struct PythonCase {
        name: String,
        stored: String,
        decrypted: Option<String>,
        or_original: String,
    }

    #[derive(Deserialize)]
    struct RustFixture {
        salt_key: String,
        cases: Vec<RustCase>,
    }

    #[derive(Deserialize)]
    struct RustCase {
        name: String,
        plaintext: String,
        stored: String,
    }

    static PYTHON: LazyLock<PythonFixture> = LazyLock::new(|| {
        serde_json::from_str(include_str!("../fixtures/python_ciphertexts.json"))
            .expect("python fixture parses")
    });

    static RUST: LazyLock<RustFixture> = LazyLock::new(|| {
        serde_json::from_str(include_str!("../fixtures/rust_ciphertexts.json"))
            .expect("rust fixture parses")
    });

    #[fixture]
    fn key() -> AtRestKey {
        AtRestKey::from_salt_key(&SecretValue::new(PYTHON.salt_key.as_str()))
    }

    fn python_case(name: &str) -> &'static PythonCase {
        PYTHON
            .cases
            .iter()
            .find(|case| case.name == name)
            .expect("case is in the python fixture")
    }

    #[rstest]
    #[case::legacy_api_key("legacy_api_key")]
    #[case::legacy_empty("legacy_empty")]
    #[case::legacy_unicode("legacy_unicode")]
    #[case::legacy_long("legacy_long")]
    #[case::gcm_api_key("gcm_api_key")]
    #[case::gcm_empty("gcm_empty")]
    #[case::gcm_unicode("gcm_unicode")]
    #[case::gcm_long("gcm_long")]
    #[case::legacy_standard_alphabet("legacy_standard_alphabet")]
    #[case::legacy_wrong_key("legacy_wrong_key")]
    #[case::gcm_wrong_key("gcm_wrong_key")]
    #[case::legacy_tampered("legacy_tampered")]
    #[case::gcm_tampered("gcm_tampered")]
    #[case::gcm_truncated("gcm_truncated")]
    #[case::gcm_not_base64("gcm_not_base64")]
    #[case::plain_model_name("plain_model_name")]
    #[case::plain_url("plain_url")]
    #[case::plain_env_reference("plain_env_reference")]
    #[case::plain_base64_alphabet("plain_base64_alphabet")]
    #[case::plain_unicode("plain_unicode")]
    #[case::plain_empty("plain_empty")]
    fn reads_python_values_as_python_does(key: AtRestKey, #[case] name: &str) {
        let case = python_case(name);

        assert_eq!(key.decrypt(&case.stored).ok(), case.decrypted);
        assert_eq!(
            key.decrypt_or_original(&case.stored).into_value(),
            case.or_original
        );
    }

    #[rstest]
    #[case::legacy_api_key("legacy_api_key")]
    #[case::legacy_empty("legacy_empty")]
    #[case::legacy_unicode("legacy_unicode")]
    #[case::gcm_api_key("gcm_api_key")]
    #[case::gcm_empty("gcm_empty")]
    #[case::gcm_unicode("gcm_unicode")]
    fn decrypts_the_values_python_verified(#[case] name: &str) {
        let key = AtRestKey::from_salt_key(&SecretValue::new(RUST.salt_key.as_str()));
        let case = RUST
            .cases
            .iter()
            .find(|case| case.name == name)
            .expect("case is in the rust fixture");

        assert_eq!(
            key.decrypt(&case.stored).as_deref(),
            Ok(case.plaintext.as_str())
        );
    }

    #[rstest]
    #[case::api_key("sk-proj-credential-value")]
    #[case::empty("")]
    #[case::unicode("clé secrète ✓ 鍵")]
    fn encrypted_values_decrypt_to_the_plaintext(
        key: AtRestKey,
        #[values(Algorithm::XSalsa20Poly1305, Algorithm::Aes256Gcm)] algorithm: Algorithm,
        #[case] plaintext: &str,
    ) {
        let stored = key.encrypt(plaintext, algorithm).expect("encrypts");

        assert_eq!(key.decrypt(&stored).as_deref(), Ok(plaintext));
    }

    #[rstest]
    #[case::legacy(Algorithm::XSalsa20Poly1305, false)]
    #[case::gcm(Algorithm::Aes256Gcm, true)]
    fn only_gcm_values_carry_the_version_prefix(
        key: AtRestKey,
        #[case] algorithm: Algorithm,
        #[case] prefixed: bool,
    ) {
        let stored = key
            .encrypt("sk-proj-credential-value", algorithm)
            .expect("encrypts");

        assert_eq!(stored.starts_with(V2_GCM_PREFIX), prefixed);
    }

    #[rstest]
    fn every_encryption_draws_a_fresh_nonce(
        key: AtRestKey,
        #[values(Algorithm::XSalsa20Poly1305, Algorithm::Aes256Gcm)] algorithm: Algorithm,
    ) {
        let first = key
            .encrypt("sk-proj-credential-value", algorithm)
            .expect("encrypts");
        let second = key
            .encrypt("sk-proj-credential-value", algorithm)
            .expect("encrypts");

        assert_ne!(first, second);
    }

    #[rstest]
    #[case::legacy_wrong_key(python_case("legacy_wrong_key").stored.clone())]
    #[case::gcm_wrong_key(python_case("gcm_wrong_key").stored.clone())]
    #[case::gcm_truncated(python_case("gcm_truncated").stored.clone())]
    #[case::legacy_shorter_than_its_nonce(URL_SAFE.encode([0u8; 12]))]
    fn values_under_another_key_or_cut_short_are_not_ciphertexts(
        key: AtRestKey,
        #[case] stored: String,
    ) {
        assert!(matches!(
            key.decrypt(&stored),
            Err(DecryptError::NotCiphertext(_))
        ));
    }

    #[rstest]
    #[case::model_name("azure/gpt-4o")]
    #[case::gcm_payload("v2:gcm:not base64!")]
    #[case::missing_padding("abcdef")]
    fn text_outside_base64_is_not_base64(key: AtRestKey, #[case] stored: &str) {
        assert!(matches!(
            key.decrypt(stored),
            Err(DecryptError::NotBase64(_))
        ));
    }

    #[rstest]
    fn a_plaintext_that_is_not_utf8_is_reported(key: AtRestKey) {
        let sealed = gcm::seal(&key.0, &[0xff, 0xfe]).expect("encrypts");
        let stored = format!("{V2_GCM_PREFIX}{}", URL_SAFE.encode(sealed));

        assert!(matches!(
            key.decrypt(&stored),
            Err(DecryptError::NotUtf8(_))
        ));
    }

    #[rstest]
    fn a_failed_read_keeps_the_original_and_the_reason(key: AtRestKey) {
        let stored = python_case("gcm_wrong_key").stored.as_str();

        assert!(matches!(
            key.decrypt_or_original(stored),
            Plaintext::Original { value, error: DecryptError::NotCiphertext(_) } if value == stored
        ));
    }

    #[rstest]
    fn text_with_no_base64_characters_is_kept_where_python_reads_it_as_empty(key: AtRestKey) {
        assert_eq!(key.decrypt_or_original("***").into_value(), "***");
    }

    #[rstest]
    fn debug_hides_the_derived_key(key: AtRestKey) {
        let debug = format!("{key:?}");

        assert!(!debug.contains(&format!("{:?}", key.0)));
        assert!(debug.contains("[REDACTED]"));
    }

    #[rstest]
    #[case::decrypted(Plaintext::Decrypted("credential-value".to_owned()))]
    #[case::original(Plaintext::Original {
        value: "credential-value",
        error: DecryptError::NotCiphertext(aes_gcm::aead::Error),
    })]
    fn debug_hides_the_plaintext(#[case] plaintext: Plaintext<'static>) {
        let debug = format!("{plaintext:?}");

        assert!(!debug.contains("credential-value"));
        assert!(debug.contains("[REDACTED]"));
    }
}
