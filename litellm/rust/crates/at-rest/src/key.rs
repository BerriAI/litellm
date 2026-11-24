//! The salt key's derived cipher key and the two stored formats it reads and writes

use std::borrow::Cow;

use base64::Engine;
use base64::engine::general_purpose::{STANDARD, URL_SAFE};
use litellm_secret_value::SecretValue;
use veil::Redact;

use crate::approved::{self, Key};
use crate::{DecryptError, EncryptError, SaltKeyTooShort, secretbox};

const V2_GCM_PREFIX: &str = "v2:gcm:";
const FIPS_KEY_LABEL: &[u8] = b"litellm-at-rest:aes-256-gcm:";
const FIPS_MIN_SALT_KEY_BYTES: usize = 14;

/// The cipher a new value is written with. Reads detect the format from the value itself
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Algorithm {
    /// NaCl SecretBox, URL-safe base64 with no prefix. What Python writes unless
    /// `general_settings.encryption_algorithm` is `aes-256-gcm` or `fips_mode` is on
    XSalsa20Poly1305,
    /// AES-256-GCM, URL-safe base64 behind `v2:gcm:`
    Aes256Gcm,
}

/// How the salt key becomes the cipher key, chosen by `general_settings.fips_mode` as in Python
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KeyDerivation {
    /// SHA-256 of the salt key, Python's `_derive_key` with `fips_mode` off. Reads and writes both formats
    Sha256,
    /// KBKDF counter-mode HMAC-SHA256 of the salt key under the label `litellm-at-rest:aes-256-gcm:`, Python's
    /// `_fips_at_rest_key` with `fips_mode` on. Reads and writes only `v2:gcm:` values
    FipsKbkdf,
}

/// The 32-byte key both formats use, derived from the salt key. The caller picks the salt key, `LITELLM_SALT_KEY`
/// when it is set and the master key otherwise
#[derive(Clone, Redact)]
pub struct AtRestKey {
    #[redact(with = "[REDACTED]")]
    key: Key,
    derivation: KeyDerivation,
}

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
    /// The key with `fips_mode` off, the same as `derive(salt_key, KeyDerivation::Sha256)`
    pub fn from_salt_key(salt_key: &SecretValue) -> Self {
        Self {
            key: approved::sha256(salt_key.expose().as_bytes()),
            derivation: KeyDerivation::Sha256,
        }
    }

    /// Only `KeyDerivation::FipsKbkdf` can fail, when the salt key is under 14 bytes of UTF-8
    pub fn derive(
        salt_key: &SecretValue,
        derivation: KeyDerivation,
    ) -> Result<Self, SaltKeyTooShort> {
        let bytes = salt_key.expose().as_bytes();
        let key = match derivation {
            KeyDerivation::Sha256 => approved::sha256(bytes),
            KeyDerivation::FipsKbkdf if bytes.len() < FIPS_MIN_SALT_KEY_BYTES => {
                return Err(SaltKeyTooShort);
            }
            KeyDerivation::FipsKbkdf => approved::kbkdf_hmac_sha256(bytes, FIPS_KEY_LABEL, b""),
        };
        Ok(Self { key, derivation })
    }

    pub fn encrypt(&self, plaintext: &str, algorithm: Algorithm) -> Result<String, EncryptError> {
        match (algorithm, self.derivation) {
            (Algorithm::Aes256Gcm, _) => Ok(format!(
                "{V2_GCM_PREFIX}{}",
                URL_SAFE.encode(approved::aes_256_gcm_seal(&self.key, plaintext.as_bytes())?)
            )),
            (Algorithm::XSalsa20Poly1305, KeyDerivation::FipsKbkdf) => {
                Err(EncryptError::XSalsa20InFipsMode)
            }
            (Algorithm::XSalsa20Poly1305, KeyDerivation::Sha256) => {
                Ok(URL_SAFE.encode(secretbox::seal(&self.key, plaintext.as_bytes())?))
            }
        }
    }

    pub fn decrypt(&self, stored: &str) -> Result<String, DecryptError> {
        let opened = match (stored.strip_prefix(V2_GCM_PREFIX), self.derivation) {
            (Some(payload), _) => approved::aes_256_gcm_open(&self.key, &decode(payload)?)?,
            (None, KeyDerivation::FipsKbkdf) => return Err(DecryptError::NotGcmInFipsMode),
            (None, KeyDerivation::Sha256) => secretbox::open(&self.key, &decode(stored)?)?,
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

    use super::{Algorithm, AtRestKey, KeyDerivation, Plaintext, V2_GCM_PREFIX};
    use crate::{DecryptError, EncryptError, SaltKeyTooShort, approved};

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

    static PYTHON_FIPS: LazyLock<PythonFixture> = LazyLock::new(|| {
        serde_json::from_str(include_str!("../fixtures/python_fips_ciphertexts.json"))
            .expect("python fips fixture parses")
    });

    static RUST: LazyLock<RustFixture> = LazyLock::new(|| {
        serde_json::from_str(include_str!("../fixtures/rust_ciphertexts.json"))
            .expect("rust fixture parses")
    });

    static RUST_FIPS: LazyLock<RustFixture> = LazyLock::new(|| {
        serde_json::from_str(include_str!("../fixtures/rust_fips_ciphertexts.json"))
            .expect("rust fips fixture parses")
    });

    #[fixture]
    fn key() -> AtRestKey {
        AtRestKey::from_salt_key(&SecretValue::new(PYTHON.salt_key.as_str()))
    }

    #[fixture]
    fn fips_key() -> AtRestKey {
        AtRestKey::derive(
            &SecretValue::new(PYTHON_FIPS.salt_key.as_str()),
            KeyDerivation::FipsKbkdf,
        )
        .expect("the fixture salt key is long enough")
    }

    fn case_in<'a>(fixture: &'a PythonFixture, name: &str) -> &'a PythonCase {
        fixture
            .cases
            .iter()
            .find(|case| case.name == name)
            .expect("case is in the python fixture")
    }

    fn python_case(name: &str) -> &'static PythonCase {
        case_in(&PYTHON, name)
    }

    fn python_fips_case(name: &str) -> &'static PythonCase {
        case_in(&PYTHON_FIPS, name)
    }

    fn rust_case(fixture: &'static RustFixture, name: &str) -> &'static RustCase {
        fixture
            .cases
            .iter()
            .find(|case| case.name == name)
            .expect("case is in the rust fixture")
    }

    fn assert_reads_as_python(key: &AtRestKey, case: &PythonCase) {
        assert_eq!(key.decrypt(&case.stored).ok(), case.decrypted);
        assert_eq!(
            key.decrypt_or_original(&case.stored).into_value(),
            case.or_original
        );
    }

    #[rstest]
    #[case::gcm_api_key("gcm_api_key")]
    #[case::gcm_empty("gcm_empty")]
    #[case::gcm_unicode("gcm_unicode")]
    #[case::gcm_long("gcm_long")]
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
    fn reads_python_values_as_python_does(key: AtRestKey, #[case] name: &str) {
        assert_reads_as_python(&key, python_case(name));
    }

    #[cfg(all(feature = "xsalsa20", not(feature = "fips")))]
    #[rstest]
    #[case::legacy_api_key("legacy_api_key")]
    #[case::legacy_empty("legacy_empty")]
    #[case::legacy_unicode("legacy_unicode")]
    #[case::legacy_long("legacy_long")]
    #[case::legacy_standard_alphabet("legacy_standard_alphabet")]
    #[case::plain_empty("plain_empty")]
    fn reads_python_nacl_values_as_python_does(key: AtRestKey, #[case] name: &str) {
        assert_reads_as_python(&key, python_case(name));
    }

    #[rstest]
    #[case::fips_api_key("fips_api_key")]
    #[case::fips_empty("fips_empty")]
    #[case::fips_unicode("fips_unicode")]
    #[case::fips_long("fips_long")]
    #[case::fips_wrong_key("fips_wrong_key")]
    #[case::fips_tampered("fips_tampered")]
    #[case::standard_gcm("standard_gcm")]
    #[case::legacy_api_key("legacy_api_key")]
    #[case::plain_model_name("plain_model_name")]
    #[case::plain_base64_alphabet("plain_base64_alphabet")]
    #[case::plain_empty("plain_empty")]
    fn reads_python_values_in_fips_mode_as_python_does(fips_key: AtRestKey, #[case] name: &str) {
        assert_reads_as_python(&fips_key, python_fips_case(name));
    }

    #[rstest]
    #[case::gcm_api_key("gcm_api_key")]
    #[case::gcm_empty("gcm_empty")]
    #[case::gcm_unicode("gcm_unicode")]
    #[cfg_attr(
        all(feature = "xsalsa20", not(feature = "fips")),
        case::legacy_api_key("legacy_api_key")
    )]
    #[cfg_attr(
        all(feature = "xsalsa20", not(feature = "fips")),
        case::legacy_empty("legacy_empty")
    )]
    #[cfg_attr(
        all(feature = "xsalsa20", not(feature = "fips")),
        case::legacy_unicode("legacy_unicode")
    )]
    fn decrypts_the_values_python_verified(#[case] name: &str) {
        let key = AtRestKey::from_salt_key(&SecretValue::new(RUST.salt_key.as_str()));
        let case = rust_case(&RUST, name);

        assert_eq!(
            key.decrypt(&case.stored).as_deref(),
            Ok(case.plaintext.as_str())
        );
    }

    #[rstest]
    #[case::api_key("fips_api_key")]
    #[case::empty("fips_empty")]
    #[case::unicode("fips_unicode")]
    fn decrypts_the_values_python_verified_in_fips_mode(#[case] name: &str) {
        let key = AtRestKey::derive(
            &SecretValue::new(RUST_FIPS.salt_key.as_str()),
            KeyDerivation::FipsKbkdf,
        )
        .expect("the fixture salt key is long enough");
        let case = rust_case(&RUST_FIPS, name);

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
        #[values(key(), fips_key())] key: AtRestKey,
        #[case] plaintext: &str,
    ) {
        let stored = key
            .encrypt(plaintext, Algorithm::Aes256Gcm)
            .expect("encrypts");

        assert_eq!(key.decrypt(&stored).as_deref(), Ok(plaintext));
    }

    #[cfg(all(feature = "xsalsa20", not(feature = "fips")))]
    #[rstest]
    #[case::api_key("sk-proj-credential-value")]
    #[case::empty("")]
    #[case::unicode("clé secrète ✓ 鍵")]
    fn nacl_values_decrypt_to_the_plaintext(key: AtRestKey, #[case] plaintext: &str) {
        let stored = key
            .encrypt(plaintext, Algorithm::XSalsa20Poly1305)
            .expect("encrypts");

        assert_eq!(key.decrypt(&stored).as_deref(), Ok(plaintext));
    }

    #[rstest]
    #[case::too_short("sk-thirteen-b", Some(SaltKeyTooShort))]
    #[case::fourteen_bytes("sk-fourteen-by", None)]
    #[case::fourteen_bytes_in_seven_characters("ééééééé", None)]
    #[case::thirteen_bytes_in_seven_characters("éééééée", Some(SaltKeyTooShort))]
    #[case::empty("", Some(SaltKeyTooShort))]
    fn fips_mode_needs_a_salt_key_of_at_least_fourteen_bytes(
        #[case] salt_key: &str,
        #[case] expected: Option<SaltKeyTooShort>,
    ) {
        assert_eq!(
            AtRestKey::derive(&SecretValue::new(salt_key), KeyDerivation::FipsKbkdf).err(),
            expected
        );
    }

    #[rstest]
    fn the_sha256_derivation_is_the_salt_key_one_and_takes_any_length(key: AtRestKey) {
        let derived = AtRestKey::derive(
            &SecretValue::new(PYTHON.salt_key.as_str()),
            KeyDerivation::Sha256,
        )
        .expect("sha256 takes any salt key");
        let short = AtRestKey::derive(&SecretValue::new("short"), KeyDerivation::Sha256);
        let stored = key
            .encrypt("sk-proj-credential-value", Algorithm::Aes256Gcm)
            .expect("encrypts");

        assert_eq!(
            derived.decrypt(&stored).as_deref(),
            Ok("sk-proj-credential-value")
        );
        assert!(short.is_ok());
    }

    #[rstest]
    fn fips_mode_refuses_nacl_values_and_keeps_the_original(fips_key: AtRestKey) {
        let stored = python_fips_case("legacy_api_key").stored.as_str();

        assert_eq!(
            fips_key.decrypt(stored),
            Err(DecryptError::NotGcmInFipsMode)
        );
        assert!(matches!(
            fips_key.decrypt_or_original(stored),
            Plaintext::Original { value, error: DecryptError::NotGcmInFipsMode } if value == stored
        ));
    }

    #[rstest]
    fn fips_mode_writes_only_gcm(fips_key: AtRestKey) {
        assert_eq!(
            fips_key.encrypt("sk-proj-credential-value", Algorithm::XSalsa20Poly1305),
            Err(EncryptError::XSalsa20InFipsMode)
        );
    }

    #[rstest]
    fn a_fips_value_is_not_a_ciphertext_under_the_sha256_key() {
        let key = AtRestKey::from_salt_key(&SecretValue::new(PYTHON_FIPS.salt_key.as_str()));

        assert_eq!(
            key.decrypt(&python_fips_case("fips_api_key").stored),
            Err(DecryptError::NotCiphertext)
        );
    }

    #[cfg(not(all(feature = "xsalsa20", not(feature = "fips"))))]
    #[rstest]
    fn without_xsalsa20_nacl_values_are_unreadable_even_with_fips_mode_off(key: AtRestKey) {
        assert_eq!(
            key.decrypt(&python_case("legacy_api_key").stored),
            Err(DecryptError::XSalsa20NotBuilt)
        );
        assert_eq!(
            key.encrypt("sk-proj-credential-value", Algorithm::XSalsa20Poly1305),
            Err(EncryptError::XSalsa20NotBuilt)
        );
    }

    #[rstest]
    #[cfg_attr(
        all(feature = "xsalsa20", not(feature = "fips")),
        case::legacy(Algorithm::XSalsa20Poly1305, false)
    )]
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
    #[cfg_attr(
        all(feature = "xsalsa20", not(feature = "fips")),
        case::legacy(key(), Algorithm::XSalsa20Poly1305)
    )]
    #[case::gcm(key(), Algorithm::Aes256Gcm)]
    #[case::fips_gcm(fips_key(), Algorithm::Aes256Gcm)]
    fn every_encryption_draws_a_fresh_nonce(#[case] key: AtRestKey, #[case] algorithm: Algorithm) {
        let first = key
            .encrypt("sk-proj-credential-value", algorithm)
            .expect("encrypts");
        let second = key
            .encrypt("sk-proj-credential-value", algorithm)
            .expect("encrypts");

        assert_ne!(first, second);
    }

    #[rstest]
    #[cfg_attr(
        all(feature = "xsalsa20", not(feature = "fips")),
        case::legacy_wrong_key(python_case("legacy_wrong_key").stored.clone())
    )]
    #[cfg_attr(
        all(feature = "xsalsa20", not(feature = "fips")),
        case::legacy_shorter_than_its_nonce(URL_SAFE.encode([0u8; 12]))
    )]
    #[case::gcm_wrong_key(python_case("gcm_wrong_key").stored.clone())]
    #[case::gcm_truncated(python_case("gcm_truncated").stored.clone())]
    fn values_under_another_key_or_cut_short_are_not_ciphertexts(
        key: AtRestKey,
        #[case] stored: String,
    ) {
        assert_eq!(key.decrypt(&stored), Err(DecryptError::NotCiphertext));
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
        let sealed = approved::aes_256_gcm_seal(&key.key, &[0xff, 0xfe]).expect("encrypts");
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
            Plaintext::Original { value, error: DecryptError::NotCiphertext } if value == stored
        ));
    }

    #[rstest]
    fn text_with_no_base64_characters_is_kept_where_python_reads_it_as_empty(key: AtRestKey) {
        assert_eq!(key.decrypt_or_original("***").into_value(), "***");
    }

    #[rstest]
    fn debug_hides_the_derived_key(#[values(key(), fips_key())] key: AtRestKey) {
        let debug = format!("{key:?}");

        assert!(!debug.contains(&format!("{:?}", key.key.as_slice())));
        assert!(debug.contains("[REDACTED]"));
    }

    #[rstest]
    #[case::decrypted(Plaintext::Decrypted("credential-value".to_owned()))]
    #[case::original(Plaintext::Original {
        value: "credential-value",
        error: DecryptError::NotCiphertext,
    })]
    fn debug_hides_the_plaintext(#[case] plaintext: Plaintext<'static>) {
        let debug = format!("{plaintext:?}");

        assert!(!debug.contains("credential-value"));
        assert!(debug.contains("[REDACTED]"));
    }
}
