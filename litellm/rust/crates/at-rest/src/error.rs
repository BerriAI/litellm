/// Why a stored value could not be read as a ciphertext under the salt key
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum DecryptError {
    #[error("the stored value is not base64")]
    NotBase64(#[source] base64::DecodeError),
    #[error("the stored value is not a ciphertext under this salt key")]
    NotCiphertext,
    #[error("the decrypted value is not UTF-8")]
    NotUtf8(#[source] std::str::Utf8Error),
    #[error("with fips_mode on, only v2:gcm: values are decrypted")]
    NotGcmInFipsMode,
    #[error("this build has no XSalsa20-Poly1305, so NaCl values cannot be read")]
    XSalsa20NotBuilt,
}

/// Why a value could not be encrypted
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum EncryptError {
    #[error("the operating system could not supply a random nonce")]
    Nonce(#[source] getrandom::Error),
    #[error("the value is longer than the cipher can encrypt")]
    TooLong,
    #[error("with fips_mode on, values are written only with AES-256-GCM")]
    XSalsa20InFipsMode,
    #[error("this build has no XSalsa20-Poly1305")]
    XSalsa20NotBuilt,
}

/// The salt key is under the 112 bits (14 bytes of UTF-8) fips_mode derives its keys from, Python's
/// `key_derivation_key` returning None
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("with fips_mode on, the salt key must be at least 112 bits (14 bytes)")]
pub struct SaltKeyTooShort;
