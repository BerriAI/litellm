/// Why a stored value could not be read as a ciphertext under the salt key
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum DecryptError {
    #[error("the stored value is not base64")]
    NotBase64(#[source] base64::DecodeError),
    #[error("the stored value is not a ciphertext under this salt key")]
    NotCiphertext(#[source] aes_gcm::aead::Error),
    #[error("the decrypted value is not UTF-8")]
    NotUtf8(#[source] std::str::Utf8Error),
}

/// Why a value could not be encrypted
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum EncryptError {
    #[error("the operating system could not supply a random nonce")]
    Nonce(#[source] getrandom::Error),
    #[error("the value is longer than the cipher can encrypt")]
    TooLong(#[source] aes_gcm::aead::Error),
}
