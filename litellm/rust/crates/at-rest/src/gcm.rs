//! AES-256-GCM as Python's `_seal_aes_gcm` lays it out, nonce (12) then ciphertext then tag (16), with no AAD

use aes_gcm::aead::{Aead, KeyInit};
use aes_gcm::{Aes256Gcm, Nonce};

use crate::{DecryptError, EncryptError, random_nonce};

const NONCE_BYTES: usize = 12;

pub(crate) fn seal(key: &[u8; 32], plaintext: &[u8]) -> Result<Vec<u8>, EncryptError> {
    let nonce = random_nonce::<NONCE_BYTES>()?;
    let ciphertext = Aes256Gcm::new(key.into())
        .encrypt(Nonce::from_slice(&nonce), plaintext)
        .map_err(EncryptError::TooLong)?;
    Ok([nonce.as_slice(), &ciphertext].concat())
}

pub(crate) fn open(key: &[u8; 32], sealed: &[u8]) -> Result<Vec<u8>, DecryptError> {
    let (nonce, ciphertext) = sealed
        .split_at_checked(NONCE_BYTES)
        .ok_or(DecryptError::NotCiphertext(aes_gcm::aead::Error))?;
    Aes256Gcm::new(key.into())
        .decrypt(Nonce::from_slice(nonce), ciphertext)
        .map_err(DecryptError::NotCiphertext)
}
