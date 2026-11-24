//! NaCl SecretBox (XSalsa20-Poly1305) as PyNaCl's `SecretBox.encrypt` lays it out, nonce (24) then tag (16) then
//! ciphertext

use crypto_secretbox::aead::{Aead, KeyInit};
use crypto_secretbox::{Nonce, XSalsa20Poly1305};

use crate::{DecryptError, EncryptError, random_nonce};

const NONCE_BYTES: usize = 24;

pub(crate) fn seal(key: &[u8; 32], plaintext: &[u8]) -> Result<Vec<u8>, EncryptError> {
    let nonce = random_nonce::<NONCE_BYTES>()?;
    let ciphertext = XSalsa20Poly1305::new(key.into())
        .encrypt(Nonce::from_slice(&nonce), plaintext)
        .map_err(EncryptError::TooLong)?;
    Ok([nonce.as_slice(), &ciphertext].concat())
}

/// An empty value opens to an empty plaintext under any key, as Python's `decrypt_value` returns `""` for no bytes
pub(crate) fn open(key: &[u8; 32], sealed: &[u8]) -> Result<Vec<u8>, DecryptError> {
    if sealed.is_empty() {
        return Ok(Vec::new());
    }
    let (nonce, ciphertext) = sealed
        .split_at_checked(NONCE_BYTES)
        .ok_or(DecryptError::NotCiphertext(crypto_secretbox::aead::Error))?;
    XSalsa20Poly1305::new(key.into())
        .decrypt(Nonce::from_slice(nonce), ciphertext)
        .map_err(DecryptError::NotCiphertext)
}
