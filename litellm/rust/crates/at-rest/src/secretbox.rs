//! NaCl SecretBox (XSalsa20-Poly1305) as PyNaCl's `SecretBox.encrypt` lays it out, nonce (24) then tag (16) then
//! ciphertext. FIPS does not approve XSalsa20, so it is built only with the `xsalsa20` feature and without `fips`,
//! and otherwise every NaCl value is refused

#[cfg(all(feature = "xsalsa20", not(feature = "fips")))]
pub(crate) use built::{open, seal};
#[cfg(not(all(feature = "xsalsa20", not(feature = "fips"))))]
pub(crate) use not_built::{open, seal};

#[cfg(all(feature = "xsalsa20", not(feature = "fips")))]
mod built {
    use crypto_secretbox::aead::{Aead, KeyInit};
    use crypto_secretbox::{Nonce, XSalsa20Poly1305};

    use crate::approved::Key;
    use crate::{DecryptError, EncryptError, random_nonce};

    const NONCE_BYTES: usize = 24;

    pub(crate) fn seal(key: &Key, plaintext: &[u8]) -> Result<Vec<u8>, EncryptError> {
        let nonce = random_nonce::<NONCE_BYTES>()?;
        let ciphertext = XSalsa20Poly1305::new(key.as_ref().into())
            .encrypt(Nonce::from_slice(&nonce), plaintext)
            .map_err(|_| EncryptError::TooLong)?;
        Ok([nonce.as_slice(), &ciphertext].concat())
    }

    /// An empty value opens to an empty plaintext under any key, as Python's `decrypt_value` returns `""` for no bytes
    pub(crate) fn open(key: &Key, sealed: &[u8]) -> Result<Vec<u8>, DecryptError> {
        if sealed.is_empty() {
            return Ok(Vec::new());
        }
        let (nonce, ciphertext) = sealed
            .split_at_checked(NONCE_BYTES)
            .ok_or(DecryptError::NotCiphertext)?;
        XSalsa20Poly1305::new(key.as_ref().into())
            .decrypt(Nonce::from_slice(nonce), ciphertext)
            .map_err(|_| DecryptError::NotCiphertext)
    }
}

#[cfg(not(all(feature = "xsalsa20", not(feature = "fips"))))]
mod not_built {
    use crate::approved::Key;
    use crate::{DecryptError, EncryptError};

    pub(crate) fn seal(_: &Key, _: &[u8]) -> Result<Vec<u8>, EncryptError> {
        Err(EncryptError::XSalsa20NotBuilt)
    }

    pub(crate) fn open(_: &Key, _: &[u8]) -> Result<Vec<u8>, DecryptError> {
        Err(DecryptError::XSalsa20NotBuilt)
    }
}
