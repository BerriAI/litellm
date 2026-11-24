//! Salt-key encryption of the values the Python proxy stores in the DB, such as model params and credentials,
//! byte-compatible with `litellm/proxy/common_utils/encrypt_decrypt_utils.py` outside FIPS mode
#![forbid(unsafe_code)]

mod error;
mod gcm;
mod key;
mod secretbox;

pub use error::{DecryptError, EncryptError};
pub use key::{Algorithm, AtRestKey, Plaintext};

fn random_nonce<const N: usize>() -> Result<[u8; N], EncryptError> {
    let mut nonce = [0u8; N];
    getrandom::fill(&mut nonce).map_err(EncryptError::Nonce)?;
    Ok(nonce)
}
