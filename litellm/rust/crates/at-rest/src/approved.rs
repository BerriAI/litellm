//! The FIPS-approved primitives the crate runs: SHA-256, SP 800-108 KBKDF over HMAC-SHA256, and AES-256-GCM laid out
//! as Python's `_seal_aes_gcm` writes it, nonce (12) then ciphertext then tag (16), with no AAD. No other module names a
//! hash or cipher library. The default build runs them on RustCrypto. The `fips` build has no validated module wired in
//! yet, so every operation there fails closed with `NoValidatedBackend`

use zeroize::Zeroizing;

const KEY_BYTES: usize = 32;

pub(crate) type Key = Zeroizing<[u8; KEY_BYTES]>;

#[cfg(feature = "fips")]
pub(crate) use no_validated_backend::{
    aes_256_gcm_open, aes_256_gcm_seal, kbkdf_hmac_sha256, sha256,
};
#[cfg(not(feature = "fips"))]
pub(crate) use rustcrypto::{aes_256_gcm_open, aes_256_gcm_seal, kbkdf_hmac_sha256, sha256};

#[cfg(not(feature = "fips"))]
mod rustcrypto {
    use aes_gcm::aead::{Aead, KeyInit};
    use aes_gcm::{Aes256Gcm, Nonce};
    use hmac::digest::FixedOutput;
    use hmac::{Hmac, Mac};
    use sha2::{Digest, Sha256};

    use super::Key;
    use crate::{DecryptError, EncryptError, KeyError, random_nonce};

    const KEY_BITS: u32 = 256;
    const GCM_NONCE_BYTES: usize = 12;

    pub(crate) fn sha256(data: &[u8]) -> Result<Key, KeyError> {
        let mut key = Key::default();
        Digest::finalize_into(
            <Sha256 as Digest>::new_with_prefix(data),
            key.as_mut().into(),
        );
        Ok(key)
    }

    /// KBKDF in counter mode with HMAC-SHA256 and OpenSSL's `EVP_KDF` defaults, which Python's `_kbkdf_hmac_sha256`
    /// runs: HMAC over a 32-bit big-endian counter, the label, a zero byte, the context and the output length in bits
    /// as a 32-bit big-endian L. A 256-bit output is one HMAC block, so the counter is only ever 1
    pub(crate) fn kbkdf_hmac_sha256(
        key: &[u8],
        label: &[u8],
        context: &[u8],
    ) -> Result<Key, KeyError> {
        let mut mac =
            <Hmac<Sha256> as Mac>::new_from_slice(key).expect("HMAC takes a key of any length");
        Mac::update(&mut mac, &1u32.to_be_bytes());
        Mac::update(&mut mac, label);
        Mac::update(&mut mac, &[0]);
        Mac::update(&mut mac, context);
        Mac::update(&mut mac, &KEY_BITS.to_be_bytes());
        let mut derived = Key::default();
        mac.finalize_into(derived.as_mut().into());
        Ok(derived)
    }

    pub(crate) fn aes_256_gcm_seal(key: &Key, plaintext: &[u8]) -> Result<Vec<u8>, EncryptError> {
        let nonce = random_nonce::<GCM_NONCE_BYTES>()?;
        let ciphertext = Aes256Gcm::new(key.as_ref().into())
            .encrypt(Nonce::from_slice(&nonce), plaintext)
            .map_err(|_| EncryptError::TooLong)?;
        Ok([nonce.as_slice(), &ciphertext].concat())
    }

    pub(crate) fn aes_256_gcm_open(key: &Key, sealed: &[u8]) -> Result<Vec<u8>, DecryptError> {
        let (nonce, ciphertext) = sealed
            .split_at_checked(GCM_NONCE_BYTES)
            .ok_or(DecryptError::NotCiphertext)?;
        Aes256Gcm::new(key.as_ref().into())
            .decrypt(Nonce::from_slice(nonce), ciphertext)
            .map_err(|_| DecryptError::NotCiphertext)
    }
}

#[cfg(feature = "fips")]
mod no_validated_backend {
    use super::Key;
    use crate::{DecryptError, EncryptError, KeyError};

    pub(crate) fn sha256(_: &[u8]) -> Result<Key, KeyError> {
        Err(KeyError::NoValidatedBackend)
    }

    pub(crate) fn kbkdf_hmac_sha256(_: &[u8], _: &[u8], _: &[u8]) -> Result<Key, KeyError> {
        Err(KeyError::NoValidatedBackend)
    }

    pub(crate) fn aes_256_gcm_seal(_: &Key, _: &[u8]) -> Result<Vec<u8>, EncryptError> {
        Err(EncryptError::NoValidatedBackend)
    }

    pub(crate) fn aes_256_gcm_open(_: &Key, _: &[u8]) -> Result<Vec<u8>, DecryptError> {
        Err(DecryptError::NoValidatedBackend)
    }
}

#[cfg(all(test, not(feature = "fips")))]
mod tests {
    use std::sync::LazyLock;

    use base64::Engine;
    use base64::engine::general_purpose::STANDARD;
    use rstest::rstest;
    use serde::Deserialize;

    use super::kbkdf_hmac_sha256;

    #[derive(Deserialize)]
    struct Fixture {
        kbkdf: Vec<KbkdfVector>,
    }

    #[derive(Deserialize)]
    struct KbkdfVector {
        name: String,
        key: String,
        label: String,
        context: String,
        derived: String,
    }

    static OPENSSL: LazyLock<Fixture> = LazyLock::new(|| {
        serde_json::from_str(include_str!("../fixtures/python_fips_ciphertexts.json"))
            .expect("python fips fixture parses")
    });

    fn bytes(base64: &str) -> Vec<u8> {
        STANDARD.decode(base64).expect("fixture holds base64")
    }

    #[rstest]
    #[case::at_rest_key("at_rest_key")]
    #[case::with_context("with_context")]
    #[case::key_longer_than_a_hmac_block("key_longer_than_a_hmac_block")]
    fn kbkdf_matches_openssl(#[case] name: &str) {
        let vector = OPENSSL
            .kbkdf
            .iter()
            .find(|vector| vector.name == name)
            .expect("vector is in the fixture");

        let derived = kbkdf_hmac_sha256(
            &bytes(&vector.key),
            &bytes(&vector.label),
            &bytes(&vector.context),
        )
        .expect("derives");

        assert_eq!(derived.as_slice(), bytes(&vector.derived));
    }
}

#[cfg(all(test, feature = "fips"))]
mod fips_build_tests {
    use rstest::rstest;

    use super::{Key, aes_256_gcm_open, aes_256_gcm_seal, kbkdf_hmac_sha256, sha256};
    use crate::{DecryptError, EncryptError, KeyError};

    #[rstest]
    fn sha256_fails_closed() {
        assert_eq!(
            sha256(b"sk-a-long-enough-salt-key").err(),
            Some(KeyError::NoValidatedBackend)
        );
    }

    #[rstest]
    fn kbkdf_fails_closed() {
        assert_eq!(
            kbkdf_hmac_sha256(b"sk-a-long-enough-salt-key", b"label", b"").err(),
            Some(KeyError::NoValidatedBackend)
        );
    }

    #[rstest]
    fn gcm_seal_fails_closed() {
        assert_eq!(
            aes_256_gcm_seal(&Key::default(), b"sk-proj-credential-value"),
            Err(EncryptError::NoValidatedBackend)
        );
    }

    #[rstest]
    fn gcm_open_fails_closed() {
        assert_eq!(
            aes_256_gcm_open(&Key::default(), &[0u8; 40]),
            Err(DecryptError::NoValidatedBackend)
        );
    }
}
