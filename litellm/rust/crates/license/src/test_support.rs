use std::sync::LazyLock;

use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use rsa::RsaPrivateKey;
use rsa::pkcs8::{EncodePublicKey, LineEnding};
use rsa::pss::Pss;
use serde_json::json;
use sha2::{Digest, Sha256};

use crate::PublicKey;

pub(crate) const MAX_SALT_FOR_2048_BIT_KEY: usize = 222;

pub(crate) static SIGNER: LazyLock<RsaPrivateKey> = LazyLock::new(key);
pub(crate) static OTHER_SIGNER: LazyLock<RsaPrivateKey> = LazyLock::new(key);

fn key() -> RsaPrivateKey {
    RsaPrivateKey::new(&mut rand::thread_rng(), 2048).unwrap()
}

pub(crate) fn public_key() -> PublicKey {
    PublicKey::from_pem(
        &SIGNER
            .to_public_key()
            .to_public_key_pem(LineEnding::LF)
            .unwrap(),
    )
    .unwrap()
}

pub(crate) fn sign_with(signer: &RsaPrivateKey, message: &[u8]) -> String {
    let signature = signer
        .sign_with_rng(
            &mut rand::thread_rng(),
            Pss::new_with_salt::<Sha256>(MAX_SALT_FOR_2048_BIT_KEY),
            &Sha256::digest(message),
        )
        .unwrap();
    STANDARD.encode([message, b".", &signature].concat())
}

pub(crate) fn sign(message: &[u8]) -> String {
    sign_with(&SIGNER, message)
}

pub(crate) fn license(expiration_date: &str) -> String {
    sign(
        json!({"expiration_date": expiration_date, "max_users": 5})
            .to_string()
            .as_bytes(),
    )
}
