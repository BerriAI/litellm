//! The FIPS build's decryption of a PKCS#8 `ENCRYPTED PRIVATE KEY`: PBES2 with PBKDF2 (HMAC-SHA1, SHA-256, SHA-384 or
//! SHA-512) and AES-CBC, both run by the AWS-LC FIPS module through aws-lc-rs, with `pkcs8` only parsing the ASN.1.
//! scrypt, des-ede3-cbc and hmacWithSHA224, which the default build decrypts in RustCrypto, are refused

use std::num::NonZeroU32;

use aws_lc_rs::{
    cipher::{self, DecryptionContext, PaddedBlockDecryptingKey, UnboundCipherKey},
    iv::FixedLength,
    pbkdf2,
};
use pkcs8::{
    EncryptedPrivateKeyInfo, SecretDocument,
    pkcs5::{self, pbes2},
};

use crate::{KeyError, UnapprovedScheme};

pub(crate) fn decrypt(
    info: &EncryptedPrivateKeyInfo<'_>,
    password: &str,
) -> Result<SecretDocument, KeyError> {
    let refused = |scheme| Err(KeyError::NotFipsApproved(scheme));
    let pkcs5::EncryptionScheme::Pbes2(params) = &info.encryption_algorithm else {
        return refused(UnapprovedScheme::Other(info.encryption_algorithm.oid()));
    };
    let kdf = match &params.kdf {
        pbes2::Kdf::Pbkdf2(kdf) => kdf,
        pbes2::Kdf::Scrypt(_) => return refused(UnapprovedScheme::Scrypt),
        kdf => return refused(UnapprovedScheme::Other(kdf.oid())),
    };
    let prf = match kdf.prf {
        pbes2::Pbkdf2Prf::HmacWithSha1 => pbkdf2::PBKDF2_HMAC_SHA1,
        pbes2::Pbkdf2Prf::HmacWithSha256 => pbkdf2::PBKDF2_HMAC_SHA256,
        pbes2::Pbkdf2Prf::HmacWithSha384 => pbkdf2::PBKDF2_HMAC_SHA384,
        pbes2::Pbkdf2Prf::HmacWithSha512 => pbkdf2::PBKDF2_HMAC_SHA512,
        pbes2::Pbkdf2Prf::HmacWithSha224 => return refused(UnapprovedScheme::HmacWithSha224),
        prf => return refused(UnapprovedScheme::Other(prf.oid())),
    };
    let (algorithm, iv) = match params.encryption {
        pbes2::EncryptionScheme::Aes128Cbc { iv } => (&cipher::AES_128, iv),
        pbes2::EncryptionScheme::Aes192Cbc { iv } => (&cipher::AES_192, iv),
        pbes2::EncryptionScheme::Aes256Cbc { iv } => (&cipher::AES_256, iv),
        pbes2::EncryptionScheme::DesEde3Cbc { .. } => {
            return refused(UnapprovedScheme::DesEde3Cbc);
        }
        scheme => return refused(UnapprovedScheme::Other(scheme.oid())),
    };
    let invalid = || KeyError::Decrypt(params.kdf.to_alg_params_invalid().into());
    let key_size = params.encryption.key_size();
    if kdf
        .key_length
        .is_some_and(|length| usize::from(length) != key_size)
    {
        return Err(invalid());
    }
    let iterations = NonZeroU32::new(kdf.iteration_count).ok_or_else(invalid)?;
    let mut key = [0; 32];
    let key = &mut key[..key_size];
    pbkdf2::derive(prf, iterations, kdf.salt, password.as_bytes(), key);
    let mut buffer = info.encrypted_data.to_vec();
    let decrypted = UnboundCipherKey::new(algorithm, key)
        .and_then(PaddedBlockDecryptingKey::cbc_pkcs7)
        .and_then(|key| key.decrypt(&mut buffer, DecryptionContext::Iv128(FixedLength::from(iv))))
        .map_err(|_| KeyError::Decrypt(pkcs5::Error::DecryptFailed.into()))?;
    SecretDocument::try_from(decrypted.to_vec()).map_err(|error| KeyError::Decrypt(error.into()))
}
