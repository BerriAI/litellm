//! The FIPS build's decryption of a PKCS#8 `ENCRYPTED PRIVATE KEY`: AWS-LC's own `PKCS8_parse_encrypted_private_key`
//! runs PBKDF2 and the cipher in the FIPS module and decides which PBES2 schemes it reads. Two kinds of key never reach
//! it, because AWS-LC decrypts them with code outside the module: keys `pkcs8` fails to parse, among them the PKCS#12
//! schemes and PBES2 with rc2-cbc, and PBES2 with des-ede3-cbc, which is refused by name

use std::{ffi::CStr, ptr, slice};

use aws_lc_fips_sys::{
    CBB, CBB_cleanup, CBB_finish, CBB_init, CBS, ERR_clear_error, ERR_get_error,
    ERR_reason_error_string, EVP_PKEY_free, EVP_marshal_private_key, OPENSSL_free,
    PKCS8_parse_encrypted_private_key,
};
use pkcs8::{EncryptedPrivateKeyInfo, SecretDocument, pkcs5::pbes2};

use crate::KeyError;

pub(crate) fn decrypt(der: &[u8], password: &str) -> Result<SecretDocument, KeyError> {
    let info = EncryptedPrivateKeyInfo::try_from(der).map_err(KeyError::Decrypt)?;
    if let Some(pbes2::Parameters {
        encryption: pbes2::EncryptionScheme::DesEde3Cbc { .. },
        ..
    }) = info.encryption_algorithm.pbes2()
    {
        return Err(KeyError::TripleDes);
    }
    let mut cbs = CBS {
        data: der.as_ptr(),
        len: der.len(),
    };
    // SAFETY: `cbs` reads `der`, which outlives the call, and the password is passed with its length, so it needs no
    // NUL terminator
    let key = unsafe {
        PKCS8_parse_encrypted_private_key(&mut cbs, password.as_ptr().cast(), password.len())
    };
    if key.is_null() {
        return Err(KeyError::FipsDecrypt(module_error()));
    }
    let (mut out, mut out_len) = (ptr::null_mut(), 0);
    // SAFETY: `key` is the key parsed above and is freed once, here. A zeroed `CBB` is what `CBB_init` writes before it
    // can fail, and `CBB_cleanup` frees the buffer only when `CBB_finish` did not hand it over in `out`
    let marshalled = unsafe {
        let mut cbb: CBB = std::mem::zeroed();
        let marshalled = CBB_init(&mut cbb, 0) == 1
            && EVP_marshal_private_key(&mut cbb, key) == 1
            && CBB_finish(&mut cbb, &mut out, &mut out_len) == 1;
        CBB_cleanup(&mut cbb);
        EVP_PKEY_free(key);
        marshalled
    };
    if !marshalled {
        return Err(KeyError::FipsDecrypt(module_error()));
    }
    // SAFETY: `CBB_finish` handed over `out_len` initialized bytes at `out`, and `OPENSSL_free` zeroes them before it
    // frees them
    let document = unsafe {
        let document = SecretDocument::try_from(slice::from_raw_parts(out, out_len));
        OPENSSL_free(out.cast());
        document
    };
    document.map_err(|error| KeyError::Decrypt(error.into()))
}

/// The reason AWS-LC gave for the earliest error it queued on this thread, such as `BAD_DECRYPT` or `UNSUPPORTED_PRF`,
/// after which the queue is cleared
fn module_error() -> String {
    // SAFETY: `ERR_reason_error_string` returns NULL or a static NUL-terminated string
    unsafe {
        let error = ERR_get_error();
        ERR_clear_error();
        let reason = ERR_reason_error_string(error);
        if reason.is_null() {
            format!("error {error:#x}")
        } else {
            CStr::from_ptr(reason).to_string_lossy().into_owned()
        }
    }
}
