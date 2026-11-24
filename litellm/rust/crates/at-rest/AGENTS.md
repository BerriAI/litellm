# at-rest

Salt-key encryption of the values the Python proxy stores in the DB (model params, credentials, config rows), byte-compatible with `litellm/proxy/common_utils/encrypt_decrypt_utils.py`. A crate that reads or writes one of those values depends on this crate and never re-implements a format

- Pure and synchronous. No I/O, no async, no environment reads. The caller picks the salt key (`LITELLM_SALT_KEY`, else the master key) and passes it in as a `SecretValue`
- Kept apart from `secrets` so the ~20 crates depending on that async crate do not pull in ciphers
- No hand-rolled crypto. RustCrypto `aes-gcm` and `crypto_secretbox` share one `aead` version, so a cipher bump moves both together. KBKDF is only the SP 800-108 input layout around RustCrypto `hmac`
- `src/approved.rs` is the seam for FIPS. SHA-256, HMAC-SHA256, KBKDF and AES-256-GCM live there and nowhere else, so pointing the `fips` build at a validated module (aws-lc-rs or OpenSSL) changes that file only
- Reads detect the format from the value. `v2:gcm:` is AES-256-GCM over base64url(nonce 12, ciphertext, tag 16) with no AAD, anything else is NaCl SecretBox over base64url(nonce 24, tag 16, ciphertext). With `KeyDerivation::Sha256` the key for both is SHA-256 of the salt key
- Writes take the `Algorithm` from the caller. Python chooses it from `general_settings.encryption_algorithm`, and that choice stays with the caller
- `decrypt_or_original` is Python's `decrypt_value_helper(..., return_original_value=True)`, for DB params that mix ciphertexts with plain strings. It keeps the `DecryptError`, so a caller can tell a value that failed to decrypt apart from one that decrypted
- `KeyDerivation::FipsKbkdf` is Python's `fips_mode`. The key is KBKDF counter-mode HMAC-SHA256 under the label `litellm-at-rest:aes-256-gcm:` with OpenSSL's default layout (`_fips_at_rest_key`), a salt key under 14 bytes is `SaltKeyTooShort`, and only `v2:gcm:` values are read or written. The caller parses `general_settings.fips_mode` and picks the derivation
- The `fips` feature compiles XSalsa20 out, so NaCl values are unreadable in that build even with `fips_mode` off. `crypto_secretbox` comes from the default `xsalsa20` feature, so `--no-default-features --features fips` also drops it from the dependency graph
- Bearer tokens (`encrypt_bearer_token`, GCM with the prefix as AAD and unpadded base64) are a different format and are not ported here

## Divergences from Python

Python decodes base64 leniently, skipping characters outside the alphabet and accepting bad trailing bits. Here a stored value must be canonical base64 in either the URL-safe or the standard alphabet, as Python always writes it. So text with no base64 characters at all, such as `***`, is kept as it is, where Python reads it as an empty plaintext. That case is pinned by its own test. Likewise a ciphertext with whitespace inside, such as a newline from a hand edit of the DB, fails here and is kept as it is, where Python decrypts it

## Fixtures

- `fixtures/python_ciphertexts.json` is written by `scripts/generate_python_fixtures.py` from Python's own functions, never by hand. Every run draws new nonces, so a regeneration rewrites every value
- `fixtures/python_fips_ciphertexts.json` is written by `scripts/generate_python_fips_fixtures.py` with `fips_mode` on. No FIPS provider is needed, `scripts/python_fips_mode.py` runs Python's own EVP_KDF and EVP_CIPHER calls on OpenSSL's default provider and checks every KBKDF output against `cryptography`'s KBKDFHMAC
- `fixtures/rust_ciphertexts.json` and `fixtures/rust_fips_ciphertexts.json` hold values this crate encrypted. `scripts/verify_rust_ciphertexts.py` checks Python decrypts each one to its plaintext, the second with `fips_mode` on. CI runs no Python, so run it after any change to how values are written, and recapture the file when the format changes
