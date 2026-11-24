# at-rest

Salt-key encryption of the values the Python proxy stores in the DB (model params, credentials, config rows), byte-compatible with `litellm/proxy/common_utils/encrypt_decrypt_utils.py`. A crate that reads or writes one of those values depends on this crate and never re-implements a format

- Pure and synchronous. No I/O, no async, no environment reads. The caller picks the salt key (`LITELLM_SALT_KEY`, else the master key) and passes it in as a `SecretValue`
- Kept apart from `secrets` so the ~20 crates depending on that async crate do not pull in ciphers
- No hand-rolled crypto. RustCrypto `aes-gcm` and `crypto_secretbox` share one `aead` version, so a cipher bump moves both together
- Reads detect the format from the value. `v2:gcm:` is AES-256-GCM over base64url(nonce 12, ciphertext, tag 16) with no AAD, anything else is NaCl SecretBox over base64url(nonce 24, tag 16, ciphertext). The key for both is SHA-256 of the salt key
- Writes take the `Algorithm` from the caller. Python chooses it from `general_settings.encryption_algorithm`, and that choice stays with the caller
- `decrypt_or_original` is Python's `decrypt_value_helper(..., return_original_value=True)`, for DB params that mix ciphertexts with plain strings. It keeps the `DecryptError`, so a caller can tell a value that failed to decrypt apart from one that decrypted
- FIPS mode is not supported. Python's `fips_mode` derives the GCM key with KBKDF counter-mode HMAC-SHA256 under the label `litellm-at-rest:aes-256-gcm:` (`_fips_at_rest_key`) and refuses legacy values. Deployments with `fips_mode` on are forwarded to Python, so nothing here may be used for them
- Bearer tokens (`encrypt_bearer_token`, GCM with the prefix as AAD and unpadded base64) are a different format and are not ported here

## Divergences from Python

Python decodes base64 leniently, skipping characters outside the alphabet and accepting bad trailing bits. Here a stored value must be canonical base64 in either the URL-safe or the standard alphabet, as Python always writes it. So text with no base64 characters at all, such as `***`, is kept as it is, where Python reads it as an empty plaintext. That case is pinned by its own test. Likewise a ciphertext with whitespace inside, such as a newline from a hand edit of the DB, fails here and is kept as it is, where Python decrypts it

## Fixtures

- `fixtures/python_ciphertexts.json` is written by `scripts/generate_python_fixtures.py` from Python's own functions, never by hand. Every run draws new nonces, so a regeneration rewrites every value
- `fixtures/rust_ciphertexts.json` holds values this crate encrypted. `scripts/verify_rust_ciphertexts.py` checks Python decrypts each one to its plaintext. CI runs no Python, so run it after any change to how values are written, and recapture the file when the format changes
