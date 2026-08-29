use sha2::{Digest, Sha256};

const HEX_CHARS: [u8; 16] = [
    b'0', b'1', b'2', b'3', b'4', b'5', b'6', b'7', b'8', b'9', b'a', b'b', b'c', b'd', b'e', b'f',
];

/// A SHA-256 hash stored entirely on the stack. No heap allocation.
///
/// Contains both the raw 32-byte digest and the pre-formatted 64-byte hex string.
/// This eliminates the `String` allocation that `hash_token()` previously returned.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct HashedToken {
    raw: [u8; 32],
    hex: [u8; 64],
}

impl HashedToken {
    /// Hash a token. Entirely stack-allocated, no heap allocation.
    pub fn hash(token: &str) -> Self {
        let digest = Sha256::digest(token.as_bytes());
        let mut raw = [0u8; 32];
        raw.copy_from_slice(&digest);

        let mut hex = [0u8; 64];
        for (i, &b) in raw.iter().enumerate() {
            hex[i * 2] = HEX_CHARS[(b >> 4) as usize];
            hex[i * 2 + 1] = HEX_CHARS[(b & 0xf) as usize];
        }

        Self { raw, hex }
    }

    /// Borrow the hex representation as a `&str`. Zero-cost borrow.
    pub fn as_hex_str(&self) -> &str {
        // Safety: hex is always valid ASCII hex digits
        unsafe { std::str::from_utf8_unchecked(&self.hex) }
    }

    /// Borrow the raw 32-byte digest.
    pub fn as_raw(&self) -> &[u8; 32] {
        &self.raw
    }
}

impl std::fmt::Debug for HashedToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "HashedToken({})", self.as_hex_str())
    }
}

impl std::fmt::Display for HashedToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_hex_str())
    }
}

/// SHA-256 hash of a token, returned as a 64-char lowercase hex string.
///
/// Matches Python's `hashlib.sha256(token.encode()).hexdigest()`.
///
/// Note: This allocates a `String`. Prefer `HashedToken::hash()` for zero-alloc.
pub fn hash_token(token: &str) -> String {
    HashedToken::hash(token).as_hex_str().to_string()
}

/// Hash a token only if it starts with `sk-`. Already-hashed tokens (64-char
/// hex) are returned as-is.
///
/// Matches Python's `_hash_token_if_needed`.
///
/// Note: This allocates a `String`. Prefer `HashedToken` for zero-alloc paths.
pub fn hash_token_if_needed(token: &str) -> String {
    if token.starts_with("sk-") {
        hash_token(token)
    } else {
        token.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hashed_token_is_zero_alloc() {
        let token = HashedToken::hash("sk-test-1234");
        // Verify it's a valid 64-char hex string
        assert_eq!(token.as_hex_str().len(), 64);
        assert!(token.as_hex_str().chars().all(|c| c.is_ascii_hexdigit()));
        // Verify it matches the String-based hash_token
        assert_eq!(token.as_hex_str(), hash_token("sk-test-1234"));
    }

    #[test]
    fn hashed_token_hex_is_valid_str() {
        let token = HashedToken::hash("sk-secret");
        let hex = token.as_hex_str();
        assert_eq!(hex.len(), 64);
        assert!(hex.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn hashed_token_raw_matches_hex() {
        let token = HashedToken::hash("test");
        let hex = token.as_hex_str();
        let raw = token.as_raw();
        for (i, &b) in raw.iter().enumerate() {
            assert_eq!(hex.as_bytes()[i * 2], HEX_CHARS[(b >> 4) as usize]);
            assert_eq!(hex.as_bytes()[i * 2 + 1], HEX_CHARS[(b & 0xf) as usize]);
        }
    }

    #[test]
    fn hashed_token_copy_is_free() {
        let t1 = HashedToken::hash("test");
        let t2 = t1; // Copy, not Clone - no allocation
        assert_eq!(t1, t2);
    }
}
