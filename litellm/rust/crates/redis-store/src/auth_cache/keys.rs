/// `is_user_key_cache_key`: user-key objects are cached under a bare `hash_token` digest, every
/// other object under a prefixed key.
pub fn is_key_object(key: &str) -> bool {
    key.len() == 64
        && key
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}
