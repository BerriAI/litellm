//! `urllib.parse.quote(text, safe=safe)` over the UTF-8 bytes of `text`.

/// `urllib.parse.quote(text, safe=safe)`: ASCII letters, digits and `_.-~` are never quoted,
/// the ASCII characters in `safe` are kept, and every other byte becomes `%XX`.
pub fn quote(text: &str, safe: &str) -> String {
    let mut encoded = String::with_capacity(text.len());
    for byte in text.bytes() {
        if byte.is_ascii_alphanumeric()
            || matches!(byte, b'_' | b'.' | b'-' | b'~')
            || (byte.is_ascii() && safe.as_bytes().contains(&byte))
        {
            encoded.push(char::from(byte));
        } else {
            encoded.push_str(&format!("%{byte:02X}"));
        }
    }
    encoded
}
