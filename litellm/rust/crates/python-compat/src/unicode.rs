//! Character classes from Python's `str` methods.

/// `c.isspace()` for one character: Unicode `White_Space` plus the four separators
/// `\x1c`..`\x1f`, which Python counts and `char::is_whitespace` does not.
pub fn isspace(c: char) -> bool {
    matches!(
        c,
        '\u{9}'..='\u{d}'
            | '\u{1c}'..='\u{20}'
            | '\u{85}'
            | '\u{a0}'
            | '\u{1680}'
            | '\u{2000}'..='\u{200a}'
            | '\u{2028}'
            | '\u{2029}'
            | '\u{202f}'
            | '\u{205f}'
            | '\u{3000}'
    )
}

/// Whether `text.encode("latin-1")` succeeds: every character is below U+0100.
pub fn is_latin1(text: &str) -> bool {
    text.chars().all(|character| u32::from(character) < 0x100)
}
