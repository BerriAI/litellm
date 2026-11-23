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

/// `unicodedata.category(c)`: the two-letter Unicode general category of a character, from the
/// Unicode tables this crate is built with, which may be newer or older than the host Python's.
pub fn category(c: char) -> &'static str {
    use unicode_general_category::{GeneralCategory, get_general_category};
    match get_general_category(c) {
        GeneralCategory::UppercaseLetter => "Lu",
        GeneralCategory::LowercaseLetter => "Ll",
        GeneralCategory::TitlecaseLetter => "Lt",
        GeneralCategory::ModifierLetter => "Lm",
        GeneralCategory::OtherLetter => "Lo",
        GeneralCategory::NonspacingMark => "Mn",
        GeneralCategory::SpacingMark => "Mc",
        GeneralCategory::EnclosingMark => "Me",
        GeneralCategory::DecimalNumber => "Nd",
        GeneralCategory::LetterNumber => "Nl",
        GeneralCategory::OtherNumber => "No",
        GeneralCategory::ConnectorPunctuation => "Pc",
        GeneralCategory::DashPunctuation => "Pd",
        GeneralCategory::OpenPunctuation => "Ps",
        GeneralCategory::ClosePunctuation => "Pe",
        GeneralCategory::InitialPunctuation => "Pi",
        GeneralCategory::FinalPunctuation => "Pf",
        GeneralCategory::OtherPunctuation => "Po",
        GeneralCategory::MathSymbol => "Sm",
        GeneralCategory::CurrencySymbol => "Sc",
        GeneralCategory::ModifierSymbol => "Sk",
        GeneralCategory::OtherSymbol => "So",
        GeneralCategory::SpaceSeparator => "Zs",
        GeneralCategory::LineSeparator => "Zl",
        GeneralCategory::ParagraphSeparator => "Zp",
        GeneralCategory::Control => "Cc",
        GeneralCategory::Format => "Cf",
        GeneralCategory::Surrogate => "Cs",
        GeneralCategory::PrivateUse => "Co",
        _ => "Cn",
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::category;

    #[rstest]
    #[case::lowercase('a', "Ll")]
    #[case::uppercase('A', "Lu")]
    #[case::other_letter('東', "Lo")]
    #[case::digit('7', "Nd")]
    #[case::roman_numeral('Ⅻ', "Nl")]
    #[case::superscript('²', "No")]
    #[case::underscore('_', "Pc")]
    #[case::space(' ', "Zs")]
    fn category_names_the_general_category(#[case] c: char, #[case] expected: &str) {
        assert_eq!(category(c), expected);
    }
}
