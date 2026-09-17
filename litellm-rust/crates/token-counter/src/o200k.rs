//! Scanner for tiktoken's `o200k_base` split regex,
//! `[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+`.
//! tiktoken runs it with a backtracking engine, so the letter alternatives
//! below reproduce where the greedy quantifiers settle, not only what the
//! classes say.

use super::scanner::{contraction_len, digit_run_len, is_newline};
use super::unicode_classes::{Case, Class, UnicodeClasses, case, case_run_len, class, run_len};

/// The alternatives in regex order: a number is never a letter piece, a
/// letter always is, and only whitespace and symbols reach the last three.
pub(super) fn piece_len(text: &str, first: char, unicode_classes: &UnicodeClasses) -> usize {
    let first_class = class(first, unicode_classes);
    if first_class == Class::Number {
        return digit_run_len(text, unicode_classes);
    }
    if let Some(len) = letter_piece_len(text, first, first_class, unicode_classes) {
        return len;
    }
    if first_class == Class::Other {
        return symbol_run_len(text, unicode_classes);
    }
    let rest = &text[first.len_utf8()..];
    if first == ' '
        && rest
            .chars()
            .next()
            .is_some_and(|character| class(character, unicode_classes) == Class::Other)
    {
        return 1 + symbol_run_len(rest, unicode_classes);
    }
    space_run_len(text, unicode_classes)
}

/// The two letter alternatives, each first with then without the optional
/// `[^\r\n\p{L}\p{N}]` prefix: the order the engine tries them in.
fn letter_piece_len(
    text: &str,
    first: char,
    first_class: Class,
    unicode_classes: &UnicodeClasses,
) -> Option<usize> {
    let prefix = (!is_newline(first) && matches!(first_class, Class::Space | Class::Other))
        .then(|| first.len_utf8());
    let after_prefix = |shape: fn(&str, &UnicodeClasses) -> Option<usize>| {
        prefix.and_then(|prefix| shape(&text[prefix..], unicode_classes).map(|len| prefix + len))
    };
    after_prefix(upper_then_lower_len)
        .or_else(|| upper_then_lower_len(text, unicode_classes))
        .or_else(|| after_prefix(upper_run_len))
        .or_else(|| upper_run_len(text, unicode_classes))
}

/// `[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?`.
/// The upper run is greedy; when no lower character follows it, the engine
/// gives characters back until the one it just gave back is lower too, and
/// that single character is the lower run.
fn upper_then_lower_len(text: &str, unicode_classes: &UnicodeClasses) -> Option<usize> {
    let upper = case_run_len(text, Case::is_upper, unicode_classes);
    let lower = case_run_len(&text[upper..], Case::is_lower, unicode_classes);
    let letters = if lower > 0 {
        upper + lower
    } else {
        let (index, last_both) = text[..upper]
            .char_indices()
            .rev()
            .find(|(_, character)| case(*character, unicode_classes).is_lower())?;
        index + last_both.len_utf8()
    };
    Some(letters + contraction_len(&text[letters..]).unwrap_or(0))
}

/// `[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?`
fn upper_run_len(text: &str, unicode_classes: &UnicodeClasses) -> Option<usize> {
    let upper = case_run_len(text, Case::is_upper, unicode_classes);
    if upper == 0 {
        return None;
    }
    let letters = upper + case_run_len(&text[upper..], Case::is_lower, unicode_classes);
    Some(letters + contraction_len(&text[letters..]).unwrap_or(0))
}

/// `[^\s\p{L}\p{N}]+[\r\n/]*`
fn symbol_run_len(text: &str, unicode_classes: &UnicodeClasses) -> usize {
    let symbols = run_len(text, Class::Other, unicode_classes);
    symbols
        + text[symbols..]
            .bytes()
            .take_while(|byte| matches!(byte, b'\r' | b'\n' | b'/'))
            .count()
}

/// `\s*[\r\n]+|\s+(?!\S)|\s+`: a run with a newline ends at its last newline,
/// even at the end of the text; otherwise whitespace to the end of the text
/// is one piece, or the run leaves its last character for the next piece's
/// optional leading space.
fn space_run_len(text: &str, unicode_classes: &UnicodeClasses) -> usize {
    let run = run_len(text, Class::Space, unicode_classes);
    if let Some(newline) = text[..run].rfind(['\r', '\n']) {
        return newline + 1;
    }
    if run == text.len() {
        return run;
    }
    let last = text[..run].chars().next_back().map_or(0, char::len_utf8);
    match run - last {
        0 => run,
        shorter => shorter,
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use tokenizers::utils::SysRegex;

    use super::*;
    use crate::scanner::pieces;

    fn split(text: &str) -> Vec<&str> {
        pieces(
            text,
            piece_len,
            UnicodeClasses::get().expect("Oniguruma exposes Unicode classes"),
        )
        .collect()
    }

    #[rstest]
    #[case("", &[])]
    #[case("Hello world", &["Hello", " world"])]
    #[case("camelCase PascalCase ABCdef ABCdeF ABC", &["camel", "Case", " Pascal", "Case", " ABCdef", " ABCde", "F", " ABC"])]
    #[case("日本ABC ABC日本 日本語abc abc日本語", &["日本", "ABC", " ABC日本", " 日本語abc", " abc日本語"])]
    #[case("\u{301}ABC \u{301}abc \u{301}\u{301}A A\u{301}\u{301} E\u{301}A aE\u{301}", &["\u{301}", "ABC", " \u{301}abc", " \u{301}\u{301}", "A", " A\u{301}\u{301}", " E\u{301}", "A", " a", "E\u{301}"])]
    #[case("ᵃbc ᵃBC Aᵃbc Aᵃ ᵃ' ᵃ's", &["ᵃbc", " ᵃ", "BC", " Aᵃbc", " Aᵃ", " ᵃ", "'", " ᵃ's"])]
    #[case("ǅungla aǅB AǅB Aǅb", &["ǅungla", " a", "ǅB", " AǅB", " Aǅb"])]
    #[case("don'tx ABC's abc'S abc'ſ ABC'ſx IT'SOK it'Dbe", &["don't", "x", " ABC's", " abc'S", " abc'ſ", " ABC'ſ", "x", " IT'S", "OK", " it'D", "be"])]
    #[case("'sabc x's 's 'Sx'Tx 9'9 a'9 ' s", &["'sabc", " x's", " '", "s", " '", "Sx'T", "x", " ", "9", "'", "9", " a", "'", "9", " '", " s"])]
    #[case("!ABC !AbC !!abc !!\u{301}a \u{a0}\u{301}A", &["!ABC", " !", "Ab", "C", " !!", "abc", " !!\u{301}", "a", " ", "\u{a0}\u{301}", "A"])]
    #[case("!!/\n/x a/b !!\n/x  /x  //", &["!!/\n/", "x", " a", "/b", " !!\n/", "x", " ", " /", "x", " ", " //"])]
    #[case("12345 6 1abc abc1", &["123", "45", " ", "6", " ", "1", "abc", " abc", "1"])]
    #[case("x \n x \r\n \r\n y", &["x", " \n", " x", " \r\n \r\n", " y"])]
    #[case("x \n ", &["x", " \n", " "])]
    #[case("a  b   \n\n  c", &["a", " ", " b", "   \n\n", " ", " c"])]
    #[case("x\t\ty x\t\t", &["x", "\t", "\ty", " x", "\t\t"])]
    #[case("end   ", &["end", "   "])]
    #[case("\u{a0}abc\u{a0}!", &["\u{a0}abc", "\u{a0}", "!"])]
    #[case("<|endoftext|>", &["<|", "endoftext", "|>"])]
    #[case("İstanbul ΣΊΣΥΦΟΣ Ελληνικά Русский", &["İstanbul", " ΣΊΣΥΦΟΣ", " Ελληνικά", " Русский"])]
    #[case("日本語 ١٢٣٤", &["日本語", " ", "١٢٣", "٤"])]
    fn scanner_splits_like_the_regex(#[case] text: &str, #[case] expected: &[&str]) {
        assert_eq!(split(text), expected);
    }

    #[test]
    fn every_scalar_alone_is_one_piece() {
        for character in (0..=0x10FFFFu32).filter_map(char::from_u32) {
            let text = character.to_string();
            assert_eq!(
                split(&text),
                [text.as_str()],
                "U+{:04X}",
                u32::from(character)
            );
        }
    }

    #[test]
    fn cases_match_oniguruma() {
        let unicode_classes = UnicodeClasses::get().expect("Oniguruma exposes Unicode classes");
        let upper = SysRegex::new(r"[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]").expect("regex");
        let lower = SysRegex::new(r"[\p{Ll}\p{Lm}\p{Lo}\p{M}]").expect("regex");
        let whole =
            |regex: &SysRegex, text: &str| regex.find_iter(text).next() == Some((0, text.len()));
        let mut text = String::new();
        for character in (0..=0x10FFFFu32).filter_map(char::from_u32) {
            text.clear();
            text.push(character);
            let expected = match (whole(&upper, &text), whole(&lower, &text)) {
                (true, true) => Case::Both,
                (true, false) => Case::Upper,
                (false, true) => Case::Lower,
                (false, false) => Case::Neither,
            };
            assert_eq!(
                case(character, unicode_classes),
                expected,
                "U+{:04X}",
                u32::from(character)
            );
        }
    }

    #[derive(serde::Deserialize)]
    struct TextFixture {
        text: String,
        pieces: Vec<String>,
    }

    #[test]
    fn scanner_splits_the_fixture_corpus_like_tiktoken_regex() {
        let fixtures: Vec<TextFixture> = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/o200k/texts.jsonl"
        ))
        .lines()
        .map(|line| serde_json::from_str(line).expect("fixture line is json"))
        .collect();
        assert!(fixtures.len() > 3000);
        let mismatches: Vec<_> = fixtures
            .iter()
            .filter(|fixture| split(&fixture.text) != fixture.pieces)
            .map(|fixture| (&fixture.text, split(&fixture.text), &fixture.pieces))
            .collect();
        assert!(mismatches.is_empty(), "{mismatches:#?}");
    }
}
