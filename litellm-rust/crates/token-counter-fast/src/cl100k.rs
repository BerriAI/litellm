//! Scanner for tiktoken's `cl100k_base` split regex,
//! `'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s`.

use super::scanner::{contraction_len, digit_run_len, is_newline};
use super::unicode_classes::{Class, UnicodeClasses, class, run_len};

/// The alternatives in regex order; the possessive quantifiers mean an
/// alternative that starts matching and runs out of input fails as a whole.
pub(super) fn piece_len(text: &str, first: char, unicode_classes: &UnicodeClasses) -> usize {
    if let Some(len) = contraction_len(text) {
        return len;
    }
    let first_class = class(first, unicode_classes);
    match first_class {
        Class::Letter => return run_len(text, Class::Letter, unicode_classes),
        Class::Number => return digit_run_len(text, unicode_classes),
        Class::Space | Class::Other => {}
    }
    let rest = &text[first.len_utf8()..];
    let second_class = rest
        .chars()
        .next()
        .map(|character| class(character, unicode_classes));
    if !is_newline(first) && second_class == Some(Class::Letter) {
        return first.len_utf8() + run_len(rest, Class::Letter, unicode_classes);
    }
    if first_class == Class::Other {
        return symbol_run_len(text, unicode_classes);
    }
    if first == ' ' && second_class == Some(Class::Other) {
        return 1 + symbol_run_len(rest, unicode_classes);
    }
    space_run_len(text, unicode_classes)
}

/// `[^\s\p{L}\p{N}]++[\r\n]*+`
fn symbol_run_len(text: &str, unicode_classes: &UnicodeClasses) -> usize {
    let symbols = run_len(text, Class::Other, unicode_classes);
    symbols
        + text[symbols..]
            .bytes()
            .take_while(|byte| matches!(byte, b'\r' | b'\n'))
            .count()
}

/// `\s++$|\s*[\r\n]|\s+(?!\S)|\s`: whitespace to the end of the text is one
/// piece; otherwise the piece ends at the last newline of the run, or leaves
/// the run's last character for the next piece's optional leading space.
fn space_run_len(text: &str, unicode_classes: &UnicodeClasses) -> usize {
    let run = run_len(text, Class::Space, unicode_classes);
    if run == text.len() {
        return run;
    }
    if let Some(newline) = text[..run].rfind(['\r', '\n']) {
        return newline + 1;
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
    #[case("don't I'LL you'Ve we'RE he'd I'm", &["don", "'t", " I", "'LL", " you", "'Ve", " we", "'RE", " he", "'d", " I", "'m"])]
    #[case("IT'SOK it'Dbe 'Sx 'Tx", &["IT", "'S", "OK", " it", "'D", "be", " '", "Sx", " '", "Tx"])]
    #[case("'Sx'Tx'Mx'LLx'VEx'REx'Dx", &["'S", "x", "'T", "x", "'M", "x", "'LL", "x", "'VE", "x", "'RE", "x", "'D", "x"])]
    #[case("'ſ 'lx", &["'ſ", " '", "lx"])]
    #[case("12345 6", &["123", "45", " ", "6"])]
    #[case("!abc !!abc", &["!abc", " !!", "abc"])]
    #[case(" !!!\r\n\r\nx", &[" !!!\r\n\r\n", "x"])]
    #[case("a  b   \n\n  c", &["a", " ", " b", "   \n\n", " ", " c"])]
    #[case("a\nb\r\nc\n\nd \n e", &["a", "\n", "b", "\r\n", "c", "\n\n", "d", " \n", " e"])]
    #[case("x \t\n \t y\n", &["x", " \t\n", " \t", " y", "\n"])]
    #[case("end   ", &["end", "   "])]
    #[case("\u{a0}abc\u{a0}!", &["\u{a0}abc", "\u{a0}", "!"])]
    #[case("<|endoftext|>", &["<|", "endoftext", "|>"])]
    #[case("e\u{301}a", &["e", "\u{301}a"])]
    #[case("日本語 ١٢٣٤", &["日本語", " ", "١٢٣", "٤"])]
    fn scanner_splits_like_the_regex(#[case] text: &str, #[case] expected: &[&str]) {
        assert_eq!(split(text), expected);
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
            "/tests/fixtures/cl100k/texts.jsonl"
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
