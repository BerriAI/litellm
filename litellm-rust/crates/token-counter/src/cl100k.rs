//! Exact token counting for tiktoken's `cl100k_base`. A scanner reproduces the
//! piece boundaries of the encoding's split regex,
//! `'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s`,
//! and each piece is merged with the rank file. Special tokens are ordinary
//! text, as with `encode(text, disallowed_special=())`.

use std::iter;

use super::tiktoken::{MergeRanks, MergeScratch};
use super::unicode_classes::{Class, UnicodeClasses, class, run_len};
use crate::Error;

const MAX_DIGITS_PER_PIECE: usize = 3;

pub(super) struct Cl100kCounter {
    ranks: MergeRanks,
    unicode_classes: &'static UnicodeClasses,
}

impl Cl100kCounter {
    pub(super) fn from_ranks(rank_file: &str) -> Result<Self, Error> {
        Ok(Self {
            ranks: MergeRanks::parse(rank_file)?,
            unicode_classes: UnicodeClasses::get().ok_or(Error::UnicodeClasses)?,
        })
    }

    pub(super) fn count(&self, text: &str) -> usize {
        let mut scratch = MergeScratch::default();
        pieces(text, self.unicode_classes)
            .map(|piece| self.ranks.count_piece(piece.as_bytes(), &mut scratch))
            .sum()
    }
}

/// The regex matches every character, so the pieces tile the text.
fn pieces<'a>(
    text: &'a str,
    unicode_classes: &'static UnicodeClasses,
) -> impl Iterator<Item = &'a str> {
    iter::successors(split_piece(text, unicode_classes), move |(_, rest)| {
        split_piece(rest, unicode_classes)
    })
    .map(|(piece, _)| piece)
}

fn split_piece<'a>(text: &'a str, unicode_classes: &UnicodeClasses) -> Option<(&'a str, &'a str)> {
    let first = text.chars().next()?;
    Some(text.split_at(piece_len(text, first, unicode_classes)))
}

/// The alternatives in regex order; the possessive quantifiers mean an
/// alternative that starts matching and runs out of input fails as a whole.
fn piece_len(text: &str, first: char, unicode_classes: &UnicodeClasses) -> usize {
    if first == '\''
        && let Some(len) = contraction_len(&text[1..])
    {
        return 1 + len;
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

/// `(?i:[sdmt]|ll|ve|re)` after the apostrophe. Simple case folding also maps
/// U+017F (long s) onto `s`.
fn contraction_len(rest: &str) -> Option<usize> {
    let mut characters = rest.chars();
    let first = characters.next()?;
    match first {
        's' | 'S' | '\u{17F}' | 'd' | 'D' | 'm' | 'M' | 't' | 'T' => Some(first.len_utf8()),
        'l' | 'L' => matches!(characters.next(), Some('l' | 'L')).then_some(2),
        'v' | 'V' | 'r' | 'R' => matches!(characters.next(), Some('e' | 'E')).then_some(2),
        _ => None,
    }
}

fn is_newline(character: char) -> bool {
    matches!(character, '\r' | '\n')
}

/// `\p{N}{1,3}+`
fn digit_run_len(text: &str, unicode_classes: &UnicodeClasses) -> usize {
    text.chars()
        .take(MAX_DIGITS_PER_PIECE)
        .take_while(|character| class(*character, unicode_classes) == Class::Number)
        .map(char::len_utf8)
        .sum()
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

    fn split(text: &str) -> Vec<&str> {
        pieces(
            text,
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
