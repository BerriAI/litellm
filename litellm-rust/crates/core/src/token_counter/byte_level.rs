//! Token counting for tokenizers shaped like Anthropic's (optional NFKC
//! normalizer, `ByteLevel` pre-tokenizer with the GPT-2 split regex, no
//! post-processor) without running the regex. Oniguruma spends ~90% of
//! `encode_fast` on `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`;
//! a hand-written scanner finds the same pieces and hands them to the model.

use std::borrow::Cow;
use std::cmp::Ordering;

use tokenizers::normalizers::NormalizerWrapper;
use tokenizers::pre_tokenizers::PreTokenizerWrapper;
use tokenizers::{Model, Tokenizer};
use unicode_normalization::{IsNormalized, UnicodeNormalization, is_nfkc_quick};

use super::unicode_classes::{LETTER_RANGES, NUMBER_RANGES, SPACE_RANGES};

const CONTRACTIONS: [&str; 7] = ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d"];

pub(super) struct ByteLevelCounter {
    nfkc: bool,
}

impl ByteLevelCounter {
    pub(super) fn detect(tokenizer: &Tokenizer) -> Option<Self> {
        let nfkc = match tokenizer.get_normalizer() {
            None => false,
            Some(NormalizerWrapper::NFKC(_)) => true,
            Some(_) => return None,
        };
        let Some(PreTokenizerWrapper::ByteLevel(byte_level)) = tokenizer.get_pre_tokenizer() else {
            return None;
        };
        let plain = !byte_level.add_prefix_space
            && byte_level.use_regex
            && tokenizer.get_post_processor().is_none()
            && tokenizer.get_truncation().is_none()
            && tokenizer.get_padding().is_none();
        plain.then_some(Self { nfkc })
    }

    /// `None` when the text contains an added token or the model rejects a
    /// piece; the caller then runs the full encoder.
    pub(super) fn count(&self, tokenizer: &Tokenizer, text: &str) -> Option<usize> {
        let normalized = self.normalize(text);
        let added_tokens = tokenizer.get_added_vocabulary().get_vocab();
        if added_tokens
            .keys()
            .any(|token| text.contains(token.as_str()) || normalized.contains(token.as_str()))
        {
            return None;
        }
        let model = tokenizer.get_model();
        let mut mapped = String::new();
        let mut total = 0;
        let pieces = Pieces {
            rest: normalized.as_ref(),
        };
        for piece in pieces {
            mapped.clear();
            mapped.extend(piece.bytes().map(|byte| BYTE_CHARS[usize::from(byte)]));
            total += model.tokenize(&mapped).ok()?.len();
        }
        Some(total)
    }

    fn normalize<'a>(&self, text: &'a str) -> Cow<'a, str> {
        if !self.nfkc || text.is_ascii() || is_nfkc_quick(text.chars()) == IsNormalized::Yes {
            return Cow::Borrowed(text);
        }
        Cow::Owned(text.nfkc().collect())
    }
}

/// GPT-2 `bytes_to_unicode`: printable Latin-1 bytes map to themselves, the
/// rest to U+0100 onwards in byte order.
const BYTE_CHARS: [char; 256] = byte_chars();

const fn byte_chars() -> [char; 256] {
    let mut table = ['\0'; 256];
    let mut next_gap = 0x100u32;
    let mut byte = 0usize;
    while byte < 256 {
        let printable = (byte >= 0x21 && byte <= 0x7E)
            || (byte >= 0xA1 && byte <= 0xAC)
            || (byte >= 0xAE && byte <= 0xFF);
        let code = if printable {
            byte as u32
        } else {
            next_gap += 1;
            next_gap - 1
        };
        table[byte] = match char::from_u32(code) {
            Some(mapped) => mapped,
            None => unreachable!(),
        };
        byte += 1;
    }
    table
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Class {
    Letter,
    Number,
    Space,
    Other,
}

const ASCII_CLASSES: [Class; 128] = ascii_classes();

const fn ascii_classes() -> [Class; 128] {
    let mut table = [Class::Other; 128];
    let mut code = 0usize;
    while code < 128 {
        table[code] = match code as u8 {
            b'A'..=b'Z' | b'a'..=b'z' => Class::Letter,
            b'0'..=b'9' => Class::Number,
            b'\t'..=b'\r' | b' ' => Class::Space,
            _ => Class::Other,
        };
        code += 1;
    }
    table
}

fn class(character: char) -> Class {
    if character.is_ascii() {
        ASCII_CLASSES[character as usize]
    } else if in_ranges(LETTER_RANGES, character) {
        Class::Letter
    } else if in_ranges(NUMBER_RANGES, character) {
        Class::Number
    } else if in_ranges(SPACE_RANGES, character) {
        Class::Space
    } else {
        Class::Other
    }
}

fn in_ranges(ranges: &[(u32, u32)], character: char) -> bool {
    let code = u32::from(character);
    ranges
        .binary_search_by(|(low, high)| {
            if *high < code {
                Ordering::Less
            } else if *low > code {
                Ordering::Greater
            } else {
                Ordering::Equal
            }
        })
        .is_ok()
}

/// The regex matches every character, so the pieces tile the text.
struct Pieces<'a> {
    rest: &'a str,
}

impl<'a> Iterator for Pieces<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<&'a str> {
        let first = self.rest.chars().next()?;
        let (piece, rest) = self.rest.split_at(piece_len(self.rest, first));
        self.rest = rest;
        Some(piece)
    }
}

fn piece_len(text: &str, first: char) -> usize {
    if let Some(contraction) = CONTRACTIONS.iter().find(|word| text.starts_with(**word)) {
        return contraction.len();
    }
    let first_class = class(first);
    if first_class != Class::Space {
        return run_len(text, first_class);
    }
    if first != ' ' {
        return space_run_len(text);
    }
    let after_space = &text[1..];
    match after_space.chars().next().map(class) {
        None | Some(Class::Space) => space_run_len(text),
        Some(run_class) => 1 + run_len(after_space, run_class),
    }
}

fn run_len(text: &str, run_class: Class) -> usize {
    text.char_indices()
        .find(|(_, character)| class(*character) != run_class)
        .map_or(text.len(), |(index, _)| index)
}

/// `\s+(?!\S)|\s+`: whitespace followed by a non-space leaves its last
/// character to start the next piece (` ?` on the following alternatives).
fn space_run_len(text: &str) -> usize {
    let run = run_len(text, Class::Space);
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
    use rand::rngs::StdRng;
    use rand::seq::SliceRandom;
    use rand::{Rng, SeedableRng};
    use tokenizers::pre_tokenizers::byte_level::ByteLevel;
    use tokenizers::utils::SysRegex;
    use tokenizers::{OffsetReferential, OffsetType, PreTokenizedString, PreTokenizer};

    use super::*;

    fn anthropic() -> Tokenizer {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../litellm/litellm_core_utils/tokenizers/anthropic_tokenizer.json"
        );
        std::fs::read_to_string(path)
            .expect("anthropic tokenizer json is in the repo")
            .parse()
            .expect("anthropic tokenizer loads")
    }

    fn reference_count(tokenizer: &Tokenizer, text: &str) -> usize {
        tokenizer.encode_fast(text, true).expect("encode").len()
    }

    const ALPHABET: &[&str] = &[
        "a",
        "Z",
        "e",
        "s",
        "t",
        "d",
        "m",
        "'",
        "'s",
        "'re",
        "'ll",
        "'S",
        "0",
        "9",
        " ",
        "  ",
        "\t",
        "\n",
        "\r\n",
        "\u{b}",
        ".",
        ",",
        "!",
        "-",
        "(",
        "\"",
        "\u{a0}",
        "\u{85}",
        "\u{2028}",
        "\u{3000}",
        "\u{200b}",
        "\u{200d}",
        "é",
        "e\u{301}",
        "ß",
        "漢",
        "字",
        "ع",
        "३",
        "½",
        "Ⅳ",
        "🙂",
        "👍🏽",
        "Ａ",
        "ﬁ",
        "㍿",
        "<",
        ">",
        "EOT",
        "<EOT>",
        "<META_START>",
    ];

    fn random_text(rng: &mut StdRng) -> String {
        let pieces = rng.gen_range(0..40);
        (0..pieces)
            .map(|_| *ALPHABET.choose(rng).expect("alphabet is not empty"))
            .collect()
    }

    #[test]
    fn anthropic_tokenizer_takes_the_fast_path() {
        let tokenizer = anthropic();
        let fast = ByteLevelCounter::detect(&tokenizer).expect("anthropic shape is supported");
        assert!(fast.nfkc);
        let text = "Hello, how are you today?";
        assert_eq!(
            fast.count(&tokenizer, text),
            Some(reference_count(&tokenizer, text))
        );
        assert_eq!(fast.count(&tokenizer, "stop <EOT> here"), None);
        assert_eq!(fast.count(&tokenizer, "stop ＜ＥＯＴ＞ here"), None);
    }

    #[test]
    fn counts_match_the_full_encoder() {
        let tokenizer = anthropic();
        let fast = ByteLevelCounter::detect(&tokenizer).expect("supported");
        let mut rng = StdRng::seed_from_u64(2026);
        for _ in 0..4000 {
            let text = random_text(&mut rng);
            let expected = reference_count(&tokenizer, &text);
            let counted = fast
                .count(&tokenizer, &text)
                .unwrap_or_else(|| reference_count(&tokenizer, &text));
            assert_eq!(counted, expected, "text {text:?}");
        }
    }

    #[test]
    fn pieces_match_the_byte_level_pre_tokenizer() {
        let byte_level = ByteLevel::new(false, true, true);
        let mut rng = StdRng::seed_from_u64(7);
        for _ in 0..4000 {
            let text: String = random_text(&mut rng).nfkc().collect();
            let mut pre_tokenized = PreTokenizedString::from(text.as_str());
            byte_level
                .pre_tokenize(&mut pre_tokenized)
                .expect("pre-tokenize");
            let expected: Vec<(String, (usize, usize))> = pre_tokenized
                .get_splits(OffsetReferential::Original, OffsetType::Byte)
                .into_iter()
                .map(|(mapped, offsets, _)| (mapped.to_string(), offsets))
                .collect();
            let mut offset = 0;
            let pieces = Pieces { rest: &text };
            let actual: Vec<(String, (usize, usize))> = pieces
                .map(|piece| {
                    let mapped: String = piece
                        .bytes()
                        .map(|byte| BYTE_CHARS[usize::from(byte)])
                        .collect();
                    let span = (offset, offset + piece.len());
                    offset += piece.len();
                    (mapped, span)
                })
                .collect();
            assert_eq!(actual, expected, "text {text:?}");
        }
    }

    #[test]
    fn classes_match_oniguruma() {
        let letter = SysRegex::new(r"\p{L}").expect("regex");
        let number = SysRegex::new(r"\p{N}").expect("regex");
        let space = SysRegex::new(r"\s").expect("regex");
        let whole =
            |regex: &SysRegex, text: &str| regex.find_iter(text).next() == Some((0, text.len()));
        let mut text = String::new();
        for character in (0..=0x10FFFFu32).filter_map(char::from_u32) {
            text.clear();
            text.push(character);
            let expected = if whole(&letter, &text) {
                Class::Letter
            } else if whole(&number, &text) {
                Class::Number
            } else if whole(&space, &text) {
                Class::Space
            } else {
                Class::Other
            };
            assert_eq!(class(character), expected, "U+{:04X}", u32::from(character));
        }
    }

    #[test]
    fn other_tokenizer_shapes_are_declined() {
        let mut tokenizer = anthropic();
        tokenizer.with_pre_tokenizer(Some(ByteLevel::new(true, true, true)));
        assert!(ByteLevelCounter::detect(&tokenizer).is_none());
    }
}
