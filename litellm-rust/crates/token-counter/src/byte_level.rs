//! Exact token counting for a supported tokenizer configuration: optional
//! NFKC normalization, `ByteLevel` pre-tokenization with the GPT-2 split regex,
//! and no post-processing. A scanner reproduces the regex's piece boundaries
//! and hands each piece to the tokenizer's model. Unsupported configurations
//! and added-token inputs fall back to the full encoder.

use std::borrow::Cow;
use std::iter;

use tokenizers::normalizers::NormalizerWrapper;
use tokenizers::pre_tokenizers::PreTokenizerWrapper;
use tokenizers::{Model, Tokenizer};
use unicode_normalization_alignments::{IsNormalized, UnicodeNormalization, is_nfkc_quick};

use super::unicode_classes::{Class, UnicodeClasses, class, run_len};

const CONTRACTIONS: [&str; 7] = ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d"];

pub(super) struct ByteLevelCounter {
    nfkc: bool,
    normalized_added_tokens: Vec<String>,
    unicode_classes: &'static UnicodeClasses,
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
        if !plain {
            return None;
        }
        let vocabulary = tokenizer.get_added_vocabulary();
        let normalized_added_tokens = vocabulary
            .get_vocab()
            .iter()
            .filter_map(|(original, id)| {
                vocabulary
                    .simple_id_to_token(*id)
                    .filter(|normalized| normalized != original)
            })
            .collect();
        Some(Self {
            nfkc,
            normalized_added_tokens,
            unicode_classes: UnicodeClasses::get()?,
        })
    }

    /// `None` when the text contains an added token or the model rejects a
    /// piece; the caller then runs the full encoder.
    pub(super) fn count(&self, tokenizer: &Tokenizer, text: &str) -> Option<usize> {
        let normalized = self.normalize(text);
        let added_tokens = tokenizer.get_added_vocabulary().get_vocab();
        if added_tokens
            .keys()
            .chain(self.normalized_added_tokens.iter())
            .any(|token| text.contains(token.as_str()) || normalized.contains(token.as_str()))
        {
            return None;
        }
        let model = tokenizer.get_model();
        let mapped: String = normalized.bytes().map(byte_char).collect();
        pieces(&normalized, self.unicode_classes)
            .try_fold((0, 0), |(start, total), piece| {
                let end = start + mapped_len(piece);
                let tokens = model.tokenize(&mapped[start..end]).ok()?;
                Some((end, total + tokens.len()))
            })
            .map(|(_, total)| total)
    }

    /// Same crate and Unicode tables as `NormalizedString::nfkc`, so the
    /// result is what the full encoder would have tokenized.
    fn normalize<'a>(&self, text: &'a str) -> Cow<'a, str> {
        if !self.nfkc || text.is_ascii() || is_nfkc_quick(text.chars()) == IsNormalized::Yes {
            return Cow::Borrowed(text);
        }
        Cow::Owned(text.nfkc().map(|(character, _)| character).collect())
    }
}

/// GPT-2 `bytes_to_unicode`: printable Latin-1 bytes map to themselves, the
/// rest to U+0100 onwards in byte order.
fn byte_char(byte: u8) -> char {
    let code = match byte {
        0x21..=0x7E | 0xA1..=0xAC | 0xAE..=0xFF => u32::from(byte),
        0x00..=0x20 => 0x100 + u32::from(byte),
        0x7F..=0xA0 => 0x121 + u32::from(byte - 0x7F),
        0xAD => 0x143,
    };
    char::from_u32(code).unwrap_or(char::REPLACEMENT_CHARACTER)
}

fn mapped_len(piece: &str) -> usize {
    piece.len()
        + piece
            .bytes()
            .filter(|byte| !byte.is_ascii_graphic())
            .count()
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

fn piece_len(text: &str, first: char, unicode_classes: &UnicodeClasses) -> usize {
    if let Some(contraction) = CONTRACTIONS.iter().find(|word| text.starts_with(**word)) {
        return contraction.len();
    }
    let first_class = class(first, unicode_classes);
    if first_class != Class::Space {
        return run_len(text, first_class, unicode_classes);
    }
    if first != ' ' {
        return space_run_len(text, unicode_classes);
    }
    let after_space = &text[1..];
    match after_space
        .chars()
        .next()
        .map(|character| class(character, unicode_classes))
    {
        None | Some(Class::Space) => space_run_len(text, unicode_classes),
        Some(run_class) => 1 + run_len(after_space, run_class, unicode_classes),
    }
}

/// `\s+(?!\S)|\s+`: whitespace followed by a non-space leaves its last
/// character to start the next piece (` ?` on the following alternatives).
fn space_run_len(text: &str, unicode_classes: &UnicodeClasses) -> usize {
    let run = run_len(text, Class::Space, unicode_classes);
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
    use rstest::{fixture, rstest};
    use tokenizers::normalizers::NFKC;
    use tokenizers::pre_tokenizers::byte_level::ByteLevel;
    use tokenizers::utils::SysRegex;
    use tokenizers::{
        NormalizedString, Normalizer, OffsetReferential, OffsetType, PreTokenizedString,
        PreTokenizer,
    };

    use super::*;

    #[fixture]
    fn anthropic_tokenizer() -> Tokenizer {
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

    fn byte_level_counter(nfkc: bool) -> ByteLevelCounter {
        ByteLevelCounter {
            nfkc,
            normalized_added_tokens: Vec::new(),
            unicode_classes: UnicodeClasses::get().expect("Oniguruma exposes Unicode classes"),
        }
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
        "㋿",
        "ꟲ",
        "𐞁",
        "a\u{30a}",
        "\u{1e0b}\u{323}",
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

    #[rstest]
    #[case::plain_text("Hello, how are you today?", true)]
    #[case::added_token("stop <EOT> here", false)]
    #[case::normalized_added_token("stop ＜ＥＯＴ＞ here", false)]
    fn anthropic_tokenizer_takes_the_fast_path(
        anthropic_tokenizer: Tokenizer,
        #[case] text: &str,
        #[case] supported: bool,
    ) {
        let fast =
            ByteLevelCounter::detect(&anthropic_tokenizer).expect("anthropic shape is supported");
        assert!(fast.nfkc);
        let count = fast.count(&anthropic_tokenizer, text);
        if supported {
            assert_eq!(count, Some(reference_count(&anthropic_tokenizer, text)));
        } else {
            assert_eq!(count, None);
        }
    }

    #[rstest]
    fn counts_match_the_full_encoder(anthropic_tokenizer: Tokenizer) {
        let fast = ByteLevelCounter::detect(&anthropic_tokenizer).expect("supported");
        let mut rng = StdRng::seed_from_u64(2026);
        for _ in 0..4000 {
            let text = random_text(&mut rng).replace('<', "(");
            let expected = reference_count(&anthropic_tokenizer, &text);
            assert_eq!(
                fast.count(&anthropic_tokenizer, &text),
                Some(expected),
                "text {text:?}"
            );
        }
    }

    #[rstest]
    fn nfkc_matches_the_tokenizer_normalizer_for_every_scalar_value() {
        let fast = byte_level_counter(true);
        let mut text = String::new();
        for character in (0..=0x10FFFFu32).filter_map(char::from_u32) {
            text.clear();
            text.push(character);
            let mut expected = NormalizedString::from(text.as_str());
            NFKC.normalize(&mut expected).expect("nfkc");
            assert_eq!(
                fast.normalize(&text),
                expected.get(),
                "U+{:04X}",
                u32::from(character)
            );
        }
    }

    #[rstest]
    fn nfkc_matches_the_tokenizer_normalizer_on_random_texts() {
        let fast = byte_level_counter(true);
        let mut rng = StdRng::seed_from_u64(11);
        for _ in 0..4000 {
            let text = random_text(&mut rng);
            let mut expected = NormalizedString::from(text.as_str());
            NFKC.normalize(&mut expected).expect("nfkc");
            assert_eq!(fast.normalize(&text), expected.get(), "text {text:?}");
        }
    }

    #[rstest]
    fn pieces_match_the_byte_level_pre_tokenizer() {
        let byte_level = ByteLevel::new(false, true, true);
        let mut rng = StdRng::seed_from_u64(7);
        for _ in 0..4000 {
            let text = byte_level_counter(true)
                .normalize(&random_text(&mut rng))
                .into_owned();
            let mut pre_tokenized = PreTokenizedString::from(text.as_str());
            byte_level
                .pre_tokenize(&mut pre_tokenized)
                .expect("pre-tokenize");
            let expected: Vec<(String, (usize, usize))> = pre_tokenized
                .get_splits(OffsetReferential::Original, OffsetType::Byte)
                .into_iter()
                .map(|(mapped, offsets, _)| (mapped.to_string(), offsets))
                .collect();
            let actual: Vec<(String, (usize, usize))> = pieces(
                &text,
                UnicodeClasses::get().expect("Oniguruma exposes Unicode classes"),
            )
            .map(|piece| {
                let start = piece.as_ptr() as usize - text.as_ptr() as usize;
                let mapped: String = piece.bytes().map(byte_char).collect();
                (mapped, (start, start + piece.len()))
            })
            .collect();
            assert_eq!(actual, expected, "text {text:?}");
        }
    }

    #[rstest]
    fn byte_chars_match_the_byte_level_alphabet() {
        let byte_level = ByteLevel::new(false, false, false);
        let characters: Vec<char> = (0..=0x10FFFFu32).filter_map(char::from_u32).collect();
        for chunk in characters.chunks(1024) {
            let text: String = chunk.iter().collect();
            let mut pre_tokenized = PreTokenizedString::from(text.as_str());
            byte_level
                .pre_tokenize(&mut pre_tokenized)
                .expect("pre-tokenize");
            let expected: String = pre_tokenized
                .get_splits(OffsetReferential::Original, OffsetType::Byte)
                .into_iter()
                .map(|(mapped, _, _)| mapped)
                .collect();
            let actual: String = text.bytes().map(byte_char).collect();
            assert_eq!(actual.len(), mapped_len(&text));
            assert_eq!(
                actual,
                expected,
                "chunk starting at U+{:04X}",
                u32::from(chunk[0])
            );
        }
    }

    #[rstest]
    fn classes_match_oniguruma() {
        let unicode_classes = UnicodeClasses::get().expect("Oniguruma exposes Unicode classes");
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
            assert_eq!(
                class(character, unicode_classes),
                expected,
                "U+{:04X}",
                u32::from(character)
            );
        }
    }

    #[rstest]
    #[case("prefix")]
    #[case("regex")]
    #[case("normalizer")]
    #[case("no_pre_tokenizer")]
    #[case("other_pre_tokenizer")]
    #[case("post_processor")]
    #[case("truncation")]
    #[case("padding")]
    fn other_tokenizer_shapes_are_declined(
        mut anthropic_tokenizer: Tokenizer,
        #[case] shape: &str,
    ) {
        use tokenizers::{PaddingParams, PaddingStrategy, TruncationParams};
        match shape {
            "prefix" => {
                anthropic_tokenizer.with_pre_tokenizer(Some(ByteLevel::new(true, true, true)));
            }
            "regex" => {
                anthropic_tokenizer.with_pre_tokenizer(Some(ByteLevel::new(false, true, false)));
            }
            "normalizer" => {
                anthropic_tokenizer
                    .with_normalizer(Some(tokenizers::normalizers::Lowercase))
                    .expect("normalizer");
            }
            "no_pre_tokenizer" => {
                anthropic_tokenizer.with_pre_tokenizer(None::<PreTokenizerWrapper>);
            }
            "other_pre_tokenizer" => {
                anthropic_tokenizer
                    .with_pre_tokenizer(Some(tokenizers::pre_tokenizers::whitespace::Whitespace));
            }
            "post_processor" => {
                anthropic_tokenizer.with_post_processor(Some(ByteLevel::default()));
            }
            "truncation" => {
                anthropic_tokenizer
                    .with_truncation(Some(TruncationParams {
                        max_length: 2,
                        ..Default::default()
                    }))
                    .expect("truncation");
            }
            "padding" => {
                anthropic_tokenizer.with_padding(Some(PaddingParams {
                    strategy: PaddingStrategy::Fixed(32),
                    ..Default::default()
                }));
            }
            _ => unreachable!(),
        }
        assert!(ByteLevelCounter::detect(&anthropic_tokenizer).is_none());
        let counter = crate::TokenCounter::from_json(
            &anthropic_tokenizer.to_string(false).expect("serialize"),
        )
        .expect("load");
        for text in ["", "Hello WORLD!  ＡＢ ﬁ Ⅳ", "<EOT> stop"] {
            assert_eq!(
                counter.count_text(text).expect("count"),
                reference_count(&anthropic_tokenizer, text)
            );
        }
    }

    #[rstest]
    #[case(false)]
    #[case(true)]
    fn arbitrary_unicode_and_long_inputs_use_fast_path(
        mut anthropic_tokenizer: Tokenizer,
        #[case] nfkc: bool,
    ) {
        if !nfkc {
            anthropic_tokenizer
                .with_normalizer(None::<NormalizerWrapper>)
                .expect("normalizer");
        }
        let fast = ByteLevelCounter::detect(&anthropic_tokenizer).expect("supported");
        let mut rng = StdRng::seed_from_u64(314159);
        for _ in 0..1000 {
            let text: String = (0..64)
                .filter_map(|_| char::from_u32(rng.gen_range(0..=0x10ffff)))
                .collect();
            assert_eq!(
                fast.count(&anthropic_tokenizer, &text),
                Some(reference_count(&anthropic_tokenizer, &text)),
                "text {text:?}"
            );
        }
        for text in [
            "",
            "'s't're've'm'll'd'S'RE",
            "  a \t\r\n b\u{85}\u{a0}c  ",
            "\0é漢🙂",
            "a\u{30a}\u{301}",
            "ＡﬁⅣ",
        ] {
            let text = text.repeat(2048);
            assert_eq!(
                fast.count(&anthropic_tokenizer, &text),
                Some(reference_count(&anthropic_tokenizer, &text))
            );
        }
    }

    #[rstest]
    #[case(false, false, false, false)]
    #[case(true, false, false, false)]
    #[case(false, true, false, false)]
    #[case(false, false, true, false)]
    #[case(false, false, false, true)]
    fn added_token_options_fall_back(
        mut anthropic_tokenizer: Tokenizer,
        #[case] special: bool,
        #[case] single_word: bool,
        #[case] lstrip: bool,
        #[case] rstrip: bool,
    ) {
        anthropic_tokenizer
            .add_tokens([tokenizers::AddedToken::from("custom token", special)
                .single_word(single_word)
                .lstrip(lstrip)
                .rstrip(rstrip)])
            .expect("add token");
        let fast = ByteLevelCounter::detect(&anthropic_tokenizer).expect("supported");
        let counter = crate::TokenCounter::from_json(
            &anthropic_tokenizer.to_string(false).expect("serialize"),
        )
        .expect("load");
        for text in [
            "custom token",
            "a custom token b",
            "acustom tokenb",
            "  custom token  ",
        ] {
            assert_eq!(fast.count(&anthropic_tokenizer, text), None);
            assert_eq!(
                counter.count_text(text).expect("count"),
                reference_count(&anthropic_tokenizer, text)
            );
        }
    }

    #[test]
    fn model_errors_reach_public_caller() {
        let mut tokenizer = Tokenizer::new(tokenizers::models::wordpiece::WordPiece::default());
        tokenizer.with_pre_tokenizer(Some(ByteLevel::new(false, true, true)));
        let fast = ByteLevelCounter::detect(&tokenizer).expect("supported");
        assert_eq!(fast.count(&tokenizer, "hello"), None);
        assert!(tokenizer.encode_fast("hello", true).is_err());
        let counter =
            crate::TokenCounter::from_json(&tokenizer.to_string(false).expect("serialize"))
                .expect("load");
        assert!(matches!(
            counter.count_text("hello"),
            Err(crate::Error::Encode(_))
        ));
    }

    #[rstest]
    fn shared_counter_matches_encoder_across_threads(anthropic_tokenizer: Tokenizer) {
        let counter = crate::TokenCounter::from_json(
            &anthropic_tokenizer.to_string(false).expect("serialize"),
        )
        .expect("load");
        let inputs = [
            "hello world",
            "ＡＢ ﬁ\n漢字🙂",
            " <EOT> stop",
            "\t  're \r\n",
        ];
        let expected = inputs.map(|text| reference_count(&anthropic_tokenizer, text));
        std::thread::scope(|scope| {
            for _ in 0..8 {
                let counter = &counter;
                scope.spawn(move || {
                    for _ in 0..100 {
                        for (text, count) in inputs.iter().zip(expected) {
                            assert_eq!(counter.count_text(text).expect("count"), count);
                        }
                    }
                });
            }
        });
    }

    #[rstest]
    fn normalized_added_token_spelling_declines_fast_path(mut anthropic_tokenizer: Tokenizer) {
        anthropic_tokenizer
            .add_tokens([tokenizers::AddedToken::from("ＡＢＣＤ　ＥＦＧＨ", false)])
            .expect("add token");
        let fast = ByteLevelCounter::detect(&anthropic_tokenizer).expect("supported");
        assert_eq!(reference_count(&anthropic_tokenizer, "ABCD EFGH"), 1);
        assert_eq!(fast.count(&anthropic_tokenizer, "ABCD EFGH"), None);
        let counter = crate::TokenCounter::from_json(
            &anthropic_tokenizer.to_string(false).expect("serialize"),
        )
        .expect("load");
        assert_eq!(counter.count_text("ABCD EFGH").expect("count"), 1);
    }
}
