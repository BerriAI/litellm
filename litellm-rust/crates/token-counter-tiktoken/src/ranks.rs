use base64::{Engine, engine::general_purpose::STANDARD};
use once_cell::sync::OnceCell;
use rustc_hash::FxHashMap;
use thiserror::Error;
use tiktoken_rs::{CoreBPE, O200K_BASE_PAT_STR, Rank};

use crate::UnsupportedTokenizer;

const CL100K: &str = "9b5ad71b2ce5302211f9c61530b329a4922fc6a4";
const O200K: &str = "fb374d419588a4632f3f557e76b4b70aebbca790";
const P50K: &str = "ec7223a39ce59f226a68acc30dc1af2788490e15";
const LEGACY_PATTERN: &str =
    r"'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s";
const CL100K_PATTERN: &str = r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s";

static CL100K_ENCODER: OnceCell<Loaded> = OnceCell::new();
static O200K_ENCODER: OnceCell<Loaded> = OnceCell::new();
static HARMONY_ENCODER: OnceCell<Loaded> = OnceCell::new();
static P50K_ENCODER: OnceCell<Loaded> = OnceCell::new();
static EDIT_ENCODER: OnceCell<Loaded> = OnceCell::new();
static R50K_ENCODER: OnceCell<Loaded> = OnceCell::new();

/// One encoding built from a rank file: the BPE engine plus the vocabulary it was built
/// from, kept because `CoreBPE` does not expose its ranks and tiktoken's Python API does
/// (`token_byte_values`, `encode_single_token`, `max_token_value`, `_special_tokens`).
pub(super) struct Loaded {
    pub(super) bpe: CoreBPE,
    pub(super) vocabulary: Vocabulary,
}

/// The byte-level vocabulary of a tiktoken encoding.
pub struct Vocabulary {
    ranks: FxHashMap<Vec<u8>, Rank>,
    special_tokens: FxHashMap<String, Rank>,
    max_token_value: Rank,
}

impl Vocabulary {
    /// Every mergeable token's bytes, sorted bytewise like tiktoken's `token_byte_values`.
    pub fn token_byte_values(&self) -> Vec<Vec<u8>> {
        let mut values: Vec<Vec<u8>> = self.ranks.keys().cloned().collect();
        values.sort_unstable();
        values
    }

    /// The rank of one whole token: a mergeable piece first, then a special token's text.
    pub fn encode_single_token(&self, piece: &[u8]) -> Option<Rank> {
        if let Some(rank) = self.ranks.get(piece) {
            return Some(*rank);
        }
        std::str::from_utf8(piece)
            .ok()
            .and_then(|text| self.special_tokens.get(text).copied())
    }

    pub fn max_token_value(&self) -> Rank {
        self.max_token_value
    }

    /// Every mergeable token with its rank, for building other tables from one parse.
    pub fn ranks(&self) -> impl Iterator<Item = (&[u8], Rank)> + '_ {
        self.ranks
            .iter()
            .map(|(bytes, rank)| (bytes.as_slice(), *rank))
    }

    /// The special tokens with their ranks, tiktoken's `_special_tokens`.
    pub fn special_tokens(&self) -> impl Iterator<Item = (&str, Rank)> + '_ {
        self.special_tokens
            .iter()
            .map(|(token, rank)| (token.as_str(), *rank))
    }

    pub fn is_special_token(&self, rank: Rank) -> bool {
        self.special_tokens.values().any(|special| *special == rank)
    }
}

#[derive(Debug, Error)]
pub enum LoadError {
    #[error(transparent)]
    Unsupported(#[from] UnsupportedTokenizer),
    #[error("failed to load tiktoken ranks: {0}")]
    Ranks(String),
}

/// Loads `name` once per process. The returned name is the one requested (`gpt2` stays
/// `gpt2`, as `tiktoken.get_encoding("gpt2").name` does), while `gpt2` and `r50k_base` share
/// one cached encoder.
pub(super) fn load(
    name: &str,
    load_file: impl FnOnce(&str) -> std::io::Result<String>,
) -> Result<(&'static Loaded, &'static str), LoadError> {
    let (requested, canonical, file, cache) = match name {
        "cl100k_base" => ("cl100k_base", "cl100k_base", CL100K, &CL100K_ENCODER),
        "o200k_base" => ("o200k_base", "o200k_base", O200K, &O200K_ENCODER),
        "o200k_harmony" => ("o200k_harmony", "o200k_harmony", O200K, &HARMONY_ENCODER),
        "p50k_base" => ("p50k_base", "p50k_base", P50K, &P50K_ENCODER),
        "p50k_edit" => ("p50k_edit", "p50k_edit", P50K, &EDIT_ENCODER),
        "r50k_base" => ("r50k_base", "r50k_base", P50K, &R50K_ENCODER),
        "gpt2" => ("gpt2", "r50k_base", P50K, &R50K_ENCODER),
        _ => return Err(UnsupportedTokenizer(name.to_owned()).into()),
    };
    let loaded = cache.get_or_try_init(|| {
        let ranks = load_file(file).map_err(|error| LoadError::Ranks(error.to_string()))?;
        build(canonical, &ranks)
    })?;
    Ok((loaded, requested))
}

fn build(name: &str, ranks: &str) -> Result<Loaded, LoadError> {
    let parsed = ranks
        .lines()
        .map(parse_rank)
        .collect::<Result<Vec<_>, _>>()?;
    let encoder: FxHashMap<_, _> = parsed
        .into_iter()
        .filter(|(_, rank)| name != "r50k_base" || *rank < 50256)
        .collect();
    if encoder
        .values()
        .collect::<std::collections::HashSet<_>>()
        .len()
        != encoder.len()
        || (0..=u8::MAX).any(|byte| !encoder.contains_key(&[byte][..]))
    {
        return Err(LoadError::Ranks("invalid vocabulary ranks".into()));
    }
    let (pattern, specials): (&str, &[(&str, Rank)]) = match name {
        "cl100k_base" => (
            CL100K_PATTERN,
            &[
                ("<|endoftext|>", 100257),
                ("<|fim_prefix|>", 100258),
                ("<|fim_middle|>", 100259),
                ("<|fim_suffix|>", 100260),
                ("<|endofprompt|>", 100276),
            ],
        ),
        "o200k_base" => (
            O200K_BASE_PAT_STR,
            &[("<|endoftext|>", 199999), ("<|endofprompt|>", 200018)],
        ),
        "o200k_harmony" => (
            O200K_BASE_PAT_STR,
            &[
                ("<|startoftext|>", 199998),
                ("<|endoftext|>", 199999),
                ("<|reserved_200000|>", 200000),
                ("<|reserved_200001|>", 200001),
                ("<|return|>", 200002),
                ("<|constrain|>", 200003),
                ("<|reserved_200004|>", 200004),
                ("<|channel|>", 200005),
                ("<|start|>", 200006),
                ("<|end|>", 200007),
                ("<|message|>", 200008),
                ("<|reserved_200009|>", 200009),
                ("<|reserved_200010|>", 200010),
                ("<|reserved_200011|>", 200011),
                ("<|call|>", 200012),
            ],
        ),
        "p50k_edit" => (
            LEGACY_PATTERN,
            &[
                ("<|endoftext|>", 50256),
                ("<|fim_prefix|>", 50281),
                ("<|fim_middle|>", 50282),
                ("<|fim_suffix|>", 50283),
            ],
        ),
        _ => (LEGACY_PATTERN, &[("<|endoftext|>", 50256)]),
    };
    let reserved = (200013..=201087)
        .filter(|_| name == "o200k_harmony")
        .map(|rank| (format!("<|reserved_{rank}|>"), rank));
    let special_tokens: FxHashMap<String, Rank> = specials
        .iter()
        .map(|(token, rank)| ((*token).to_owned(), *rank))
        .chain(reserved)
        .collect();
    let max_token_value = encoder
        .values()
        .chain(special_tokens.values())
        .copied()
        .max()
        .ok_or_else(|| LoadError::Ranks("empty vocabulary".into()))?;
    let bpe = CoreBPE::new(encoder.clone(), special_tokens.clone(), pattern)
        .map_err(|error| LoadError::Ranks(error.to_string()))?;
    Ok(Loaded {
        bpe,
        vocabulary: Vocabulary {
            ranks: encoder,
            special_tokens,
            max_token_value,
        },
    })
}

fn parse_rank(line: &str) -> Result<(Vec<u8>, Rank), LoadError> {
    let (token, rank) = line
        .split_once(' ')
        .ok_or_else(|| LoadError::Ranks("missing rank".into()))?;
    let bytes = STANDARD
        .decode(token)
        .map_err(|error| LoadError::Ranks(error.to_string()))?;
    let rank = rank
        .parse()
        .map_err(|error: std::num::ParseIntError| LoadError::Ranks(error.to_string()))?;
    Ok((bytes, rank))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TiktokenTokenizer;

    fn read_packaged_ranks(file: &str) -> std::io::Result<String> {
        std::fs::read_to_string(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../../litellm/litellm_core_utils/tokenizers")
                .join(file),
        )
    }

    #[test]
    fn packaged_encodings_match_embedded_encodings_and_reuse_successful_loads() {
        for name in [
            "cl100k_base",
            "o200k_base",
            "o200k_harmony",
            "p50k_base",
            "p50k_edit",
            "r50k_base",
            "gpt2",
        ] {
            if name != "gpt2" {
                assert!(
                    TiktokenTokenizer::from_cached_ranks(name, |_| {
                        Err(std::io::Error::other("unreadable vocabulary"))
                    })
                    .is_err()
                );
            }
            let loads = std::sync::atomic::AtomicUsize::new(0);
            let barrier = std::sync::Barrier::new(4);
            let encoders = std::thread::scope(|scope| {
                let tasks: Vec<_> = (0..4)
                    .map(|_| {
                        scope.spawn(|| {
                            barrier.wait();
                            TiktokenTokenizer::from_cached_ranks(name, |file| {
                                loads.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                read_packaged_ranks(file)
                            })
                            .unwrap()
                        })
                    })
                    .collect();
                tasks
                    .into_iter()
                    .map(|task| task.join().unwrap())
                    .collect::<Vec<_>>()
            });
            assert_eq!(loads.into_inner(), usize::from(name != "gpt2"));
            let actual = &encoders[0];
            let expected = TiktokenTokenizer::from_name(name).unwrap();
            assert_eq!(actual.special_tokens(), expected.special_tokens());
            let specials: Vec<_> = expected.special_tokens().into_iter().collect();
            let special_text = specials.join(" ");
            assert_eq!(
                actual.encode_special(&special_text, &specials).unwrap(),
                expected.encode_special(&special_text, &specials).unwrap()
            );
            for text in [
                "",
                "café 漢字 ع 🙂",
                "a\r\nb\t ",
                "       hello 123456789",
                &special_text,
            ] {
                let ids = expected.encode(text);
                assert_eq!(actual.encode(text), ids, "{name}: {text:?}");
                assert_eq!(actual.count_tokens(text), ids.len(), "{name}: {text:?}");
                assert_eq!(
                    actual.decode_bytes(&ids).unwrap(),
                    expected.decode_bytes(&ids).unwrap()
                );
            }
            let cached = TiktokenTokenizer::from_cached_ranks(name, |_| {
                panic!("reloaded cached vocabulary")
            })
            .unwrap();
            assert_eq!(cached.encode("cached"), expected.encode("cached"));
            assert_eq!(cached.name(), name);
            assert!(expected.vocabulary().is_none());
            assert_vocabulary_lookups(name, actual);
        }
    }

    /// The token-level lookups tiktoken's Python `Encoding` exposes, checked against the
    /// encoder itself and against the known vocabulary sizes.
    fn assert_vocabulary_lookups(name: &str, tokenizer: &TiktokenTokenizer) {
        let max_token_value = match name {
            "cl100k_base" => 100_276,
            "o200k_base" => 200_018,
            "o200k_harmony" => 201_087,
            "p50k_base" => 50_280,
            "p50k_edit" => 50_283,
            "r50k_base" | "gpt2" => 50_256,
            _ => unreachable!("{name}"),
        };
        let vocabulary = tokenizer.vocabulary().unwrap();
        assert_eq!(vocabulary.max_token_value(), max_token_value, "{name}");
        let values = vocabulary.token_byte_values();
        assert!(values.windows(2).all(|pair| pair[0] < pair[1]), "{name}");
        for piece in values.iter().step_by(997) {
            let rank = vocabulary.encode_single_token(piece).unwrap();
            assert_eq!(tokenizer.decode_bytes(&[rank]).unwrap(), *piece, "{name}");
            assert!(!vocabulary.is_special_token(rank), "{name}");
        }
        for (token, rank) in vocabulary.special_tokens() {
            assert_eq!(vocabulary.encode_single_token(token.as_bytes()), Some(rank));
            assert!(vocabulary.is_special_token(rank), "{name}: {token}");
        }
        assert_eq!(vocabulary.encode_single_token(b"<|not-a-token|>"), None);
    }

    #[test]
    fn malformed_ranks_return_errors_instead_of_panicking() {
        for ranks in ["", "IQ==", "IQ== x", "!!! 1", "IQ== 1"] {
            assert!(build("cl100k_base", ranks).is_err());
        }
        let repeated_rank = (0..=u8::MAX)
            .map(|byte| format!("{} 0\n", STANDARD.encode([byte])))
            .collect::<String>();
        assert!(build("cl100k_base", &repeated_rank).is_err());
    }
}
