use litellm_token_counter_fast::Error as BackendError;
pub use litellm_token_counter_fast::FastTokenizer;

use crate::{Error, TokenCounter, Tokenizer};

/// A count-only counter derived from a loaded codec, sharing its parsed model rather than
/// loading the artifact twice. Opt-in: it trades the codec's generality for count speed.
pub trait FastCounter {
    fn fast_counter(&self) -> Result<FastTokenizer, Error>;
}

#[cfg(feature = "huggingface")]
impl FastCounter for crate::huggingface::HuggingFaceTokenizer {
    fn fast_counter(&self) -> Result<FastTokenizer, Error> {
        Ok(FastTokenizer::from_shared(self.shared()))
    }
}

#[cfg(feature = "tiktoken")]
impl FastCounter for crate::tiktoken::TiktokenTokenizer {
    /// Only the encodings the fast scanner reproduces; the rest stay on the codec.
    fn fast_counter(&self) -> Result<FastTokenizer, Error> {
        let vocabulary = self.vocabulary().ok_or_else(|| {
            Error::Ranks("this encoding was built without its vocabulary".to_owned())
        })?;
        match self.name() {
            "cl100k_base" => FastTokenizer::from_cl100k_pairs(vocabulary.ranks()),
            "o200k_base" | "o200k_harmony" => FastTokenizer::from_o200k_pairs(vocabulary.ranks()),
            name => return Err(Error::UnsupportedTokenizer(name.to_owned())),
        }
        .map_err(Error::from)
    }
}

impl TokenCounter {
    pub fn from_json_fast(tokenizer_json: &str) -> Result<Self, Error> {
        FastTokenizer::from_json(tokenizer_json)
            .map(Self::new)
            .map_err(Error::from)
    }

    pub fn from_cl100k_ranks(rank_file: &str) -> Result<Self, Error> {
        FastTokenizer::from_cl100k_ranks(rank_file)
            .map(Self::new)
            .map_err(Error::from)
    }

    pub fn from_o200k_ranks(rank_file: &str) -> Result<Self, Error> {
        FastTokenizer::from_o200k_ranks(rank_file)
            .map(Self::new)
            .map_err(Error::from)
    }
}

impl Tokenizer for FastTokenizer {
    fn count_tokens(&self, text: &str) -> Result<usize, Error> {
        FastTokenizer::count_tokens(self, text).map_err(Error::from)
    }
}

impl From<BackendError> for Error {
    fn from(error: BackendError) -> Self {
        match error {
            BackendError::Load(source) => Self::Load(source),
            BackendError::Ranks(message) => Self::Ranks(message),
            BackendError::UnicodeClasses => Self::UnicodeClasses,
            BackendError::Encode(source) => Self::Encode(source),
        }
    }
}

#[cfg(all(test, feature = "huggingface", feature = "tiktoken"))]
mod tests {
    use super::*;
    use crate::huggingface::HuggingFaceTokenizer;
    use crate::tiktoken::TiktokenTokenizer;

    const TEXTS: [&str; 4] = [
        "",
        "hello world <|endoftext|>",
        "café 漢字 ع 🙂 line\r\n  indented 123456789",
        "<SOS>system<EOT> a\u{301} ﬁ",
    ];

    fn packaged(file: &str) -> String {
        std::fs::read_to_string(format!(
            "{}/../../../litellm/litellm_core_utils/tokenizers/{file}",
            env!("CARGO_MANIFEST_DIR")
        ))
        .unwrap()
    }

    #[test]
    fn fast_counters_derived_from_codecs_count_like_the_codecs() {
        let huggingface =
            HuggingFaceTokenizer::from_json(&packaged("anthropic_tokenizer.json")).unwrap();
        let fast = huggingface.fast_counter().unwrap();
        for text in TEXTS {
            assert_eq!(
                fast.count_tokens(text).unwrap(),
                Tokenizer::count_tokens(&huggingface, text).unwrap(),
                "{text:?}"
            );
        }

        for name in ["cl100k_base", "o200k_base", "o200k_harmony"] {
            let tiktoken =
                TiktokenTokenizer::from_cached_ranks(name, |file| Ok(packaged(file))).unwrap();
            let fast = tiktoken.fast_counter().unwrap();
            for text in TEXTS {
                assert_eq!(
                    fast.count_tokens(text).unwrap(),
                    tiktoken.count_tokens(text),
                    "{name}: {text:?}"
                );
            }
        }
    }

    #[test]
    fn encodings_without_a_fast_scanner_are_refused() {
        let tiktoken =
            TiktokenTokenizer::from_cached_ranks("p50k_base", |file| Ok(packaged(file))).unwrap();
        assert!(matches!(
            tiktoken.fast_counter(),
            Err(Error::UnsupportedTokenizer(name)) if name == "p50k_base"
        ));
        assert!(matches!(
            TiktokenTokenizer::from_name("cl100k_base")
                .unwrap()
                .fast_counter(),
            Err(Error::Ranks(_))
        ));
    }
}
