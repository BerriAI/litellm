#![forbid(unsafe_code)]

mod byte_level;
mod cl100k;
mod error;
mod o200k;
mod scanner;
mod tiktoken;
mod unicode_classes;

use std::sync::Arc;

use byte_level::ByteLevelCounter;
use scanner::{SplitPattern, TiktokenCounter};

pub use error::Error;

enum Encoder {
    HuggingFace {
        tokenizer: Arc<tokenizers::Tokenizer>,
        byte_level: Option<ByteLevelCounter>,
    },
    Tiktoken(TiktokenCounter),
}

/// A count-only tokenizer. Its model tables are immutable, so one built from an already
/// loaded model (`from_shared`, `from_*_pairs`) adds only the count-specific tables.
pub struct FastTokenizer(Encoder);

impl FastTokenizer {
    pub fn from_json(json: &str) -> Result<Self, Error> {
        let tokenizer = json.parse::<tokenizers::Tokenizer>().map_err(Error::Load)?;
        Ok(Self::from_shared(Arc::new(tokenizer)))
    }

    /// Counts with a Hugging Face model another codec already holds; nothing is re-parsed.
    pub fn from_shared(tokenizer: Arc<tokenizers::Tokenizer>) -> Self {
        let byte_level = ByteLevelCounter::detect(&tokenizer);
        Self(Encoder::HuggingFace {
            tokenizer,
            byte_level,
        })
    }

    pub fn from_cl100k_ranks(ranks: &str) -> Result<Self, Error> {
        Self::from_ranks(SplitPattern::Cl100k, ranks)
    }

    pub fn from_o200k_ranks(ranks: &str) -> Result<Self, Error> {
        Self::from_ranks(SplitPattern::O200k, ranks)
    }

    /// `cl100k_base` from ranks another loader already parsed.
    pub fn from_cl100k_pairs<'a>(
        pairs: impl IntoIterator<Item = (&'a [u8], u32)>,
    ) -> Result<Self, Error> {
        Self::from_pairs(SplitPattern::Cl100k, pairs)
    }

    /// `o200k_base` (and `o200k_harmony`, whose ordinary tokens are the same) from ranks
    /// another loader already parsed.
    pub fn from_o200k_pairs<'a>(
        pairs: impl IntoIterator<Item = (&'a [u8], u32)>,
    ) -> Result<Self, Error> {
        Self::from_pairs(SplitPattern::O200k, pairs)
    }

    fn from_ranks(split: SplitPattern, ranks: &str) -> Result<Self, Error> {
        TiktokenCounter::from_ranks(split, ranks)
            .map(Encoder::Tiktoken)
            .map(Self)
    }

    fn from_pairs<'a>(
        split: SplitPattern,
        pairs: impl IntoIterator<Item = (&'a [u8], u32)>,
    ) -> Result<Self, Error> {
        TiktokenCounter::from_pairs(split, pairs)
            .map(Encoder::Tiktoken)
            .map(Self)
    }

    pub fn count_tokens(&self, text: &str) -> Result<usize, Error> {
        match &self.0 {
            Encoder::Tiktoken(counter) => Ok(counter.count(text)),
            Encoder::HuggingFace {
                tokenizer,
                byte_level,
            } => {
                if let Some(count) = byte_level
                    .as_ref()
                    .and_then(|counter| counter.count(tokenizer, text))
                {
                    return Ok(count);
                }
                tokenizer
                    .encode_fast(text, true)
                    .map(|encoding| encoding.len())
                    .map_err(Error::Encode)
            }
        }
    }
}
