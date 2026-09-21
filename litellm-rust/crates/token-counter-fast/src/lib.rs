#![forbid(unsafe_code)]

mod byte_level;
mod cl100k;
mod error;
mod o200k;
mod scanner;
mod tiktoken;
mod unicode_classes;

use byte_level::ByteLevelCounter;
use scanner::{SplitPattern, TiktokenCounter};

pub use error::Error;

enum Encoder {
    HuggingFace {
        tokenizer: Box<tokenizers::Tokenizer>,
        byte_level: Option<ByteLevelCounter>,
    },
    Tiktoken(TiktokenCounter),
}

pub struct FastTokenizer(Encoder);

impl FastTokenizer {
    pub fn from_json(json: &str) -> Result<Self, Error> {
        let tokenizer = json.parse::<tokenizers::Tokenizer>().map_err(Error::Load)?;
        let byte_level = ByteLevelCounter::detect(&tokenizer);
        Ok(Self(Encoder::HuggingFace {
            tokenizer: Box::new(tokenizer),
            byte_level,
        }))
    }

    pub fn from_cl100k_ranks(ranks: &str) -> Result<Self, Error> {
        Self::from_ranks(SplitPattern::Cl100k, ranks)
    }

    pub fn from_o200k_ranks(ranks: &str) -> Result<Self, Error> {
        Self::from_ranks(SplitPattern::O200k, ranks)
    }

    fn from_ranks(split: SplitPattern, ranks: &str) -> Result<Self, Error> {
        TiktokenCounter::from_ranks(split, ranks)
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
