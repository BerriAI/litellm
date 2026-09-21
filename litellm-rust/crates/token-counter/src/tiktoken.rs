pub use litellm_token_counter_tiktoken::TiktokenTokenizer;
use litellm_token_counter_tiktoken::UnsupportedTokenizer;

use crate::{Error, TokenCounter, Tokenizer};

impl TokenCounter {
    pub fn from_tiktoken(encoding: &str) -> Result<Self, Error> {
        TiktokenTokenizer::from_name(encoding)
            .map(Self::new)
            .map_err(Error::from)
    }
}

impl Tokenizer for TiktokenTokenizer {
    fn count_tokens(&self, text: &str) -> Result<usize, Error> {
        Ok(TiktokenTokenizer::count_tokens(self, text))
    }
}

impl From<UnsupportedTokenizer> for Error {
    fn from(error: UnsupportedTokenizer) -> Self {
        Self::UnsupportedTokenizer(error.0)
    }
}
