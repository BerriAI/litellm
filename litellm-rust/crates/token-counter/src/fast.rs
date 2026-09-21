use litellm_token_counter_fast::Error as BackendError;
pub use litellm_token_counter_fast::FastTokenizer;

use crate::{Error, TokenCounter, Tokenizer};

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
