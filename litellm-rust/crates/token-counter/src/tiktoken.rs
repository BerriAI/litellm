use litellm_token_counter_tiktoken::{LoadError, UnsupportedTokenizer};
pub use litellm_token_counter_tiktoken::{TiktokenTokenizer, Vocabulary, encoding_for_model};

use crate::{Error, TextCodec, TokenCounter, Tokenizer};

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

impl TextCodec for TiktokenTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>, Error> {
        Ok(TiktokenTokenizer::encode(self, text))
    }

    fn decode(&self, ids: &[u32], _skip_special_tokens: bool) -> Result<String, Error> {
        TiktokenTokenizer::decode(self, ids).map_err(|error| Error::Decode(error.to_string()))
    }

    fn name(&self) -> &str {
        TiktokenTokenizer::name(self)
    }
}

impl From<UnsupportedTokenizer> for Error {
    fn from(error: UnsupportedTokenizer) -> Self {
        Self::UnsupportedTokenizer(error.0)
    }
}

impl From<LoadError> for Error {
    fn from(error: LoadError) -> Self {
        match error {
            LoadError::Unsupported(error) => error.into(),
            LoadError::Ranks(message) => Self::Ranks(message),
        }
    }
}
