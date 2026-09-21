use litellm_token_counter_huggingface::Error as BackendError;
pub use litellm_token_counter_huggingface::HuggingFaceTokenizer;

use crate::{Error, TokenCounter, Tokenizer};

impl TokenCounter {
    pub fn from_json(tokenizer_json: &str) -> Result<Self, Error> {
        HuggingFaceTokenizer::from_json(tokenizer_json)
            .map(Self::new)
            .map_err(Error::from)
    }
}

impl Tokenizer for HuggingFaceTokenizer {
    fn count_tokens(&self, text: &str) -> Result<usize, Error> {
        HuggingFaceTokenizer::count_tokens(self, text).map_err(Error::from)
    }
}

impl From<BackendError> for Error {
    fn from(error: BackendError) -> Self {
        match error {
            BackendError::Load(source) => Self::Load(source),
            BackendError::Encode(source) => Self::Encode(source),
        }
    }
}
