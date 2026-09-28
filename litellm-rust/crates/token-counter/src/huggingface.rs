use litellm_token_counter_huggingface::Error as BackendError;
pub use litellm_token_counter_huggingface::{
    AddedToken, EncodeInput, Encoding, HuggingFaceTokenizer, InputSequence, PaddingDirection,
    PaddingParams, PaddingStrategy, TruncationDirection, TruncationParams, encoding_from_json,
    encoding_to_json,
};

use crate::{Error, TextCodec, TokenCounter, Tokenizer};

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

impl TextCodec for HuggingFaceTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>, Error> {
        HuggingFaceTokenizer::encode(self, text).map_err(Error::from)
    }

    fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String, Error> {
        HuggingFaceTokenizer::decode(self, ids, skip_special_tokens).map_err(Error::from)
    }

    fn name(&self) -> &str {
        HuggingFaceTokenizer::name(self)
    }
}

impl From<BackendError> for Error {
    fn from(error: BackendError) -> Self {
        match error {
            BackendError::Load(source) => Self::Load(source),
            BackendError::Encode(source) => Self::Encode(source),
            BackendError::Decode(source) => Self::Decode(source.to_string()),
        }
    }
}
