#![forbid(unsafe_code)]

mod error;

pub use error::Error;

pub struct HuggingFaceTokenizer(Box<tokenizers::Tokenizer>);

impl HuggingFaceTokenizer {
    pub fn from_json(json: &str) -> Result<Self, Error> {
        json.parse::<tokenizers::Tokenizer>()
            .map(Box::new)
            .map(Self)
            .map_err(Error::Load)
    }

    pub fn count_tokens(&self, text: &str) -> Result<usize, Error> {
        self.0
            .encode_fast(text, true)
            .map(|encoding| encoding.len())
            .map_err(Error::Encode)
    }
}
