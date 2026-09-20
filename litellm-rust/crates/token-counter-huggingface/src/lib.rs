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

    pub fn from_pretrained(
        identifier: &str,
        revision: &str,
        token: Option<&str>,
    ) -> Result<Self, Error> {
        tokenizers::Tokenizer::from_pretrained(
            identifier,
            Some(tokenizers::FromPretrainedParameters {
                revision: revision.to_owned(),
                token: token.map(str::to_owned),
                ..Default::default()
            }),
        )
        .map(Box::new)
        .map(Self)
        .map_err(Error::Download)
    }

    pub fn count_tokens(&self, text: &str) -> Result<usize, Error> {
        self.0
            .encode_fast(text, true)
            .map(|encoding| encoding.len())
            .map_err(Error::Encode)
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>, Error> {
        self.0
            .encode_fast(text, true)
            .map(|encoding| encoding.get_ids().to_vec())
            .map_err(Error::Encode)
    }

    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String, Error> {
        self.0
            .decode(ids, skip_special_tokens)
            .map_err(Error::Decode)
    }

    pub fn name(&self) -> &str {
        "huggingface"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codecs_round_trip_and_skip_special_tokens() {
        let json = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../litellm/litellm_core_utils/tokenizers/anthropic_tokenizer.json"
        ));
        let tokenizer = HuggingFaceTokenizer::from_json(json).unwrap();
        let ids = tokenizer.encode("<SOS>hello<EOT>").unwrap();

        assert!(tokenizer.decode(&ids, false).unwrap().contains("<SOS>"));
        assert_eq!(tokenizer.decode(&ids, true).unwrap(), "hello");
    }
}
