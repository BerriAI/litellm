#![forbid(unsafe_code)]

mod error;

use std::collections::HashSet;

pub use error::Error;
pub use tokenizers::{EncodeInput, Encoding, InputSequence};

pub fn encoding_from_json(json: &str) -> Result<Encoding, Error> {
    serde_json::from_str(json).map_err(|error| Error::Load(error.into()))
}

pub fn encoding_to_json(encoding: &Encoding) -> Result<String, Error> {
    serde_json::to_string(encoding).map_err(|error| Error::Load(error.into()))
}

pub struct HuggingFaceTokenizer {
    tokenizer: Box<tokenizers::Tokenizer>,
    special_token_ids: HashSet<u32>,
}

impl HuggingFaceTokenizer {
    pub fn from_json(json: &str) -> Result<Self, Error> {
        json.parse::<tokenizers::Tokenizer>()
            .map(Self::new)
            .map_err(Error::Load)
    }

    fn new(tokenizer: tokenizers::Tokenizer) -> Self {
        let special_token_ids: HashSet<u32> = tokenizer
            .get_added_tokens_decoder()
            .into_iter()
            .filter_map(|(id, token)| token.special.then_some(id))
            .collect();
        Self {
            tokenizer: Box::new(tokenizer),
            special_token_ids,
        }
    }

    pub fn count_tokens(&self, text: &str) -> Result<usize, Error> {
        self.tokenizer
            .encode_fast(text, true)
            .map(|encoding| encoding.len())
            .map_err(Error::Encode)
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>, Error> {
        self.tokenizer
            .encode_fast(text, true)
            .map(|encoding| encoding.get_ids().to_vec())
            .map_err(Error::Encode)
    }

    pub fn encode_result<'a>(
        &self,
        input: EncodeInput<'a>,
        add_special_tokens: bool,
        fast: bool,
    ) -> Result<Encoding, Error> {
        if fast {
            return self
                .tokenizer
                .encode_fast(input, add_special_tokens)
                .map_err(Error::Encode);
        }
        self.tokenizer
            .encode_char_offsets(input, add_special_tokens)
            .map_err(Error::Encode)
    }

    pub fn encode_batch_result<'a>(
        &self,
        inputs: Vec<EncodeInput<'a>>,
        add_special_tokens: bool,
        fast: bool,
    ) -> Result<Vec<Encoding>, Error> {
        if fast {
            return self
                .tokenizer
                .encode_batch_fast(inputs, add_special_tokens)
                .map_err(Error::Encode);
        }
        self.tokenizer
            .encode_batch_char_offsets(inputs, add_special_tokens)
            .map_err(Error::Encode)
    }

    pub fn to_json(&self, pretty: bool) -> Result<String, Error> {
        self.tokenizer.to_string(pretty).map_err(Error::Load)
    }

    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String, Error> {
        if !skip_special_tokens {
            return self.tokenizer.decode(ids, false).map_err(Error::Decode);
        }
        let filtered_ids: Vec<u32> = ids
            .iter()
            .copied()
            .filter(|id| !self.special_token_ids.contains(id))
            .collect();
        self.tokenizer
            .decode(&filtered_ids, true)
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

    #[test]
    fn decode_filters_special_added_tokens() {
        let json = r#"{
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [
                {
                    "id": 1,
                    "content": "<s>",
                    "single_word": false,
                    "lstrip": false,
                    "rstrip": false,
                    "normalized": false,
                    "special": true
                }
            ],
            "normalizer": null,
            "pre_tokenizer": {"type": "Whitespace"},
            "post_processor": null,
            "decoder": null,
            "model": {
                "type": "WordLevel",
                "vocab": {"<unk>": 0, "<s>": 1, "hello": 2},
                "unk_token": "<unk>"
            }
        }"#;
        let tokenizer = HuggingFaceTokenizer::from_json(json).unwrap();

        assert!(!tokenizer.decode(&[1, 2], true).unwrap().contains("<s>"));
        assert!(tokenizer.decode(&[1, 2], false).unwrap().contains("<s>"));
    }
}
