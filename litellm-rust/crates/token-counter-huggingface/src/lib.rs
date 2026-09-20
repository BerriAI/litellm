#![forbid(unsafe_code)]

mod error;

use std::collections::HashSet;

pub use error::Error;

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
        .map(Self::new)
        .map_err(Error::Download)
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
