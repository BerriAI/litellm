#![forbid(unsafe_code)]

mod error;

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

pub use error::Error;
use tokenizers::PostProcessor;
pub use tokenizers::{
    AddedToken, EncodeInput, Encoding, InputSequence, PaddingDirection, PaddingParams,
    PaddingStrategy, TruncationDirection, TruncationParams,
};

pub fn encoding_from_json(json: &str) -> Result<Encoding, Error> {
    serde_json::from_str(json).map_err(|error| Error::Load(error.into()))
}

pub fn encoding_to_json(encoding: &Encoding) -> Result<String, Error> {
    serde_json::to_string(encoding).map_err(|error| Error::Load(error.into()))
}

pub struct HuggingFaceTokenizer {
    tokenizer: Arc<tokenizers::Tokenizer>,
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
            tokenizer: Arc::new(tokenizer),
            special_token_ids,
        }
    }

    /// The parsed model, for a count-only counter to share instead of parsing it again.
    pub fn shared(&self) -> Arc<tokenizers::Tokenizer> {
        Arc::clone(&self.tokenizer)
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

    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.tokenizer.token_to_id(token)
    }

    pub fn id_to_token(&self, id: u32) -> Option<String> {
        self.tokenizer.id_to_token(id)
    }

    pub fn vocab(&self, with_added_tokens: bool) -> HashMap<String, u32> {
        self.tokenizer.get_vocab(with_added_tokens)
    }

    pub fn vocab_size(&self, with_added_tokens: bool) -> usize {
        self.tokenizer.get_vocab_size(with_added_tokens)
    }

    /// The added tokens by id, in id order.
    pub fn added_tokens_decoder(&self) -> Vec<(u32, AddedToken)> {
        let mut added: Vec<(u32, AddedToken)> = self
            .tokenizer
            .get_added_tokens_decoder()
            .into_iter()
            .collect();
        added.sort_unstable_by_key(|(id, _)| *id);
        added
    }

    pub fn padding(&self) -> Option<&PaddingParams> {
        self.tokenizer.get_padding()
    }

    pub fn truncation(&self) -> Option<&TruncationParams> {
        self.tokenizer.get_truncation()
    }

    /// How many special tokens the post-processor adds to a single sequence or a pair.
    pub fn num_special_tokens_to_add(&self, is_pair: bool) -> usize {
        self.tokenizer
            .get_post_processor()
            .map_or(0, |processor| processor.added_tokens(is_pair))
    }

    pub fn encode_special_tokens(&self) -> bool {
        self.tokenizer.get_encode_special_tokens()
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

    #[test]
    fn vocabulary_lookups_mirror_the_tokenizers_api() {
        let json = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../litellm/litellm_core_utils/tokenizers/anthropic_tokenizer.json"
        ));
        let tokenizer = HuggingFaceTokenizer::from_json(json).unwrap();
        let ids = tokenizer.encode("hello").unwrap();

        let token = tokenizer.id_to_token(ids[0]).unwrap();
        assert_eq!(tokenizer.token_to_id(&token), Some(ids[0]));
        assert_eq!(tokenizer.id_to_token(u32::MAX), None);
        assert_eq!(tokenizer.vocab(true).len(), tokenizer.vocab_size(true));
        assert!(tokenizer.vocab_size(true) >= tokenizer.vocab_size(false));
        let added = tokenizer.added_tokens_decoder();
        assert!(added.windows(2).all(|pair| pair[0].0 < pair[1].0));
        assert!(added.iter().any(|(_, token)| token.special));
        assert!(tokenizer.padding().is_none());
        assert!(tokenizer.truncation().is_none());
        assert!(!tokenizer.encode_special_tokens());
        assert_eq!(
            tokenizer.num_special_tokens_to_add(false),
            tokenizer.encode("").unwrap().len()
        );
    }
}
