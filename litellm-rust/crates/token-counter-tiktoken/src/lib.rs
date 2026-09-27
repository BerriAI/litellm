#![forbid(unsafe_code)]

mod error;
mod ranks;

use std::collections::HashSet;

pub use error::UnsupportedTokenizer;
pub use ranks::{LoadError, Vocabulary};

pub struct TiktokenTokenizer {
    encoder: &'static tiktoken_rs::CoreBPE,
    /// Present for encodings built from a rank file; the embedded tiktoken-rs singletons
    /// behind [`from_name`](Self::from_name) keep their ranks private.
    vocabulary: Option<&'static Vocabulary>,
    name: &'static str,
}

impl TiktokenTokenizer {
    /// Builds `name` from its packaged rank file (read through `load`), once per process.
    /// The tokenizer reports the requested name, so `gpt2` stays `gpt2` like tiktoken does.
    pub fn from_cached_ranks(
        name: &str,
        load: impl FnOnce(&str) -> std::io::Result<String>,
    ) -> Result<Self, LoadError> {
        let (loaded, name) = ranks::load(name, load)?;
        Ok(Self {
            encoder: &loaded.bpe,
            vocabulary: Some(&loaded.vocabulary),
            name,
        })
    }

    /// The encodings tiktoken-rs embeds, for hosts without the packaged rank files.
    pub fn from_name(name: &str) -> Result<Self, UnsupportedTokenizer> {
        let (encoder, name) = match name {
            "cl100k_base" => (tiktoken_rs::cl100k_base_singleton(), "cl100k_base"),
            "o200k_base" => (tiktoken_rs::o200k_base_singleton(), "o200k_base"),
            "o200k_harmony" => (tiktoken_rs::o200k_harmony_singleton(), "o200k_harmony"),
            "p50k_base" => (tiktoken_rs::p50k_base_singleton(), "p50k_base"),
            "p50k_edit" => (tiktoken_rs::p50k_edit_singleton(), "p50k_edit"),
            "r50k_base" => (tiktoken_rs::r50k_base_singleton(), "r50k_base"),
            "gpt2" => (tiktoken_rs::r50k_base_singleton(), "gpt2"),
            _ => return Err(UnsupportedTokenizer(name.to_owned())),
        };
        Ok(Self {
            encoder,
            vocabulary: None,
            name,
        })
    }

    pub fn vocabulary(&self) -> Option<&Vocabulary> {
        self.vocabulary
    }

    pub fn count_tokens(&self, text: &str) -> usize {
        self.encoder.count_ordinary(text)
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        self.encoder.encode_ordinary(text)
    }

    pub fn encode_special(&self, text: &str, allowed: &[String]) -> Result<Vec<u32>, String> {
        let allowed = allowed.iter().map(String::as_str).collect();
        self.encoder
            .encode(text, &allowed)
            .map(|(ids, _)| ids)
            .map_err(|error| error.to_string())
    }

    pub fn special_tokens(&self) -> HashSet<String> {
        self.encoder
            .special_tokens()
            .into_iter()
            .map(str::to_owned)
            .collect()
    }

    /// tiktoken's `encode_with_unstable`: the stable prefix of `text`'s tokens and every
    /// token sequence the unstable tail could still become, sorted for a stable order.
    pub fn encode_with_unstable(
        &self,
        text: &str,
        allowed: &[String],
    ) -> (Vec<u32>, Vec<Vec<u32>>) {
        let allowed = allowed.iter().map(String::as_str).collect();
        let (stable, completions) = self.encoder._encode_unstable_native(text, &allowed);
        let mut completions: Vec<Vec<u32>> = completions.into_iter().collect();
        completions.sort_unstable();
        (stable, completions)
    }

    pub fn decode_bytes(&self, ids: &[u32]) -> Result<Vec<u8>, String> {
        self.encoder
            .decode_bytes(ids)
            .map_err(|error| error.to_string())
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String, String> {
        self.encoder
            .decode_bytes(ids)
            .map(|bytes| String::from_utf8_lossy(&bytes).into_owned())
            .map_err(|error| error.to_string())
    }

    pub fn name(&self) -> &str {
        self.name
    }
}

pub fn encoding_for_model(model: &str) -> Option<&'static str> {
    match tiktoken_rs::tokenizer::get_tokenizer(model)? {
        tiktoken_rs::tokenizer::Tokenizer::Cl100kBase => Some("cl100k_base"),
        tiktoken_rs::tokenizer::Tokenizer::O200kBase => Some("o200k_base"),
        tiktoken_rs::tokenizer::Tokenizer::O200kHarmony => Some("o200k_harmony"),
        tiktoken_rs::tokenizer::Tokenizer::P50kBase => Some("p50k_base"),
        tiktoken_rs::tokenizer::Tokenizer::P50kEdit => Some("p50k_edit"),
        tiktoken_rs::tokenizer::Tokenizer::R50kBase | tiktoken_rs::tokenizer::Tokenizer::Gpt2 => {
            Some("r50k_base")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn named_encodings_match_their_reference_counts() {
        let encodings = [
            ("cl100k_base", tiktoken_rs::cl100k_base_singleton()),
            ("o200k_base", tiktoken_rs::o200k_base_singleton()),
            ("o200k_harmony", tiktoken_rs::o200k_harmony_singleton()),
            ("p50k_base", tiktoken_rs::p50k_base_singleton()),
            ("p50k_edit", tiktoken_rs::p50k_edit_singleton()),
            ("r50k_base", tiktoken_rs::r50k_base_singleton()),
            ("gpt2", tiktoken_rs::r50k_base_singleton()),
        ];
        let texts = [
            "",
            "Hello, how are you today?",
            "é e\u{301} 漢字 ع ३ 🙂 ＡﬁⅣ",
            "  def function():\n    return 123456789\r\n",
            "<|endoftext|><|fim_prefix|><|start|>assistant<|message|>",
        ];
        for (name, reference) in encodings {
            let counter = TiktokenTokenizer::from_name(name).unwrap();
            for text in texts {
                assert_eq!(
                    counter.count_tokens(text),
                    reference.encode_ordinary(text).len(),
                    "{name}: {text:?}",
                );
            }
        }
    }

    #[test]
    fn unsupported_encoding_preserves_its_name() {
        let Err(UnsupportedTokenizer(name)) = TiktokenTokenizer::from_name("unknown-encoding")
        else {
            panic!("unknown encoding must be rejected");
        };
        assert_eq!(name, "unknown-encoding");
        assert_eq!(TiktokenTokenizer::from_name("gpt2").unwrap().name(), "gpt2");
    }

    #[test]
    fn codecs_round_trip_named_encodings() {
        let encodings = [
            "cl100k_base",
            "o200k_base",
            "o200k_harmony",
            "p50k_base",
            "p50k_edit",
            "r50k_base",
            "gpt2",
        ];
        let texts = ["hello world", "café 漢字 مرحبا 🙂", "line one\nline two"];
        for name in encodings {
            let tokenizer = TiktokenTokenizer::from_name(name).unwrap();
            for text in texts {
                assert_eq!(
                    tokenizer.decode(&tokenizer.encode(text)).unwrap(),
                    text,
                    "{name}: {text:?}",
                );
            }
        }
    }

    #[test]
    fn decoding_token_prefixes_replaces_incomplete_utf8() {
        let tokenizer = TiktokenTokenizer::from_name("cl100k_base").unwrap();
        let reference = tiktoken_rs::cl100k_base_singleton();
        let ids = tokenizer.encode("🙂漢字");
        for end in 1..=ids.len() {
            let bytes = reference.decode_bytes(&ids[..end]).unwrap();
            assert_eq!(
                tokenizer.decode(&ids[..end]).unwrap(),
                String::from_utf8_lossy(&bytes),
            );
        }
        assert!(tokenizer.decode(&[u32::MAX]).is_err());
    }

    #[test]
    fn unstable_encoding_prefixes_stay_consistent_with_full_encoding() {
        let tokenizer = TiktokenTokenizer::from_name("cl100k_base").unwrap();
        let text = "hello fanta";
        let (stable, completions) = tokenizer.encode_with_unstable(text, &[]);
        assert!(
            text.as_bytes()
                .starts_with(&tokenizer.decode_bytes(&stable).unwrap())
        );
        assert!(!completions.is_empty());
        for completion in &completions {
            let mut ids = stable.clone();
            ids.extend(completion);
            assert!(
                tokenizer
                    .decode_bytes(&ids)
                    .unwrap()
                    .starts_with(text.as_bytes())
            );
        }
        assert!(completions.windows(2).all(|pair| pair[0] < pair[1]));
    }

    #[test]
    fn encoding_for_model_maps_known_models() {
        assert_eq!(encoding_for_model("gpt-4o"), Some("o200k_base"));
        assert_eq!(encoding_for_model("text-davinci-003"), Some("p50k_base"));
        assert_eq!(encoding_for_model("unknown-model"), None);
    }
}
