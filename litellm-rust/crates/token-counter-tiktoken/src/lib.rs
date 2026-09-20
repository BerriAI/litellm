#![forbid(unsafe_code)]

mod error;

pub use error::UnsupportedTokenizer;

pub struct TiktokenTokenizer {
    encoder: &'static tiktoken_rs::CoreBPE,
    name: &'static str,
}

impl TiktokenTokenizer {
    pub fn from_name(name: &str) -> Result<Self, UnsupportedTokenizer> {
        let (encoder, canonical_name) = match name {
            "cl100k_base" => (tiktoken_rs::cl100k_base_singleton(), "cl100k_base"),
            "o200k_base" => (tiktoken_rs::o200k_base_singleton(), "o200k_base"),
            "o200k_harmony" => (tiktoken_rs::o200k_harmony_singleton(), "o200k_harmony"),
            "p50k_base" => (tiktoken_rs::p50k_base_singleton(), "p50k_base"),
            "p50k_edit" => (tiktoken_rs::p50k_edit_singleton(), "p50k_edit"),
            "r50k_base" => (tiktoken_rs::r50k_base_singleton(), "r50k_base"),
            "gpt2" => (tiktoken_rs::r50k_base_singleton(), "r50k_base"),
            _ => return Err(UnsupportedTokenizer(name.to_owned())),
        };
        Ok(Self {
            encoder,
            name: canonical_name,
        })
    }

    pub fn count_tokens(&self, text: &str) -> usize {
        self.encoder.count_ordinary(text)
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        self.encoder.encode_ordinary(text)
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String, String> {
        self.encoder.decode(ids).map_err(|error| error.to_string())
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
        assert_eq!(
            TiktokenTokenizer::from_name("gpt2").unwrap().name(),
            "r50k_base"
        );
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
    fn encoding_for_model_maps_known_models() {
        assert_eq!(encoding_for_model("gpt-4o"), Some("o200k_base"));
        assert_eq!(encoding_for_model("text-davinci-003"), Some("p50k_base"));
        assert_eq!(encoding_for_model("unknown-model"), None);
    }
}
