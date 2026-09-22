#![forbid(unsafe_code)]

mod error;

pub use error::UnsupportedTokenizer;

pub struct TiktokenTokenizer(&'static tiktoken_rs::CoreBPE);

impl TiktokenTokenizer {
    pub fn from_name(name: &str) -> Result<Self, UnsupportedTokenizer> {
        let tokenizer = match name {
            "cl100k_base" => tiktoken_rs::cl100k_base_singleton(),
            "o200k_base" => tiktoken_rs::o200k_base_singleton(),
            "o200k_harmony" => tiktoken_rs::o200k_harmony_singleton(),
            "p50k_base" => tiktoken_rs::p50k_base_singleton(),
            "p50k_edit" => tiktoken_rs::p50k_edit_singleton(),
            "r50k_base" | "gpt2" => tiktoken_rs::r50k_base_singleton(),
            _ => return Err(UnsupportedTokenizer(name.to_owned())),
        };
        Ok(Self(tokenizer))
    }

    pub fn count_tokens(&self, text: &str) -> usize {
        self.0.count_ordinary(text)
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
    }
}
