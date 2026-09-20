#![forbid(unsafe_code)]

use thiserror::Error as ThisError;

#[derive(Debug, ThisError)]
#[error("unsupported tokenizer: {0}")]
pub struct UnsupportedTokenizer(pub String);

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
    fn special_tokens_are_counted_as_ordinary_text() {
        let counter = TiktokenTokenizer::from_name("cl100k_base").unwrap();
        assert!(counter.count_tokens("<|endoftext|>") > 1);
    }

    #[test]
    fn all_python_tiktoken_encodings_are_available() {
        for name in [
            "cl100k_base",
            "o200k_base",
            "o200k_harmony",
            "p50k_base",
            "p50k_edit",
            "r50k_base",
            "gpt2",
        ] {
            assert!(TiktokenTokenizer::from_name(name).is_ok(), "{name}");
        }
    }
}
