//! Input token counting for a request body, mirroring `litellm.token_counter`
//! for the shapes it can count exactly. Everything else is declined so the host
//! keeps its own counter as the reference.

#![forbid(unsafe_code)]

mod byte_level;
mod cl100k;
mod counter;
mod error;
mod o200k;
mod python_json;
mod scanner;
mod tiktoken;
mod tools;
mod types;
mod unicode_classes;

pub use counter::{InputTokenCount, TokenCounter};
pub use error::Error;
pub use types::CountableRequest;

pub fn admit_tokenizer(
    kind: Option<&str>,
    encoding: &str,
    disabled: bool,
    legacy_accounting: bool,
) -> Result<&'static str, Error> {
    if disabled {
        return Err(Error::UnsupportedTokenizer);
    }
    match (kind, encoding, legacy_accounting) {
        (Some("anthropic"), _, _) => Ok("anthropic"),
        (None, "cl100k_base", false) => Ok("cl100k_base"),
        (None, "o200k_base", false) => Ok("o200k_base"),
        _ => Err(Error::UnsupportedTokenizer),
    }
}
