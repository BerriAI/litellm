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
