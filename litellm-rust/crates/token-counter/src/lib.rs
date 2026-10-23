//! Input token counting for a request body, mirroring `litellm.token_counter`
//! for the shapes it can count exactly. Everything else is declined so the host
//! keeps its own counter as the reference.

#![forbid(unsafe_code)]

mod counter;
mod error;
mod python_json;
mod tokenizer;
mod tools;
mod types;

#[cfg(feature = "fast")]
pub mod fast;
#[cfg(feature = "huggingface")]
pub mod huggingface;
#[cfg(feature = "tiktoken")]
pub mod tiktoken;

pub use counter::{InputTokenCount, TokenCounter};
pub use error::Error;
pub use tokenizer::{TextCodec, Tokenizer};
pub use types::CountableRequest;
