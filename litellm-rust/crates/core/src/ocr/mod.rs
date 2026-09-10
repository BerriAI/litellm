mod adapters;
pub mod client;
mod codecs;
pub mod error;
mod handler;
pub mod hooks;
mod prepare;
mod registry;
pub mod transformation;
pub mod types;
pub mod wire;

pub use client::{OcrClient, ocr};
pub use types::{LiteLLMOcrRequest, LiteLLMOcrResponse, OcrConnection, OcrDocument};

#[cfg(test)]
#[path = "../../tests/ocr/support.rs"]
pub(crate) mod test_support;
#[cfg(test)]
#[path = "../../tests/ocr.rs"]
pub(crate) mod tests;
