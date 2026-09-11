mod adapters;
pub mod client;
mod codecs;
mod document;
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
#[path = "../../tests/azure_ai_ocr.rs"]
mod azure_ai_tests;
#[cfg(test)]
#[path = "../../tests/azure_document_intelligence_ocr.rs"]
mod azure_document_intelligence_tests;
#[cfg(test)]
#[path = "../../tests/reducto_ocr.rs"]
mod reducto_tests;
#[cfg(test)]
#[path = "../../tests/ocr/support.rs"]
pub(crate) mod test_support;
#[cfg(test)]
#[path = "../../tests/ocr.rs"]
pub(crate) mod tests;
#[cfg(test)]
#[path = "../../tests/vertex_ai_ocr.rs"]
mod vertex_ai_tests;
