pub(crate) mod backends;
pub mod client;
pub(crate) mod document;
pub mod error;
pub mod formats;
mod handler;
pub mod hooks;
pub mod prepare;
pub mod registry;
pub mod types;
pub mod wire;

pub use client::{OcrClient, ocr};
pub use types::{OcrRequest, OcrResponseData};

#[cfg(test)]
#[path = "../../tests/azure_ai_ocr.rs"]
mod azure_ai_tests;
#[cfg(test)]
#[path = "../../tests/azure_document_intelligence_ocr.rs"]
mod azure_document_intelligence_tests;
#[cfg(test)]
#[path = "../../tests/mistral_ocr.rs"]
mod mistral_tests;
#[cfg(test)]
#[path = "../../tests/reducto_ocr.rs"]
mod reducto_tests;
#[cfg(test)]
#[path = "../../tests/ocr.rs"]
pub(crate) mod tests;
#[cfg(test)]
#[path = "../../tests/vertex_ai_deepseek_ocr.rs"]
mod vertex_ai_deepseek_tests;
#[cfg(test)]
#[path = "../../tests/vertex_ai_ocr.rs"]
mod vertex_ai_tests;
