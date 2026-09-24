pub mod arguments;
pub mod client;
pub mod document;
pub(crate) mod handler;
pub(crate) mod prepare;
pub mod provider_config;
pub mod route;
pub mod types;
pub mod wire;

#[cfg(test)]
#[path = "../../tests/ocr/aws_textract.rs"]
mod aws_textract_tests;

#[cfg(test)]
#[path = "../../tests/ocr/azure_ai.rs"]
mod azure_ai_tests;
#[cfg(test)]
#[path = "../../tests/ocr/azure_document_intelligence.rs"]
mod azure_document_intelligence_tests;
#[cfg(test)]
#[path = "../../tests/ocr/cohere.rs"]
mod cohere_tests;
#[cfg(test)]
#[path = "../../tests/ocr/deepseek.rs"]
mod deepseek_tests;
#[cfg(test)]
#[path = "../../tests/ocr/document.rs"]
mod document_tests;
#[cfg(test)]
#[path = "../../tests/ocr/reducto.rs"]
mod reducto_tests;
#[cfg(test)]
#[path = "../../tests/ocr/support.rs"]
pub(crate) mod test_support;
#[cfg(test)]
#[path = "../../tests/ocr/contract.rs"]
pub(crate) mod tests;
#[cfg(test)]
#[path = "../../tests/ocr/vertex_ai_deepseek.rs"]
mod vertex_ai_deepseek_tests;
#[cfg(test)]
#[path = "../../tests/ocr/vertex_ai.rs"]
mod vertex_ai_tests;
