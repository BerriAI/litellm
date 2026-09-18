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
#[path = "../../tests/azure_ai_ocr.rs"]
mod azure_ai_tests;
#[cfg(test)]
#[path = "../../tests/azure_document_intelligence_ocr.rs"]
mod azure_document_intelligence_tests;
#[cfg(test)]
#[path = "../../tests/cohere_ocr.rs"]
mod cohere_tests;
#[cfg(test)]
#[path = "../../tests/deepseek_ocr.rs"]
mod deepseek_tests;
#[cfg(test)]
#[path = "../../tests/ocr/passthrough.rs"]
mod passthrough_tests;
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
#[path = "../../tests/vertex_ai_deepseek_ocr.rs"]
mod vertex_ai_deepseek_tests;
#[cfg(test)]
#[path = "../../tests/vertex_ai_ocr.rs"]
mod vertex_ai_tests;
