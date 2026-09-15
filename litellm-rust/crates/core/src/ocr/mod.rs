mod adapters;
pub mod client;
mod codecs;
mod document;
pub mod error;
mod handler;
pub mod hooks;
mod lifecycle;
mod prepare;
mod registry;
pub mod types;
pub mod wire;

pub use crate::call_lifecycle::host::LifecycleBackend as OcrHost;
pub use client::{OcrClient, ocr};
pub use document::{encode_file_document, mime_type_for_name, upload_mime_type};
pub use lifecycle::{
    NativeOutcome, NativeResult, NoopOcrHost, OcrAdmission, OcrCall, OcrCallStep, OcrDecline,
    OcrHookHost, OcrHostOperation, OcrHostResult,
};
pub use types::{LiteLLMOcrRequest, LiteLLMOcrResponse, OcrConnection, OcrDocument};

pub fn admit_value(
    model: &str,
    provider: Option<&str>,
) -> Result<(), crate::call_lifecycle::admission::AdmissionDecline> {
    registry::resolve_wire_adapter(model, provider)
        .map(|_| ())
        .map_err(|_| crate::call_lifecycle::admission::AdmissionDecline::Provider)
}

#[cfg(test)]
#[path = "../../tests/azure_ai_ocr.rs"]
mod azure_ai_tests;
#[cfg(test)]
#[path = "../../tests/azure_document_intelligence_ocr.rs"]
mod azure_document_intelligence_tests;
#[cfg(test)]
#[path = "../../tests/deepseek_ocr.rs"]
mod deepseek_tests;
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
