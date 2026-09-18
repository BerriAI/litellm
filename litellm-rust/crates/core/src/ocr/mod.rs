mod arguments;
pub mod client;
pub(crate) mod document;
pub mod error;
pub use error::Error;
pub(crate) mod handler;
pub(crate) mod json;
pub(crate) mod prepare;
mod provider_config;
pub mod route;
pub mod types;
pub mod wire;

pub use arguments::{
    consumed_optional_param_names, consumed_optional_params, is_supported_request,
};
pub use client::{OcrClient, ocr};
pub use document::{encode_file_document, mime_type_for_name, read_path_document};
pub use provider_config::{get_api_key_env_var, get_health_check_document};
pub use route::{LocalOcrHost, Ocr, OcrHost, OcrMachine, OcrOp, OcrOpResult, ocr_machine};
pub use types::{
    LiteLLMOcrRequest, LiteLLMOcrResponse, OcrConnection, OcrConnectionInputs, OcrCredentialInputs,
    OcrDocument, OcrDocumentInput, OcrFileContent, OcrPage, OcrPageDimensions, OcrPageImage,
    OcrTransportConfig, OcrUsageInfo,
};

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
