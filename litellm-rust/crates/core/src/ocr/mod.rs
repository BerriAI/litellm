mod error;
pub use error::Error;
pub mod client;
pub(crate) mod document;
pub(crate) mod handler;
pub mod hooks;
mod lifecycle;
pub(crate) mod prepare;
mod provider_config;
pub mod types;
pub mod wire;

pub use client::{OcrClient, ocr};
pub use document::{encode_file_document, mime_type_for_name, upload_mime_type};
pub use lifecycle::{
    NativeOutcome, NativeResult, NoopOcrHost, OcrAdmission, OcrCall, OcrCallStep, OcrDecline,
    OcrHookHost, OcrHost, OcrHostOperation, OcrHostResult,
};
pub use provider_config::{get_api_key_env_var, get_health_check_document};
pub use types::{
    LiteLLMOcrRequest, LiteLLMOcrResponse, OcrConnection, OcrDocument, OcrPage, OcrPageDimensions,
    OcrPageImage, OcrUsageInfo,
};

#[cfg(test)]
pub(crate) mod test_support;
