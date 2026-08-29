//! The OCR call, the Rust equivalent of Python's `litellm.ocr()`.

mod client;
mod document_fetch;
mod document_intelligence;
mod handler;
mod hooks;
mod prepare;
mod provider_config;
pub mod transformation;
pub mod types;

use serde_json::Value;

use crate::CoreResult;
use crate::call_lifecycle::CallLifecycle;

use handler::execute_ocr_provider_call;
use prepare::{PreparedOcrCall, prepare_ocr_call};
pub use types::{OcrDocument, OcrRequest};

pub async fn ocr(request: OcrRequest<'_>) -> CoreResult<Value> {
    let PreparedOcrCall { request, hooks } = prepare_ocr_call(request);
    CallLifecycle::default()
        .run_request(request, &hooks, execute_ocr_provider_call)
        .await
}

#[cfg(test)]
mod tests;
