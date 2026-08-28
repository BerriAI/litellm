//! The OCR call, the Rust equivalent of Python's `litellm.ocr()`.
//!
//! [`ocr`] is the top-level entrypoint: give it a model, a document, and
//! credentials, and it resolves the provider, transforms the request, calls
//! the provider, and returns the normalized response JSON.

mod client;
mod common_utils;
mod handler;
mod hooks;
mod prepare;
pub mod transformation;
pub mod types;

use serde_json::Value;

use crate::CoreResult;
use crate::call_lifecycle::CallLifecycle;

use handler::execute_ocr_provider_call;
use prepare::{PreparedOcrCall, prepare_ocr_call};
pub use types::OcrRequest;

pub async fn ocr(request: OcrRequest<'_>) -> CoreResult<Value> {
    let PreparedOcrCall { request, hooks } = prepare_ocr_call(request);
    CallLifecycle::default()
        .run_request(request, &hooks, execute_ocr_provider_call)
        .await
}

#[cfg(test)]
mod tests;
