use litellm_core::Error;
use litellm_core::call_lifecycle::CallLifecycle;
use serde_json::Value;

mod common_utils;
mod file_input;
mod handler;
mod hooks;
mod prepare;
mod request_body;
mod types;

pub use file_input::{
    FileInput, build_document_from_upload, convert_file_document_to_url_document, get_mime_type,
};
pub use request_body::{parse_ocr_json_body, parse_ocr_multipart_form};
pub use types::OcrRequest;

use handler::execute_ocr_provider_call;
use prepare::{PreparedOcrCall, prepare_ocr_call};

#[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
pub async fn ocr(request: OcrRequest<'_>) -> Result<Value, Error> {
    let PreparedOcrCall { request, hooks } = prepare_ocr_call(request);
    CallLifecycle::default()
        .run_request(request, &hooks, |request| {
            execute_ocr_provider_call(request, &hooks)
        })
        .await
}

#[cfg(test)]
mod tests;
