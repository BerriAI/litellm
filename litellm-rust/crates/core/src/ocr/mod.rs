//! The OCR call, the Rust equivalent of Python's `litellm.ocr()`.

mod client;
mod document_fetch;
mod handler;
mod hooks;
mod prepare;
mod provider_config;
pub mod transformation;
pub mod types;

use serde_json::Value;

use crate::CoreResult;
use crate::call_lifecycle::{CallInterceptor, CallRuntime};
use crate::callbacks::CallbackOptions;

use handler::execute_ocr_provider_call;
use hooks::OcrCallbackInterceptor;
use prepare::{PreparedOcrCall, prepare_ocr_call, prepare_provider_request};
pub use types::{OcrCall, OcrDocument, OcrRequest, PreparedOcrRequest, ProviderOcrRequest};

pub async fn ocr(request: OcrRequest<'_>) -> CoreResult<Value> {
    ocr_with_interceptor(request, &crate::call_lifecycle::NoopCallInterceptor).await
}

pub async fn ocr_with_callbacks(
    request: OcrRequest<'_>,
    callbacks: CallbackOptions,
) -> CoreResult<Value> {
    ocr_with_interceptor(request, &OcrCallbackInterceptor::new(callbacks)).await
}

pub async fn ocr_with_interceptor<Interceptor>(
    request: OcrRequest<'_>,
    interceptor: &Interceptor,
) -> CoreResult<Value>
where
    Interceptor: CallInterceptor<OcrCall>,
{
    let PreparedOcrCall { context, request } = prepare_ocr_call(request);
    CallRuntime::new(interceptor)
        .run::<OcrCall, _, _, _, _>(
            context,
            request,
            prepare_provider_request,
            execute_ocr_provider_call,
        )
        .await
}

#[cfg(test)]
mod tests;
