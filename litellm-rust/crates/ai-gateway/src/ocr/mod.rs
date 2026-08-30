use litellm_core::CoreResult;
use serde_json::Value;

mod hooks;
mod types;

pub use types::OcrRequest;

use hooks::OcrLifecycleHooks;

pub async fn ocr(request: OcrRequest<'_>) -> CoreResult<Value> {
    let hooks = OcrLifecycleHooks::new(
        crate::integrations::custom_logger::CustomLoggerRunner::new(request.callbacks),
        crate::integrations::custom_guardrail::CustomGuardrailRunner::new(request.guardrails),
        request.request_metadata,
    );
    let request = litellm_runtime::ocr::prepare_ocr_request(litellm_runtime::ocr::OcrRequest {
        model: request.model,
        document: request.document,
        api_key: request.api_key,
        api_base: request.api_base,
        custom_llm_provider: request.custom_llm_provider,
        extra_headers: request.extra_headers,
        optional_params: request.optional_params,
        timeout: request.timeout,
        litellm_call_id: request.litellm_call_id,
    });
    litellm_runtime::ocr::ocr_with_hooks(request, &hooks).await
}

#[cfg(test)]
mod tests;
