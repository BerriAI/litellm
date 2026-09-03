use litellm_core::ocr::{OcrRequest as CoreOcrRequest, PreparedOcrRequest};

use super::hooks::OcrLifecycleHooks;
use super::types::OcrRequest;
use crate::integrations::custom_guardrail::CustomGuardrailRunner;
use crate::integrations::custom_logger::CustomLoggerRunner;

pub(crate) struct PreparedOcrCall {
    pub(crate) request: PreparedOcrRequest,
    pub(crate) hooks: OcrLifecycleHooks,
}

pub(crate) fn prepare_ocr_call(request: OcrRequest<'_>) -> PreparedOcrCall {
    let OcrRequest {
        model,
        document,
        api_key,
        api_base,
        custom_llm_provider,
        extra_headers,
        optional_params,
        timeout,
        callbacks,
        guardrails,
        request_metadata,
        litellm_call_id,
    } = request;
    PreparedOcrCall {
        request: litellm_core::ocr::prepare_ocr_call(CoreOcrRequest {
            model,
            document,
            api_key,
            api_base,
            custom_llm_provider,
            extra_headers,
            optional_params,
            timeout,
            litellm_call_id,
        }),
        hooks: OcrLifecycleHooks::new(
            CustomLoggerRunner::new(callbacks),
            CustomGuardrailRunner::new(guardrails),
            request_metadata,
        ),
    }
}
