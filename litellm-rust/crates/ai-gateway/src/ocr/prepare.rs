use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use litellm_core::routing_utils::provider::{get_custom_llm_provider, CustomLlmProvider};
use serde_json::{json, Value};

use super::hooks::OcrLifecycleHooks;
use super::types::{OcrRequest, PreparedOcrRequest};
use litellm_core::integrations::custom_guardrail::CustomGuardrailRunner;
use litellm_core::integrations::custom_logger::CustomLoggerRunner;

pub(crate) struct PreparedOcrCall {
    pub(crate) request: PreparedOcrRequest,
    pub(crate) hooks: OcrLifecycleHooks,
}

pub(crate) fn prepare_ocr_call(request: OcrRequest<'_>) -> PreparedOcrCall {
    let call_id = request
        .litellm_call_id
        .map(str::to_string)
        .unwrap_or_else(new_ocr_call_id);
    let provider_info = get_custom_llm_provider(request.model, request.custom_llm_provider)
        .unwrap_or(CustomLlmProvider {
            model: request.model,
            custom_llm_provider: "mistral",
        });
    let model = provider_info.model.to_string();
    let custom_llm_provider = provider_info.custom_llm_provider.to_string();
    let logging_kwargs = ocr_logging_kwargs(&request, &model, &custom_llm_provider);

    PreparedOcrCall {
        request: PreparedOcrRequest {
            model,
            custom_llm_provider,
            litellm_call_id: call_id,
            document: request.document,
            api_key: request.api_key.map(str::to_string),
            api_base: request.api_base.map(str::to_string),
            extra_headers: request.extra_headers,
            optional_params: request.optional_params,
            timeout: request.timeout,
        },
        hooks: OcrLifecycleHooks::new(
            CustomLoggerRunner::new(request.callbacks),
            CustomGuardrailRunner::new(request.guardrails),
            request.request_metadata,
            logging_kwargs,
        ),
    }
}

fn ocr_logging_kwargs(
    request: &OcrRequest<'_>,
    model: &str,
    custom_llm_provider: &str,
) -> HashMap<String, Value> {
    let mut kwargs = HashMap::new();
    kwargs.insert("model".to_string(), json!(model));
    kwargs.insert(
        "custom_llm_provider".to_string(),
        json!(custom_llm_provider),
    );
    kwargs.insert("call_type".to_string(), json!("ocr"));
    kwargs.insert("document".to_string(), request.document.clone());
    kwargs.insert(
        "optional_params".to_string(),
        Value::Object(request.optional_params.clone()),
    );
    kwargs.insert(
        "api_base".to_string(),
        request
            .api_base
            .map(|api_base| json!(api_base))
            .unwrap_or(Value::Null),
    );
    kwargs.insert(
        "headers".to_string(),
        request
            .extra_headers
            .clone()
            .map(Value::Object)
            .unwrap_or(Value::Null),
    );
    kwargs.insert(
        "timeout".to_string(),
        request
            .timeout
            .map(|timeout| json!(timeout.as_secs_f64()))
            .unwrap_or(Value::Null),
    );
    kwargs
}

fn new_ocr_call_id() -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(1);
    let sequence = COUNTER.fetch_add(1, Ordering::Relaxed);
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    format!("ocr-{timestamp}-{sequence}")
}
