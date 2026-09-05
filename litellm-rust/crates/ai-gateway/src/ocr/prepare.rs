use crate::integrations::types::RequestHooks;
use litellm_core::call_lifecycle::CallLifecycleContext;
use litellm_core::request_context::LiteLlmRequestContext;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use litellm_core::routing_utils::provider::{CustomLlmProvider, get_custom_llm_provider};
use serde_json::{Map, Value};

use super::common_utils::ocr_provider_config;
use super::hooks::OcrLifecycleHooks;
use super::types::{OcrRequest, PreparedOcrRequest};
use crate::integrations::custom_guardrail::CustomGuardrailRunner;
use crate::integrations::custom_logger::CustomLoggerRunner;

pub(crate) struct PreparedOcrCall {
    pub(crate) context: CallLifecycleContext,
    pub(crate) request: PreparedOcrRequest,
    pub(crate) hooks: OcrLifecycleHooks,
}

#[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
pub(crate) fn prepare_ocr_call(
    request: OcrRequest<'_>,
    context: &LiteLlmRequestContext,
    hooks: RequestHooks,
) -> PreparedOcrCall {
    let call_id = context
        .litellm_call_id
        .clone()
        .unwrap_or_else(new_ocr_call_id);
    let provider_info = get_custom_llm_provider(
        request.model,
        request.options.custom_llm_provider.as_deref(),
    )
    .unwrap_or(CustomLlmProvider {
        model: request.model,
        custom_llm_provider: "mistral",
    });
    let model = provider_info.model.to_string();
    let custom_llm_provider = provider_info.custom_llm_provider.to_string();
    let config = ocr_provider_config(&custom_llm_provider, &model)
        .ok_or_else(|| litellm_core::Error::InvalidProvider(custom_llm_provider.clone()))
        .and_then(|config| {
            validate_request_format(config, &request.optional_params, &custom_llm_provider)?;
            Ok(config)
        });
    let optional_params = match &config {
        Ok(config) => {
            let supported = config.supported_ocr_params();
            config.map_ocr_params(
                &request
                    .optional_params
                    .iter()
                    .filter(|(name, _)| supported.contains(&name.as_str()))
                    .map(|(name, value)| (name.clone(), value.clone()))
                    .collect(),
            )
        }
        Err(_) => request.optional_params,
    };

    PreparedOcrCall {
        context: CallLifecycleContext::new(
            "ocr",
            model.clone(),
            custom_llm_provider.clone(),
            call_id,
        ),
        request: PreparedOcrRequest {
            config,
            model,
            custom_llm_provider,
            document: request.document,
            provider_connection: request.options.provider_connection,
            api_key: request.options.api_key,
            api_base: request.options.api_base,
            extra_headers: request.options.extra_headers,
            optional_params,
            timeout: request.options.timeout,
        },
        hooks: OcrLifecycleHooks::new(
            CustomLoggerRunner::new(hooks.callbacks),
            CustomGuardrailRunner::new(hooks.guardrails),
            context.attribution.clone(),
        ),
    }
}

fn validate_request_format(
    config: &'static dyn litellm_core::ocr::transformation::OcrProviderConfig,
    optional_params: &Map<String, Value>,
    provider: &str,
) -> Result<(), litellm_core::Error> {
    let Some(format) = optional_params.get("req_format") else {
        return Ok(());
    };
    match format.as_str() {
        Some("litellm") => Ok(()),
        Some("native") if config.supported_ocr_params().contains(&"req_format") => Ok(()),
        Some("native") => Err(litellm_core::Error::InvalidRequest(format!(
            "`req_format=native` is not supported for provider {provider}"
        ))),
        _ => Err(litellm_core::Error::InvalidRequest(format!(
            "Invalid `req_format`: {format}. Expected `litellm` or `native`"
        ))),
    }
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

#[cfg(test)]
mod tests {
    use crate::integrations::types::RequestHooks;
    use litellm_core::error::Error;
    use litellm_core::request_context::LiteLlmRequestContext;
    use litellm_core::request_options::RequestOptions;
    use serde_json::{Map, json};

    use super::{OcrRequest, prepare_ocr_call};

    fn base_ocr_request(model: &str) -> OcrRequest<'_> {
        OcrRequest {
            model,
            document: json!({
                "type": "document_url",
                "document_url": "https://example.com/doc.pdf"
            }),
            optional_params: Map::new(),

            options: RequestOptions {
                api_key: (Some("sk-test")).map(|value| value.to_string()),
                api_base: None,
                custom_llm_provider: None,
                extra_headers: None,
                timeout: None,
                ..Default::default()
            },
        }
    }

    fn request_with_format(format: &str) -> OcrRequest<'_> {
        let mut request = base_ocr_request("mistral/mistral-ocr-latest");
        request.optional_params = Map::from_iter([("req_format".to_string(), json!(format))]);
        request
    }

    #[test]
    fn native_format_rejected_for_provider_without_support_as_bad_request() {
        let prepared = prepare_ocr_call(
            request_with_format("native"),
            &LiteLlmRequestContext {
                ..Default::default()
            },
            RequestHooks {
                ..Default::default()
            },
        );
        assert!(
            matches!(prepared.request.config, Err(Error::InvalidRequest(message)) if message.contains("not supported for provider"))
        );
    }

    #[test]
    fn unknown_format_rejected_for_provider_without_support_as_bad_request() {
        let prepared = prepare_ocr_call(
            request_with_format("raw"),
            &LiteLlmRequestContext {
                ..Default::default()
            },
            RequestHooks {
                ..Default::default()
            },
        );
        assert!(
            matches!(prepared.request.config, Err(Error::InvalidRequest(message)) if message.contains("Invalid `req_format`"))
        );
    }
}
