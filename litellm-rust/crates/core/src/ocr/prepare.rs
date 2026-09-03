use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::Error;
use crate::http_utils::string_headers;
use crate::ocr::common_utils::convert_document_url_to_data_uri;
use crate::ocr::transformation::OcrProviderConfig;
use crate::ocr::types::{OcrRequest, PreparedOcrRequest, ProviderOcrRequest};
use crate::providers::azure_ai::ocr::transformation::{
    AZURE_AI_OCR_CONFIG, AZURE_DOCUMENT_INTELLIGENCE_OCR_CONFIG,
};
use crate::providers::mistral::ocr::transformation::MISTRAL_OCR_CONFIG;
use crate::providers::vertex_ai::ocr::transformation as vertex_ai;
use crate::providers::vertex_ai::ocr::transformation::{
    VERTEX_AI_DEEPSEEK_OCR_CONFIG, VERTEX_AI_OCR_CONFIG,
};
use crate::routing_utils::provider::{CustomLlmProvider, get_custom_llm_provider};

#[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
pub(crate) fn ocr_provider_config(
    provider: &str,
    model: &str,
) -> Option<&'static dyn OcrProviderConfig> {
    match provider {
        "mistral" => Some(&MISTRAL_OCR_CONFIG),
        "azure_ai" if is_azure_document_intelligence_model(model) => {
            Some(&AZURE_DOCUMENT_INTELLIGENCE_OCR_CONFIG)
        }
        "azure_ai" => Some(&AZURE_AI_OCR_CONFIG),
        "vertex_ai" if vertex_ai::is_deepseek_model(model) => Some(&VERTEX_AI_DEEPSEEK_OCR_CONFIG),
        "vertex_ai" => Some(&VERTEX_AI_OCR_CONFIG),
        _ => None,
    }
}

fn is_azure_document_intelligence_model(model: &str) -> bool {
    let model = model.to_ascii_lowercase();
    model.contains("doc-intelligence") || model.contains("documentintelligence")
}

#[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
pub fn prepare_ocr_call(request: OcrRequest<'_>) -> PreparedOcrRequest {
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
    let config = ocr_provider_config(&custom_llm_provider, &model)
        .ok_or_else(|| Error::InvalidProvider(custom_llm_provider.clone()));
    let optional_params = match &config {
        Ok(config) => {
            let supported = config.supported_ocr_params();
            config.map_ocr_params(
                &request
                    .optional_params
                    .into_iter()
                    .filter(|(name, _)| supported.contains(&name.as_str()))
                    .collect(),
            )
        }
        Err(_) => request.optional_params,
    };

    PreparedOcrRequest {
        config,
        model,
        custom_llm_provider,
        litellm_call_id: call_id,
        document: request.document,
        api_key: request.api_key.map(str::to_string),
        api_base: request.api_base.map(str::to_string),
        extra_headers: request.extra_headers,
        optional_params,
        timeout: request.timeout,
    }
}

#[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
pub async fn prepare_ocr_provider_call(
    request: PreparedOcrRequest,
) -> Result<ProviderOcrRequest, Error> {
    let config = request.config?;
    let env_lookup = |key: &str| std::env::var(key).ok();
    let upstream_headers = config.validate_environment(
        string_headers("OCR", request.extra_headers)?,
        request.api_key.as_deref(),
        &env_lookup,
    )?;
    let url = config.complete_url(
        request.api_base.as_deref(),
        &request.model,
        &request.optional_params,
        &env_lookup,
    )?;
    let model = request.model.clone();
    let custom_llm_provider = request.custom_llm_provider.clone();
    let document = if config.requires_data_uri_document() {
        convert_document_url_to_data_uri(request.document).await?
    } else {
        request.document
    };
    let body = config
        .transform_ocr_request(&request.model, document, request.optional_params)?
        .data;
    Ok(ProviderOcrRequest {
        model,
        custom_llm_provider,
        config,
        url,
        body,
        upstream_headers,
        timeout: request.timeout,
    })
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
