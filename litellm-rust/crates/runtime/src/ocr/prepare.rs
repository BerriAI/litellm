use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use litellm_core::CoreResult;
use litellm_core::error::CoreError;
use litellm_core::routing_utils::provider::{CustomLlmProvider, get_custom_llm_provider};

use super::common_utils::{convert_document_url_to_data_uri, has_header, string_headers};
use super::provider::ocr_provider_config;
use super::types::{OcrAuthStrategy, OcrRequest, PreparedOcrRequest, ProviderOcrRequest};

pub fn prepare_ocr_request(request: OcrRequest<'_>) -> PreparedOcrRequest {
    let litellm_call_id = request
        .litellm_call_id
        .map(str::to_string)
        .unwrap_or_else(new_ocr_call_id);
    let provider_info = get_custom_llm_provider(request.model, request.custom_llm_provider)
        .unwrap_or(CustomLlmProvider {
            model: request.model,
            custom_llm_provider: "mistral",
        });

    PreparedOcrRequest {
        model: provider_info.model.to_string(),
        custom_llm_provider: provider_info.custom_llm_provider.to_string(),
        litellm_call_id,
        document: request.document,
        api_key: request.api_key.map(str::to_string),
        api_base: request.api_base.map(str::to_string),
        extra_headers: request.extra_headers,
        optional_params: request.optional_params,
        timeout: request.timeout,
    }
}

pub async fn prepare_provider_request(
    request: PreparedOcrRequest,
) -> CoreResult<ProviderOcrRequest> {
    let config = ocr_provider_config(&request.custom_llm_provider, &request.model)
        .ok_or_else(|| CoreError::InvalidProvider(request.custom_llm_provider.clone()))?;
    let env_lookup = |key: &str| std::env::var(key).ok();
    let headers = string_headers(request.extra_headers)?;
    let auth_strategy = config.auth_strategy();
    let api_key = (!has_header(&headers, auth_strategy.header_name()))
        .then(|| config.resolve_api_key(request.api_key.as_deref(), &env_lookup))
        .transpose()?;
    let url = config.complete_url(
        request.api_base.as_deref(),
        &request.model,
        &request.optional_params,
        &env_lookup,
    )?;
    let transformation = config.transformation();
    let supported_params = transformation.get_supported_ocr_params();
    let non_default_params = request
        .optional_params
        .iter()
        .filter(|(param, _)| supported_params.contains(&param.as_str()))
        .map(|(param, value)| (param.clone(), value.clone()))
        .collect();
    let filtered_params = transformation.map_ocr_params(&non_default_params);
    let document = if config.requires_data_uri_document() {
        convert_document_url_to_data_uri(request.document, request.timeout).await?
    } else {
        request.document
    };
    let body = transformation
        .transform_ocr_request(&request.model, document, filtered_params)?
        .data;

    Ok(ProviderOcrRequest {
        model: request.model,
        config,
        url,
        body,
        upstream_headers: upstream_headers(&headers, auth_strategy, api_key.as_deref()),
        timeout: request.timeout,
    })
}

fn upstream_headers(
    headers: &[(String, String)],
    auth_strategy: OcrAuthStrategy,
    api_key: Option<&str>,
) -> Vec<(String, String)> {
    api_key
        .map(|api_key| match auth_strategy {
            OcrAuthStrategy::Bearer => ("Authorization".to_string(), format!("Bearer {api_key}")),
            OcrAuthStrategy::Header(header_name) => (header_name.to_string(), api_key.to_string()),
        })
        .into_iter()
        .chain(headers.iter().cloned())
        .collect()
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
