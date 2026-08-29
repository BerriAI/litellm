use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::CoreResult;
use crate::call_lifecycle::CallContext;
use crate::error::CoreError;
use crate::http_utils::{has_header, string_headers};
use crate::ocr::transformation::OcrAuthStrategy;
use crate::routing_utils::provider::{CustomLlmProvider, get_custom_llm_provider};

use super::document_fetch::convert_document_url_to_data_uri;
use super::provider_config::ocr_provider_config;
use super::types::{OcrRequest, PreparedOcrRequest, ProviderOcrRequest};

pub(crate) struct PreparedOcrCall {
    pub(crate) context: CallContext,
    pub(crate) request: PreparedOcrRequest,
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
    let context = CallContext::new(&model, &custom_llm_provider, &call_id);
    let (private_params, optional_params) = request
        .optional_params
        .into_iter()
        .partition(|(key, _)| is_private_param(key));

    PreparedOcrCall {
        context,
        request: PreparedOcrRequest {
            model,
            custom_llm_provider,
            document: request.document,
            api_key: request.api_key.map(str::to_string),
            api_base: request.api_base.map(str::to_string),
            extra_headers: request.extra_headers,
            optional_params,
            private_params,
            timeout: request.timeout,
        },
    }
}

pub(crate) async fn prepare_provider_request(
    request: PreparedOcrRequest,
) -> CoreResult<ProviderOcrRequest> {
    if let Some(key) = request
        .optional_params
        .keys()
        .find(|key| is_private_param(key))
    {
        return Err(CoreError::invalid_request(format!(
            "OCR interceptor cannot set private parameter '{key}'"
        )));
    }
    let optional_params = request
        .private_params
        .into_iter()
        .chain(request.optional_params)
        .collect();
    let config = ocr_provider_config(&request.custom_llm_provider, &request.model)
        .ok_or_else(|| CoreError::invalid_provider(request.custom_llm_provider.clone()))?;
    let env_lookup = |key: &str| std::env::var(key).ok();
    let headers = string_headers("OCR", request.extra_headers)?;
    let auth_strategy = config.auth_strategy();
    let api_key = (!has_header(&headers, auth_strategy.header_name()))
        .then(|| config.resolve_api_key(request.api_key.as_deref(), &env_lookup))
        .transpose()?;
    let url = config.complete_url(
        request.api_base.as_deref(),
        &request.model,
        &optional_params,
        &env_lookup,
    )?;
    let filtered_params = config.map_ocr_params(&optional_params);
    let document = if config.requires_data_uri_document() {
        convert_document_url_to_data_uri(request.document).await?
    } else {
        request.document
    };
    let document = serde_json::to_value(document)
        .map_err(|error| CoreError::invalid_request(format!("invalid OCR document: {error}")))?;
    let body = config
        .transform_ocr_request(&request.model, document, filtered_params)?
        .data;

    Ok(ProviderOcrRequest {
        model: request.model,
        custom_llm_provider: request.custom_llm_provider,
        config,
        url,
        body,
        upstream_headers: upstream_headers(&headers, auth_strategy, api_key.as_deref()),
        timeout: request.timeout,
    })
}

fn is_private_param(key: &str) -> bool {
    matches!(
        key,
        "vertex_project" | "vertex_ai_project" | "vertex_location" | "vertex_ai_location"
    )
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
