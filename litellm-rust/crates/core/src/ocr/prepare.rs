use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::Error;
use crate::auth::{
    AuthorizationProvider, BearerAuthorizationProvider, CredentialProvenance, SecretString,
    StaticHeaderAuthorizationProvider, SystemClock, TokenProvider,
};
use crate::http_utils::string_headers;
use crate::ocr::common_utils::convert_document_url_to_data_uri;
use crate::ocr::transformation::{OcrAuthStrategy, OcrProviderConfig};
use crate::ocr::types::{OcrRequest, PendingOcrUpload, PreparedOcrRequest, ProviderOcrRequest};
use crate::providers::azure_ai::ocr::transformation::{
    AZURE_AI_OCR_CONFIG, AZURE_DOCUMENT_INTELLIGENCE_OCR_CONFIG,
};
use crate::providers::mistral::ocr::transformation::MISTRAL_OCR_CONFIG;
use crate::providers::reducto::ocr::transformation as reducto;
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
        "reducto" => reducto::config_for_model(model),
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
    let mut context = request.context;
    context
        .litellm_call_id
        .get_or_insert_with(|| call_id.clone());
    context
        .request_model
        .get_or_insert_with(|| request.model.to_string());
    let provider_info = get_custom_llm_provider(request.model, request.custom_llm_provider)
        .unwrap_or(CustomLlmProvider {
            model: request.model,
            custom_llm_provider: "mistral",
        });
    let model = provider_info.model.to_string();
    let custom_llm_provider = provider_info.custom_llm_provider.to_string();
    let config = ocr_provider_config(&custom_llm_provider, &model)
        .ok_or_else(|| Error::InvalidProvider(custom_llm_provider.clone()));
    let mut provider_params = request.optional_params;
    provider_params.retain(|name, _| {
        !matches!(
            name.as_str(),
            "callbacks"
                | "litellm_logging_obj"
                | "litellm_call_id"
                | "max_retries"
                | "metadata"
                | "num_retries"
                | "tags"
        )
    });
    let optional_params = match &config {
        Ok(config) => config.map_ocr_params(&provider_params),
        Err(_) => provider_params,
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
        context,
    }
}

#[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
pub async fn prepare_ocr_provider_call(
    request: PreparedOcrRequest,
) -> Result<ProviderOcrRequest, Error> {
    prepare_ocr_provider_call_with_token(request, None).await
}

#[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
pub async fn prepare_ocr_provider_call_with_token(
    request: PreparedOcrRequest,
    token_provider: Option<Arc<dyn TokenProvider>>,
) -> Result<ProviderOcrRequest, Error> {
    let config = request.config?;
    let env_lookup = |key: &str| std::env::var(key).ok();
    let url = config.complete_url(
        request.api_base.as_deref(),
        &request.model,
        &request.optional_params,
        &env_lookup,
    )?;
    let mut upstream_headers = string_headers("OCR", request.extra_headers)?;
    let authorization = select_authorization(
        config,
        &mut upstream_headers,
        request.api_key.as_deref(),
        token_provider,
        &env_lookup,
    )?;
    let model = request.model.clone();
    let custom_llm_provider = request.custom_llm_provider.clone();
    let document = if config.requires_data_uri_document() {
        convert_document_url_to_data_uri(request.document).await?
    } else {
        request.document
    };
    let pending_upload = if request.custom_llm_provider == "reducto" {
        match reducto::extract_document_source(&document)? {
            reducto::ReductoDocumentSource::FileId(_) => None,
            reducto::ReductoDocumentSource::Upload { bytes, mime_type } => Some(PendingOcrUpload {
                url: reducto::upload_url(request.api_base.as_deref()),
                bytes,
                mime_type,
            }),
        }
    } else {
        None
    };
    let body = if pending_upload.is_none() {
        Some(
            config
                .transform_ocr_request(&request.model, document, request.optional_params.clone())?
                .data,
        )
    } else {
        None
    };
    Ok(ProviderOcrRequest {
        model,
        custom_llm_provider,
        config,
        url,
        body,
        pending_upload,
        optional_params: request.optional_params,
        upstream_headers,
        authorization,
        timeout: request.timeout,
        context: request.context,
    })
}

fn select_authorization(
    config: &dyn OcrProviderConfig,
    headers: &mut Vec<(String, String)>,
    api_key: Option<&str>,
    token_provider: Option<Arc<dyn TokenProvider>>,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Result<Arc<dyn AuthorizationProvider>, Error> {
    let strategy = config.auth_strategy();
    let selected_header = strategy.header_name();
    if let Some(value) = take_header(headers, selected_header) {
        return Ok(static_authorization(
            strategy,
            value,
            CredentialProvenance::ForwardedHeader(
                selected_header
                    .parse()
                    .expect("provider auth header is valid"),
            ),
        ));
    }

    match config.resolve_api_key(api_key, env_lookup) {
        Ok(key) => {
            let value = match strategy {
                OcrAuthStrategy::Bearer => format!("Bearer {key}"),
                OcrAuthStrategy::Header(_) => key,
            };
            return Ok(static_authorization(
                strategy,
                value,
                CredentialProvenance::CallerSupplied,
            ));
        }
        Err(Error::Auth(_)) => {}
        Err(error) => return Err(error),
    }

    if strategy != OcrAuthStrategy::Bearer
        && let Some(value) = take_header(headers, "authorization")
    {
        return Ok(static_authorization(
            OcrAuthStrategy::Bearer,
            value,
            CredentialProvenance::ForwardedHeader(reqwest::header::AUTHORIZATION),
        ));
    }
    let provider = token_provider.ok_or_else(|| Error::Auth("missing OCR credential".into()))?;
    Ok(Arc::new(BearerAuthorizationProvider::new(
        provider,
        Arc::new(SystemClock),
        auth_conflicts(OcrAuthStrategy::Bearer),
        CredentialProvenance::ExternalProvider,
    )))
}

fn static_authorization(
    strategy: OcrAuthStrategy,
    value: String,
    provenance: CredentialProvenance,
) -> Arc<dyn AuthorizationProvider> {
    Arc::new(StaticHeaderAuthorizationProvider::new(
        strategy
            .header_name()
            .parse()
            .expect("provider auth header is valid"),
        SecretString::new(value),
        auth_conflicts(strategy),
        provenance,
    ))
}

fn auth_conflicts(strategy: OcrAuthStrategy) -> Vec<reqwest::header::HeaderName> {
    match strategy {
        OcrAuthStrategy::Bearer => vec![
            "ocp-apim-subscription-key"
                .parse()
                .expect("Azure subscription header is valid"),
        ],
        OcrAuthStrategy::Header(_) => vec![reqwest::header::AUTHORIZATION],
    }
}

fn take_header(headers: &mut Vec<(String, String)>, name: &str) -> Option<String> {
    let index = headers
        .iter()
        .position(|(candidate, _)| candidate.eq_ignore_ascii_case(name))?;
    Some(headers.remove(index).1)
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
