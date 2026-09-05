use std::time::Duration;

use reqwest::Url;
use serde_json::{Map, Value};

use crate::Error;
use crate::auth::AuthPreflight;
use crate::http_utils::string_headers;
use crate::providers::azure_ai::ocr::transformation::{
    AZURE_AI_OCR_CONFIG, AZURE_DOCUMENT_INTELLIGENCE_OCR_CONFIG,
};
use crate::providers::mistral::ocr::transformation::MISTRAL_OCR_CONFIG;
use crate::routing_utils::provider::{CustomLlmProvider, get_custom_llm_provider};

use super::transformation::OcrProviderConfig;
use super::types::{OcrAuthentication, OcrDocument, OcrRequest};

pub struct OcrPlan {
    pub(super) model: String,
    pub(super) provider: String,
    pub(super) config: &'static dyn OcrProviderConfig,
    pub(super) document: OcrDocument,
    pub(super) optional_params: Map<String, Value>,
    pub(super) url: Url,
    pub(super) auth: OcrAuthentication,
    pub(super) timeout: Option<Duration>,
}

pub(super) fn provider_config(
    model: &str,
    provider: &str,
) -> Option<&'static dyn OcrProviderConfig> {
    match provider {
        "mistral" => Some(&MISTRAL_OCR_CONFIG),
        "azure_ai"
            if model.to_ascii_lowercase().contains("doc-intelligence")
                || model.to_ascii_lowercase().contains("documentintelligence") =>
        {
            Some(&AZURE_DOCUMENT_INTELLIGENCE_OCR_CONFIG)
        }
        "azure_ai" => Some(&AZURE_AI_OCR_CONFIG),
        _ => None,
    }
}

pub fn parameter_names() -> impl Iterator<Item = &'static str> {
    [
        &MISTRAL_OCR_CONFIG as &dyn OcrProviderConfig,
        &AZURE_AI_OCR_CONFIG,
        &AZURE_DOCUMENT_INTELLIGENCE_OCR_CONFIG,
    ]
    .into_iter()
    .flat_map(|config| config.supported_ocr_params().iter().copied())
    .chain(std::iter::once("req_format"))
}

pub fn decode_document(document: Value) -> AuthPreflight<OcrDocument> {
    match serde_json::from_value(document) {
        Ok(document) => AuthPreflight::Ready(document),
        Err(_) => {
            AuthPreflight::Declined("document shape is not supported by the native OCR consumer")
        }
    }
}

pub fn preflight(
    request: OcrRequest<'_>,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Result<AuthPreflight<OcrPlan>, Error> {
    let Some(provider) = get_custom_llm_provider(
        request.model,
        request.options.custom_llm_provider.as_deref(),
    )
    .or_else(|| {
        request
            .options
            .custom_llm_provider
            .as_deref()
            .map(|provider| CustomLlmProvider {
                model: request.model,
                custom_llm_provider: provider,
            })
    }) else {
        return Ok(AuthPreflight::Declined(
            "OCR provider requires Python resolution",
        ));
    };
    let Some(config) = provider_config(provider.model, provider.custom_llm_provider) else {
        return Ok(AuthPreflight::Declined(
            "OCR provider is not integrated with shared auth",
        ));
    };
    let source = match &request.document {
        OcrDocument::DocumentUrl { document_url } => document_url,
        OcrDocument::ImageUrl { image_url } => image_url,
    };
    if config.requires_data_uri_document() && !source.starts_with("data:") {
        return Ok(AuthPreflight::Declined(
            "OCR document download requires Python",
        ));
    }
    if request
        .optional_params
        .get("req_format")
        .is_some_and(|value| value != "litellm")
    {
        return Ok(AuthPreflight::Declined(
            "OCR response format requires Python",
        ));
    }
    let mut connection = request.options.provider_connection;
    let default_key = connection
        .remove("sdk_api_key")
        .and_then(|value| value.as_str().map(str::to_owned));
    let default_base = connection
        .remove("sdk_api_base")
        .and_then(|value| value.as_str().map(str::to_owned));
    let api_key = request
        .options
        .api_key
        .as_deref()
        .filter(|value| !value.trim().is_empty())
        .or(default_key.as_deref());
    let api_base = request
        .options
        .api_base
        .as_deref()
        .filter(|value| !value.trim().is_empty())
        .or(default_base.as_deref());
    let headers = string_headers("OCR", request.options.extra_headers)?;
    let auth = match config.select_auth(api_key, headers, &connection, env_lookup) {
        Ok(AuthPreflight::Ready(auth)) => auth,
        Ok(AuthPreflight::Declined(reason)) => return Ok(AuthPreflight::Declined(reason)),
        Err(Error::Auth(_)) if provider.custom_llm_provider == "azure_ai" && api_key.is_none() => {
            return Ok(AuthPreflight::Declined(
                "Azure credential discovery is not implemented",
            ));
        }
        Err(error) => return Err(error),
    };
    let optional_params = config.map_ocr_params(&request.optional_params);
    let url = config.complete_url(api_base, provider.model, &optional_params, env_lookup)?;
    let url =
        Url::parse(&url).map_err(|_| Error::InvalidRequest("invalid OCR provider URL".into()))?;
    if request
        .options
        .extra_query
        .is_some_and(|query| !query.is_empty())
    {
        return Ok(AuthPreflight::Declined(
            "OCR query overrides require Python",
        ));
    }
    Ok(AuthPreflight::Ready(OcrPlan {
        model: provider.model.to_owned(),
        provider: provider.custom_llm_provider.to_owned(),
        config,
        document: request.document,
        optional_params,
        url,
        auth,
        timeout: request.options.timeout,
    }))
}
