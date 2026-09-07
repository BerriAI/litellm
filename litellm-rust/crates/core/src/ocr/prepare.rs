use std::time::Duration;

use serde_json::{Map, Value};

use crate::Error;
use crate::providers::azure_ai::ocr::transformation as azure_ai;
use crate::providers::mistral::ocr::transformation::MISTRAL_OCR_CONFIG;
use crate::providers::vertex_ai::ocr::transformation as vertex_ai;
use crate::routing_utils::provider::get_custom_llm_provider;

use super::transformation::{OcrProviderConfig, OcrResponseHandling};
use super::types::OcrRequest;
pub use super::types::PreparedOcr;

pub fn prepare(request: OcrRequest) -> Result<PreparedOcr, Error> {
    match request.request_format.as_deref() {
        None | Some("litellm") => {}
        Some("native") => return Err(Error::Unsupported("native OCR request format")),
        Some(_) => {
            return Err(Error::Unsupported(
                "OCR request format must be litellm or native",
            ));
        }
    }
    let provider = get_custom_llm_provider(&request.model, request.custom_llm_provider.as_deref())
        .ok_or_else(|| Error::InvalidProvider("unable to resolve OCR provider".into()))?;
    let config = provider_config(provider.custom_llm_provider, provider.model)?;
    if provider.model.trim().is_empty() {
        return Err(Error::InvalidRequest("OCR model must not be empty".into()));
    }
    Duration::try_from_secs_f64(request.timeout_seconds)
        .ok()
        .filter(|timeout| !timeout.is_zero())
        .ok_or_else(|| Error::InvalidRequest("timeout must be positive and finite".into()))?;

    let env_lookup = |key: &str| std::env::var(key).ok();
    let headers = config
        .validate_credentials(
            request.extra_headers,
            request.api_key.as_deref(),
            request.azure_ad_token.as_deref(),
            &env_lookup,
        )
        .map_err(
            |error| match (error, config.credential_acquisition_operation()) {
                (Error::Auth(_), Some(operation)) => Error::Unsupported(operation),
                (error, _) => error,
            },
        )?;
    validate_capabilities(config)?;
    request
        .document
        .validate(config.requires_data_uri_document())?;
    if request.stream {
        return Err(Error::Unsupported("OCR streaming response handling"));
    }
    let url_params = [
        ("vertex_project", request.vertex_project),
        ("vertex_location", request.vertex_location),
    ]
    .into_iter()
    .filter_map(|(name, value)| value.map(|value| (name.into(), Value::String(value))))
    .collect();
    let url = config.complete_url(
        request.api_base.as_deref(),
        provider.model,
        &url_params,
        &env_lookup,
    )?;
    let parsed_url = reqwest::Url::parse(&url)
        .map_err(|_| Error::InvalidRequest("invalid OCR API URL".into()))?;
    if !matches!(parsed_url.scheme(), "http" | "https") || parsed_url.host_str().is_none() {
        return Err(Error::InvalidRequest("invalid OCR API URL".into()));
    }
    let document = serde_json::to_value(request.document)
        .map_err(|_| Error::InvalidRequest("could not project OCR document".into()))?;
    let template = config.transform_ocr_request(provider.model, document, Map::new())?;
    if template.files.is_some() {
        return Err(Error::Unsupported("OCR multipart document preparation"));
    }
    let Value::Object(body) = template.data else {
        return Err(Error::Unsupported("non-object OCR request template"));
    };
    Ok(PreparedOcr {
        model: provider.model.to_string(),
        custom_llm_provider: provider.custom_llm_provider.to_string(),
        url,
        headers,
        body,
        document_projection: config.document_projection(),
        parameter_fields: config.supported_ocr_params(),
        timeout_seconds: request.timeout_seconds,
    })
}

pub(super) fn validate_capabilities(config: &dyn OcrProviderConfig) -> Result<(), Error> {
    match config.response_handling() {
        OcrResponseHandling::Json => Ok(()),
        OcrResponseHandling::AzureDocumentIntelligencePoll => Err(Error::Unsupported(
            "Azure Document Intelligence OCR polling",
        )),
    }
}

pub(super) fn provider_config(
    provider: &str,
    model: &str,
) -> Result<&'static dyn OcrProviderConfig, Error> {
    match provider {
        "mistral" => Ok(&MISTRAL_OCR_CONFIG),
        "azure_ai" => azure_ai::config_for_model(model),
        "vertex_ai" => vertex_ai::config_for_model(model),
        _ => Err(Error::Unsupported("OCR provider")),
    }
}
