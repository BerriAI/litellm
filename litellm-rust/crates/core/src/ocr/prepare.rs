use serde_json::{Map, Value};

use crate::error::Error;
use crate::ocr::transformation::OcrProviderConfig;
use crate::ocr::types::OcrAuthInputs;
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

pub struct PreparedOcrProvider {
    pub config: Result<&'static dyn OcrProviderConfig, Error>,
    pub model: String,
    pub custom_llm_provider: String,
    pub optional_params: Map<String, Value>,
    pub auth_inputs: OcrAuthInputs,
}

pub fn prepare_ocr_provider(
    model: &str,
    custom_llm_provider: Option<&str>,
    optional_params: Map<String, Value>,
) -> PreparedOcrProvider {
    let provider_info =
        get_custom_llm_provider(model, custom_llm_provider).unwrap_or(CustomLlmProvider {
            model,
            custom_llm_provider: "mistral",
        });
    let model = provider_info.model.to_string();
    let custom_llm_provider = provider_info.custom_llm_provider.to_string();
    let parsed_config = ocr_provider_config(&custom_llm_provider, &model)
        .ok_or_else(|| Error::InvalidProvider(custom_llm_provider.clone()))
        .and_then(|config| {
            validate_request_format(config, &optional_params, &custom_llm_provider)?;
            Ok(config)
        });
    let (config, auth_inputs) = match parsed_config {
        Ok(config) => match config.parse_auth_inputs(&optional_params) {
            Ok(auth_inputs) => (Ok(config), auth_inputs),
            Err(error) => (Err(error.into()), Default::default()),
        },
        Err(error) => (Err(error), Default::default()),
    };
    let optional_params = match &config {
        Ok(config) => map_ocr_params(*config, &optional_params),
        Err(_) => optional_params,
    };

    PreparedOcrProvider {
        config,
        model,
        custom_llm_provider,
        optional_params,
        auth_inputs,
    }
}

pub fn map_ocr_params(
    config: &'static dyn OcrProviderConfig,
    optional_params: &Map<String, Value>,
) -> Map<String, Value> {
    let supported = config.supported_ocr_params();
    config.map_ocr_params(
        &optional_params
            .iter()
            .filter(|(name, _)| supported.contains(&name.as_str()))
            .map(|(name, value)| (name.clone(), value.clone()))
            .collect(),
    )
}

#[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
pub fn ocr_provider_config(provider: &str, model: &str) -> Option<&'static dyn OcrProviderConfig> {
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

fn validate_request_format(
    config: &'static dyn OcrProviderConfig,
    optional_params: &Map<String, Value>,
    provider: &str,
) -> Result<(), Error> {
    let Some(format) = optional_params.get("req_format") else {
        return Ok(());
    };
    match format.as_str() {
        Some("litellm") => Ok(()),
        Some("native") if config.supported_ocr_params().contains(&"req_format") => Ok(()),
        Some("native") => Err(Error::InvalidRequest(format!(
            "`req_format=native` is not supported for provider {provider}"
        ))),
        _ => Err(Error::InvalidRequest(format!(
            "Invalid `req_format`: {format}. Expected `litellm` or `native`"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use crate::ocr::transformation::OcrResponseHandling;

    use super::ocr_provider_config;

    #[test]
    fn provider_selection_is_owned_by_core() {
        assert!(ocr_provider_config("mistral", "mistral-ocr-latest").is_some());
        assert_eq!(
            ocr_provider_config("azure_ai", "doc-intelligence/prebuilt-read")
                .expect("Azure Document Intelligence config resolves")
                .response_handling(),
            OcrResponseHandling::AzureDocumentIntelligencePoll
        );
        assert!(ocr_provider_config("openai", "gpt-4o").is_none());
    }
}
