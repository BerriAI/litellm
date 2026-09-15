use crate::llms::azure_ai::ocr::cohere_parse_transformation::AzureAICohereParseConfig;
use crate::llms::azure_ai::ocr::document_intelligence::transformation::AzureDocumentIntelligenceOCRConfig;
use crate::llms::azure_ai::ocr::transformation::AzureAIOCRConfig;
use crate::llms::base_llm::ocr::transformation::BaseOcrConfig;
use crate::llms::cohere::ocr::transformation::CohereParseConfig;
use crate::llms::mistral::ocr::transformation::MistralOCRConfig;
use crate::llms::reducto::ocr::transformation::{ReductoParseLegacyConfig, ReductoParseV3Config};
use crate::llms::vertex_ai::ocr::deepseek_transformation::VertexAIDeepSeekOCRConfig;
use crate::llms::vertex_ai::ocr::transformation::VertexAIOCRConfig;
use crate::routing_utils::provider::{CustomLlmProvider, get_custom_llm_provider};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum OcrConfigKind {
    Cohere,
    Mistral,
    AzureAi,
    AzureCohere,
    AzureDocumentIntelligence,
    ReductoLegacy,
    ReductoV3,
    VertexAi,
    VertexDeepSeek,
}

impl OcrConfigKind {
    pub(crate) const fn provider(self) -> OcrProvider {
        match self {
            Self::Cohere => OcrProvider::Cohere,
            Self::Mistral => OcrProvider::Mistral,
            Self::AzureAi | Self::AzureCohere | Self::AzureDocumentIntelligence => {
                OcrProvider::AzureAi
            }
            Self::ReductoLegacy | Self::ReductoV3 => OcrProvider::Reducto,
            Self::VertexAi | Self::VertexDeepSeek => OcrProvider::VertexAi,
        }
    }

    pub(crate) fn get_supported_ocr_params(self, model: &str) -> &'static [&'static str] {
        match self {
            Self::Cohere => CohereParseConfig.get_supported_ocr_params(model),
            Self::Mistral => MistralOCRConfig.get_supported_ocr_params(model),
            Self::AzureAi => AzureAIOCRConfig.get_supported_ocr_params(model),
            Self::AzureCohere => AzureAICohereParseConfig.get_supported_ocr_params(model),
            Self::AzureDocumentIntelligence => {
                AzureDocumentIntelligenceOCRConfig.get_supported_ocr_params(model)
            }
            Self::ReductoLegacy => ReductoParseLegacyConfig.get_supported_ocr_params(model),
            Self::ReductoV3 => ReductoParseV3Config.get_supported_ocr_params(model),
            Self::VertexAi => VertexAIOCRConfig.get_supported_ocr_params(model),
            Self::VertexDeepSeek => VertexAIDeepSeekOCRConfig.get_supported_ocr_params(model),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum OcrProvider {
    Cohere,
    Mistral,
    AzureAi,
    Reducto,
    VertexAi,
}

impl OcrProvider {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Cohere => "cohere",
            Self::Mistral => "mistral",
            Self::AzureAi => "azure_ai",
            Self::Reducto => "reducto",
            Self::VertexAi => "vertex_ai",
        }
    }
}

pub(crate) fn resolve_provider_config(
    model: &str,
    custom_llm_provider: Option<&str>,
) -> Result<(String, OcrConfigKind), super::Error> {
    let provider =
        get_custom_llm_provider(model, custom_llm_provider).unwrap_or(CustomLlmProvider {
            model,
            custom_llm_provider: OcrProvider::Mistral.as_str(),
        });
    let config = match provider.custom_llm_provider {
        "cohere" => OcrConfigKind::Cohere,
        "mistral" => OcrConfigKind::Mistral,
        "azure_ai" if is_document_intelligence_model(provider.model) => {
            OcrConfigKind::AzureDocumentIntelligence
        }
        "azure_ai"
            if provider.model.to_ascii_lowercase().contains("cohere")
                && provider.model.to_ascii_lowercase().contains("parse") =>
        {
            OcrConfigKind::AzureCohere
        }
        "azure_ai" => OcrConfigKind::AzureAi,
        "reducto" if provider.model.eq_ignore_ascii_case("parse-legacy") => {
            OcrConfigKind::ReductoLegacy
        }
        "reducto" => OcrConfigKind::ReductoV3,
        "vertex_ai" if provider.model.to_ascii_lowercase().contains("deepseek") => {
            OcrConfigKind::VertexDeepSeek
        }
        "vertex_ai" => OcrConfigKind::VertexAi,
        value => return Err(super::Error::InvalidProvider(value.to_string())),
    };
    Ok((provider.model.to_string(), config))
}

fn is_document_intelligence_model(model: &str) -> bool {
    let model = model.to_ascii_lowercase();
    model.contains("doc-intelligence") || model.contains("documentintelligence")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_models_are_preserved_without_a_local_allowlist() {
        for (qualified_model, expected_config) in [
            ("mistral/future-ocr-model", OcrConfigKind::Mistral),
            ("azure_ai/future-ocr-model", OcrConfigKind::AzureAi),
        ] {
            let expected_model = qualified_model.split_once('/').unwrap().1;
            let (model, config) = resolve_provider_config(qualified_model, None).unwrap();
            assert_eq!(model, expected_model);
            assert_eq!(config, expected_config);
        }
    }

    #[test]
    fn provider_specific_models_select_their_config() {
        assert_eq!(
            resolve_provider_config("reducto/parse-legacy", None)
                .unwrap()
                .1,
            OcrConfigKind::ReductoLegacy
        );
        assert_eq!(
            resolve_provider_config("reducto/future-parse-model", None)
                .unwrap()
                .1,
            OcrConfigKind::ReductoV3
        );
        assert_eq!(
            resolve_provider_config("azure_ai/doc-intelligence/prebuilt-layout", None)
                .unwrap()
                .1,
            OcrConfigKind::AzureDocumentIntelligence
        );
    }
}
