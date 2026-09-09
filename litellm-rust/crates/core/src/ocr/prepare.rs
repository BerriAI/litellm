use super::transformation::OcrProviderConfig;
use super::types::{OcrConnection, OcrDocument};
use crate::Error;
use crate::providers::azure_ai::ocr::document_intelligence::types::DocumentIntelligenceInputParams;
use crate::providers::mistral::ocr::types::MistralOcrParams;
use crate::providers::reducto::ocr::types::{ReductoLegacyParams, ReductoV3Params};
use crate::providers::vertex_ai::ocr::deepseek::types::DeepSeekOcrParams;
use serde::Serialize;

#[derive(Clone, Debug, Serialize)]
#[serde(untagged)]
pub enum OcrProviderRequest {
    Mistral(MistralOcrParams),
    AzureAi(MistralOcrParams),
    AzureDocumentIntelligence(DocumentIntelligenceInputParams),
    VertexAi(MistralOcrParams),
    VertexAiDeepSeek(DeepSeekOcrParams),
    ReductoV3(ReductoV3Params),
    ReductoLegacy(ReductoLegacyParams),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OcrProviderKind {
    Mistral,
    AzureAi,
    AzureDocumentIntelligence,
    VertexAi,
    VertexAiDeepSeek,
    ReductoV3,
    ReductoLegacy,
}
impl OcrProviderKind {
    pub fn provider_name(self) -> &'static str {
        match self {
            Self::Mistral => "mistral",
            Self::AzureAi | Self::AzureDocumentIntelligence => "azure_ai",
            Self::VertexAi | Self::VertexAiDeepSeek => "vertex_ai",
            Self::ReductoV3 | Self::ReductoLegacy => "reducto",
        }
    }
}
impl OcrProviderRequest {
    pub fn kind(&self) -> OcrProviderKind {
        match self {
            Self::Mistral(_) => OcrProviderKind::Mistral,
            Self::AzureAi(_) => OcrProviderKind::AzureAi,
            Self::AzureDocumentIntelligence(_) => OcrProviderKind::AzureDocumentIntelligence,
            Self::VertexAi(_) => OcrProviderKind::VertexAi,
            Self::VertexAiDeepSeek(_) => OcrProviderKind::VertexAiDeepSeek,
            Self::ReductoV3(_) => OcrProviderKind::ReductoV3,
            Self::ReductoLegacy(_) => OcrProviderKind::ReductoLegacy,
        }
    }
}

#[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
pub fn ocr_provider_config(provider: &str, model: &str) -> Result<OcrProviderKind, Error> {
    let lower = model.to_ascii_lowercase();
    match provider {
        "mistral" => Ok(OcrProviderKind::Mistral),
        "azure_ai"
            if lower.contains("doc-intelligence") || lower.contains("documentintelligence") =>
        {
            Ok(OcrProviderKind::AzureDocumentIntelligence)
        }
        "azure_ai" => Ok(OcrProviderKind::AzureAi),
        "vertex_ai" if lower.contains("deepseek") => Ok(OcrProviderKind::VertexAiDeepSeek),
        "vertex_ai" => Ok(OcrProviderKind::VertexAi),
        "reducto" if model == "parse-v3" => Ok(OcrProviderKind::ReductoV3),
        "reducto" if model == "parse-legacy" => Ok(OcrProviderKind::ReductoLegacy),
        _ => Err(Error::InvalidProvider(provider.into())),
    }
}

pub(crate) struct PreparedOcrRequest<C: OcrProviderConfig> {
    pub config: C,
    pub model: String,
    pub document: OcrDocument,
    pub params: C::MappedParams,
    pub connection: OcrConnection,
}

#[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
pub(crate) fn prepare_ocr_call<C: OcrProviderConfig>(
    config: C,
    model: String,
    document: OcrDocument,
    params: C::InputParams,
    connection: OcrConnection,
) -> Result<PreparedOcrRequest<C>, Error> {
    let params = config.map_ocr_params(params)?;
    Ok(PreparedOcrRequest {
        config,
        model,
        document,
        params,
        connection,
    })
}
