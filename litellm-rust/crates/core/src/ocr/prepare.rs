use super::transformation::OcrProviderConfig;
use super::types::{OcrConnection, OcrDocument};
use crate::Error;
use crate::providers::azure_ai::ocr::document_intelligence::types::DocumentIntelligenceInputParams;
use crate::providers::mistral::ocr::types::MistralOcrParams;
use crate::providers::reducto::ocr::types::{ReductoLegacyParams, ReductoV3Params};
use serde::Serialize;
use strum::EnumString;

#[derive(Clone, Debug, Serialize)]
#[serde(untagged)]
pub enum OcrProviderRequest {
    Mistral(MistralOcrParams),
    AzureAi(MistralOcrParams),
    AzureDocumentIntelligence(DocumentIntelligenceInputParams),
    ReductoV3(ReductoV3Params),
    ReductoLegacy(ReductoLegacyParams),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OcrProviderKind {
    Mistral,
    AzureAi,
    AzureDocumentIntelligence,
    ReductoV3,
    ReductoLegacy,
}

#[derive(Clone, Copy, Debug, EnumString, PartialEq, Eq)]
#[strum(serialize_all = "snake_case")]
pub enum OcrProvider {
    Mistral,
    AzureAi,
    Reducto,
}

#[derive(Clone, Debug, EnumString, PartialEq, Eq)]
pub enum OcrModel {
    #[strum(serialize = "parse-v3")]
    ReductoV3,
    #[strum(serialize = "parse-legacy")]
    ReductoLegacy,
    #[strum(default)]
    Passthrough(String),
}

impl OcrModel {
    pub fn as_str(&self) -> &str {
        match self {
            Self::ReductoV3 => "parse-v3",
            Self::ReductoLegacy => "parse-legacy",
            Self::Passthrough(model) => model,
        }
    }
}

impl OcrProviderKind {
    pub fn provider_name(self) -> &'static str {
        match self {
            Self::Mistral => "mistral",
            Self::AzureAi | Self::AzureDocumentIntelligence => "azure_ai",
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
            Self::ReductoV3(_) => OcrProviderKind::ReductoV3,
            Self::ReductoLegacy(_) => OcrProviderKind::ReductoLegacy,
        }
    }
}

#[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
pub fn ocr_provider_config(provider: OcrProvider, model: &OcrModel) -> OcrProviderKind {
    let lower = model.as_str().to_ascii_lowercase();
    match (provider, model) {
        (OcrProvider::Mistral, _) => OcrProviderKind::Mistral,
        (OcrProvider::AzureAi, _)
            if lower.contains("doc-intelligence") || lower.contains("documentintelligence") =>
        {
            OcrProviderKind::AzureDocumentIntelligence
        }
        (OcrProvider::AzureAi, _) => OcrProviderKind::AzureAi,
        (OcrProvider::Reducto, OcrModel::ReductoLegacy) => OcrProviderKind::ReductoLegacy,
        (OcrProvider::Reducto, _) => OcrProviderKind::ReductoV3,
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

#[cfg(test)]
#[path = "../../tests/ocr_prepare.rs"]
mod tests;
