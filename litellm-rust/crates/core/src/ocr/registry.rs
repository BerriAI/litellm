use serde::Serialize;
use strum::EnumString;

use super::backends::OcrBackend;
use super::backends::azure_ai::AzureAiOcrBackend;
use super::backends::mistral::MistralOcrBackend;
use super::backends::reducto::ReductoOcrBackend;
use super::backends::vertex_ai::VertexAiOcrBackend;
use super::formats::OcrFormat;
use super::formats::deepseek::{DeepSeekOcrFormat, types::DeepSeekOcrParams};
use super::formats::document_intelligence::{
    AzureDocumentIntelligenceOcrFormat, types::DocumentIntelligenceInputParams,
};
use super::formats::mistral::{MistralOcrFormat, types::MistralOcrParams};
use super::formats::reducto::{
    ReductoParseLegacyFormat, ReductoParseV3Format,
    types::{ReductoLegacyParams, ReductoV3Params},
};

pub(crate) struct OcrIntegration<F, B>
where
    F: OcrFormat,
    B: OcrBackend<F>,
{
    pub(crate) format: F,
    pub(crate) backend: B,
}

impl<F, B> OcrIntegration<F, B>
where
    F: OcrFormat,
    B: OcrBackend<F>,
{
    const fn new(format: F, backend: B) -> Self {
        Self { format, backend }
    }

    #[cfg(test)]
    pub(crate) fn complete_url(
        &self,
        connection: &super::types::OcrConnection,
        model: &str,
        params: &F::MappedParams,
    ) -> Result<String, super::error::OcrError> {
        self.backend.complete_url(connection, model, params)
    }
}

pub(crate) const MISTRAL: OcrIntegration<MistralOcrFormat, MistralOcrBackend> =
    OcrIntegration::new(MistralOcrFormat, MistralOcrBackend);
pub(crate) const AZURE_MISTRAL: OcrIntegration<MistralOcrFormat, AzureAiOcrBackend> =
    OcrIntegration::new(MistralOcrFormat, AzureAiOcrBackend);
pub(crate) const AZURE_DOCUMENT_INTELLIGENCE: OcrIntegration<
    AzureDocumentIntelligenceOcrFormat,
    AzureAiOcrBackend,
> = OcrIntegration::new(AzureDocumentIntelligenceOcrFormat, AzureAiOcrBackend);
pub(crate) const VERTEX_MISTRAL: OcrIntegration<MistralOcrFormat, VertexAiOcrBackend> =
    OcrIntegration::new(MistralOcrFormat, VertexAiOcrBackend);
pub(crate) const VERTEX_DEEPSEEK: OcrIntegration<DeepSeekOcrFormat, VertexAiOcrBackend> =
    OcrIntegration::new(DeepSeekOcrFormat, VertexAiOcrBackend);
pub(crate) const REDUCTO_V3: OcrIntegration<ReductoParseV3Format, ReductoOcrBackend> =
    OcrIntegration::new(ReductoParseV3Format, ReductoOcrBackend);
pub(crate) const REDUCTO_LEGACY: OcrIntegration<ReductoParseLegacyFormat, ReductoOcrBackend> =
    OcrIntegration::new(ReductoParseLegacyFormat, ReductoOcrBackend);

#[derive(Clone, Debug, Serialize)]
#[serde(untagged)]
pub enum OcrIntegrationRequest {
    Mistral(MistralOcrParams),
    AzureMistral(MistralOcrParams),
    AzureDocumentIntelligence(DocumentIntelligenceInputParams),
    VertexMistral(MistralOcrParams),
    VertexDeepSeek(DeepSeekOcrParams),
    ReductoV3(ReductoV3Params),
    ReductoLegacy(ReductoLegacyParams),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OcrIntegrationKind {
    Mistral,
    AzureMistral,
    AzureDocumentIntelligence,
    VertexMistral,
    VertexDeepSeek,
    ReductoV3,
    ReductoLegacy,
}

#[derive(Clone, Copy, Debug, EnumString, PartialEq, Eq)]
#[strum(serialize_all = "snake_case")]
pub enum OcrProvider {
    Mistral,
    AzureAi,
    VertexAi,
    Reducto,
}

impl OcrProvider {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Mistral => "mistral",
            Self::AzureAi => "azure_ai",
            Self::VertexAi => "vertex_ai",
            Self::Reducto => "reducto",
        }
    }
}

impl OcrIntegrationKind {
    pub const fn provider(self) -> OcrProvider {
        match self {
            Self::Mistral => OcrProvider::Mistral,
            Self::AzureMistral | Self::AzureDocumentIntelligence => OcrProvider::AzureAi,
            Self::VertexMistral | Self::VertexDeepSeek => OcrProvider::VertexAi,
            Self::ReductoV3 | Self::ReductoLegacy => OcrProvider::Reducto,
        }
    }
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

impl OcrIntegrationRequest {
    pub fn kind(&self) -> OcrIntegrationKind {
        match self {
            Self::Mistral(_) => OcrIntegrationKind::Mistral,
            Self::AzureMistral(_) => OcrIntegrationKind::AzureMistral,
            Self::AzureDocumentIntelligence(_) => OcrIntegrationKind::AzureDocumentIntelligence,
            Self::VertexMistral(_) => OcrIntegrationKind::VertexMistral,
            Self::VertexDeepSeek(_) => OcrIntegrationKind::VertexDeepSeek,
            Self::ReductoV3(_) => OcrIntegrationKind::ReductoV3,
            Self::ReductoLegacy(_) => OcrIntegrationKind::ReductoLegacy,
        }
    }
}

#[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
pub fn resolve_ocr_integration(provider: OcrProvider, model: &OcrModel) -> OcrIntegrationKind {
    let lower = model.as_str().to_ascii_lowercase();
    match (provider, model) {
        (OcrProvider::Mistral, _) => OcrIntegrationKind::Mistral,
        (OcrProvider::AzureAi, _)
            if lower.contains("doc-intelligence") || lower.contains("documentintelligence") =>
        {
            OcrIntegrationKind::AzureDocumentIntelligence
        }
        (OcrProvider::AzureAi, _) => OcrIntegrationKind::AzureMistral,
        (OcrProvider::VertexAi, _) if lower.contains("deepseek") => {
            OcrIntegrationKind::VertexDeepSeek
        }
        (OcrProvider::VertexAi, _) => OcrIntegrationKind::VertexMistral,
        (OcrProvider::Reducto, OcrModel::ReductoLegacy) => OcrIntegrationKind::ReductoLegacy,
        (OcrProvider::Reducto, _) => OcrIntegrationKind::ReductoV3,
    }
}
