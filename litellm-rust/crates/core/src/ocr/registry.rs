use serde_json::{Map, Value};
use strum::EnumString;

use crate::Error;
use crate::auth::azure::AzureAuthInputs;
use crate::providers::vertex_ai::auth::VertexAuthInputs;
use crate::routing_utils::provider::{CustomLlmProvider, get_custom_llm_provider};

use super::backends::OcrBackend;
use super::backends::azure_ai::{AzureDocumentIntelligenceOcrBackend, AzureMistralOcrBackend};
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
use super::types::OcrRequestFormat;

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

    pub(crate) fn decode_input_params(
        &self,
        params: Map<String, Value>,
        prefix: &str,
    ) -> Result<F::InputParams, super::error::OcrRequestError> {
        validate_request_format(
            &params,
            self.backend.supports_native_request_format(),
            self.backend.provider_name(),
        )?;
        F::validate_input_params(&params)?;
        super::wire::decode_request_value(Value::Object(params), prefix)
    }
}

fn validate_request_format(
    params: &Map<String, Value>,
    supports_native: bool,
    provider: &'static str,
) -> Result<(), super::error::OcrRequestError> {
    let Some(format) = params.get("req_format") else {
        return Ok(());
    };
    let format: OcrRequestFormat = serde_json::from_value(format.clone())
        .map_err(|_| super::error::OcrRequestError::RequestFormat)?;
    if format == OcrRequestFormat::Native && !supports_native {
        return Err(super::error::OcrRequestError::NativeUnsupported(provider));
    }
    Ok(())
}

pub(crate) const MISTRAL: OcrIntegration<MistralOcrFormat, MistralOcrBackend> =
    OcrIntegration::new(MistralOcrFormat, MistralOcrBackend);
pub(crate) const AZURE_MISTRAL: OcrIntegration<MistralOcrFormat, AzureMistralOcrBackend> =
    OcrIntegration::new(MistralOcrFormat, AzureMistralOcrBackend);
pub(crate) const AZURE_DOCUMENT_INTELLIGENCE: OcrIntegration<
    AzureDocumentIntelligenceOcrFormat,
    AzureDocumentIntelligenceOcrBackend,
> = OcrIntegration::new(
    AzureDocumentIntelligenceOcrFormat,
    AzureDocumentIntelligenceOcrBackend,
);
pub(crate) const VERTEX_MISTRAL: OcrIntegration<MistralOcrFormat, VertexAiOcrBackend> =
    OcrIntegration::new(MistralOcrFormat, VertexAiOcrBackend);
pub(crate) const VERTEX_DEEPSEEK: OcrIntegration<DeepSeekOcrFormat, VertexAiOcrBackend> =
    OcrIntegration::new(DeepSeekOcrFormat, VertexAiOcrBackend);
pub(crate) const REDUCTO_V3: OcrIntegration<ReductoParseV3Format, ReductoOcrBackend> =
    OcrIntegration::new(ReductoParseV3Format, ReductoOcrBackend);
pub(crate) const REDUCTO_LEGACY: OcrIntegration<ReductoParseLegacyFormat, ReductoOcrBackend> =
    OcrIntegration::new(ReductoParseLegacyFormat, ReductoOcrBackend);

#[derive(Clone, Debug)]
pub struct OcrIntegrationInput<P, C> {
    pub params: P,
    pub backend_config: C,
}

#[derive(Clone, Debug)]
pub enum OcrIntegrationRequest {
    Mistral(OcrIntegrationInput<MistralOcrParams, ()>),
    AzureMistral(OcrIntegrationInput<MistralOcrParams, AzureAuthInputs>),
    AzureDocumentIntelligence(
        OcrIntegrationInput<DocumentIntelligenceInputParams, AzureAuthInputs>,
    ),
    VertexMistral(OcrIntegrationInput<MistralOcrParams, VertexAuthInputs>),
    VertexDeepSeek(OcrIntegrationInput<DeepSeekOcrParams, VertexAuthInputs>),
    ReductoV3(OcrIntegrationInput<ReductoV3Params, ()>),
    ReductoLegacy(OcrIntegrationInput<ReductoLegacyParams, ()>),
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

pub fn resolve_wire_integration(
    model: &str,
    custom_llm_provider: Option<&str>,
) -> Result<(OcrModel, OcrIntegrationKind), Error> {
    let provider =
        get_custom_llm_provider(model, custom_llm_provider).unwrap_or(CustomLlmProvider {
            model,
            custom_llm_provider: OcrProvider::Mistral.as_str(),
        });
    let typed_provider = provider
        .custom_llm_provider
        .parse::<OcrProvider>()
        .map_err(|_| Error::InvalidProvider(provider.custom_llm_provider.to_string()))?;
    let model = OcrModel::from(provider.model);
    let integration = resolve_ocr_integration(typed_provider, &model);
    Ok((model, integration))
}

pub fn decode_integration_request(
    kind: OcrIntegrationKind,
    params: Map<String, Value>,
) -> Result<OcrIntegrationRequest, Error> {
    let request = match kind {
        OcrIntegrationKind::Mistral => OcrIntegrationRequest::Mistral(OcrIntegrationInput {
            params: MISTRAL.decode_input_params(params, "optional_params")?,
            backend_config: (),
        }),
        OcrIntegrationKind::AzureMistral => {
            let backend = AzureAuthInputs::from_optional_params(&params)?;
            OcrIntegrationRequest::AzureMistral(OcrIntegrationInput {
                params: AZURE_MISTRAL.decode_input_params(params, "optional_params")?,
                backend_config: backend,
            })
        }
        OcrIntegrationKind::AzureDocumentIntelligence => {
            let backend = AzureAuthInputs::from_optional_params(&params)?;
            OcrIntegrationRequest::AzureDocumentIntelligence(OcrIntegrationInput {
                params: AZURE_DOCUMENT_INTELLIGENCE
                    .decode_input_params(params, "optional_params")?,
                backend_config: backend,
            })
        }
        OcrIntegrationKind::VertexMistral => {
            let backend = VertexAuthInputs::from_optional_params(&params)?;
            OcrIntegrationRequest::VertexMistral(OcrIntegrationInput {
                params: VERTEX_MISTRAL.decode_input_params(params, "optional_params")?,
                backend_config: backend,
            })
        }
        OcrIntegrationKind::VertexDeepSeek => {
            let backend = VertexAuthInputs::from_optional_params(&params)?;
            OcrIntegrationRequest::VertexDeepSeek(OcrIntegrationInput {
                params: VERTEX_DEEPSEEK.decode_input_params(params, "optional_params")?,
                backend_config: backend,
            })
        }
        OcrIntegrationKind::ReductoV3 => OcrIntegrationRequest::ReductoV3(OcrIntegrationInput {
            params: REDUCTO_V3.decode_input_params(params, "optional_params")?,
            backend_config: (),
        }),
        OcrIntegrationKind::ReductoLegacy => {
            OcrIntegrationRequest::ReductoLegacy(OcrIntegrationInput {
                params: REDUCTO_LEGACY.decode_input_params(params, "optional_params")?,
                backend_config: (),
            })
        }
    };
    Ok(request)
}
