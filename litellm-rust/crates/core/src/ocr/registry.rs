use serde_json::{Map, Value};
use strum::EnumString;

use crate::Error;
use crate::auth::azure::AzureAuthInputs;
use crate::providers::vertex_ai::auth::VertexAuthInputs;
use crate::routing_utils::provider::{CustomLlmProvider, get_custom_llm_provider};

pub use super::backends::azure_ai::{AzureDocumentIntelligence, AzureMistral};
pub use super::backends::mistral::MistralDirect;
pub use super::backends::reducto::{ReductoLegacy, ReductoV3};
pub use super::backends::vertex_ai::{VertexDeepSeek, VertexMistral};
use super::backends::{HostConfig, InputParams};
pub use super::backends::{OcrHost, OcrIntegration};
use super::types::OcrRequestFormat;

pub(crate) fn validate_request_format(
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

pub(crate) const MISTRAL: MistralDirect = MistralDirect;
pub(crate) const AZURE_MISTRAL: AzureMistral = AzureMistral;
pub(crate) const AZURE_DOCUMENT_INTELLIGENCE: AzureDocumentIntelligence = AzureDocumentIntelligence;
pub(crate) const VERTEX_MISTRAL: VertexMistral = VertexMistral;
pub(crate) const VERTEX_DEEPSEEK: VertexDeepSeek = VertexDeepSeek;
pub(crate) const REDUCTO_V3: ReductoV3 = ReductoV3;
pub(crate) const REDUCTO_LEGACY: ReductoLegacy = ReductoLegacy;

/// ```
/// use litellm_core::ocr::registry::{MistralDirect, OcrIntegrationInput};
/// use litellm_core::ocr::formats::mistral::types::MistralOcrParams;
/// let input = OcrIntegrationInput::<MistralDirect> {
///     params: MistralOcrParams::default(),
///     backend_config: (),
/// };
/// ```
///
/// ```compile_fail
/// use litellm_core::ocr::registry::{MistralDirect, OcrIntegrationInput};
/// use litellm_core::ocr::formats::deepseek::types::DeepSeekOcrParams;
/// let input = OcrIntegrationInput::<MistralDirect> {
///     params: DeepSeekOcrParams::default(),
///     backend_config: (),
/// };
/// ```
///
/// ```compile_fail
/// use litellm_core::ocr::registry::{VertexMistral, OcrIntegrationInput};
/// use litellm_core::ocr::formats::mistral::types::MistralOcrParams;
/// let input = OcrIntegrationInput::<VertexMistral> {
///     params: MistralOcrParams::default(),
///     backend_config: (),
/// };
/// ```
#[derive(Clone, Debug)]
pub struct OcrIntegrationInput<I: OcrIntegration> {
    pub params: InputParams<I>,
    pub backend_config: HostConfig<I>,
}

#[derive(Clone, Debug)]
pub enum OcrIntegrationRequest {
    Mistral(OcrIntegrationInput<MistralDirect>),
    AzureMistral(OcrIntegrationInput<AzureMistral>),
    AzureDocumentIntelligence(OcrIntegrationInput<AzureDocumentIntelligence>),
    VertexMistral(OcrIntegrationInput<VertexMistral>),
    VertexDeepSeek(OcrIntegrationInput<VertexDeepSeek>),
    ReductoV3(OcrIntegrationInput<ReductoV3>),
    ReductoLegacy(OcrIntegrationInput<ReductoLegacy>),
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
