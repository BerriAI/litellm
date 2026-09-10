use serde_json::{Map, Value};
use strum::EnumString;

use crate::Error;
use crate::routing_utils::provider::{CustomLlmProvider, get_custom_llm_provider};

pub use super::backends::OcrBackend;
pub use super::integrations::{
    AzureDocumentIntelligence, AzureMistral, MistralDirect, ReductoLegacy, ReductoV3,
};
use super::integrations::{BackendConfig, InputParams, OcrIntegration};

/// ```
/// use litellm_core::ocr::registry::{MistralDirect, OcrIntegrationInput};
/// use litellm_core::ocr::formats::mistral::types::MistralOcrParams;
/// let input = OcrIntegrationInput::<MistralDirect> {
///     params: MistralOcrParams::default(),
///     backend_config: (),
/// };
/// ```
///
#[derive(Clone, Debug)]
pub struct OcrIntegrationInput<I: OcrIntegration> {
    pub params: InputParams<I>,
    pub backend_config: BackendConfig<I>,
}

macro_rules! define_integration_types {
    ($( $variant:ident, $integration:ty, $instance:expr, $provider:ident; )+) => {
        #[derive(Clone, Debug)]
        pub enum OcrIntegrationRequest {
            $( $variant(OcrIntegrationInput<$integration>), )+
        }

        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub enum OcrIntegrationKind {
            $( $variant, )+
        }

        impl OcrIntegrationKind {
            pub const fn provider(self) -> OcrProvider {
                match self {
                    $( Self::$variant => OcrProvider::$provider, )+
                }
            }
        }

        impl OcrIntegrationRequest {
            pub fn kind(&self) -> OcrIntegrationKind {
                match self {
                    $( Self::$variant(_) => OcrIntegrationKind::$variant, )+
                }
            }
        }
    };
}

super::integrations::for_each_ocr_integration!(define_integration_types);

#[derive(Clone, Copy, Debug, EnumString, PartialEq, Eq)]
#[strum(serialize_all = "snake_case")]
pub enum OcrProvider {
    Mistral,
    AzureAi,
    Reducto,
}

impl OcrProvider {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Mistral => "mistral",
            Self::AzureAi => "azure_ai",
            Self::Reducto => "reducto",
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

#[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
pub fn resolve_ocr_integration(provider: OcrProvider, model: &OcrModel) -> OcrIntegrationKind {
    match provider {
        OcrProvider::Mistral => OcrIntegrationKind::Mistral,
        OcrProvider::AzureAi => super::backends::azure_ai::resolve_integration(model),
        OcrProvider::Reducto => super::backends::reducto::resolve_integration(model),
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
    macro_rules! decode_selected_integration {
        ($( $variant:ident, $integration:ty, $instance:expr, $provider:ident; )+) => {
            match kind {
                $(
                    OcrIntegrationKind::$variant => OcrIntegrationRequest::$variant(
                        decode_input::<$integration>($instance, params)?,
                    ),
                )+
            }
        };
    }

    Ok(super::integrations::for_each_ocr_integration!(
        decode_selected_integration
    ))
}

fn decode_input<I>(
    integration: I,
    params: Map<String, Value>,
) -> Result<OcrIntegrationInput<I>, Error>
where
    I: OcrIntegration,
{
    let backend_config = I::Backend::decode_config(&params)?;
    let params = integration.decode_input_params(params, "optional_params")?;
    Ok(OcrIntegrationInput {
        params,
        backend_config,
    })
}
