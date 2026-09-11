use super::adapters::OcrAdapter;
use crate::Error;
use crate::routing_utils::provider::{CustomLlmProvider, get_custom_llm_provider};

macro_rules! define_adapter_types {
    ($( $variant:ident, $adapter:ty, $instance:expr, $provider:ident; )+) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub(crate) enum OcrAdapterKind {
            $( $variant, )+
        }

        impl OcrAdapterKind {
            pub(crate) const fn provider(self) -> OcrProvider {
                match self {
                    $( Self::$variant => <$adapter>::PROVIDER, )+
                }
            }
        }
    };
}

super::adapters::for_each_ocr_adapter!(define_adapter_types);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum OcrProvider {
    Mistral,
    AzureAi,
}

impl OcrProvider {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Mistral => "mistral",
            Self::AzureAi => "azure_ai",
        }
    }
}

pub(crate) fn resolve_wire_adapter(
    model: &str,
    custom_llm_provider: Option<&str>,
) -> Result<(String, OcrAdapterKind), Error> {
    let provider =
        get_custom_llm_provider(model, custom_llm_provider).unwrap_or(CustomLlmProvider {
            model,
            custom_llm_provider: OcrProvider::Mistral.as_str(),
        });
    let typed_provider = match provider.custom_llm_provider {
        "mistral" => OcrProvider::Mistral,
        "azure_ai" => OcrProvider::AzureAi,
        value => return Err(Error::InvalidProvider(value.to_string())),
    };
    match typed_provider {
        OcrProvider::Mistral => Ok((provider.model.to_string(), OcrAdapterKind::Mistral)),
        OcrProvider::AzureAi if is_document_intelligence_model(provider.model) => {
            Err(Error::InvalidProvider("azure_ai".into()))
        }
        OcrProvider::AzureAi => Ok((provider.model.to_string(), OcrAdapterKind::AzureMistral)),
    }
}

fn is_document_intelligence_model(model: &str) -> bool {
    let model = model.to_ascii_lowercase();
    model.contains("doc-intelligence") || model.contains("documentintelligence")
}
