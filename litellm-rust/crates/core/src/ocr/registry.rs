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
    Reducto,
}

impl OcrProvider {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Mistral => "mistral",
            Self::AzureAi => "azure_ai",
            Self::Reducto => "reducto",
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
        "reducto" => OcrProvider::Reducto,
        value => return Err(Error::InvalidProvider(value.to_string())),
    };
    let adapter = match typed_provider {
        OcrProvider::Mistral => OcrAdapterKind::Mistral,
        OcrProvider::AzureAi if is_document_intelligence_model(provider.model) => {
            OcrAdapterKind::AzureDocumentIntelligence
        }
        OcrProvider::AzureAi => OcrAdapterKind::AzureMistral,
        OcrProvider::Reducto if provider.model.eq_ignore_ascii_case("parse-legacy") => {
            OcrAdapterKind::ReductoLegacy
        }
        OcrProvider::Reducto => OcrAdapterKind::ReductoV3,
    };
    Ok((provider.model.to_string(), adapter))
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
        let cases = [
            ("mistral/future-ocr-model", OcrAdapterKind::Mistral),
            ("azure_ai/future-ocr-model", OcrAdapterKind::AzureMistral),
            ("reducto/future-parse-model", OcrAdapterKind::ReductoV3),
        ];

        for (qualified_model, expected_adapter) in cases {
            let expected_model = qualified_model.split_once('/').unwrap().1;
            let (model, adapter) = resolve_wire_adapter(qualified_model, None).unwrap();
            assert_eq!(model, expected_model);
            assert_eq!(adapter, expected_adapter);
        }
    }

    #[test]
    fn known_protocol_models_still_select_specialized_adapters() {
        let (model, adapter) = resolve_wire_adapter("reducto/parse-legacy", None).unwrap();
        assert_eq!(model, "parse-legacy");
        assert_eq!(adapter, OcrAdapterKind::ReductoLegacy);

        let (model, adapter) =
            resolve_wire_adapter("azure_ai/doc-intelligence/prebuilt-layout", None).unwrap();
        assert_eq!(model, "doc-intelligence/prebuilt-layout");
        assert_eq!(adapter, OcrAdapterKind::AzureDocumentIntelligence);
    }
}
