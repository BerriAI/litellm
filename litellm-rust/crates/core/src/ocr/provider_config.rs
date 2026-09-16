use super::types::{OcrConnection, OcrDocument};
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
use litellm_auth::Sourced;
use strum::{EnumString, IntoStaticStr};

macro_rules! dispatch_config {
    ($config:expr, $method:ident($($argument:expr),* $(,)?)) => {
        match $config {
            OcrConfigKind::Cohere => CohereParseConfig.$method($($argument),*),
            OcrConfigKind::Mistral => MistralOCRConfig.$method($($argument),*),
            OcrConfigKind::AzureAi => AzureAIOCRConfig.$method($($argument),*),
            OcrConfigKind::AzureCohere => AzureAICohereParseConfig.$method($($argument),*),
            OcrConfigKind::AzureDocumentIntelligence => AzureDocumentIntelligenceOCRConfig.$method($($argument),*),
            OcrConfigKind::ReductoLegacy => ReductoParseLegacyConfig.$method($($argument),*),
            OcrConfigKind::ReductoV3 => ReductoParseV3Config.$method($($argument),*),
            OcrConfigKind::VertexAi => VertexAIOCRConfig.$method($($argument),*),
            OcrConfigKind::VertexDeepSeek => VertexAIDeepSeekOCRConfig.$method($($argument),*),
        }
    };
}

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
        dispatch_config!(self, get_supported_ocr_params(model))
    }

    pub(crate) fn get_api_key_env_var(self) -> Option<&'static str> {
        dispatch_config!(self, get_api_key_env_var())
    }

    pub(crate) fn get_health_check_document(self) -> OcrDocument {
        dispatch_config!(self, get_health_check_document())
    }

    pub(crate) fn resolve_connection_params(self, connection: OcrConnection) -> OcrConnection {
        let api_key = connection
            .api_key
            .map(|value| Sourced::new(value, connection.api_key_source));
        let api_base = connection
            .api_base
            .map(|value| Sourced::new(value, connection.api_base_source));
        let (api_key, api_base) = dispatch_config!(
            self,
            resolve_connection_params(
                api_key,
                api_base,
                connection.dynamic_api_key,
                connection.dynamic_api_base,
            )
        );
        OcrConnection {
            api_key_source: api_key
                .as_ref()
                .map(Sourced::source)
                .unwrap_or(connection.api_key_source),
            api_base_source: api_base
                .as_ref()
                .map(Sourced::source)
                .unwrap_or(connection.api_base_source),
            api_key: api_key.map(Sourced::into_value),
            api_base: api_base.map(Sourced::into_value),
            dynamic_api_key: None,
            dynamic_api_base: None,
            ..connection
        }
    }

    pub(crate) fn get_error_class(
        self,
        message: String,
        status: u16,
        headers: Vec<(String, String)>,
    ) -> super::Error {
        dispatch_config!(self, get_error_class(message, status, headers))
    }
}

pub fn get_api_key_env_var(
    model: &str,
    custom_llm_provider: Option<&str>,
) -> Result<Option<&'static str>, super::Error> {
    Ok(resolve_provider_config(model, custom_llm_provider)?
        .1
        .get_api_key_env_var())
}

pub fn get_health_check_document(
    model: &str,
    custom_llm_provider: Option<&str>,
) -> Result<OcrDocument, super::Error> {
    Ok(resolve_provider_config(model, custom_llm_provider)?
        .1
        .get_health_check_document())
}

#[derive(Clone, Copy, Debug, EnumString, IntoStaticStr, PartialEq, Eq)]
#[strum(serialize_all = "snake_case")]
pub(crate) enum OcrProvider {
    Cohere,
    Mistral,
    AzureAi,
    Reducto,
    VertexAi,
}

pub(crate) fn resolve_provider_config(
    model: &str,
    custom_llm_provider: Option<&str>,
) -> Result<(String, OcrConfigKind), super::Error> {
    let provider =
        get_custom_llm_provider(model, custom_llm_provider).unwrap_or(CustomLlmProvider {
            model,
            custom_llm_provider: OcrProvider::Mistral.into(),
        });
    let ocr_provider = provider
        .custom_llm_provider
        .parse::<OcrProvider>()
        .map_err(|_| super::Error::InvalidProvider(provider.custom_llm_provider.to_string()))?;
    let config = match ocr_provider {
        OcrProvider::Cohere => OcrConfigKind::Cohere,
        OcrProvider::Mistral => OcrConfigKind::Mistral,
        OcrProvider::AzureAi if is_document_intelligence_model(provider.model) => {
            OcrConfigKind::AzureDocumentIntelligence
        }
        OcrProvider::AzureAi
            if provider.model.to_ascii_lowercase().contains("cohere")
                && provider.model.to_ascii_lowercase().contains("parse") =>
        {
            OcrConfigKind::AzureCohere
        }
        OcrProvider::AzureAi => OcrConfigKind::AzureAi,
        OcrProvider::Reducto if provider.model.eq_ignore_ascii_case("parse-legacy") => {
            OcrConfigKind::ReductoLegacy
        }
        OcrProvider::Reducto => OcrConfigKind::ReductoV3,
        OcrProvider::VertexAi if provider.model.to_ascii_lowercase().contains("deepseek") => {
            OcrConfigKind::VertexDeepSeek
        }
        OcrProvider::VertexAi => OcrConfigKind::VertexAi,
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
    use litellm_auth::InputSource;

    #[test]
    fn provider_names_round_trip_exactly() {
        for provider in ["cohere", "mistral", "azure_ai", "reducto", "vertex_ai"] {
            let (_, config) = resolve_provider_config("model", Some(provider)).unwrap();
            let resolved: &'static str = config.provider().into();
            assert_eq!(resolved, provider);
        }

        for provider in ["Mistral", "unknown"] {
            assert_eq!(
                resolve_provider_config("model", Some(provider)),
                Err(crate::ocr::Error::InvalidProvider(provider.into()))
            );
        }
    }

    #[test]
    fn health_check_documents_are_valid_for_each_provider() {
        for model in [
            "mistral/ocr",
            "azure_ai/ocr",
            "azure_ai/doc-intelligence/prebuilt-layout",
            "reducto/parse-v3",
            "vertex_ai/mistral-ocr",
            "vertex_ai/deepseek-ocr",
        ] {
            let document = get_health_check_document(model, None).unwrap();
            assert!(matches!(document, OcrDocument::DocumentUrl { .. }));
            let inline = crate::ocr::document::InlineDocument::parse(document.source())
                .unwrap()
                .unwrap();
            assert_eq!(inline.mime_type().to_string(), "application/pdf");
            assert!(inline.decode(4096).unwrap().starts_with(b"%PDF-"));
        }
        for model in ["cohere/parse", "azure_ai/cohere-parse"] {
            let document = get_health_check_document(model, None).unwrap();
            crate::llms::cohere::ocr::validate_document(&document).unwrap();
            let inline = crate::ocr::document::InlineDocument::parse(document.source())
                .unwrap()
                .unwrap();
            assert_eq!(inline.mime_type().to_string(), "image/png");
            assert!(
                inline
                    .decode(4096)
                    .unwrap()
                    .starts_with(b"\x89PNG\r\n\x1a\n")
            );
        }
    }

    #[test]
    fn api_key_metadata_follows_provider_overrides_and_python_defaults() {
        for (model, expected) in [
            ("mistral/ocr", Some("MISTRAL_API_KEY")),
            ("cohere/parse", Some("COHERE_API_KEY")),
            ("azure_ai/ocr", Some("AZURE_AI_API_KEY")),
            ("azure_ai/cohere-parse", Some("AZURE_AI_API_KEY")),
            (
                "azure_ai/doc-intelligence/prebuilt-layout",
                Some("AZURE_DOCUMENT_INTELLIGENCE_API_KEY"),
            ),
            ("vertex_ai/mistral-ocr", Some("VERTEX_AI_API_KEY")),
            ("vertex_ai/deepseek-ocr", Some("VERTEX_AI_API_KEY")),
            ("reducto/parse-v3", None),
            ("reducto/parse-legacy", None),
        ] {
            assert_eq!(
                get_api_key_env_var(model, None).unwrap(),
                expected,
                "{model}"
            );
        }
    }

    #[test]
    fn connection_resolution_preserves_dynamic_precedence_and_input_sources() {
        let connection = OcrConfigKind::Mistral.resolve_connection_params(OcrConnection {
            api_key: Some("explicit-key".into()),
            api_base: Some("https://explicit.test".into()),
            dynamic_api_key: Some(Sourced::new("dynamic-key".into(), InputSource::Environment)),
            dynamic_api_base: Some(Sourced::new(
                "https://dynamic.test".into(),
                InputSource::Request,
            )),
            ..Default::default()
        });
        assert_eq!(connection.api_key.as_deref(), Some("dynamic-key"));
        assert_eq!(connection.api_base.as_deref(), Some("https://dynamic.test"));
        assert_eq!(connection.api_key_source, InputSource::Environment);
        assert_eq!(connection.api_base_source, InputSource::Request);
        for dynamic in [
            None,
            Some(Sourced::new(String::new(), InputSource::Environment)),
        ] {
            let connection = OcrConfigKind::Mistral.resolve_connection_params(OcrConnection {
                api_key: Some("explicit-key".into()),
                api_base: Some("https://explicit.test".into()),
                dynamic_api_key: dynamic.clone(),
                dynamic_api_base: dynamic,
                ..Default::default()
            });
            assert_eq!(connection.api_key.as_deref(), Some("explicit-key"));
            assert_eq!(
                connection.api_base.as_deref(),
                Some("https://explicit.test")
            );
        }
    }

    #[test]
    fn document_intelligence_only_accepts_dynamic_values_for_explicit_fields() {
        for (explicit_key, explicit_base) in [
            (None, None),
            (Some("key"), None),
            (None, Some("base")),
            (Some("key"), Some("base")),
        ] {
            let connection =
                OcrConfigKind::AzureDocumentIntelligence.resolve_connection_params(OcrConnection {
                    api_key: explicit_key.map(str::to_string),
                    api_base: explicit_base.map(str::to_string),
                    dynamic_api_key: Some(Sourced::new(
                        "dynamic-key".into(),
                        InputSource::Environment,
                    )),
                    dynamic_api_base: Some(Sourced::new(
                        "https://dynamic.test".into(),
                        InputSource::Deployment,
                    )),
                    ..Default::default()
                });
            assert_eq!(
                connection.api_key.as_deref(),
                explicit_key.map(|_| "dynamic-key")
            );
            assert_eq!(
                connection.api_base.as_deref(),
                explicit_base.map(|_| "https://dynamic.test")
            );
        }
    }

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
