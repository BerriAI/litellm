use strum::{EnumString, IntoStaticStr};

use super::OcrClient;
use super::types::{
    LiteLLMOcrResponse, OcrCredentialInputs, OcrDocument, PreparedOcrRequest,
    ResolvedOcrCredentials,
};
use crate::litellm_core_utils::get_llm_provider_logic::{
    CustomLlmProvider, get_custom_llm_provider,
};
use crate::llms::azure_ai::ocr::cohere_parse_transformation::AzureAICohereParseConfig;
use crate::llms::azure_ai::ocr::document_intelligence::transformation::AzureDocumentIntelligenceOcrConfig;
use crate::llms::azure_ai::ocr::transformation::AzureAiOcrConfig;
use crate::llms::base_llm::ocr::transformation::{BaseOcrConfig, OcrResponseContext};
use crate::llms::cohere::ocr::transformation::CohereParseConfig;
use crate::llms::mistral::ocr::transformation::MistralOcrConfig;
use crate::llms::reducto::ocr::transformation::{ReductoParseLegacyConfig, ReductoParseV3Config};
use crate::llms::vertex_ai::ocr::deepseek_transformation::VertexAIDeepSeekOCRConfig;
use crate::llms::vertex_ai::ocr::transformation::VertexAiOcrConfig;

macro_rules! dispatch_config {
    ($config:expr, $method:ident($($argument:expr),* $(,)?)) => {
        dispatch_config!(@arms $config, $method($($argument),*), )
    };
    ($config:expr, $method:ident($($argument:expr),* $(,)?).await) => {
        dispatch_config!(@arms $config, $method($($argument),*), .await)
    };
    (@arms $config:expr, $method:ident($($argument:expr),*), $($suffix:tt)*) => {
        match $config {
            OcrConfigKind::Cohere => CohereParseConfig.$method($($argument),*)$($suffix)*,
            OcrConfigKind::Mistral => MistralOcrConfig.$method($($argument),*)$($suffix)*,
            OcrConfigKind::AzureAi => AzureAiOcrConfig.$method($($argument),*)$($suffix)*,
            OcrConfigKind::AzureCohere => AzureAICohereParseConfig.$method($($argument),*)$($suffix)*,
            OcrConfigKind::AzureDocumentIntelligence => AzureDocumentIntelligenceOcrConfig.$method($($argument),*)$($suffix)*,
            OcrConfigKind::ReductoLegacy => ReductoParseLegacyConfig.$method($($argument),*)$($suffix)*,
            OcrConfigKind::ReductoV3 => ReductoParseV3Config.$method($($argument),*)$($suffix)*,
            OcrConfigKind::VertexAi => VertexAiOcrConfig.$method($($argument),*)$($suffix)*,
            OcrConfigKind::VertexDeepSeek => VertexAIDeepSeekOCRConfig.$method($($argument),*)$($suffix)*,
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

    pub(crate) fn resolve_connection_params(
        self,
        inputs: OcrCredentialInputs,
    ) -> ResolvedOcrCredentials {
        dispatch_config!(self, resolve_connection_params(inputs))
    }

    pub(crate) fn get_error_class(
        self,
        message: String,
        status: u16,
        headers: Vec<(String, String)>,
    ) -> super::Error {
        dispatch_config!(self, get_error_class(message, status, headers))
    }

    pub(crate) async fn prepare_request(
        self,
        request: &PreparedOcrRequest,
        client: &OcrClient,
    ) -> Result<reqwest::Request, super::Error> {
        dispatch_config!(self, prepare_request(request, client).await)
    }

    pub(crate) async fn async_transform_ocr_response(
        self,
        model: &str,
        raw_response: reqwest::Response,
        context: OcrResponseContext<'_>,
    ) -> Result<LiteLLMOcrResponse, super::Error> {
        dispatch_config!(
            self,
            async_transform_ocr_response(model, raw_response, context).await
        )
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
    use litellm_auth::{InputSource, Sourced};
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case("cohere")]
    #[case("mistral")]
    #[case("azure_ai")]
    #[case("reducto")]
    #[case("vertex_ai")]
    fn provider_names_round_trip_exactly(#[case] provider: &str) {
        let (_, config) = resolve_provider_config("model", Some(provider)).unwrap();
        let resolved: &'static str = config.provider().into();
        assert_eq!(resolved, provider);
    }

    #[rstest]
    #[case("Mistral")]
    #[case("unknown")]
    fn invalid_provider_names_are_rejected(#[case] provider: &str) {
        assert!(matches!(
            resolve_provider_config("model", Some(provider)),
            Err(crate::ocr::Error::InvalidProvider(value)) if value == provider
        ));
    }

    #[rstest]
    #[case("mistral/ocr")]
    #[case("azure_ai/ocr")]
    #[case("azure_ai/doc-intelligence/prebuilt-layout")]
    #[case("reducto/parse-v3")]
    #[case("vertex_ai/mistral-ocr")]
    #[case("vertex_ai/deepseek-ocr")]
    fn pdf_health_check_documents_are_valid(#[case] model: &str) {
        let document = get_health_check_document(model, None).unwrap();
        assert!(matches!(document, OcrDocument::DocumentUrl { .. }));
        let inline = crate::ocr::document::InlineDocument::parse(document.source())
            .unwrap()
            .unwrap();
        assert_eq!(inline.mime_type().to_string(), "application/pdf");
        assert!(inline.decode(4096).unwrap().starts_with(b"%PDF-"));
    }

    #[rstest]
    #[case("cohere/parse")]
    #[case("azure_ai/cohere-parse")]
    fn png_health_check_documents_are_valid(#[case] model: &str) {
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

    #[rstest]
    #[case("mistral/ocr", Some("MISTRAL_API_KEY"))]
    #[case("cohere/parse", Some("COHERE_API_KEY"))]
    #[case("azure_ai/ocr", Some("AZURE_AI_API_KEY"))]
    #[case("azure_ai/cohere-parse", Some("AZURE_AI_API_KEY"))]
    #[case(
        "azure_ai/doc-intelligence/prebuilt-layout",
        Some("AZURE_DOCUMENT_INTELLIGENCE_API_KEY")
    )]
    #[case("vertex_ai/mistral-ocr", Some("VERTEX_AI_API_KEY"))]
    #[case("vertex_ai/deepseek-ocr", Some("VERTEX_AI_API_KEY"))]
    #[case("reducto/parse-v3", None)]
    #[case("reducto/parse-legacy", None)]
    fn api_key_metadata_follows_provider_overrides_and_python_defaults(
        #[case] model: &str,
        #[case] expected: Option<&str>,
    ) {
        assert_eq!(get_api_key_env_var(model, None).unwrap(), expected);
    }

    #[test]
    fn connection_resolution_preserves_dynamic_precedence_and_input_sources() {
        let connection = OcrConfigKind::Mistral.resolve_connection_params(OcrCredentialInputs {
            api_key: Some(Sourced::new("explicit-key".into(), InputSource::Deployment)),
            api_base: Some(Sourced::new(
                "https://explicit.test".into(),
                InputSource::Deployment,
            )),
            dynamic_api_key: Some(Sourced::new("dynamic-key".into(), InputSource::Environment)),
            dynamic_api_base: Some(Sourced::new(
                "https://dynamic.test".into(),
                InputSource::Request,
            )),
        });
        assert_eq!(
            connection
                .api_key
                .as_ref()
                .map(|value| value.value().as_str()),
            Some("dynamic-key")
        );
        assert_eq!(
            connection
                .api_base
                .as_ref()
                .map(|value| value.value().as_str()),
            Some("https://dynamic.test")
        );
        assert_eq!(
            connection.api_key.as_ref().map(Sourced::source),
            Some(InputSource::Environment)
        );
        assert_eq!(
            connection.api_base.as_ref().map(Sourced::source),
            Some(InputSource::Request)
        );
    }

    #[rstest]
    #[case(None)]
    #[case(Some(""))]
    fn empty_or_missing_dynamic_credentials_preserve_explicit_values(
        #[case] dynamic_value: Option<&str>,
    ) {
        let dynamic =
            dynamic_value.map(|value| Sourced::new(value.into(), InputSource::Environment));
        let connection = OcrConfigKind::Mistral.resolve_connection_params(OcrCredentialInputs {
            api_key: Some(Sourced::new("explicit-key".into(), InputSource::Deployment)),
            api_base: Some(Sourced::new(
                "https://explicit.test".into(),
                InputSource::Deployment,
            )),
            dynamic_api_key: dynamic.clone(),
            dynamic_api_base: dynamic,
        });
        assert_eq!(
            connection
                .api_key
                .as_ref()
                .map(|value| value.value().as_str()),
            Some("explicit-key")
        );
        assert_eq!(
            connection
                .api_base
                .as_ref()
                .map(|value| value.value().as_str()),
            Some("https://explicit.test")
        );
    }

    #[rstest]
    #[case(None, None)]
    #[case(Some("key"), None)]
    #[case(None, Some("base"))]
    #[case(Some("key"), Some("base"))]
    fn document_intelligence_only_accepts_dynamic_values_for_explicit_fields(
        #[case] explicit_key: Option<&str>,
        #[case] explicit_base: Option<&str>,
    ) {
        let connection = OcrConfigKind::AzureDocumentIntelligence.resolve_connection_params(
            OcrCredentialInputs {
                api_key: explicit_key
                    .map(|value| Sourced::new(value.into(), InputSource::Deployment)),
                api_base: explicit_base
                    .map(|value| Sourced::new(value.into(), InputSource::Deployment)),
                dynamic_api_key: Some(Sourced::new("dynamic-key".into(), InputSource::Environment)),
                dynamic_api_base: Some(Sourced::new(
                    "https://dynamic.test".into(),
                    InputSource::Deployment,
                )),
            },
        );
        assert_eq!(
            connection
                .api_key
                .as_ref()
                .map(|value| value.value().as_str()),
            explicit_key.map(|_| "dynamic-key")
        );
        assert_eq!(
            connection
                .api_base
                .as_ref()
                .map(|value| value.value().as_str()),
            explicit_base.map(|_| "https://dynamic.test")
        );
    }

    #[rstest]
    #[case("mistral/future-ocr-model", OcrConfigKind::Mistral)]
    #[case("azure_ai/future-ocr-model", OcrConfigKind::AzureAi)]
    fn provider_models_are_preserved_without_a_local_allowlist(
        #[case] qualified_model: &str,
        #[case] expected_config: OcrConfigKind,
    ) {
        let expected_model = qualified_model.split_once('/').unwrap().1;
        let (model, config) = resolve_provider_config(qualified_model, None).unwrap();
        assert_eq!(model, expected_model);
        assert_eq!(config, expected_config);
    }

    #[rstest]
    #[case("reducto/parse-legacy", OcrConfigKind::ReductoLegacy)]
    #[case("reducto/future-parse-model", OcrConfigKind::ReductoV3)]
    #[case("azure_ai/Cohere-parse-v5", OcrConfigKind::AzureCohere)]
    #[case("azure_ai/cohere-parse-v5", OcrConfigKind::AzureCohere)]
    #[case("azure_ai/cohere/parse-v5", OcrConfigKind::AzureCohere)]
    #[case("azure_ai/invoice-parser", OcrConfigKind::AzureAi)]
    #[case("azure_ai/parse-v5", OcrConfigKind::AzureAi)]
    #[case("azure_ai/mistral-ocr-4-0", OcrConfigKind::AzureAi)]
    #[case("azure_ai/mistral-document-ai-2512", OcrConfigKind::AzureAi)]
    #[case(
        "azure_ai/doc-intelligence/prebuilt-layout",
        OcrConfigKind::AzureDocumentIntelligence
    )]
    fn provider_specific_models_select_their_config(
        #[case] model: &str,
        #[case] expected_config: OcrConfigKind,
    ) {
        assert_eq!(
            resolve_provider_config(model, None).unwrap().1,
            expected_config
        );
        assert_eq!(
            resolve_provider_config(model, None).unwrap().0,
            model.split_once('/').unwrap().1
        );
    }

    #[rstest]
    #[case::prefix("not_a_provider/model", None)]
    #[case::explicit("model", Some("not_a_provider"))]
    fn ocr_contract_unknown_provider_is_bad_request(
        #[case] model: &str,
        #[case] provider: Option<&str>,
    ) {
        let error = resolve_provider_config(model, provider).unwrap_err();
        assert!(
            matches!(&error, crate::ocr::Error::InvalidProvider(provider) if provider == "not_a_provider")
        );
        assert_eq!(error.http_status_code(), Some(400));
    }
}
