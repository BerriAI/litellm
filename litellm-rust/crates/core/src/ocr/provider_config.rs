use std::collections::BTreeMap;

use litellm_auth::{InputSource, Sourced};
use litellm_core_utils::{
    call_arguments::CallArguments,
    get_llm_provider_logic::{CustomLlmProvider, get_custom_llm_provider},
};
use litellm_llms::{
    aws_textract::ocr::{
        analyze_transformation::TextractAnalyzeDocumentConfig, common_utils::TextractOperation,
        transformation::TextractDetectTextConfig,
    },
    azure_ai::ocr::{
        cohere_parse_transformation::AzureAICohereParseConfig,
        document_intelligence::transformation::AzureDocumentIntelligenceOcrConfig,
        transformation::AzureAiOcrConfig,
    },
    base_llm::ocr::{
        error::Error,
        handler::{self, CallHooks, OcrClient},
        settings::OcrSettings,
        transformation::{
            BaseOcrConfig, LiteLLMOcrResponse, OcrConnection, OcrCredentialInputs, OcrDocument,
            OcrResponseFormat, OcrTransportConfig, PreparedOcrRequest, ResolvedOcrCredentials,
        },
    },
    cohere::ocr::transformation::CohereParseConfig,
    mistral::ocr::transformation::MistralOcrConfig,
    reducto::ocr::transformation::{ReductoParseLegacyConfig, ReductoParseV3Config},
    vertex_ai::ocr::{
        deepseek_transformation::VertexAIDeepSeekOCRConfig, transformation::VertexAiOcrConfig,
    },
};
use strum::{EnumString, IntoStaticStr};

macro_rules! with_config {
    ($kind:expr, $config:ident => $body:expr) => {
        match $kind {
            OcrConfigKind::AwsTextract => {
                let $config = TextractDetectTextConfig;
                $body
            }
            OcrConfigKind::AwsTextractAnalyze => {
                let $config = TextractAnalyzeDocumentConfig;
                $body
            }
            OcrConfigKind::Cohere => {
                let $config = CohereParseConfig;
                $body
            }
            OcrConfigKind::Mistral => {
                let $config = MistralOcrConfig;
                $body
            }
            OcrConfigKind::AzureAi => {
                let $config = AzureAiOcrConfig;
                $body
            }
            OcrConfigKind::AzureCohere => {
                let $config = AzureAICohereParseConfig;
                $body
            }
            OcrConfigKind::AzureDocumentIntelligence => {
                let $config = AzureDocumentIntelligenceOcrConfig;
                $body
            }
            OcrConfigKind::ReductoLegacy => {
                let $config = ReductoParseLegacyConfig;
                $body
            }
            OcrConfigKind::ReductoV3 => {
                let $config = ReductoParseV3Config;
                $body
            }
            OcrConfigKind::VertexAi => {
                let $config = VertexAiOcrConfig;
                $body
            }
            OcrConfigKind::VertexDeepSeek => {
                let $config = VertexAIDeepSeekOCRConfig;
                $body
            }
        }
    };
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum OcrConfigKind {
    AwsTextract,
    AwsTextractAnalyze,
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
            Self::AwsTextract | Self::AwsTextractAnalyze => OcrProvider::AwsTextract,
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
        with_config!(self, config => config.get_supported_ocr_params(model))
    }

    pub(crate) fn get_api_key_env_var(self) -> Option<&'static str> {
        with_config!(self, config => config.get_api_key_env_var())
    }

    pub(crate) fn secret_names(self) -> Vec<&'static str> {
        with_config!(self, config => config.secret_names())
    }

    pub(crate) fn get_health_check_document(self) -> OcrDocument {
        with_config!(self, config => config.get_health_check_document())
    }

    pub(crate) fn resolve_connection_params(
        self,
        inputs: OcrCredentialInputs,
    ) -> ResolvedOcrCredentials {
        with_config!(self, config => config.resolve_connection_params(inputs))
    }

    pub(crate) async fn ocr(
        self,
        client: &OcrClient,
        request: &PreparedOcrRequest,
        hooks: &dyn CallHooks<Error>,
    ) -> Result<LiteLLMOcrResponse, Error> {
        with_config!(self, config => handler::ocr(&config, client, request, hooks).await)
    }
}

pub fn get_api_key_env_var(
    model: &str,
    custom_llm_provider: Option<&str>,
) -> Result<Option<&'static str>, Error> {
    Ok(resolve_provider_config(model, custom_llm_provider)?
        .1
        .get_api_key_env_var())
}

pub fn get_health_check_document(
    model: &str,
    custom_llm_provider: Option<&str>,
) -> Result<OcrDocument, Error> {
    Ok(resolve_provider_config(model, custom_llm_provider)?
        .1
        .get_health_check_document())
}

/// The URL the resolved Azure AI OCR config would call for `model` on
/// `api_base`, used by the passthrough logger to tell whether a relayed
/// response is OCR-costable.
pub fn passthrough_url(
    model: &str,
    api_base: &str,
    settings: OcrSettings,
) -> Result<String, Error> {
    let (model, config) = resolve_provider_config(model, Some("azure_ai"))?;
    let arguments = CallArguments::default();
    let request = PreparedOcrRequest {
        model: model.clone(),
        document: OcrDocument::DocumentUrl {
            document_url: String::new(),
            extra_fields: BTreeMap::new(),
        },
        connection: OcrConnection::new(
            ResolvedOcrCredentials {
                api_key: None,
                api_base: Some(Sourced::new(api_base.to_owned(), InputSource::Deployment)),
            },
            OcrTransportConfig::default(),
            settings,
            std::sync::Arc::new(|_: &str| -> Option<String> { None }),
        ),
        caller_document: false,
        optional_params: arguments.clone(),
        input_sources: BTreeMap::new(),
        azure_ad_token_provider: None,
    };
    let environment: Vec<(String, String)> = Vec::new();
    match config {
        OcrConfigKind::AzureAi => AzureAiOcrConfig.get_complete_url(
            &request,
            &AzureAiOcrConfig.map_ocr_params(&arguments, &model)?,
            &environment,
        ),
        OcrConfigKind::AzureCohere => AzureAICohereParseConfig.get_complete_url(
            &request,
            &AzureAICohereParseConfig.map_ocr_params(&arguments, &model)?,
            &environment,
        ),
        OcrConfigKind::AzureDocumentIntelligence => AzureDocumentIntelligenceOcrConfig
            .get_complete_url(
                &request,
                &AzureDocumentIntelligenceOcrConfig.map_ocr_params(&arguments, &model)?,
                &environment,
            ),
        other => {
            let provider: &'static str = other.provider().into();
            Err(Error::InvalidProvider(provider.to_owned()))
        }
    }
}

/// Normalize a relayed Azure AI OCR response body into the LiteLLM OCR shape.
pub fn passthrough_transform(model: &str, body: &[u8]) -> Result<LiteLLMOcrResponse, Error> {
    let (model, config) = resolve_provider_config(model, Some("azure_ai"))?;
    match config {
        OcrConfigKind::AzureAi => {
            AzureAiOcrConfig.transform_ocr_response(&model, body, OcrResponseFormat::Litellm)
        }
        OcrConfigKind::AzureCohere => AzureAICohereParseConfig.transform_ocr_response(
            &model,
            body,
            OcrResponseFormat::Litellm,
        ),
        OcrConfigKind::AzureDocumentIntelligence => AzureDocumentIntelligenceOcrConfig
            .transform_ocr_response(&model, body, OcrResponseFormat::Litellm),
        other => {
            let provider: &'static str = other.provider().into();
            Err(Error::InvalidProvider(provider.to_owned()))
        }
    }
}

#[derive(Clone, Copy, Debug, EnumString, IntoStaticStr, PartialEq, Eq)]
#[strum(serialize_all = "snake_case")]
pub(crate) enum OcrProvider {
    AwsTextract,
    Cohere,
    Mistral,
    AzureAi,
    Reducto,
    VertexAi,
}

pub(crate) fn resolve_provider_config(
    model: &str,
    custom_llm_provider: Option<&str>,
) -> Result<(String, OcrConfigKind), Error> {
    let provider =
        get_custom_llm_provider(model, custom_llm_provider).unwrap_or(CustomLlmProvider {
            model,
            custom_llm_provider: OcrProvider::Mistral.into(),
        });
    let ocr_provider = provider
        .custom_llm_provider
        .parse::<OcrProvider>()
        .map_err(|_| Error::InvalidProvider(provider.custom_llm_provider.to_string()))?;
    let config = match ocr_provider {
        OcrProvider::AwsTextract => match TextractOperation::from_model(provider.model)? {
            TextractOperation::DetectDocumentText => OcrConfigKind::AwsTextract,
            TextractOperation::AnalyzeDocument => OcrConfigKind::AwsTextractAnalyze,
        },
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
    use std::collections::HashSet;

    use litellm_auth::{InputSource, Sourced};
    use litellm_llms::{
        base_llm::ocr::document::InlineDocument, cohere::ocr::transformation::validate_document,
    };
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case(OcrConfigKind::AwsTextract)]
    #[case(OcrConfigKind::AwsTextractAnalyze)]
    #[case(OcrConfigKind::Cohere)]
    #[case(OcrConfigKind::Mistral)]
    #[case(OcrConfigKind::AzureAi)]
    #[case(OcrConfigKind::AzureCohere)]
    #[case(OcrConfigKind::AzureDocumentIntelligence)]
    #[case(OcrConfigKind::ReductoLegacy)]
    #[case(OcrConfigKind::ReductoV3)]
    #[case(OcrConfigKind::VertexAi)]
    #[case(OcrConfigKind::VertexDeepSeek)]
    fn secret_names_include_api_keys_without_duplicates(#[case] config: OcrConfigKind) {
        let names = config.secret_names();
        let unique = names.iter().collect::<HashSet<_>>();
        assert_eq!(names.len(), unique.len());
        if let Some(api_key) = config.get_api_key_env_var() {
            assert!(names.contains(&api_key));
        }
    }

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
            Err(Error::InvalidProvider(value)) if value == provider
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
        let inline = InlineDocument::parse(document.source()).unwrap().unwrap();
        assert_eq!(inline.mime_type().to_string(), "application/pdf");
        assert!(inline.decode(4096).unwrap().starts_with(b"%PDF-"));
    }

    #[rstest]
    #[case("cohere/parse")]
    #[case("azure_ai/cohere-parse")]
    fn png_health_check_documents_are_valid(#[case] model: &str) {
        let document = get_health_check_document(model, None).unwrap();
        validate_document(&document).unwrap();
        let inline = InlineDocument::parse(document.source()).unwrap().unwrap();
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
            api_key: Some(Sourced::new(
                litellm_auth::SecretValue::new("explicit-key"),
                InputSource::Deployment,
            )),
            api_base: Some(Sourced::new(
                "https://explicit.test".into(),
                InputSource::Deployment,
            )),
            dynamic_api_key: Some(Sourced::new(
                litellm_auth::SecretValue::new("dynamic-key"),
                InputSource::Environment,
            )),
            dynamic_api_base: Some(Sourced::new(
                "https://dynamic.test".into(),
                InputSource::Request,
            )),
        });
        assert_eq!(
            connection
                .api_key
                .as_ref()
                .map(|value| value.value().expose()),
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
        let dynamic_key = dynamic_value.map(|value| {
            Sourced::new(
                litellm_auth::SecretValue::new(value),
                InputSource::Environment,
            )
        });
        let dynamic_base =
            dynamic_value.map(|value| Sourced::new(value.into(), InputSource::Environment));
        let connection = OcrConfigKind::Mistral.resolve_connection_params(OcrCredentialInputs {
            api_key: Some(Sourced::new(
                litellm_auth::SecretValue::new("explicit-key"),
                InputSource::Deployment,
            )),
            api_base: Some(Sourced::new(
                "https://explicit.test".into(),
                InputSource::Deployment,
            )),
            dynamic_api_key: dynamic_key,
            dynamic_api_base: dynamic_base,
        });
        assert_eq!(
            connection
                .api_key
                .as_ref()
                .map(|value| value.value().expose()),
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
                api_key: explicit_key.map(|value| {
                    Sourced::new(
                        litellm_auth::SecretValue::new(value),
                        InputSource::Deployment,
                    )
                }),
                api_base: explicit_base
                    .map(|value| Sourced::new(value.into(), InputSource::Deployment)),
                dynamic_api_key: Some(Sourced::new(
                    litellm_auth::SecretValue::new("dynamic-key"),
                    InputSource::Environment,
                )),
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
                .map(|value| value.value().expose()),
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
    #[case::misspelled_operation("aws_textract/analyse-document")]
    #[case::operation_name_from_the_api("aws_textract/AnalyzeDocument")]
    fn textract_models_outside_its_two_operations_are_refused(#[case] model: &str) {
        assert!(matches!(
            resolve_provider_config(model, None),
            Err(Error::InvalidModel {
                provider: "aws_textract",
                ..
            })
        ));
    }

    #[rstest]
    #[case("aws_textract/detect-document-text", OcrConfigKind::AwsTextract)]
    #[case("aws_textract/analyze-document", OcrConfigKind::AwsTextractAnalyze)]
    #[case("aws_textract/Analyze-Document", OcrConfigKind::AwsTextractAnalyze)]
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
        assert!(matches!(&error, Error::InvalidProvider(provider) if provider == "not_a_provider"));
        assert_eq!(error.http_status_code(), Some(400));
    }

    fn passthrough_url_for(model: &str, api_base: &str) -> String {
        passthrough_url(model, api_base, OcrSettings::default()).unwrap()
    }

    #[rstest]
    #[case(
        "azure_ai/mistral-document-ai-2512",
        "https://foundry.services.ai.azure.com",
        "/providers/mistral/azure/ocr"
    )]
    #[case(
        "azure_ai/Cohere-parse-v5",
        "https://foundry.services.ai.azure.com",
        "/providers/cohere/v2/parse"
    )]
    #[case(
        "azure_ai/doc-intelligence/prebuilt-layout",
        "https://foundry.services.ai.azure.com",
        "/documentintelligence/documentModels/prebuilt-layout:analyze"
    )]
    fn passthrough_url_builds_the_resolved_provider_ocr_endpoint(
        #[case] model: &str,
        #[case] api_base: &str,
        #[case] expected_path: &str,
    ) {
        let url = passthrough_url_for(model, api_base);
        assert_eq!(url::Url::parse(&url).unwrap().path(), expected_path);
    }

    #[test]
    fn passthrough_transform_normalizes_the_provider_body() {
        let body = br#"{
            "pages": [{"index": 0, "markdown": "page one"}, {"index": 1, "markdown": "page two"}],
            "model": "mistral-document-ai-2512",
            "usage_info": {"pages_processed": 2}
        }"#;
        let response = passthrough_transform("azure_ai/mistral-document-ai-2512", body).unwrap();
        let json = response.into_json();
        assert_eq!(json["usage_info"]["pages_processed"], 2);
        assert_eq!(json["pages"][0]["markdown"], "page one");
    }

    #[test]
    fn passthrough_transform_rejects_unparseable_bodies() {
        assert!(passthrough_transform("azure_ai/mistral-document-ai-2512", b"not json").is_err());
    }
}
