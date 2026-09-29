use litellm_core_utils::{call_arguments::CallArguments, url_utils::ApiUrl};
use serde_json::Value;

use crate::{
    base_llm::ocr::{
        document::{inline_remote_document, validate_inline_document},
        error::Error,
        handler::OcrClient,
        transformation::{
            BaseOcrConfig, LiteLLMOcrResponse, OcrDocument, OcrRequestContext, OcrResponseFormat,
            PreparedOcrRequest,
        },
    },
    cohere::ocr::transformation::{
        CohereOptions, CohereParseConfig, CohereRequest, validate_document,
    },
};

pub const AZURE_COHERE_PARSE_PATH: [&str; 4] = ["providers", "cohere", "v2", "parse"];

#[derive(Default)]
pub struct AzureAICohereParseConfig;

impl BaseOcrConfig for AzureAICohereParseConfig {
    type OcrParams = CohereOptions;
    type ProviderRequest = CohereRequest;
    type Environment = Vec<(String, String)>;

    fn get_api_key_env_var(&self) -> Option<&'static str> {
        super::transformation::AzureAiOcrConfig.get_api_key_env_var()
    }

    fn secret_names(&self) -> Vec<&'static str> {
        super::transformation::AzureAiOcrConfig.secret_names()
    }

    fn get_health_check_document(&self) -> OcrDocument {
        CohereParseConfig.get_health_check_document()
    }

    async fn validate_environment(
        &self,
        request: &PreparedOcrRequest,
        client: &OcrClient,
    ) -> Result<Self::Environment, Error> {
        BaseOcrConfig::validate_environment(
            &super::transformation::AzureAiOcrConfig,
            request,
            client,
        )
        .await
    }

    fn get_complete_url(
        &self,
        request: &PreparedOcrRequest,
        _params: &Self::OcrParams,
        _environment: &Self::Environment,
    ) -> Result<url::Url, Error> {
        let base = super::transformation::AzureAiOcrConfig::resolve_api_base(
            request.connection.api_base.as_deref(),
            &|name: &str| request.connection.secret(name),
        )?;
        self.get_complete_url(&base)
    }

    fn transform_ocr_request(
        &self,
        model: &str,
        document: OcrDocument,
        params: &CohereOptions,
        headers: &[(String, String)],
    ) -> Result<CohereRequest, Error> {
        CohereParseConfig.transform_ocr_request(model, document, params, headers)
    }

    fn get_supported_ocr_params(&self, model: &str) -> &'static [&'static str] {
        CohereParseConfig.get_supported_ocr_params(model)
    }

    fn map_ocr_params(
        &self,
        arguments: &CallArguments,
        model: &str,
    ) -> Result<CohereOptions, Error> {
        CohereParseConfig.map_ocr_params(arguments, model)
    }

    async fn async_transform_ocr_request(
        &self,
        model: &str,
        document: OcrDocument,
        optional_params: &CohereOptions,
        headers: &[(String, String)],
        context: OcrRequestContext<'_>,
    ) -> Result<CohereRequest, Error> {
        validate_document(&document)?;
        let document = inline_remote_document(
            context.client.document_fetcher(),
            document,
            context.connection,
        )
        .await?;
        self.transform_ocr_request(model, document, optional_params, headers)
    }

    fn transform_ocr_response(
        &self,
        model: &str,
        raw_response: &[u8],
        request_format: OcrResponseFormat,
    ) -> Result<LiteLLMOcrResponse, Error> {
        CohereParseConfig.transform_ocr_response(model, raw_response, request_format)
    }

    fn validate_request_body(&self, body: &Value) -> Result<(), Error> {
        let document = crate::base_llm::ocr::handler::body_document(body)?;
        validate_document(&document)?;
        validate_inline_document(&document)
    }
}

impl AzureAICohereParseConfig {
    fn get_complete_url(&self, base: &str) -> Result<url::Url, Error> {
        let mut url = reqwest::Url::parse(base).map_err(|_| invalid_api_base())?;
        if !matches!(url.scheme(), "http" | "https") {
            return Err(invalid_api_base());
        }
        let segments: Vec<_> = url.path_segments().ok_or_else(invalid_api_base)?.collect();
        let segments = segments.strip_suffix(&[""]).unwrap_or(&segments);
        if segments.ends_with(&["v2", "parse"]) {
            return Ok(url);
        }
        if segments.last() == Some(&"models") {
            url.path_segments_mut()
                .map_err(|()| invalid_api_base())?
                .pop_if_empty()
                .pop();
        }
        ApiUrl::from_url(url)
            .and_then(|url| url.complete_path(&AZURE_COHERE_PARSE_PATH))
            .map(|url| url.into_url())
            .map_err(|_| invalid_api_base())
    }
}

fn invalid_api_base() -> Error {
    Error::RequestField {
        path: "api_base".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[rstest::rstest]
    fn completes_foundry_urls_without_duplicate_paths_and_preserves_queries() {
        for suffix in [
            "",
            "/models",
            "/providers/cohere/v2",
            "/providers/cohere/v2/parse",
        ] {
            assert_eq!(
                AzureAICohereParseConfig
                    .get_complete_url(&format!("https://example.com{suffix}?tenant=a"))
                    .unwrap()
                    .to_string(),
                "https://example.com/providers/cohere/v2/parse?tenant=a"
            );
        }
        assert_eq!(
            AzureAICohereParseConfig
                .get_complete_url("https://example.com/v2/parse?tenant=a")
                .unwrap()
                .to_string(),
            "https://example.com/v2/parse?tenant=a"
        );
        assert!(
            AzureAICohereParseConfig
                .get_complete_url("relative/path")
                .is_err()
        );
    }
}
