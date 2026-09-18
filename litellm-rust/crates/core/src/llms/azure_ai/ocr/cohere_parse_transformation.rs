use litellm_core_utils::{call_arguments::CallArguments, url_utils::ApiUrl};
use serde_json::Value;

use crate::{
    llms::{
        base_llm::ocr::transformation::{BaseOcrConfig, OcrRequestContext},
        cohere::ocr::{
            CohereOptions,
            transformation::{CohereParseConfig, CohereRequest},
            validate_document,
        },
    },
    ocr::{
        OcrClient,
        document::{inline_remote_document, validate_inline_document},
        types::{LiteLLMOcrResponse, OcrDocument, PreparedOcrRequest},
    },
};

#[derive(Default)]
pub(crate) struct AzureAICohereParseConfig;

impl BaseOcrConfig for AzureAICohereParseConfig {
    type OcrParams = CohereOptions;
    type ProviderRequest = CohereRequest;
    type Environment = Vec<(String, String)>;

    fn get_api_key_env_var(&self) -> Option<&'static str> {
        super::transformation::AzureAiOcrConfig.get_api_key_env_var()
    }

    fn get_health_check_document(&self) -> OcrDocument {
        CohereParseConfig.get_health_check_document()
    }

    async fn validate_environment(
        &self,
        request: &PreparedOcrRequest,
        client: &OcrClient,
    ) -> Result<Self::Environment, crate::ocr::Error> {
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
    ) -> Result<String, crate::ocr::Error> {
        let base = super::transformation::AzureAiOcrConfig::resolve_api_base(
            request.connection.api_base.as_deref(),
            &crate::ocr::prepare::credential_env,
        )?;
        self.get_complete_url(&base)
    }

    fn transform_ocr_request(
        &self,
        model: &str,
        document: OcrDocument,
        params: &CohereOptions,
        headers: &[(String, String)],
    ) -> Result<CohereRequest, crate::ocr::Error> {
        CohereParseConfig.transform_ocr_request(model, document, params, headers)
    }

    fn get_supported_ocr_params(&self, model: &str) -> &'static [&'static str] {
        CohereParseConfig.get_supported_ocr_params(model)
    }

    fn map_ocr_params(
        &self,
        arguments: &CallArguments,
        model: &str,
    ) -> Result<CohereOptions, crate::ocr::Error> {
        CohereParseConfig.map_ocr_params(arguments, model)
    }

    async fn async_transform_ocr_request(
        &self,
        model: &str,
        document: OcrDocument,
        optional_params: &CohereOptions,
        headers: &[(String, String)],
        context: OcrRequestContext<'_>,
    ) -> Result<CohereRequest, crate::ocr::Error> {
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
        request_format: crate::ocr::types::OcrResponseFormat,
    ) -> Result<LiteLLMOcrResponse, crate::ocr::Error> {
        CohereParseConfig.transform_ocr_response(model, raw_response, request_format)
    }

    fn validate_request_body(&self, body: &Value) -> Result<(), crate::ocr::Error> {
        let document = crate::ocr::prepare::body_document(body)?;
        validate_document(&document)?;
        validate_inline_document(&document)
    }
}

impl AzureAICohereParseConfig {
    fn get_complete_url(&self, base: &str) -> Result<String, crate::ocr::Error> {
        let mut url = reqwest::Url::parse(base).map_err(|_| invalid_api_base())?;
        if !matches!(url.scheme(), "http" | "https") {
            return Err(invalid_api_base());
        }
        let path = url.path().trim_end_matches('/').to_string();
        if path.ends_with("/v2/parse") {
            url.set_path(&path);
            return Ok(url.into());
        }
        url.set_path(path.strip_suffix("/models").unwrap_or(&path));
        ApiUrl::parse(url.as_str())
            .and_then(|url| url.complete_path(&["providers", "cohere", "v2", "parse"]))
            .map(|url| url.into_string())
            .map_err(|_| invalid_api_base())
    }
}

fn invalid_api_base() -> crate::ocr::Error {
    crate::ocr::Error::RequestField {
        path: "api_base".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
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
                    .unwrap(),
                "https://example.com/providers/cohere/v2/parse?tenant=a"
            );
        }
        assert_eq!(
            AzureAICohereParseConfig
                .get_complete_url("https://example.com/v2/parse?tenant=a")
                .unwrap(),
            "https://example.com/v2/parse?tenant=a"
        );
        assert!(
            AzureAICohereParseConfig
                .get_complete_url("relative/path")
                .is_err()
        );
    }
}
