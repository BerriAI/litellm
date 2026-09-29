use crate::base_llm::endpoint::ResolvedEndpoint;
use litellm_core_utils::{call_arguments::CallArguments, url_utils::ApiUrl};
use serde_json::Value;

use crate::{
    base_llm::ocr::{
        document::{inline_remote_document, validate_inline_document},
        error::Error,
        handler::OcrClient,
        transformation::{BaseOcrConfig, OcrRequestContext, PreparedOcrRequest},
    },
    cohere::ocr::transformation::{
        CohereOptions, CohereParseConfig, CohereRequest, validate_document,
    },
};
use litellm_llms_types::formats::ocr::{LiteLLMOcrResponse, OcrDocument, OcrResponseFormat};

#[derive(Default)]
pub struct AzureAICohereParseConfig;

impl BaseOcrConfig for AzureAICohereParseConfig {
    type OcrParams = CohereOptions;
    type ProviderRequest = CohereRequest;
    type Environment = Vec<(String, String)>;

    fn connection_env_vars(&self) -> (Option<&'static str>, Option<&'static str>) {
        super::transformation::AzureAiOcrConfig.connection_env_vars()
    }

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
    ) -> Result<ResolvedEndpoint, Error> {
        let base = super::transformation::AzureAiOcrConfig::resolve_api_base(
            request.connection.api_base.as_deref(),
            &|name: &str| request.connection.secret(name),
        )?;
        let url = self.get_complete_url(&base)?;
        ResolvedEndpoint::parse_exact(reqwest::Method::POST, &url).map_err(|_| {
            Error::RequestField {
                path: "api_base".into(),
            }
        })
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

    fn matches_endpoint(&self, _model: &str, path: &str) -> Result<bool, Error> {
        Ok(crate::azure_ai::endpoints::AzureEndpoint::CohereParse
            .path()
            .matches(path))
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
    fn get_complete_url(&self, base: &str) -> Result<String, Error> {
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
            .and_then(|url| {
                url.complete_path(
                    &crate::azure_ai::endpoints::AzureEndpoint::CohereParse
                        .path()
                        .segments()
                        .collect::<Vec<_>>(),
                )
            })
            .map(|url| url.into_string())
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
