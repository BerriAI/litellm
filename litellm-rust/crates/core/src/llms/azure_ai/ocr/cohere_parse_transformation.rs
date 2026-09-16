use crate::llms::base_llm::ocr::transformation::{BaseOcrConfig, OcrRequestContext};
use crate::llms::cohere::ocr::transformation::{CohereParseConfig, CohereRequest};
use crate::llms::cohere::ocr::{CohereParams, CohereResponse, validate_document};
use crate::ocr::OcrArguments;
use crate::ocr::OcrClient;
use crate::ocr::document::{inline_remote_document, validate_inline_document};
use crate::ocr::prepare::{credential_env, transform_request_body};
use crate::ocr::types::{LiteLLMOcrRequest, LiteLLMOcrResponse, OcrDocument};
use crate::url_utils::ApiUrl;
use litellm_auth_azure::AzureAuthInputs;

const AZURE_AI_API_BASE_ENV: &str = "AZURE_AI_API_BASE";

#[derive(Default)]
pub(crate) struct AzureAICohereParseConfig;

impl BaseOcrConfig for AzureAICohereParseConfig {
    type OcrParams = CohereParams;
    type ProviderRequest = CohereRequest;
    type ProviderResponse = CohereResponse;

    fn get_supported_ocr_params(&self, model: &str) -> &'static [&'static str] {
        CohereParseConfig.get_supported_ocr_params(model)
    }

    fn map_ocr_params(
        &self,
        non_default_params: &OcrArguments,
        optional_params: &OcrArguments,
        model: &str,
    ) -> Result<OcrArguments, crate::ocr::Error> {
        CohereParseConfig.map_ocr_params(non_default_params, optional_params, model)
    }

    async fn async_transform_ocr_request(
        &self,
        model: &str,
        document: OcrDocument,
        optional_params: &CohereParams,
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
        CohereParseConfig.transform_ocr_request(model, document, optional_params, headers)
    }

    fn normalize_response(
        &self,
        model: &str,
        response: CohereResponse,
    ) -> Result<LiteLLMOcrResponse, crate::ocr::Error> {
        CohereParseConfig.normalize_response(model, response)
    }
}

impl AzureAICohereParseConfig {
    pub(crate) async fn prepare_request(
        &self,
        request: &LiteLLMOcrRequest,
        client: &OcrClient,
    ) -> Result<reqwest::Request, crate::ocr::Error> {
        let params = self.parse_options(&request.optional_params, &request.model)?;
        let config = AzureAuthInputs {
            azure_ad_token_provider: request.azure_ad_token_provider.clone(),
            ..AzureAuthInputs::from_sourced_optional_params(
                &request.optional_params,
                &request.input_sources,
            )
            .map_err(crate::ocr::Error::from)?
        };
        let base = request
            .connection
            .api_base
            .clone()
            .or_else(|| credential_env(AZURE_AI_API_BASE_ENV))
            .filter(|base| !base.trim().is_empty())
            .ok_or_else(|| {
                crate::ocr::Error::Auth(litellm_auth::Error::ProviderAuthentication(
                    "Missing Azure AI API Base - Set AZURE_AI_API_BASE or pass api_base".into(),
                ))
            })?;
        let headers = super::transformation::AzureAIOCRConfig
            .validate_environment(&request.connection, &config, &credential_env)
            .await?;
        let remote = request.document.source().starts_with("http://")
            || request.document.source().starts_with("https://");
        let body = self
            .async_transform_ocr_request(
                &request.model,
                request.document.clone(),
                &params,
                &headers,
                OcrRequestContext {
                    client,
                    connection: &request.connection,
                },
            )
            .await?;
        transform_request_body(
            client,
            request,
            &self.get_complete_url(&base)?,
            &headers,
            !remote,
            body,
            |body| {
                validate_document(&body.document.as_document())?;
                validate_inline_document(&body.document.as_document())
            },
        )
        .await
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
