use super::super::OcrAdapter;
use crate::Error;
use crate::ocr::OcrClient;
use crate::ocr::codecs::cohere::{
    CohereParams, CohereResponse, transform_request, transform_response, validate_document,
};
use crate::ocr::document::{inline_remote_document, validate_inline_document};
use crate::ocr::error::{OcrError, OcrRequestError, OcrResponseError};
use crate::ocr::prepare::{credential_env, transform_request_body};
use crate::ocr::registry::OcrProvider;
use crate::ocr::types::{LiteLLMOcrRequest, LiteLLMOcrResponse};
use crate::providers::azure_ai::auth::AzureAuthInputs;
use crate::url_utils::ApiUrl;

const AZURE_AI_API_BASE_ENV: &str = "AZURE_AI_API_BASE";

pub(crate) struct AzureCohereAdapter;

impl OcrAdapter for AzureCohereAdapter {
    type ProviderResponse = CohereResponse;
    const PROVIDER: OcrProvider = OcrProvider::AzureAi;

    async fn prepare_request(
        &self,
        request: &LiteLLMOcrRequest,
        client: &OcrClient,
    ) -> Result<reqwest::Request, OcrError> {
        let params = super::super::super::wire::decode_request_value::<CohereParams>(
            serde_json::Value::Object(request.optional_params.clone()),
            "optional_params",
        )?;
        let mut config = AzureAuthInputs::from_sourced_optional_params(
            &request.optional_params,
            &request.input_sources,
        )
        .map_err(Error::from)?;
        config.azure_ad_token_provider = request.azure_ad_token_provider.clone();
        let base = request
            .connection
            .api_base
            .clone()
            .or_else(|| credential_env(AZURE_AI_API_BASE_ENV))
            .filter(|base| !base.trim().is_empty())
            .ok_or_else(|| {
                Error::Auth(
                    "Missing Azure AI API Base - Set AZURE_AI_API_BASE or pass api_base".into(),
                )
            })?;
        let headers =
            super::validate_ai_environment(&request.connection, &config, &credential_env).await?;
        validate_document(&request.document)?;
        let remote = request.document.source().starts_with("http://")
            || request.document.source().starts_with("https://");
        let document = inline_remote_document(
            client.document_fetcher(),
            request.document.clone(),
            &request.connection,
        )
        .await?;
        let body = transform_request(&request.model, document, params)?;
        transform_request_body(
            client,
            request,
            &complete_url(&base)?,
            &headers,
            !remote,
            body,
            |body| {
                validate_document(&body.document)?;
                validate_inline_document(&body.document)
            },
        )
        .await
    }

    fn transform_ocr_response(
        &self,
        request: &LiteLLMOcrRequest,
        response: Self::ProviderResponse,
    ) -> Result<LiteLLMOcrResponse, OcrResponseError> {
        transform_response(&request.model, response)
    }
}

fn complete_url(base: &str) -> Result<String, OcrError> {
    let mut url = reqwest::Url::parse(base).map_err(|_| invalid_api_base())?;
    if !matches!(url.scheme(), "http" | "https") {
        return Err(invalid_api_base().into());
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
        .map_err(|_| invalid_api_base().into())
}

fn invalid_api_base() -> OcrRequestError {
    OcrRequestError::RequestField {
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
                complete_url(&format!("https://example.com{suffix}?tenant=a")).unwrap(),
                "https://example.com/providers/cohere/v2/parse?tenant=a"
            );
        }
        assert_eq!(
            complete_url("https://example.com/v2/parse?tenant=a").unwrap(),
            "https://example.com/v2/parse?tenant=a"
        );
        assert!(complete_url("relative/path").is_err());
    }
}
