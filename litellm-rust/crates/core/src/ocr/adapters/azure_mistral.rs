use super::OcrAdapter;
use crate::Error;
use crate::constants::AZURE_AI_OCR_PATH;
use crate::ocr::OcrClient;
use crate::ocr::codecs::mistral::{self, MistralOcrParams, MistralOcrResponse};
use crate::ocr::document::{inline_remote_document, validate_inline_document};
use crate::ocr::error::{OcrError, OcrRequestError, OcrResponseError};
use crate::ocr::prepare::{_prepare_ocr_request, credential_env, transform_request_body};
use crate::ocr::registry::OcrProvider;
use crate::ocr::types::{LiteLLMOcrRequest, LiteLLMOcrResponse, OcrConnection};
use crate::url_utils::ApiUrl;

const AZURE_AI_API_KEY_ENV: &str = "AZURE_AI_API_KEY";
const AZURE_AI_API_BASE_ENV: &str = "AZURE_AI_API_BASE";

#[derive(Clone, Debug)]
pub(crate) struct AzureMistralAdapter;

impl OcrAdapter for AzureMistralAdapter {
    type ProviderResponse = MistralOcrResponse;
    const PROVIDER: OcrProvider = OcrProvider::AzureAi;

    #[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
    async fn transform_ocr_request(
        &self,
        request: &LiteLLMOcrRequest,
        client: &OcrClient,
    ) -> Result<reqwest::Request, OcrError> {
        let params: MistralOcrParams = _prepare_ocr_request(request)?;
        let headers = authenticate(&request.connection, &credential_env)?;
        let url = get_complete_url(request.connection.api_base.as_deref(), &credential_env)?;
        let document = inline_remote_document(
            client.document_fetcher(),
            request.document.clone(),
            &request.connection,
        )
        .await?;
        let body = mistral::transform_ocr_request(&request.model, document, &params)?;
        transform_request_body(client, request, &url, &headers, body, |body| {
            validate_inline_document(&body.document)
        })
        .await
    }

    fn transform_ocr_response(
        &self,
        request: &LiteLLMOcrRequest,
        response: Self::ProviderResponse,
    ) -> Result<LiteLLMOcrResponse, OcrResponseError> {
        mistral::transform_ocr_response(&request.model, response)
    }
}

fn get_complete_url(
    api_base: Option<&str>,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Result<String, OcrError> {
    let base = nonblank(api_base.map(str::to_string))
        .or_else(|| nonblank(env_lookup(AZURE_AI_API_BASE_ENV)))
        .ok_or_else(|| Error::Auth(
            "Missing Azure AI API Base - Set AZURE_AI_API_BASE environment variable or pass api_base parameter".into(),
        ))?;
    let path: Vec<&str> = AZURE_AI_OCR_PATH.trim_matches('/').split('/').collect();
    ApiUrl::parse(&base)
        .and_then(|url| url.complete_path(&path))
        .map(|url| url.into_string())
        .map_err(|_| {
            OcrRequestError::RequestField {
                path: "api_base".into(),
            }
            .into()
        })
}

fn authenticate(
    connection: &OcrConnection,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Result<Vec<(String, String)>, OcrError> {
    if crate::http_utils::has_header(&connection.extra_headers, "authorization") {
        return Ok(connection.extra_headers.clone());
    }
    let key = nonblank(connection.api_key.clone())
        .or_else(|| nonblank(env_lookup(AZURE_AI_API_KEY_ENV)))
        .ok_or(Error::MissingAzureAiCredentials)?;
    Ok(
        std::iter::once(("Authorization".into(), format!("Bearer {key}")))
            .chain(connection.extra_headers.clone())
            .collect(),
    )
}

fn nonblank(value: Option<String>) -> Option<String> {
    value
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn completes_azure_path_and_preserves_query() {
        assert_eq!(
            get_complete_url(Some("https://example.com/?tenant=a"), &|_| None).unwrap(),
            "https://example.com/providers/mistral/azure/ocr?tenant=a"
        );
        assert_eq!(
            get_complete_url(
                Some("https://example.com/providers/mistral/azure/ocr"),
                &|_| None
            )
            .unwrap(),
            "https://example.com/providers/mistral/azure/ocr"
        );
    }

    #[test]
    fn supplied_authorization_precedes_keys() {
        let connection = OcrConnection {
            api_key: Some("request-key".into()),
            extra_headers: vec![("authorization".into(), "Bearer prepared".into())],
            ..Default::default()
        };
        assert_eq!(
            authenticate(&connection, &|_| Some("environment-key".into())).unwrap(),
            connection.extra_headers
        );
    }

    #[test]
    fn request_key_precedes_environment_key() {
        let connection = OcrConnection {
            api_key: Some("request-key".into()),
            ..Default::default()
        };
        assert_eq!(
            authenticate(&connection, &|_| Some("environment-key".into())).unwrap()[0],
            ("Authorization".into(), "Bearer request-key".into())
        );
    }
}
