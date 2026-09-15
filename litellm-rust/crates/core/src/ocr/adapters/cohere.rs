use super::OcrAdapter;
use crate::Error;
use crate::constants::{COHERE_API_KEY_ENV, COHERE_PARSE_API_BASE};
use crate::ocr::OcrClient;
use crate::ocr::codecs::cohere::{
    CohereParams, CohereResponse, transform_request, transform_response, validate_document,
};
use crate::ocr::error::{OcrError, OcrRequestError, OcrResponseError};
use crate::ocr::prepare::{credential_env, transform_request_body};
use crate::ocr::registry::OcrProvider;
use crate::ocr::types::{LiteLLMOcrRequest, LiteLLMOcrResponse, OcrConnection};
use crate::url_utils::ApiUrl;

pub(crate) struct CohereAdapter;

impl OcrAdapter for CohereAdapter {
    type ProviderResponse = CohereResponse;
    const PROVIDER: OcrProvider = OcrProvider::Cohere;

    async fn prepare_request(
        &self,
        request: &LiteLLMOcrRequest,
        client: &OcrClient,
    ) -> Result<reqwest::Request, OcrError> {
        let params = super::super::wire::decode_request_value::<CohereParams>(
            serde_json::Value::Object(request.optional_params.clone()),
            "optional_params",
        )?;
        let headers = validate_environment(&request.connection, &credential_env)?;
        let url = complete_url(
            request
                .connection
                .api_base
                .as_deref()
                .unwrap_or(COHERE_PARSE_API_BASE),
        )?;
        let body = transform_request(&request.model, request.document.clone(), params)?;
        transform_request_body(client, request, &url, &headers, true, body, |body| {
            validate_document(&body.document)
        })
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
    let parsed = reqwest::Url::parse(base).map_err(|_| invalid_api_base())?;
    if !matches!(parsed.scheme(), "http" | "https") {
        return Err(invalid_api_base().into());
    }
    ApiUrl::parse(base)
        .and_then(|url| url.complete_path(&["v2", "parse"]))
        .map(|url| url.into_string())
        .map_err(|_| invalid_api_base().into())
}

fn invalid_api_base() -> OcrRequestError {
    OcrRequestError::RequestField {
        path: "api_base".into(),
    }
}

fn validate_environment(
    connection: &OcrConnection,
    env_lookup: &(dyn Fn(&str) -> Option<String> + Sync),
) -> Result<Vec<(String, String)>, OcrError> {
    if crate::http_utils::has_header(&connection.extra_headers, "authorization") {
        return Ok(connection.extra_headers.clone());
    }
    let key = connection
        .api_key
        .as_deref()
        .map(str::trim)
        .filter(|key| !key.is_empty())
        .map(str::to_string)
        .or_else(|| env_lookup(COHERE_API_KEY_ENV).filter(|key| !key.trim().is_empty()))
        .ok_or_else(|| {
            Error::Auth("Missing COHERE_API_KEY - set it in the environment or pass api_key".into())
        })?;
    Ok(
        std::iter::once(("Authorization".into(), format!("Bearer {key}")))
            .chain(connection.extra_headers.clone())
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn completes_provider_urls_without_duplicate_paths_and_preserves_queries() {
        for suffix in ["", "/v2", "/v2/parse"] {
            assert_eq!(
                complete_url(&format!("https://example.com{suffix}?tenant=a")).unwrap(),
                "https://example.com/v2/parse?tenant=a"
            );
        }
    }

    #[test]
    fn rejects_invalid_urls_and_blank_keys() {
        assert!(complete_url("relative/path").is_err());
        assert!(complete_url("ftp://example.com").is_err());
        assert!(matches!(
            validate_environment(
                &OcrConnection {
                    api_key: Some("  ".into()),
                    ..Default::default()
                },
                &|_| None,
            ),
            Err(OcrError::Public(Error::Auth(_)))
        ));
    }
}
