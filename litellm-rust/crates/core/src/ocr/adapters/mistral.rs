use super::OcrAdapter;
use crate::Error;
use crate::constants::MISTRAL_OCR_API_BASE;
use crate::ocr::OcrClient;
use crate::ocr::codecs::mistral::{self, MistralOcrParams, MistralOcrResponse};
use crate::ocr::error::{OcrError, OcrRequestError, OcrResponseError};
use crate::ocr::prepare::{
    _prepare_ocr_request, ParsedProviderParams, credential_env, transform_request_body,
};
use crate::ocr::registry::OcrProvider;
use crate::ocr::types::{LiteLLMOcrRequest, LiteLLMOcrResponse, OcrConnection};
use crate::url_utils::ApiUrl;

const MISTRAL_API_KEY_ENV: &str = "MISTRAL_API_KEY";

#[derive(Clone, Debug)]
pub(crate) struct MistralAdapter;

impl OcrAdapter for MistralAdapter {
    type ProviderResponse = MistralOcrResponse;
    const PROVIDER: OcrProvider = OcrProvider::Mistral;

    async fn prepare_request(
        &self,
        request: &LiteLLMOcrRequest,
        client: &OcrClient,
    ) -> Result<reqwest::Request, OcrError> {
        let ParsedProviderParams {
            known: params,
            extra_params: _extra_params,
        } = _prepare_ocr_request::<MistralOcrParams>(request)?;
        let headers = validate_environment(&request.connection, &credential_env)?;
        let url = get_complete_url(request.connection.api_base.as_deref())?;
        let body =
            mistral::transform_ocr_request(&request.model, request.document.clone(), &params)?;
        transform_request_body(client, request, &url, &headers, body, |_| Ok(())).await
    }

    fn transform_ocr_response(
        &self,
        request: &LiteLLMOcrRequest,
        response: Self::ProviderResponse,
    ) -> Result<LiteLLMOcrResponse, OcrResponseError> {
        mistral::transform_ocr_response(&request.model, response)
    }
}

pub(crate) fn get_complete_url(api_base: Option<&str>) -> Result<String, OcrError> {
    let base = api_base
        .map(str::trim)
        .filter(|base| !base.is_empty())
        .unwrap_or(MISTRAL_OCR_API_BASE);
    ApiUrl::parse(base)
        .and_then(|url| url.complete_path(&["v1", "ocr"]))
        .map(|url| url.into_string())
        .map_err(|_| {
            OcrRequestError::RequestField {
                path: "api_base".into(),
            }
            .into()
        })
}

fn validate_environment(
    connection: &OcrConnection,
    env_lookup: &(dyn Fn(&str) -> Option<String> + Sync),
) -> Result<Vec<(String, String)>, OcrError> {
    if crate::http_utils::has_header(&connection.extra_headers, "authorization") {
        return Ok(connection.extra_headers.clone());
    }
    let api_key = connection
        .api_key
        .as_deref()
        .map(str::trim)
        .filter(|key| !key.is_empty())
        .map(str::to_string)
        .or_else(|| env_lookup(MISTRAL_API_KEY_ENV).filter(|key| !key.trim().is_empty()))
        .ok_or(Error::MissingApiKey {
            provider: "Mistral",
        })?;
    Ok(
        std::iter::once(("Authorization".into(), format!("Bearer {api_key}")))
            .chain(connection.extra_headers.clone())
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complete_url_defaults_and_dedupes_v1() {
        assert_eq!(
            get_complete_url(None).unwrap(),
            "https://api.mistral.ai/v1/ocr"
        );
        assert_eq!(
            get_complete_url(Some("https://example.com/v1?tenant=a")).unwrap(),
            "https://example.com/v1/ocr?tenant=a"
        );
        assert_eq!(
            get_complete_url(Some("https://example.com/v1/ocr?tenant=a")).unwrap(),
            "https://example.com/v1/ocr?tenant=a"
        );
    }

    #[test]
    fn environment_prefers_explicit_key_then_environment() {
        let explicit = OcrConnection {
            api_key: Some("explicit".into()),
            ..OcrConnection::default()
        };
        assert_eq!(
            validate_environment(&explicit, &|_| Some("environment".into())).unwrap()[0],
            ("Authorization".into(), "Bearer explicit".into())
        );

        assert_eq!(
            validate_environment(&OcrConnection::default(), &|_| Some("environment".into()))
                .unwrap()[0],
            ("Authorization".into(), "Bearer environment".into())
        );
    }

    #[test]
    fn environment_preserves_forwarded_authorization() {
        let connection = OcrConnection {
            extra_headers: vec![("authorization".into(), "Bearer forwarded".into())],
            ..OcrConnection::default()
        };
        assert_eq!(
            validate_environment(&connection, &|_| None).unwrap(),
            connection.extra_headers
        );
    }

    #[test]
    fn environment_rejects_missing_key() {
        assert!(matches!(
            validate_environment(&OcrConnection::default(), &|_| None),
            Err(OcrError::Public(Error::MissingApiKey {
                provider: "Mistral"
            }))
        ));
    }
}
