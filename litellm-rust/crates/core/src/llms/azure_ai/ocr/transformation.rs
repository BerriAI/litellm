use crate::constants::AZURE_AI_OCR_PATH;
use crate::llms::base_llm::ocr::transformation::{BaseOcrConfig, OcrRequestContext};
use crate::llms::mistral::ocr::MistralOcrResponse;
use crate::llms::mistral::ocr::transformation::{MistralOCRConfig, MistralOcrRequest};
use crate::ocr::OcrClient;
use crate::ocr::document::{inline_remote_document, validate_inline_document};
use crate::ocr::prepare::{credential_env, transform_request_body};
use crate::ocr::types::{LiteLLMOcrRequest, LiteLLMOcrResponse, OcrConnection, OcrDocument};
use crate::params::OpaqueParams;
use crate::url_utils::ApiUrl;
use litellm_auth::{InputSource, Sourced};
use litellm_auth_azure::AzureAuthInputs;

const AZURE_AI_API_KEY_ENV: &str = "AZURE_AI_API_KEY";
const AZURE_AI_API_BASE_ENV: &str = "AZURE_AI_API_BASE";

#[derive(Clone, Debug, Default)]
pub(crate) struct AzureAIOCRConfig;

impl BaseOcrConfig for AzureAIOCRConfig {
    type OcrParams = OpaqueParams;
    type ProviderRequest = MistralOcrRequest;
    type ProviderResponse = MistralOcrResponse;

    fn get_supported_ocr_params(&self, model: &str) -> &'static [&'static str] {
        MistralOCRConfig.get_supported_ocr_params(model)
    }

    fn map_ocr_params(
        &self,
        non_default_params: &OpaqueParams,
        optional_params: &OpaqueParams,
        model: &str,
    ) -> Result<OpaqueParams, crate::ocr::Error> {
        MistralOCRConfig.map_ocr_params(non_default_params, optional_params, model)
    }

    async fn async_transform_ocr_request(
        &self,
        model: &str,
        document: OcrDocument,
        optional_params: &OpaqueParams,
        headers: &[(String, String)],
        context: OcrRequestContext<'_>,
    ) -> Result<MistralOcrRequest, crate::ocr::Error> {
        let document = inline_remote_document(
            context.client.document_fetcher(),
            document,
            context.connection,
        )
        .await?;
        MistralOCRConfig.transform_ocr_request(model, document, optional_params, headers)
    }

    fn normalize_response(
        &self,
        model: &str,
        response: MistralOcrResponse,
    ) -> Result<LiteLLMOcrResponse, crate::ocr::Error> {
        MistralOCRConfig.normalize_response(model, response)
    }
}

impl AzureAIOCRConfig {
    pub(crate) async fn prepare_request(
        &self,
        request: &LiteLLMOcrRequest,
        client: &OcrClient,
    ) -> Result<reqwest::Request, crate::ocr::Error> {
        let params = self.map_ocr_params(
            &request.optional_params,
            &OpaqueParams::default(),
            &request.model,
        )?;
        let config = AzureAuthInputs {
            azure_ad_token_provider: request.azure_ad_token_provider.clone(),
            ..AzureAuthInputs::from_sourced_optional_params(
                &request.optional_params,
                &request.input_sources,
            )
            .map_err(crate::ocr::Error::from)?
        };
        let url = self.get_complete_url(request.connection.api_base.as_deref(), &credential_env)?;
        let headers = self
            .validate_environment(&request.connection, &config, &credential_env)
            .await?;
        let retains_document = !request.document.source().starts_with("http://")
            && !request.document.source().starts_with("https://");
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
            &url,
            &headers,
            retains_document,
            body,
            |body| validate_inline_document(&body.document),
        )
        .await
    }
}

impl AzureAIOCRConfig {
    fn get_complete_url(
        &self,
        api_base: Option<&str>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, crate::ocr::Error> {
        let base = nonblank(api_base.map(str::to_string))
        .or_else(|| nonblank(env_lookup(AZURE_AI_API_BASE_ENV)))
        .ok_or_else(|| crate::ocr::Error::Auth(litellm_auth::Error::ProviderAuthentication("Missing Azure AI API Base - Set AZURE_AI_API_BASE environment variable or pass api_base parameter".into())))?;
        let path: Vec<&str> = AZURE_AI_OCR_PATH.trim_matches('/').split('/').collect();
        ApiUrl::parse(&base)
            .and_then(|url| url.complete_path(&path))
            .map(|url| url.into_string())
            .map_err(|_| crate::ocr::Error::RequestField {
                path: "api_base".into(),
            })
    }

    pub(super) async fn validate_environment(
        &self,
        connection: &OcrConnection,
        config: &AzureAuthInputs,
        env_lookup: &(dyn Fn(&str) -> Option<String> + Sync),
    ) -> Result<Vec<(String, String)>, crate::ocr::Error> {
        if crate::http_utils::has_header(&connection.extra_headers, "authorization") {
            if config.azure_ad_token_provider.is_some() {
                super::common_utils::resolve_entra(config, env_lookup).await?;
            }
            super::common_utils::validate_destination(connection, connection.extra_headers_source)?;
            return Ok(connection.extra_headers.clone());
        }
        let key = nonblank(connection.api_key.clone())
            .map(|value| Sourced::new(value, connection.api_key_source))
            .or_else(|| {
                nonblank(env_lookup(AZURE_AI_API_KEY_ENV))
                    .map(|value| Sourced::new(value, InputSource::Environment))
            });
        if let Some(key) = key {
            super::common_utils::validate_destination(connection, key.source())?;
            return Ok(bearer_headers(connection, key.value()));
        }
        let key = super::common_utils::resolve_entra(config, env_lookup)
            .await?
            .ok_or(crate::ocr::Error::MissingAzureAiCredentials)?;
        super::common_utils::validate_destination(connection, key.source())?;
        Ok(bearer_headers(connection, key.value()))
    }
}

fn bearer_headers(connection: &OcrConnection, key: &str) -> Vec<(String, String)> {
    std::iter::once(("Authorization".into(), format!("Bearer {key}")))
        .chain(connection.extra_headers.clone())
        .collect()
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
            AzureAIOCRConfig
                .get_complete_url(Some("https://example.com/?tenant=a"), &|_| None)
                .unwrap(),
            "https://example.com/providers/mistral/azure/ocr?tenant=a"
        );
        assert_eq!(
            AzureAIOCRConfig
                .get_complete_url(
                    Some("https://example.com/providers/mistral/azure/ocr"),
                    &|_| None
                )
                .unwrap(),
            "https://example.com/providers/mistral/azure/ocr"
        );
    }

    #[tokio::test]
    async fn supplied_authorization_precedes_keys() {
        let connection = OcrConnection {
            api_key: Some("request-key".into()),
            extra_headers: vec![("authorization".into(), "Bearer prepared".into())],
            ..Default::default()
        };
        assert_eq!(
            AzureAIOCRConfig
                .validate_environment(&connection, &Default::default(), &|_| {
                    Some("environment-key".into())
                })
                .await
                .unwrap(),
            connection.extra_headers
        );
    }

    #[tokio::test]
    async fn request_key_precedes_environment_key() {
        let connection = OcrConnection {
            api_key: Some("request-key".into()),
            ..Default::default()
        };
        assert_eq!(
            AzureAIOCRConfig
                .validate_environment(&connection, &Default::default(), &|_| {
                    Some("environment-key".into())
                })
                .await
                .unwrap()[0],
            ("Authorization".into(), "Bearer request-key".into())
        );
    }

    #[tokio::test]
    async fn request_endpoint_cannot_receive_environment_key() {
        let connection = OcrConnection {
            api_base: Some("https://request.example".into()),
            api_base_source: InputSource::Request,
            ..Default::default()
        };

        let error = AzureAIOCRConfig
            .validate_environment(&connection, &Default::default(), &|name| {
                (name == AZURE_AI_API_KEY_ENV).then(|| "environment-key".into())
            })
            .await
            .unwrap_err();

        assert!(
            error
                .to_string()
                .contains("request-controlled Azure endpoint")
        );
    }

    #[tokio::test]
    async fn request_endpoint_accepts_request_owned_key() {
        let connection = OcrConnection {
            api_key: Some("request-key".into()),
            api_key_source: InputSource::Request,
            api_base: Some("https://request.example".into()),
            api_base_source: InputSource::Request,
            ..Default::default()
        };

        let headers = AzureAIOCRConfig
            .validate_environment(&connection, &Default::default(), &|_| None)
            .await
            .unwrap();

        assert_eq!(
            headers[0],
            ("Authorization".into(), "Bearer request-key".into())
        );
    }
}
