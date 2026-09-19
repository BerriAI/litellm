use litellm_auth::{InputSource, Sourced};
use litellm_auth_azure::AzureAuthInputs;
use litellm_core_utils::{call_arguments::CallArguments, params::OpaqueParams, url_utils::ApiUrl};
use serde_json::Value;

use crate::{
    base_llm::ocr::{
        document::{inline_remote_document, validate_inline_document},
        error::Error,
        handler::OcrClient,
        transformation::{
            BaseOcrConfig, LiteLLMOcrResponse, OcrConnection, OcrDocument, OcrRequestContext,
            OcrResponseFormat, PreparedOcrRequest,
        },
    },
    mistral::ocr::transformation::{MistralOcrConfig, MistralOcrRequest},
};

const AZURE_AI_OCR_PATH: &str = "/providers/mistral/azure/ocr";

const AZURE_AI_API_KEY_ENV: &str = "AZURE_AI_API_KEY";
const AZURE_AI_API_BASE_ENV: &str = "AZURE_AI_API_BASE";

#[derive(Clone, Debug, Default)]
pub struct AzureAiOcrConfig;

impl BaseOcrConfig for AzureAiOcrConfig {
    type OcrParams = OpaqueParams;
    type ProviderRequest = MistralOcrRequest;
    type Environment = Vec<(String, String)>;

    fn get_supported_ocr_params(&self, model: &str) -> &'static [&'static str] {
        MistralOcrConfig.get_supported_ocr_params(model)
    }

    fn get_api_key_env_var(&self) -> Option<&'static str> {
        Some(AZURE_AI_API_KEY_ENV)
    }

    fn map_ocr_params(
        &self,
        non_default_params: &CallArguments,
        model: &str,
    ) -> Result<OpaqueParams, Error> {
        MistralOcrConfig.map_ocr_params(non_default_params, model)
    }

    async fn validate_environment(
        &self,
        request: &PreparedOcrRequest,
        _client: &OcrClient,
    ) -> Result<Self::Environment, Error> {
        let config = crate::azure_ai::ocr::common_utils::azure_auth_inputs(request)?;
        self.resolve_headers(&request.connection, &config, &|name: &str| {
            request.connection.secret(name)
        })
        .await
    }

    fn get_complete_url(
        &self,
        request: &PreparedOcrRequest,
        _optional_params: &Self::OcrParams,
        _environment: &Self::Environment,
    ) -> Result<String, Error> {
        self.build_ocr_url(request.connection.api_base.as_deref(), &|name: &str| {
            request.connection.secret(name)
        })
    }

    fn transform_ocr_request(
        &self,
        model: &str,
        document: OcrDocument,
        optional_params: &OpaqueParams,
        headers: &[(String, String)],
    ) -> Result<MistralOcrRequest, Error> {
        MistralOcrConfig.transform_ocr_request(model, document, optional_params, headers)
    }

    async fn async_transform_ocr_request(
        &self,
        model: &str,
        document: OcrDocument,
        optional_params: &OpaqueParams,
        headers: &[(String, String)],
        context: OcrRequestContext<'_>,
    ) -> Result<MistralOcrRequest, Error> {
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
        MistralOcrConfig.transform_ocr_response(model, raw_response, request_format)
    }

    fn validate_request_body(&self, body: &Value) -> Result<(), Error> {
        validate_inline_document(&crate::base_llm::ocr::handler::body_document(body)?)
    }
}

impl AzureAiOcrConfig {
    /// Python `AzureAIOCRConfig.validate_environment` requires the endpoint
    /// before it resolves credentials; keep that order so a missing base is
    /// reported without invoking any token provider.
    pub(super) fn resolve_api_base(
        api_base: Option<&str>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, Error> {
        nonblank(api_base.map(str::to_string))
            .or_else(|| nonblank(env_lookup(AZURE_AI_API_BASE_ENV)))
            .ok_or(Error::Auth(litellm_auth::Error::MissingApiBase {
                provider: "Azure AI",
                environment_variable: AZURE_AI_API_BASE_ENV,
            }))
    }

    async fn resolve_headers(
        &self,
        connection: &OcrConnection,
        config: &AzureAuthInputs,
        env_lookup: &(dyn Fn(&str) -> Option<String> + Sync),
    ) -> Result<Vec<(String, String)>, Error> {
        Self::resolve_api_base(connection.api_base.as_deref(), env_lookup)?;
        if litellm_http::request::has_header(&connection.extra_headers, "authorization") {
            if config.azure_ad_token_provider.is_some() {
                super::common_utils::resolve_entra(config, env_lookup).await?;
            }
            super::common_utils::validate_destination(connection, connection.extra_headers_source)?;
            return Ok(connection.extra_headers.clone());
        }
        let key = nonblank(
            connection
                .api_key
                .as_ref()
                .map(|key| key.expose().to_string()),
        )
        .map(|value| Sourced::new(value, connection.api_key_source))
        .or_else(|| {
            nonblank(self.get_api_key_env_var().and_then(env_lookup))
                .map(|value| Sourced::new(value, InputSource::Environment))
        });
        if let Some(key) = key {
            super::common_utils::validate_destination(connection, key.source())?;
            return Ok(bearer_headers(connection, key.value()));
        }
        let key = super::common_utils::resolve_entra(config, env_lookup)
            .await?
            .ok_or(Error::MissingAzureAiCredentials)?;
        super::common_utils::validate_destination(connection, key.source())?;
        Ok(bearer_headers(connection, key.value()))
    }

    fn build_ocr_url(
        &self,
        api_base: Option<&str>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, Error> {
        let base = Self::resolve_api_base(api_base, env_lookup)?;
        let path: Vec<&str> = AZURE_AI_OCR_PATH.trim_matches('/').split('/').collect();
        ApiUrl::parse(&base)
            .and_then(|url| url.complete_path(&path))
            .map(|url| url.into_string())
            .map_err(|_| Error::RequestField {
                path: "api_base".into(),
            })
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
    use rstest::{fixture, rstest};

    use super::*;

    #[fixture]
    fn connection() -> OcrConnection {
        OcrConnection {
            api_key: Some(litellm_auth::SecretValue::new("request-key")),
            api_base: Some("https://example.com".into()),
            ..Default::default()
        }
    }

    #[rstest]
    #[case::base_with_query(
        "https://example.com/?tenant=a",
        "https://example.com/providers/mistral/azure/ocr?tenant=a"
    )]
    #[case::complete_endpoint(
        "https://example.com/providers/mistral/azure/ocr",
        "https://example.com/providers/mistral/azure/ocr"
    )]
    fn completes_azure_path_and_preserves_query(#[case] api_base: &str, #[case] expected: &str) {
        assert_eq!(
            AzureAiOcrConfig
                .build_ocr_url(Some(api_base), &|_| None)
                .unwrap(),
            expected
        );
    }

    #[test]
    fn missing_api_base_is_structured() {
        assert!(matches!(
            AzureAiOcrConfig::resolve_api_base(None, &|_| None),
            Err(Error::Auth(litellm_auth::Error::MissingApiBase {
                provider: "Azure AI",
                environment_variable: AZURE_AI_API_BASE_ENV,
            }))
        ));
    }

    #[rstest]
    #[tokio::test]
    async fn supplied_authorization_precedes_keys(connection: OcrConnection) {
        let connection = OcrConnection {
            extra_headers: vec![("authorization".into(), "Bearer prepared".into())],
            ..connection
        };
        assert_eq!(
            AzureAiOcrConfig
                .resolve_headers(&connection, &Default::default(), &|_| {
                    Some("environment-key".into())
                })
                .await
                .unwrap(),
            connection.extra_headers
        );
    }

    #[rstest]
    #[tokio::test]
    async fn request_key_precedes_environment_key(connection: OcrConnection) {
        assert_eq!(
            AzureAiOcrConfig
                .resolve_headers(&connection, &Default::default(), &|_| {
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

        let error = AzureAiOcrConfig
            .resolve_headers(&connection, &Default::default(), &|name| {
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
            api_key: Some(litellm_auth::SecretValue::new("request-key")),
            api_key_source: InputSource::Request,
            api_base: Some("https://request.example".into()),
            api_base_source: InputSource::Request,
            ..Default::default()
        };

        let headers = AzureAiOcrConfig
            .resolve_headers(&connection, &Default::default(), &|_| None)
            .await
            .unwrap();

        assert_eq!(
            headers[0],
            ("Authorization".into(), "Bearer request-key".into())
        );
    }

    #[tokio::test]
    async fn environment_supplies_api_base_and_bearer_key() {
        let env = |name: &str| match name {
            AZURE_AI_API_BASE_ENV => Some("https://env.example".to_string()),
            AZURE_AI_API_KEY_ENV => Some("env-key".to_string()),
            _ => None,
        };
        let connection = OcrConnection::default();

        let headers = AzureAiOcrConfig
            .resolve_headers(&connection, &Default::default(), &env)
            .await
            .unwrap();
        let url = AzureAiOcrConfig.build_ocr_url(None, &env).unwrap();

        assert_eq!(
            headers,
            [("Authorization".to_string(), "Bearer env-key".to_string())]
        );
        assert_eq!(url, "https://env.example/providers/mistral/azure/ocr");
    }
}
