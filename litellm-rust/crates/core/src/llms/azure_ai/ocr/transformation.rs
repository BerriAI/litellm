use litellm_auth::{InputSource, Sourced};
use litellm_auth_azure::AzureAuthInputs;
use serde_json::Value;

use crate::call_arguments::CallArguments;
use crate::constants::AZURE_AI_OCR_PATH;
use crate::llms::base_llm::ocr::transformation::{BaseOcrConfig, OcrRequestContext};
use crate::llms::mistral::ocr::transformation::{MistralOcrConfig, MistralOcrRequest};
use crate::ocr::OcrClient;
use crate::ocr::document::{inline_remote_document, validate_inline_document};
use crate::ocr::prepare::credential_env;
use crate::ocr::types::{LiteLLMOcrResponse, OcrConnection, OcrDocument, PreparedOcrRequest};
use crate::params::OpaqueParams;
use crate::url_utils::ApiUrl;

const AZURE_AI_API_KEY_ENV: &str = "AZURE_AI_API_KEY";
const AZURE_AI_API_BASE_ENV: &str = "AZURE_AI_API_BASE";

#[derive(Clone, Debug, Default)]
pub(crate) struct AzureAiOcrConfig;

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
    ) -> Result<OpaqueParams, crate::ocr::Error> {
        MistralOcrConfig.map_ocr_params(non_default_params, model)
    }

    async fn validate_environment(
        &self,
        request: &PreparedOcrRequest,
        _client: &OcrClient,
    ) -> Result<Self::Environment, crate::ocr::Error> {
        let config = AzureAuthInputs {
            azure_ad_token_provider: request.azure_ad_token_provider.clone(),
            ..AzureAuthInputs::from_sourced_optional_params(
                &request.optional_params,
                &request.input_sources,
            )?
        };
        self.resolve_headers(&request.connection, &config, &credential_env)
            .await
    }

    fn get_complete_url(
        &self,
        request: &PreparedOcrRequest,
        _optional_params: &Self::OcrParams,
        _environment: &Self::Environment,
    ) -> Result<String, crate::ocr::Error> {
        self.build_ocr_url(request.connection.api_base.as_deref(), &credential_env)
    }

    fn transform_ocr_request(
        &self,
        model: &str,
        document: OcrDocument,
        optional_params: &OpaqueParams,
        headers: &[(String, String)],
    ) -> Result<MistralOcrRequest, crate::ocr::Error> {
        MistralOcrConfig.transform_ocr_request(model, document, optional_params, headers)
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
        self.transform_ocr_request(model, document, optional_params, headers)
    }

    fn transform_ocr_response(
        &self,
        model: &str,
        raw_response: &[u8],
        request_format: crate::ocr::types::OcrResponseFormat,
    ) -> Result<LiteLLMOcrResponse, crate::ocr::Error> {
        MistralOcrConfig.transform_ocr_response(model, raw_response, request_format)
    }

    fn validate_request_body(&self, body: &Value) -> Result<(), crate::ocr::Error> {
        validate_inline_document(&crate::ocr::prepare::body_document(body)?)
    }
}

impl AzureAiOcrConfig {
    /// Python `AzureAIOCRConfig.validate_environment` requires the endpoint
    /// before it resolves credentials; keep that order so a missing base is
    /// reported without invoking any token provider.
    pub(super) fn resolve_api_base(
        api_base: Option<&str>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, crate::ocr::Error> {
        nonblank(api_base.map(str::to_string))
            .or_else(|| nonblank(env_lookup(AZURE_AI_API_BASE_ENV)))
            .ok_or(crate::ocr::Error::Auth(
                litellm_auth::Error::MissingApiBase {
                    provider: "Azure AI",
                    environment_variable: AZURE_AI_API_BASE_ENV,
                },
            ))
    }

    async fn resolve_headers(
        &self,
        connection: &OcrConnection,
        config: &AzureAuthInputs,
        env_lookup: &(dyn Fn(&str) -> Option<String> + Sync),
    ) -> Result<Vec<(String, String)>, crate::ocr::Error> {
        Self::resolve_api_base(connection.api_base.as_deref(), env_lookup)?;
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
                nonblank(self.get_api_key_env_var().and_then(env_lookup))
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

    fn build_ocr_url(
        &self,
        api_base: Option<&str>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, crate::ocr::Error> {
        let base = Self::resolve_api_base(api_base, env_lookup)?;
        let path: Vec<&str> = AZURE_AI_OCR_PATH.trim_matches('/').split('/').collect();
        ApiUrl::parse(&base)
            .and_then(|url| url.complete_path(&path))
            .map(|url| url.into_string())
            .map_err(|_| crate::ocr::Error::RequestField {
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
            api_key: Some("request-key".into()),
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
            Err(crate::ocr::Error::Auth(
                litellm_auth::Error::MissingApiBase {
                    provider: "Azure AI",
                    environment_variable: AZURE_AI_API_BASE_ENV,
                }
            ))
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
            api_key: Some("request-key".into()),
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

    use serde_json::json;

    use crate::ocr::LocalOcrHost;
    use crate::ocr::test_support::{
        MockResponse, mock_server, perform_ocr, perform_ocr_with, wire_request,
    };

    #[tokio::test]
    async fn facade_executes_azure_mistral_with_prepared_auth() {
        let (base, seen, server) = mock_server(vec![MockResponse::json(json!({
            "pages":[{"index":0,"markdown":"hello"}],
            "usage_info":{"pages_processed":1}
        }))])
        .await;
        let mut request = wire_request(
            "azure_ai/model",
            &base,
            json!({"include_image_base64":true}),
        );
        request.credentials.api_key = None;
        request.transport.extra_headers = vec![(
            "Authorization".into(),
            "Bearer python-prepared-token".into(),
        )];

        let result = perform_ocr(request).await.unwrap();
        server.await.unwrap();
        assert_eq!(result.pages[0].markdown, "hello");
        let requests = seen.lock().unwrap();
        assert_eq!(requests.len(), 1);
        assert!(requests[0].starts_with("POST /providers/mistral/azure/ocr "));
        assert!(
            requests[0]
                .to_ascii_lowercase()
                .contains("authorization: bearer python-prepared-token\r\n")
        );
        let body: Value =
            serde_json::from_str(requests[0].split_once("\r\n\r\n").unwrap().1).unwrap();
        assert_eq!(
            body,
            json!({
                "model":"model",
                "document":{"type":"document_url","document_url":"data:application/pdf;base64,YWJj"},
                "include_image_base64":true
            })
        );
    }

    #[tokio::test]
    async fn facade_acquires_supplied_entra_token_for_final_request() {
        let (base, seen, server) = mock_server(vec![MockResponse::json(json!({"pages":[]}))]).await;
        let mut request = wire_request(
            "azure_ai/model",
            &base,
            json!({"azure_ad_token":"rust-owned-token"}),
        );
        request.credentials.api_key = None;

        perform_ocr(request).await.unwrap();
        server.await.unwrap();

        let requests = seen.lock().unwrap();
        assert_eq!(requests.len(), 1);
        assert!(
            requests[0]
                .to_ascii_lowercase()
                .contains("authorization: bearer rust-owned-token\r\n")
        );
    }

    #[tokio::test]
    async fn rejects_non_inline_body_after_guardrails() {
        let request = wire_request("azure_ai/model", "http://127.0.0.1:1", json!({}));
        let host = LocalOcrHost::new(request).with_before_send(|mut wire, _| {
            wire.body["document"] = json!({
                "type":"document_url",
                "document_url":"https://example.com/not-inline.pdf"
            });
            Ok(wire)
        });
        let error = perform_ocr_with(host).await.unwrap_err();
        assert!(error.to_string().contains("data URI"));
    }
}
