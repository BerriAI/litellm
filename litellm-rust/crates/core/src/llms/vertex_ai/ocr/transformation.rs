use litellm_auth_gcp::{self as vertex, VertexConfig};
use serde_json::Value;

use super::common_utils::validate_destination;
use crate::call_arguments::CallArguments;
use crate::llms::base_llm::ocr::transformation::{
    BaseOcrConfig, OcrEnvironment, OcrRequestContext,
};
use crate::llms::mistral::ocr::transformation::{MistralOCRConfig, MistralOcrRequest};
use crate::ocr::OcrClient;
use crate::ocr::document::{inline_remote_document, validate_inline_document};
use crate::ocr::prepare::credential_env;
use crate::ocr::types::{LiteLLMOcrResponse, OcrConnection, OcrDocument, PreparedOcrRequest};
use crate::params::OpaqueParams;
use crate::url_utils::ApiUrl;

const DEFAULT_LOCATION: &str = "us-central1";

#[derive(Clone, Debug, Default)]
pub(crate) struct VertexAIOCRConfig;

impl BaseOcrConfig for VertexAIOCRConfig {
    type OcrParams = OpaqueParams;
    type ProviderRequest = MistralOcrRequest;
    type Environment = vertex::VertexEnvironment;

    fn get_api_key_env_var(&self) -> Option<&'static str> {
        Some("VERTEX_AI_API_KEY")
    }

    async fn validate_environment(
        &self,
        request: &PreparedOcrRequest,
        client: &OcrClient,
    ) -> Result<Self::Environment, crate::ocr::Error> {
        let config = VertexConfig::from_sourced_optional_params(
            &request.optional_params,
            &request.input_sources,
        )?;
        self.validate_environment(&request.connection, &config, client)
            .await
    }

    fn get_complete_url(
        &self,
        request: &PreparedOcrRequest,
        _params: &Self::OcrParams,
        environment: &Self::Environment,
    ) -> Result<String, crate::ocr::Error> {
        let config = VertexConfig::from_sourced_optional_params(
            &request.optional_params,
            &request.input_sources,
        )?;
        let location = vertex::get_vertex_ai_location(&config, &credential_env)
            .unwrap_or_else(|| DEFAULT_LOCATION.to_string());
        self.get_complete_url(
            request.connection.api_base.as_deref(),
            &environment.project_id,
            &location,
            &request.model,
        )
    }

    fn transform_ocr_request(
        &self,
        model: &str,
        document: OcrDocument,
        params: &OpaqueParams,
        headers: &[(String, String)],
    ) -> Result<MistralOcrRequest, crate::ocr::Error> {
        MistralOCRConfig.transform_ocr_request(model, document, params, headers)
    }

    fn get_supported_ocr_params(&self, model: &str) -> &'static [&'static str] {
        MistralOCRConfig.get_supported_ocr_params(model)
    }

    fn map_ocr_params(
        &self,
        arguments: &CallArguments,
        model: &str,
    ) -> Result<OpaqueParams, crate::ocr::Error> {
        MistralOCRConfig.map_ocr_params(arguments, model)
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
        MistralOCRConfig.transform_ocr_response(model, raw_response, request_format)
    }

    fn retains_document(&self, document: &OcrDocument) -> bool {
        !document.is_remote()
    }

    fn validate_request_body(&self, body: &Value) -> Result<(), crate::ocr::Error> {
        validate_inline_document(&crate::ocr::prepare::body_document(body)?)
    }
}

impl OcrEnvironment for vertex::VertexEnvironment {
    fn headers(&self) -> &[(String, String)] {
        &self.headers
    }
}

impl VertexAIOCRConfig {
    pub(super) async fn validate_environment(
        &self,
        connection: &OcrConnection,
        config: &VertexConfig,
        client: &OcrClient,
    ) -> Result<vertex::VertexEnvironment, crate::ocr::Error> {
        validate_destination(connection)?;
        client
            .vertex_auth()
            .validate_environment(
                connection.extra_headers.clone(),
                connection.api_key.as_deref(),
                config,
                &credential_env,
            )
            .await
            .map_err(crate::ocr::Error::from)
    }

    fn get_complete_url(
        &self,
        api_base: Option<&str>,
        project: &str,
        location: &str,
        model: &str,
    ) -> Result<String, crate::ocr::Error> {
        validate_location(location)?;
        let default_base = format!("https://{location}-aiplatform.googleapis.com");
        let base = api_base
            .map(str::trim)
            .filter(|base| !base.is_empty())
            .unwrap_or(&default_base);
        let prediction = format!("{model}:rawPredict");
        ApiUrl::parse(base)
            .and_then(|url| {
                url.complete_path(&[
                    "v1",
                    "projects",
                    project,
                    "locations",
                    location,
                    "publishers",
                    "mistralai",
                    "models",
                    &prediction,
                ])
            })
            .map(|url| url.into_string())
            .map_err(|_| crate::ocr::Error::RequestField {
                path: "api_base".into(),
            })
    }
}

fn validate_location(location: &str) -> Result<(), crate::ocr::Error> {
    let valid = !location.is_empty()
        && location
            .bytes()
            .all(|value| value.is_ascii_lowercase() || value.is_ascii_digit() || value == b'-')
        && location
            .as_bytes()
            .first()
            .is_some_and(u8::is_ascii_alphanumeric)
        && location
            .as_bytes()
            .last()
            .is_some_and(u8::is_ascii_alphanumeric);
    if valid {
        return Ok(());
    }
    Err(crate::ocr::Error::RequestField {
        path: "vertex_location".into(),
    })
}

#[cfg(test)]
mod tests {
    use super::VertexAIOCRConfig;

    #[test]
    fn endpoint_uses_location_project_and_model() {
        assert_eq!(
            VertexAIOCRConfig
                .get_complete_url(None, "proj-1", "europe-west4", "mistral-ocr-maas")
                .unwrap(),
            "https://europe-west4-aiplatform.googleapis.com/v1/projects/proj-1/locations/europe-west4/publishers/mistralai/models/mistral-ocr-maas:rawPredict"
        );
        assert!(
            VertexAIOCRConfig
                .get_complete_url(None, "proj-1", "attacker.example/path", "model")
                .is_err()
        );
    }

    use serde_json::{Value, json};

    use crate::ocr::test_support::{MockResponse, mock_server, perform_ocr, wire_request};
    use litellm_auth::InputSource;

    fn request_body(request: &str) -> Value {
        serde_json::from_str(request.split_once("\r\n\r\n").unwrap().1).unwrap()
    }

    #[tokio::test]
    async fn facade_executes_vertex_mistral_with_resolved_project_and_location() {
        let (base, seen, server) = mock_server(vec![MockResponse::json(json!({
            "pages":[{"index":0,"markdown":"hello"}],
            "usage_info":{"pages_processed":1}
        }))])
        .await;
        let request = wire_request(
            "vertex_ai/mistral-ocr-maas",
            &base,
            json!({
                "vertex_project":"project-1",
                "vertex_location":"europe-west4",
                "extract_footer":true
            }),
        );

        let response = perform_ocr(request).await.unwrap();
        server.await.unwrap();
        assert_eq!(response.pages[0].markdown, "hello");
        let requests = seen.lock().unwrap();
        assert_eq!(requests.len(), 1);
        assert!(requests[0].starts_with(
            "POST /v1/projects/project-1/locations/europe-west4/publishers/mistralai/models/mistral-ocr-maas:rawPredict "
        ));
        assert!(
            requests[0]
                .to_ascii_lowercase()
                .contains("authorization: bearer test-key")
        );
        assert_eq!(
            request_body(&requests[0]),
            json!({
                "model":"mistral-ocr-maas",
                "document":{"type":"document_url","document_url":"data:application/pdf;base64,YWJj"},
                "extract_footer":true
            })
        );
    }

    #[tokio::test]
    async fn supplied_authorization_is_forwarded_without_a_static_token() {
        let (base, seen, server) = mock_server(vec![MockResponse::json(json!({"pages":[]}))]).await;
        let mut request = wire_request(
            "vertex_ai/model",
            &base,
            json!({"vertex_project":"project-1"}),
        );
        request.credentials.api_key = None;
        request.transport.extra_headers = vec![("authorization".into(), "Bearer supplied".into())];

        perform_ocr(request).await.unwrap();
        server.await.unwrap();
        assert!(
            seen.lock().unwrap()[0]
                .to_ascii_lowercase()
                .contains("authorization: bearer supplied")
        );
    }

    #[tokio::test]
    async fn invalid_credentials_fail_before_provider_http() {
        let request = wire_request(
            "vertex_ai/model",
            "http://127.0.0.1:1",
            json!({"vertex_credentials": true}),
        );
        let error = perform_ocr(request).await.unwrap_err();
        assert!(error.to_string().contains("vertex_credentials"));
    }

    #[tokio::test]
    async fn request_controlled_api_base_is_rejected_before_vertex_auth() {
        let mut request = wire_request(
            "vertex_ai/mistral-ocr-maas",
            "https://caller.example",
            json!({"vertex_project":"project-1"}),
        );
        request.credentials.api_base = Some(litellm_auth::Sourced::new(
            "https://caller.example".into(),
            InputSource::Request,
        ));

        let error = perform_ocr(request).await.unwrap_err();
        assert!(
            error
                .to_string()
                .contains("request-controlled Vertex AI endpoint")
        );
    }

    #[tokio::test]
    async fn configs_build_complete_requests_and_share_mistral_normalization() {
        use std::time::Duration;

        use crate::llms::base_llm::ocr::transformation::BaseOcrConfig;
        use crate::llms::mistral::ocr::transformation::MistralOCRConfig;
        use crate::llms::vertex_ai::ocr::transformation::VertexAIOCRConfig;
        use crate::ocr::test_support::ocr_client;

        let client = ocr_client();
        let options = json!({
            "pages": [0, 2],
            "include_image_base64": true,
            "vertex_project": "project-1",
            "vertex_location": "us-central1",
            "unknown": "preserved"
        });
        let direct = wire_request(
            "mistral/mistral-ocr-maas",
            "https://mistral.test",
            options.clone(),
        );
        let vertex = wire_request("vertex_ai/mistral-ocr-maas", "https://vertex.test", options);
        let direct = crate::ocr::prepare::prepare_request(direct);
        let vertex = crate::ocr::prepare::prepare_request(vertex);
        let direct_http = MistralOCRConfig
            .prepare_request(&direct, &client)
            .await
            .unwrap();
        let vertex_http = VertexAIOCRConfig
            .prepare_request(&vertex, &client)
            .await
            .unwrap();
        assert_eq!(direct_http.url().as_str(), "https://mistral.test/v1/ocr");
        assert_eq!(
            vertex_http.url().as_str(),
            "https://vertex.test/v1/projects/project-1/locations/us-central1/publishers/mistralai/models/mistral-ocr-maas:rawPredict"
        );
        for http in [&direct_http, &vertex_http] {
            assert_eq!(http.method(), reqwest::Method::POST);
            assert_eq!(http.headers()["authorization"], "Bearer test-key");
            assert_eq!(http.headers()["content-type"], "application/json");
            assert_eq!(http.timeout(), Some(&Duration::from_secs(2)));
            let body: Value =
                serde_json::from_slice(http.body().unwrap().as_bytes().unwrap()).unwrap();
            assert_eq!(
                body,
                json!({
                    "model": "mistral-ocr-maas",
                    "document": {"type": "document_url", "document_url": "data:application/pdf;base64,YWJj"},
                    "pages": [0, 2],
                    "include_image_base64": true,
                    "unknown": "preserved"
                })
            );
        }
        let payload = serde_json::to_vec(
            &json!({"pages": [{"index": 0, "markdown": "hello"}], "extra": "preserved"}),
        )
        .unwrap();
        let direct_response = MistralOCRConfig
            .transform_ocr_response(&direct.model, &payload, Default::default())
            .unwrap()
            .into_json();
        let vertex_response = VertexAIOCRConfig
            .transform_ocr_response(&vertex.model, &payload, Default::default())
            .unwrap()
            .into_json();
        assert_eq!(direct_response, vertex_response);
        assert_eq!(direct_response["model"], "mistral-ocr-maas");
        assert_eq!(direct_response["object"], "ocr");
        assert!(direct_response.get("extra").is_none());
    }
}
