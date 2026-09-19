use litellm_auth_gcp::{self as vertex, VertexConfig};
use litellm_core_utils::{call_arguments::CallArguments, params::OpaqueParams, url_utils::ApiUrl};
use serde_json::Value;

use super::common_utils::validate_destination;
use crate::{
    base_llm::ocr::{
        document::{inline_remote_document, validate_inline_document},
        error::Error,
        transformation::{
            BaseOcrConfig, LiteLLMOcrResponse, OcrConnection, OcrDocument, OcrEnvironment,
            OcrRequestContext, OcrResponseFormat, PreparedOcrRequest, credential_env,
        },
    },
    custom_httpx::llm_http_handler::OcrClient,
    mistral::ocr::transformation::{MistralOcrConfig, MistralOcrRequest},
};

const DEFAULT_LOCATION: &str = "us-central1";

#[derive(Clone, Debug, Default)]
pub struct VertexAiOcrConfig;

impl BaseOcrConfig for VertexAiOcrConfig {
    type OcrParams = OpaqueParams;
    type ProviderRequest = MistralOcrRequest;
    type Environment = vertex::VertexEnvironment;

    fn get_supported_ocr_params(&self, model: &str) -> &'static [&'static str] {
        MistralOcrConfig.get_supported_ocr_params(model)
    }

    fn get_api_key_env_var(&self) -> Option<&'static str> {
        Some("VERTEX_AI_API_KEY")
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
        client: &OcrClient,
    ) -> Result<Self::Environment, Error> {
        let config = VertexConfig::from_sourced_optional_params(
            &request.optional_params,
            &request.input_sources,
        )?;
        self.resolve_environment(&request.connection, &config, client)
            .await
    }

    fn get_complete_url(
        &self,
        request: &PreparedOcrRequest,
        _optional_params: &Self::OcrParams,
        environment: &Self::Environment,
    ) -> Result<String, Error> {
        let config = VertexConfig::from_sourced_optional_params(
            &request.optional_params,
            &request.input_sources,
        )?;
        let location = vertex::get_vertex_ai_location(&config, &credential_env)
            .unwrap_or_else(|| DEFAULT_LOCATION.to_string());
        self.build_ocr_url(
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
        validate_inline_document(&crate::custom_httpx::llm_http_handler::body_document(body)?)
    }
}

impl OcrEnvironment for vertex::VertexEnvironment {
    fn headers(&self) -> &[(String, String)] {
        &self.headers
    }
}

impl VertexAiOcrConfig {
    async fn resolve_environment(
        &self,
        connection: &OcrConnection,
        config: &VertexConfig,
        client: &OcrClient,
    ) -> Result<vertex::VertexEnvironment, Error> {
        validate_destination(connection)?;
        client
            .vertex_auth()
            .validate_environment(
                connection.extra_headers.clone(),
                connection
                    .api_key
                    .as_ref()
                    .map(litellm_auth::SecretValue::expose),
                config,
                &credential_env,
            )
            .await
            .map_err(Error::from)
    }

    fn build_ocr_url(
        &self,
        api_base: Option<&str>,
        project: &str,
        location: &str,
        model: &str,
    ) -> Result<String, Error> {
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
            .map_err(|_| Error::RequestField {
                path: "api_base".into(),
            })
    }
}

fn validate_location(location: &str) -> Result<(), Error> {
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
    Err(Error::RequestField {
        path: "vertex_location".into(),
    })
}

#[cfg(test)]
mod tests {

    use super::VertexAiOcrConfig;

    #[test]
    fn endpoint_uses_location_project_and_model() {
        assert_eq!(
            VertexAiOcrConfig
                .build_ocr_url(None, "proj-1", "europe-west4", "mistral-ocr-maas")
                .unwrap(),
            "https://europe-west4-aiplatform.googleapis.com/v1/projects/proj-1/locations/europe-west4/publishers/mistralai/models/mistral-ocr-maas:rawPredict"
        );
    }

    #[test]
    fn endpoint_rejects_invalid_location() {
        assert!(
            VertexAiOcrConfig
                .build_ocr_url(None, "proj-1", "attacker.example/path", "model")
                .is_err()
        );
    }
}
