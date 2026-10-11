use litellm_auth_gcp::{self as vertex, VertexConfig};
use litellm_core_utils::{call_arguments::CallArguments, params::OpaqueParams};
use serde_json::Value;

use super::common_utils::{validate_destination, vertex_connection};
use crate::{
    base_llm::ocr::{
        document::{inline_remote_document, validate_inline_document},
        error::Error,
        handler::OcrClient,
        transformation::{
            BaseOcrConfig, OcrConnection, OcrEnvironment, OcrRequestContext, PreparedOcrRequest,
        },
    },
    mistral::ocr::transformation::{MistralOcrConfig, MistralOcrRequest},
};
use litellm_llms_types::formats::ocr::{LiteLLMOcrResponse, OcrDocument, OcrResponseFormat};

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

    fn secret_names(&self) -> Vec<&'static str> {
        litellm_auth_gcp::secret_names().to_vec()
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
        let connection = vertex_connection(request)?;
        self.resolve_environment(&request.connection, &connection.config, client)
            .await
    }

    fn get_complete_url(
        &self,
        request: &PreparedOcrRequest,
        _optional_params: &Self::OcrParams,
        environment: &Self::Environment,
    ) -> Result<String, Error> {
        let prediction = format!("{}:rawPredict", request.model);
        Ok(vertex_connection(request)?.resource_url(
            &environment.project_id,
            &["publishers", "mistralai", "models", &prediction],
        )?)
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
            .auth()
            .gcp
            .validate_environment(
                connection.extra_headers.clone(),
                connection
                    .api_key
                    .as_ref()
                    .map(litellm_auth::SecretValue::expose),
                config,
                &|name: &str| connection.secret(name),
            )
            .await
            .map_err(Error::from)
    }
}
