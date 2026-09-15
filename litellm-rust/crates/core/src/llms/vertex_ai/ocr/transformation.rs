use super::common_utils::validate_destination;
use crate::llms::base_llm::ocr::transformation::{BaseOcrConfig, OcrRequestContext};
use crate::llms::mistral::ocr::MistralOcrResponse;
use crate::llms::mistral::ocr::transformation::{MistralOCRConfig, MistralOcrRequest};
use crate::ocr::OcrClient;
use crate::ocr::document::{inline_remote_document, validate_inline_document};
use crate::ocr::prepare::{credential_env, transform_request_body};
use crate::ocr::types::{LiteLLMOcrRequest, LiteLLMOcrResponse, OcrConnection, OcrDocument};
use crate::params::OpaqueParams;
use crate::url_utils::ApiUrl;
use litellm_auth_gcp::{self as vertex, VertexConfig};
const DEFAULT_LOCATION: &str = "us-central1";

#[derive(Clone, Debug, Default)]
pub(crate) struct VertexAIOCRConfig;

impl BaseOcrConfig for VertexAIOCRConfig {
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

impl VertexAIOCRConfig {
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
        let config = VertexConfig::from_sourced_optional_params(
            &request.optional_params,
            &request.input_sources,
        )
        .map_err(crate::ocr::Error::from)?;
        let authentication = self
            .validate_environment(&request.connection, &config, client)
            .await?;
        let location = vertex::get_vertex_ai_location(&config, &credential_env)
            .unwrap_or_else(|| DEFAULT_LOCATION.to_string());
        let url = self.get_complete_url(
            request.connection.api_base.as_deref(),
            &authentication.project_id,
            &location,
            &request.model,
        )?;
        let retains_document = !request.document.source().starts_with("http://")
            && !request.document.source().starts_with("https://");
        let body = self
            .async_transform_ocr_request(
                &request.model,
                request.document.clone(),
                &params,
                &authentication.headers,
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
            &authentication.headers,
            retains_document,
            body,
            |body| validate_inline_document(&body.document),
        )
        .await
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
}
