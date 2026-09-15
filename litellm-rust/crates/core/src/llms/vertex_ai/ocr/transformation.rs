use super::common_utils::validate_destination;
use crate::llms::base_llm::ocr::transformation::BaseOcrConfig;
use crate::llms::mistral::ocr::MistralOcrResponse;
use crate::llms::mistral::ocr::transformation::MistralOCRConfig;
use crate::ocr::OcrClient;
use crate::ocr::document::{inline_remote_document, validate_inline_document};
use crate::ocr::prepare::{credential_env, transform_request_body};
use crate::ocr::types::{LiteLLMOcrRequest, LiteLLMOcrResponse};
use crate::url_utils::ApiUrl;
use litellm_auth_gcp::{self as vertex, VertexConfig};
const DEFAULT_LOCATION: &str = "us-central1";

#[derive(Clone, Debug, Default)]
pub(crate) struct VertexAIOCRConfig;

impl BaseOcrConfig for VertexAIOCRConfig {
    type ProviderResponse = MistralOcrResponse;

    fn get_supported_ocr_params(&self, model: &str) -> &'static [&'static str] {
        MistralOCRConfig.get_supported_ocr_params(model)
    }

    async fn prepare_request(
        &self,
        request: &LiteLLMOcrRequest,
        client: &OcrClient,
    ) -> Result<reqwest::Request, crate::ocr::Error> {
        validate_destination(&request.connection)?;
        let params = self.map_ocr_params(&request.model, &request.optional_params);
        let config = VertexConfig::from_sourced_optional_params(
            &request.optional_params,
            &request.input_sources,
        )
        .map_err(crate::ocr::Error::from)?;
        let authentication = client
            .vertex_auth()
            .validate_environment(
                request.connection.extra_headers.clone(),
                request.connection.api_key.as_deref(),
                &config,
                &credential_env,
            )
            .await
            .map_err(crate::ocr::Error::from)?;
        let location = vertex::get_vertex_ai_location(&config, &credential_env)
            .unwrap_or_else(|| DEFAULT_LOCATION.to_string());
        let url = get_complete_url(
            request.connection.api_base.as_deref(),
            &authentication.project_id,
            &location,
            &request.model,
        )?;
        let retains_document = !request.document.source().starts_with("http://")
            && !request.document.source().starts_with("https://");
        let document = inline_remote_document(
            client.document_fetcher(),
            request.document.clone(),
            &request.connection,
        )
        .await?;
        let body = MistralOCRConfig.transform_ocr_request(&request.model, document, &params)?;
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

    fn transform_ocr_response(
        &self,
        request: &LiteLLMOcrRequest,
        response: MistralOcrResponse,
    ) -> Result<LiteLLMOcrResponse, crate::ocr::Error> {
        MistralOCRConfig.transform_ocr_response(request, response)
    }
}

fn get_complete_url(
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
    use super::get_complete_url;

    #[test]
    fn endpoint_uses_location_project_and_model() {
        assert_eq!(
            get_complete_url(None, "proj-1", "europe-west4", "mistral-ocr-maas").unwrap(),
            "https://europe-west4-aiplatform.googleapis.com/v1/projects/proj-1/locations/europe-west4/publishers/mistralai/models/mistral-ocr-maas:rawPredict"
        );
        assert!(get_complete_url(None, "proj-1", "attacker.example/path", "model").is_err());
    }
}
