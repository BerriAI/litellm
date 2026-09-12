use super::super::OcrAdapter;
use super::validate_destination;
use crate::Error;
use crate::auth::vertex::{self, VertexConfig};
use crate::ocr::OcrClient;
use crate::ocr::codecs::deepseek::{self, DeepSeekOcrParams, DeepSeekOcrResponse};
use crate::ocr::error::{OcrError, OcrRequestError, OcrResponseError};
use crate::ocr::prepare::{
    _prepare_ocr_request, ParsedProviderParams, credential_env, transform_request_body,
};
use crate::ocr::registry::OcrProvider;
use crate::ocr::types::{LiteLLMOcrRequest, LiteLLMOcrResponse};
use crate::url_utils::ApiUrl;
const DEFAULT_API_BASE: &str = "https://aiplatform.googleapis.com";
const MODEL_NAMESPACE: &str = "deepseek-ai";
const DEFAULT_LOCATION: &str = "us-central1";

#[derive(Clone, Debug)]
pub(crate) struct VertexDeepSeekAdapter;

impl OcrAdapter for VertexDeepSeekAdapter {
    type ProviderResponse = DeepSeekOcrResponse;
    const PROVIDER: OcrProvider = OcrProvider::VertexAi;

    async fn prepare_request(
        &self,
        request: &LiteLLMOcrRequest,
        client: &OcrClient,
    ) -> Result<reqwest::Request, OcrError> {
        validate_destination(&request.connection)?;
        let ParsedProviderParams {
            known: params,
            extra_params: _extra_params,
        } = _prepare_ocr_request::<DeepSeekOcrParams>(request)?;
        let config = VertexConfig::from_sourced_optional_params(
            &request.optional_params,
            &request.input_sources,
        )
        .map_err(Error::from)?;
        let authentication = client
            .vertex_auth()
            .validate_environment(
                request.connection.extra_headers.clone(),
                request.connection.api_key.as_deref(),
                &config,
                &credential_env,
            )
            .await
            .map_err(Error::from)?;
        let location = vertex::get_vertex_ai_location(&config, &credential_env)
            .unwrap_or_else(|| DEFAULT_LOCATION.to_string());
        let url = get_complete_url(
            request.connection.api_base.as_deref(),
            &authentication.project_id,
            &location,
        )?;
        let document = request.document.clone();
        let body =
            deepseek::transform_ocr_request(&provider_model(&request.model), document, &params)?;
        transform_request_body(
            client,
            request,
            &url,
            &authentication.headers,
            false,
            body,
            |_| Ok(()),
        )
        .await
    }

    fn transform_ocr_response(
        &self,
        request: &LiteLLMOcrRequest,
        response: Self::ProviderResponse,
    ) -> Result<LiteLLMOcrResponse, OcrResponseError> {
        deepseek::transform_ocr_response(&request.model, response)
    }
}

fn provider_model(model: &str) -> String {
    if model.starts_with(&format!("{MODEL_NAMESPACE}/")) {
        model.to_string()
    } else {
        format!("{MODEL_NAMESPACE}/{model}")
    }
}

fn get_complete_url(
    api_base: Option<&str>,
    project: &str,
    location: &str,
) -> Result<String, OcrError> {
    let base = api_base
        .map(str::trim)
        .filter(|base| !base.is_empty())
        .unwrap_or(DEFAULT_API_BASE);
    ApiUrl::parse(base)
        .and_then(|url| {
            url.complete_path(&[
                "v1",
                "projects",
                project,
                "locations",
                location,
                "endpoints",
                "openapi",
                "chat",
                "completions",
            ])
        })
        .map(|url| url.into_string())
        .map_err(|_| {
            OcrRequestError::RequestField {
                path: "api_base".into(),
            }
            .into()
        })
}

#[cfg(test)]
mod tests {
    use super::{get_complete_url, provider_model};

    #[test]
    fn adapter_owns_model_namespace_and_endpoint() {
        assert_eq!(
            provider_model("deepseek-ocr-maas"),
            "deepseek-ai/deepseek-ocr-maas"
        );
        assert_eq!(
            provider_model("deepseek-ai/deepseek-ocr-maas"),
            "deepseek-ai/deepseek-ocr-maas"
        );
        assert_eq!(
            get_complete_url(None, "proj-1", "europe-west4").unwrap(),
            "https://aiplatform.googleapis.com/v1/projects/proj-1/locations/europe-west4/endpoints/openapi/chat/completions"
        );
    }
}
