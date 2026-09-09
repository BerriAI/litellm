use crate::auth::AuthError;
use crate::auth::{CredentialPlacement, http::apply_credential};
use crate::constants::VERTEX_OCR_DEFAULT_LOCATION;
use crate::ocr::error::OcrError;
use crate::ocr::error::OcrRequestError;
use crate::ocr::error::OcrResponseError;
use crate::ocr::transformation::OcrProviderConfig;
use crate::ocr::types::{OcrConnection, OcrDocument, OcrResponseData};
use crate::providers::mistral::ocr::transformation::MISTRAL_OCR_CONFIG;
use crate::providers::mistral::ocr::types::{
    MistralOcrParams, MistralOcrRequest, MistralOcrResponse,
};

pub struct VertexAiOcrConfig;
pub const VERTEX_AI_OCR_CONFIG: VertexAiOcrConfig = VertexAiOcrConfig;

pub(super) fn project(connection: &OcrConnection) -> Result<&str, OcrRequestError> {
    connection
        .vertex
        .project
        .as_deref()
        .ok_or(OcrRequestError::MissingField("vertex_project"))
}
pub(super) fn location(connection: &OcrConnection) -> &str {
    connection
        .vertex
        .location
        .as_deref()
        .unwrap_or(VERTEX_OCR_DEFAULT_LOCATION)
}

pub(super) async fn authenticate_vertex(
    connection: &OcrConnection,
) -> Result<Vec<(String, String)>, AuthError> {
    if crate::http_utils::has_header(&connection.extra_headers, "Authorization") {
        return Ok(connection.extra_headers.clone());
    }
    let key = connection
        .api_key
        .as_deref()
        .ok_or(crate::AuthError::MissingApiKey {
            provider: "Vertex AI",
        })?;
    apply_credential(
        connection.extra_headers.clone(),
        key,
        CredentialPlacement::Bearer,
    )
}

impl OcrProviderConfig for VertexAiOcrConfig {
    type InputParams = MistralOcrParams;
    type MappedParams = MistralOcrParams;
    type PreparedDocument = OcrDocument;
    type RequestBody = MistralOcrRequest;
    type ResponseBody = MistralOcrResponse;
    fn map_ocr_params(
        &self,
        params: MistralOcrParams,
    ) -> Result<MistralOcrParams, OcrRequestError> {
        MISTRAL_OCR_CONFIG.map_ocr_params(params)
    }

    #[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
    fn transform_ocr_request(
        &self,
        model: &str,
        document: OcrDocument,
        params: &MistralOcrParams,
    ) -> Result<MistralOcrRequest, OcrRequestError> {
        MISTRAL_OCR_CONFIG.transform_ocr_request(model, document, params)
    }

    fn transform_ocr_response(
        &self,
        model: &str,
        response: MistralOcrResponse,
        params: &MistralOcrParams,
    ) -> Result<OcrResponseData, OcrResponseError> {
        MISTRAL_OCR_CONFIG.transform_ocr_response(model, response, params)
    }

    #[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
    fn complete_url(
        &self,
        connection: &OcrConnection,
        model: &str,
        _params: &MistralOcrParams,
    ) -> Result<String, OcrError> {
        let location = location(connection);
        let default_base = format!("https://{location}-aiplatform.googleapis.com");
        let base = connection
            .api_base
            .as_deref()
            .unwrap_or(&default_base)
            .trim_end_matches('/');
        Ok(format!(
            "{base}/v1/projects/{}/locations/{location}/publishers/mistralai/models/{model}:rawPredict",
            project(connection)?
        ))
    }

    async fn prepare_document(
        &self,
        http_client: &reqwest::Client,
        document: OcrDocument,
        connection: &OcrConnection,
        _headers: &[(String, String)],
    ) -> Result<OcrDocument, OcrError> {
        crate::ocr::client::convert_document_url_to_data_uri(http_client, document, connection)
            .await
    }

    async fn authenticate(
        &self,
        connection: &OcrConnection,
    ) -> Result<Vec<(String, String)>, AuthError> {
        authenticate_vertex(connection).await
    }
}
