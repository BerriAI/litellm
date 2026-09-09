use crate::auth::AuthError;
use crate::constants::{VERTEX_DEEPSEEK_API_BASE, VERTEX_OCR_DEFAULT_LOCATION};
use crate::ocr::backends::OcrBackend;
use crate::ocr::error::OcrError;
use crate::ocr::error::OcrRequestError;
use crate::ocr::formats::deepseek::{DeepSeekOcrFormat, types::DeepSeekOcrParams};
use crate::ocr::formats::mistral::{MistralOcrFormat, types::MistralOcrParams};
use crate::ocr::types::{OcrConnection, OcrDocument};
use crate::providers::vertex_ai::auth;

pub struct VertexAiOcrBackend;

fn project(connection: &OcrConnection) -> Result<&str, OcrRequestError> {
    connection
        .vertex
        .project
        .as_deref()
        .ok_or(OcrRequestError::MissingField("vertex_project"))
}

fn location(connection: &OcrConnection) -> &str {
    connection
        .vertex
        .location
        .as_deref()
        .unwrap_or(VERTEX_OCR_DEFAULT_LOCATION)
}

async fn authenticate_vertex(
    connection: &OcrConnection,
) -> Result<Vec<(String, String)>, AuthError> {
    auth::authenticate(
        connection.extra_headers.clone(),
        connection.api_key.as_deref(),
        &connection.vertex_auth,
        connection.vertex.project.clone(),
        &|name| std::env::var(name).ok(),
    )
    .await
    .map(|authentication| authentication.headers)
}

impl OcrBackend<MistralOcrFormat> for VertexAiOcrBackend {
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

impl OcrBackend<DeepSeekOcrFormat> for VertexAiOcrBackend {
    fn complete_url(
        &self,
        connection: &OcrConnection,
        _model: &str,
        _params: &DeepSeekOcrParams,
    ) -> Result<String, OcrError> {
        let base = connection
            .api_base
            .as_deref()
            .unwrap_or(VERTEX_DEEPSEEK_API_BASE)
            .trim_end_matches('/');
        Ok(format!(
            "{base}/v1/projects/{}/locations/{}/endpoints/openapi/chat/completions",
            project(connection)?,
            location(connection)
        ))
    }

    async fn prepare_document(
        &self,
        _http_client: &reqwest::Client,
        document: OcrDocument,
        _connection: &OcrConnection,
        _headers: &[(String, String)],
    ) -> Result<OcrDocument, OcrError> {
        Ok(document)
    }

    async fn authenticate(
        &self,
        connection: &OcrConnection,
    ) -> Result<Vec<(String, String)>, AuthError> {
        authenticate_vertex(connection).await
    }
}
