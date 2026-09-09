use crate::auth::AuthError;
use crate::ocr::error::OcrError;
use crate::ocr::transformation::OcrBackend;
use crate::ocr::types::{OcrConnection, OcrDocument};
use crate::providers::mistral::ocr::transformation::MistralOcrFormat;
use crate::providers::mistral::ocr::types::MistralOcrParams;
use crate::providers::vertex_ai::ocr::{authenticate_vertex, location, project};

pub struct VertexMistralOcrBackend;
pub const VERTEX_MISTRAL_OCR_BACKEND: VertexMistralOcrBackend = VertexMistralOcrBackend;

impl OcrBackend for VertexMistralOcrBackend {
    type Format = MistralOcrFormat;
    const FORMAT: Self::Format = MistralOcrFormat;

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
