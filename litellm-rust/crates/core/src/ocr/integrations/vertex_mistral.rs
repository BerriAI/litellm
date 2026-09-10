use crate::ocr::backends::PreparedOcrBackend;
use crate::ocr::backends::vertex_ai::{self, VertexBackend};
use crate::ocr::document::InlineOcrDocument;
use crate::ocr::error::{OcrError, OcrRequestError};
use crate::ocr::formats::mistral::{MistralOcrFormat, types::MistralOcrParams};
use crate::ocr::types::OcrConnection;
use crate::url_utils::ApiUrl;

use super::{BackendConfig, OcrIntegration};

#[derive(Clone, Debug)]
pub struct VertexMistral;

impl OcrIntegration for VertexMistral {
    type Backend = VertexBackend;
    type Format = MistralOcrFormat;
    type DocumentPreparation = super::RequireInline;
    const FORMAT: Self::Format = MistralOcrFormat;

    async fn prepare(
        &self,
        connection: &OcrConnection,
        config: &BackendConfig<Self>,
        model: &str,
        _params: &MistralOcrParams,
        env_lookup: &(dyn Fn(&str) -> Option<String> + Sync),
    ) -> Result<PreparedOcrBackend, OcrError> {
        let authentication = vertex_ai::authenticate(connection, config, env_lookup).await?;
        let location = vertex_ai::location(config, env_lookup);
        let default_base = format!("https://{location}-aiplatform.googleapis.com");
        let base = connection.api_base.as_deref().unwrap_or(&default_base);
        let prediction = format!("{model}:rawPredict");
        let path = [
            "v1",
            "projects",
            authentication.project_id.as_str(),
            "locations",
            location.as_str(),
            "publishers",
            "mistralai",
            "models",
            prediction.as_str(),
        ];
        let url = ApiUrl::parse(base)
            .and_then(|url| url.complete_path(&path))
            .map(|url| url.into_string())
            .map_err(|_| OcrRequestError::RequestField {
                path: "api_base".into(),
            })?;
        Ok(PreparedOcrBackend {
            url,
            headers: authentication.headers,
        })
    }

    fn validate_request_body(
        &self,
        body: &crate::ocr::formats::mistral::types::MistralOcrRequest,
    ) -> Result<(), OcrRequestError> {
        InlineOcrDocument::try_from(body.document.clone())?;
        Ok(())
    }
}
