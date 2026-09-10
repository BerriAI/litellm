use crate::constants::VERTEX_DEEPSEEK_API_BASE;
use crate::ocr::backends::PreparedOcrBackend;
use crate::ocr::backends::vertex_ai::{self, VertexBackend};
use crate::ocr::error::{OcrError, OcrRequestError};
use crate::ocr::formats::deepseek::{DeepSeekOcrFormat, types::DeepSeekOcrParams};
use crate::ocr::types::OcrConnection;
use crate::url_utils::ApiUrl;

use super::{BackendConfig, OcrIntegration};

#[derive(Clone, Debug)]
pub struct VertexDeepSeek;

impl OcrIntegration for VertexDeepSeek {
    type Backend = VertexBackend;
    type Format = DeepSeekOcrFormat;
    type DocumentPreparation = super::PassThrough;

    async fn prepare(
        &self,
        connection: &OcrConnection,
        config: &BackendConfig<Self>,
        _model: &str,
        _params: &DeepSeekOcrParams,
        env_lookup: &(dyn Fn(&str) -> Option<String> + Sync),
    ) -> Result<PreparedOcrBackend, OcrError> {
        let authentication = vertex_ai::authenticate(connection, config, env_lookup).await?;
        let location = vertex_ai::location(config, env_lookup);
        let base = connection
            .api_base
            .as_deref()
            .unwrap_or(VERTEX_DEEPSEEK_API_BASE);
        let url = ApiUrl::parse(base)
            .and_then(|url| {
                url.complete_path(&[
                    "v1",
                    "projects",
                    authentication.project_id.as_str(),
                    "locations",
                    location.as_str(),
                    "endpoints",
                    "openapi",
                    "chat",
                    "completions",
                ])
            })
            .map(|url| url.into_string())
            .map_err(|_| OcrRequestError::RequestField {
                path: "api_base".into(),
            })?;
        Ok(PreparedOcrBackend {
            url,
            headers: authentication.headers,
        })
    }
}
