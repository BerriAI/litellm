use crate::constants::{VERTEX_DEEPSEEK_API_BASE, VERTEX_OCR_DEFAULT_LOCATION};
use crate::ocr::backends::{HostConfig, OcrHost, OcrIntegration, PreparedOcrBackend};
use crate::ocr::error::OcrError;
use crate::ocr::formats::deepseek::{DeepSeekOcrFormat, types::DeepSeekOcrParams};
use crate::ocr::formats::mistral::{MistralOcrFormat, types::MistralOcrParams};
use crate::ocr::types::{OcrConnection, OcrDocument};
use crate::providers::vertex_ai::auth::{self, VertexAuthInputs};

async fn authenticate(
    connection: &OcrConnection,
    config: &VertexAuthInputs,
    env_lookup: &(dyn Fn(&str) -> Option<String> + Sync),
) -> Result<auth::VertexAuthentication, OcrError> {
    Ok(auth::authenticate(
        connection.extra_headers.clone(),
        connection.api_key.as_deref(),
        config,
        None,
        env_lookup,
    )
    .await?)
}

fn location(config: &VertexAuthInputs, env_lookup: &dyn Fn(&str) -> Option<String>) -> String {
    auth::resolve_location(config, env_lookup)
        .unwrap_or_else(|| VERTEX_OCR_DEFAULT_LOCATION.to_string())
}

#[derive(Clone, Debug)]
pub struct VertexMistral;

impl OcrIntegration for VertexMistral {
    type Host = VertexHost;
    type Format = MistralOcrFormat;
    type PreparedDocument = crate::ocr::document::InlineOcrDocument;
    const FORMAT: Self::Format = MistralOcrFormat;

    #[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
    async fn prepare(
        &self,
        connection: &OcrConnection,
        config: &HostConfig<Self>,
        model: &str,
        _params: &MistralOcrParams,
        env_lookup: &(dyn Fn(&str) -> Option<String> + Sync),
    ) -> Result<PreparedOcrBackend, OcrError> {
        let authentication = authenticate(connection, config, env_lookup).await?;
        let location = location(config, env_lookup);
        let default_base = format!("https://{location}-aiplatform.googleapis.com");
        let base = connection
            .api_base
            .as_deref()
            .unwrap_or(&default_base)
            .trim_end_matches('/');
        Ok(PreparedOcrBackend {
            url: format!(
                "{base}/v1/projects/{}/locations/{location}/publishers/mistralai/models/{model}:rawPredict",
                authentication.project_id
            ),
            headers: authentication.headers,
        })
    }

    fn validate_request_body(
        &self,
        body: &crate::ocr::formats::mistral::types::MistralOcrRequest,
    ) -> Result<(), crate::ocr::error::OcrRequestError> {
        crate::ocr::document::InlineOcrDocument::try_from(body.document.clone())?;
        Ok(())
    }

    async fn prepare_document(
        &self,
        client: &crate::ocr::OcrClient,
        document: OcrDocument,
        connection: &OcrConnection,
        _headers: &[(String, String)],
    ) -> Result<Self::PreparedDocument, OcrError> {
        crate::ocr::document::inline_remote_document(
            client.document_fetcher(),
            document,
            connection,
        )
        .await
    }
}

#[derive(Clone, Debug)]
pub struct VertexDeepSeek;

impl OcrIntegration for VertexDeepSeek {
    type Host = VertexHost;
    type Format = DeepSeekOcrFormat;
    type PreparedDocument = OcrDocument;
    const FORMAT: Self::Format = DeepSeekOcrFormat;

    async fn prepare(
        &self,
        connection: &OcrConnection,
        config: &HostConfig<Self>,
        _model: &str,
        _params: &DeepSeekOcrParams,
        env_lookup: &(dyn Fn(&str) -> Option<String> + Sync),
    ) -> Result<PreparedOcrBackend, OcrError> {
        let authentication = authenticate(connection, config, env_lookup).await?;
        let location = location(config, env_lookup);
        let base = connection
            .api_base
            .as_deref()
            .unwrap_or(VERTEX_DEEPSEEK_API_BASE)
            .trim_end_matches('/');
        Ok(PreparedOcrBackend {
            url: format!(
                "{base}/v1/projects/{}/locations/{location}/endpoints/openapi/chat/completions",
                authentication.project_id
            ),
            headers: authentication.headers,
        })
    }

    async fn prepare_document(
        &self,
        _client: &crate::ocr::OcrClient,
        document: OcrDocument,
        _connection: &OcrConnection,
        _headers: &[(String, String)],
    ) -> Result<Self::PreparedDocument, OcrError> {
        Ok(document)
    }
}

#[derive(Clone, Debug)]
pub struct VertexHost;

impl OcrHost for VertexHost {
    type Config = VertexAuthInputs;
    const PROVIDER: crate::ocr::registry::OcrProvider = crate::ocr::registry::OcrProvider::VertexAi;
}
