use super::types::*;
use crate::auth::AuthError;
use crate::constants::VERTEX_DEEPSEEK_API_BASE;
use crate::ocr::error::OcrError;
use crate::ocr::error::OcrRequestError;
use crate::ocr::error::OcrResponseError;
use crate::ocr::transformation::{OcrBackend, OcrFormat};
use crate::ocr::types::{OcrConnection, OcrDocument, OcrPage, OcrResponseData};
use crate::providers::vertex_ai::ocr::{authenticate_vertex, location, project};

pub struct VertexDeepSeekOcrBackend;
pub const VERTEX_DEEPSEEK_OCR_BACKEND: VertexDeepSeekOcrBackend = VertexDeepSeekOcrBackend;

pub struct DeepSeekOcrFormat;

impl OcrFormat for DeepSeekOcrFormat {
    type InputParams = DeepSeekOcrParams;
    type MappedParams = DeepSeekOcrParams;
    type PreparedDocument = OcrDocument;
    type RequestBody = DeepSeekOcrRequest;
    type ResponseBody = DeepSeekOcrResponse;
    #[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
    fn map_ocr_params(
        &self,
        params: Self::InputParams,
    ) -> Result<Self::MappedParams, OcrRequestError> {
        Ok(params)
    }

    #[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
    fn transform_ocr_request(
        &self,
        model: &str,
        document: OcrDocument,
        params: &Self::MappedParams,
    ) -> Result<Self::RequestBody, OcrRequestError> {
        if document.source().is_empty() {
            return Err(OcrRequestError::MissingField("document URL"));
        }
        let model = model.strip_prefix("vertex_ai/").unwrap_or(model);
        let model = if model.contains('/') {
            model.to_string()
        } else {
            format!("deepseek-ai/{model}")
        };
        Ok(DeepSeekOcrRequest {
            model,
            messages: vec![DeepSeekOcrMessage {
                role: UserRole::User,
                content: vec![document],
            }],
            params: params.clone(),
        })
    }

    #[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
    fn transform_ocr_response(
        &self,
        model: &str,
        response: Self::ResponseBody,
        _params: &Self::MappedParams,
    ) -> Result<OcrResponseData, OcrResponseError> {
        let content = response
            .choices
            .into_iter()
            .next()
            .and_then(|choice| choice.message.content)
            .ok_or(OcrResponseError::EmptyContent)?;
        let (result, fallback) = match content {
            DeepSeekContent::Text(text) if text.is_empty() => {
                return Err(OcrResponseError::EmptyContent);
            }
            DeepSeekContent::Text(text) => {
                let result = if text.trim_start().starts_with('{') {
                    crate::ocr::wire::decode_deepseek_content(&text)?
                } else {
                    None
                };
                (result, text)
            }
            DeepSeekContent::Object(object) => {
                let fallback = serde_json::to_string(&object).map_err(|_| {
                    OcrResponseError::ResponseField {
                        path: "choices[0].message.content".into(),
                    }
                })?;
                (Some(object), fallback)
            }
        };
        let result = result.unwrap_or_default();
        let pages = match result.pages {
            Some(pages) if !pages.is_empty() => pages
                .into_iter()
                .map(|page| OcrPage {
                    images: page.images,
                    dimensions: page.dimensions,
                    ..OcrPage::text(page.index, page.markdown)
                })
                .collect(),
            _ => vec![OcrPage::text(0, fallback)],
        };
        Ok(OcrResponseData {
            document_annotation: result.document_annotation,
            usage_info: result.usage_info.or(response.usage),
            ..OcrResponseData::new(result.model.unwrap_or_else(|| model.to_string()), pages)
        })
    }
}

impl OcrBackend for VertexDeepSeekOcrBackend {
    type Format = DeepSeekOcrFormat;
    const FORMAT: Self::Format = DeepSeekOcrFormat;

    #[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
    fn complete_url(
        &self,
        connection: &OcrConnection,
        _model: &str,
        _params: &<Self::Format as OcrFormat>::MappedParams,
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
