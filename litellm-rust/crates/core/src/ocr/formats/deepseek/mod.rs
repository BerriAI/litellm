pub mod types;

use self::types::*;
use crate::ocr::error::OcrRequestError;
use crate::ocr::error::OcrResponseError;
use crate::ocr::formats::OcrFormat;
use crate::ocr::types::{OcrDocument, OcrPage, OcrResponseData};
use serde::de::IntoDeserializer;
use serde_json::Value;

pub struct DeepSeekOcrFormat;

fn decode_content(text: &str) -> Result<Option<DeepSeekOcrResult>, OcrResponseError> {
    match serde_json::from_str::<Value>(text) {
        Err(_) => Ok(None),
        Ok(value) => serde_path_to_error::deserialize(value.into_deserializer())
            .map(Some)
            .map_err(|error| OcrResponseError::ResponseField {
                path: format!("choices[0].message.content.{}", error.path()),
            }),
    }
}

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
                    decode_content(&text)?
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
