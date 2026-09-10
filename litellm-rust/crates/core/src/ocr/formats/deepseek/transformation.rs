use serde::de::IntoDeserializer;
use serde_json::Value;

use super::types::*;
use crate::constants::DEEPSEEK_OCR_MODEL_NAMESPACE;
use crate::ocr::error::{OcrRequestError, OcrResponseError};
use crate::ocr::types::{OcrDocument, OcrPage, OcrResponseData};

fn decode_json_content(text: &str) -> Result<Option<DeepSeekOcrResult>, OcrResponseError> {
    match serde_json::from_str::<Value>(text) {
        Err(_) => Ok(None),
        Ok(value) => serde_path_to_error::deserialize(value.into_deserializer())
            .map(Some)
            .map_err(|error| OcrResponseError::ResponseField {
                path: format!("choices[0].message.content.{}", error.path()),
            }),
    }
}

pub(super) fn request(
    model: &str,
    document: OcrDocument,
    params: &DeepSeekOcrParams,
) -> Result<DeepSeekOcrRequest, OcrRequestError> {
    if document.source().is_empty() {
        return Err(OcrRequestError::MissingField("document URL"));
    }
    let model = if model.contains('/') {
        model.to_string()
    } else {
        format!("{DEEPSEEK_OCR_MODEL_NAMESPACE}/{model}")
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

pub(super) fn response(
    model: &str,
    response: DeepSeekOcrResponse,
    _params: &DeepSeekOcrParams,
) -> Result<OcrResponseData, OcrResponseError> {
    let content = response
        .choices
        .into_iter()
        .next()
        .and_then(|choice| choice.message.content)
        .ok_or(OcrResponseError::EmptyContent)?;
    let DecodedContent {
        result,
        fallback_markdown,
    } = decode_content(content)?;
    let pages = match result.pages {
        Some(pages) if !pages.is_empty() => pages
            .into_iter()
            .map(|page| OcrPage {
                images: page.images,
                dimensions: page.dimensions,
                ..OcrPage::text(page.index, page.markdown)
            })
            .collect(),
        _ => vec![OcrPage::text(0, fallback_markdown)],
    };
    Ok(OcrResponseData {
        document_annotation: result.document_annotation,
        usage_info: result.usage_info.or(response.usage),
        ..OcrResponseData::new(result.model.unwrap_or_else(|| model.to_string()), pages)
    })
}

struct DecodedContent {
    result: DeepSeekOcrResult,
    fallback_markdown: String,
}

fn decode_content(content: DeepSeekContent) -> Result<DecodedContent, OcrResponseError> {
    let (result, fallback) = match content {
        DeepSeekContent::Text(text) if text.is_empty() => {
            return Err(OcrResponseError::EmptyContent);
        }
        DeepSeekContent::Text(text) => {
            let result = if text.trim_start().starts_with('{') {
                decode_json_content(&text)?
            } else {
                None
            };
            (result, text)
        }
        DeepSeekContent::Object(object) => {
            let fallback =
                serde_json::to_string(&object).map_err(|_| OcrResponseError::ResponseField {
                    path: "choices[0].message.content".into(),
                })?;
            (Some(object), fallback)
        }
    };
    Ok(DecodedContent {
        result: result.unwrap_or_default(),
        fallback_markdown: fallback,
    })
}
