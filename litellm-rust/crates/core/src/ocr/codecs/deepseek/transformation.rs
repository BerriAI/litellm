use serde::de::IntoDeserializer;
use serde_json::{Value, json};

use super::types::*;
use crate::ocr::error::{OcrRequestError, OcrResponseError};
use crate::ocr::types::{LiteLLMOcrResponse, OcrDocument};

#[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
pub(crate) fn transform_ocr_request(
    provider_model: &str,
    document: OcrDocument,
    params: &DeepSeekOcrParams,
) -> Result<DeepSeekOcrRequest, OcrRequestError> {
    if document.source().is_empty() {
        return Err(OcrRequestError::MissingField("document URL"));
    }
    let content = OcrDocument::ImageUrl {
        image_url: document.source().to_string(),
        extra_fields: serde_json::Map::new(),
    };
    Ok(DeepSeekOcrRequest {
        model: provider_model.to_string(),
        messages: vec![DeepSeekOcrMessage {
            role: UserRole::User,
            content: vec![content],
        }],
        params: params.clone(),
    })
}

pub(crate) fn transform_ocr_response(
    model: &str,
    response: DeepSeekOcrResponse,
) -> Result<LiteLLMOcrResponse, OcrResponseError> {
    let content = response
        .choices
        .into_iter()
        .next()
        .and_then(|choice| choice.message.content)
        .ok_or(OcrResponseError::EmptyContent)?;
    let decoded = decode_content(content)?;
    let pages = match decoded.result.pages {
        Some(pages) if !pages.is_empty() => pages
            .into_iter()
            .map(|page| serde_json::to_value(page).expect("DeepSeek page serializes"))
            .collect(),
        _ => vec![json!({
            "index":0,
            "markdown":decoded.fallback_markdown,
            "images":null
        })],
    };
    Ok(LiteLLMOcrResponse {
        pages,
        model: decoded.result.model.unwrap_or_else(|| model.to_string()),
        document_annotation: decoded.result.document_annotation,
        usage_info: decoded.result.usage_info.or(response.usage),
        object: "ocr".into(),
        extra_fields: decoded.result.extra_fields,
        provider_native_response: None,
    })
}

struct DecodedContent {
    result: DeepSeekOcrResult,
    fallback_markdown: String,
}

fn decode_content(content: DeepSeekContent) -> Result<DecodedContent, OcrResponseError> {
    let (result, fallback_markdown) = match content {
        DeepSeekContent::Text(text) if text.is_empty() => {
            return Err(OcrResponseError::EmptyContent);
        }
        DeepSeekContent::Text(text) => (decode_json_content(&text)?, text),
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
        fallback_markdown,
    })
}

fn decode_json_content(text: &str) -> Result<Option<DeepSeekOcrResult>, OcrResponseError> {
    if !text.trim_start().starts_with('{') {
        return Ok(None);
    }
    let value = match serde_json::from_str::<Value>(text) {
        Ok(value) => value,
        Err(_) => return Ok(None),
    };
    serde_path_to_error::deserialize(value.into_deserializer())
        .map(Some)
        .map_err(|error| OcrResponseError::ResponseField {
            path: format!("choices[0].message.content.{}", error.path()),
        })
}
