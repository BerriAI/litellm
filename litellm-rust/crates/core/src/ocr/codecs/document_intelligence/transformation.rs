use base64::{Engine, engine::general_purpose::STANDARD};
use serde_json::{Map, Value, json};

use super::types::*;
use crate::constants::{AZURE_DI_DEFAULT_DPI, AZURE_DI_DEFAULT_HEIGHT, AZURE_DI_DEFAULT_WIDTH};
use crate::ocr::document::InlineDocument;
use crate::ocr::error::{OcrRequestError, OcrResponseError};
use crate::ocr::types::{LiteLLMOcrResponse, OcrDocument};

#[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
pub(crate) fn transform_ocr_request(
    document: OcrDocument,
) -> Result<DocumentIntelligenceRequest, OcrRequestError> {
    let source = document.source();
    if source.is_empty() {
        return Err(OcrRequestError::MissingField("document URL"));
    }
    Ok(if let Some(document) = InlineDocument::parse(source)? {
        DocumentIntelligenceRequest::Base64Source(
            STANDARD.encode(document.decode(crate::constants::OCR_INLINE_MAX_BYTES)?),
        )
    } else {
        DocumentIntelligenceRequest::UrlSource(source.to_string())
    })
}

pub(crate) fn transform_ocr_response(
    model: &str,
    response: AzureDocumentIntelligenceOperation,
) -> Result<LiteLLMOcrResponse, OcrResponseError> {
    if response.status != Some(OperationStatus::Succeeded) {
        return Err(OcrResponseError::OperationStatus(
            response
                .status
                .map(|status| status.to_string())
                .unwrap_or_else(|| "None".into()),
        ));
    }
    let result = response.analyze_result.unwrap_or_default();
    let pages = result
        .pages
        .into_iter()
        .map(normalize_page)
        .collect::<Result<Vec<_>, _>>()?;
    let pages_processed = pages.len();
    let mut extra_fields = Map::new();
    extra_fields.insert("content".into(), option_value(result.content));
    extra_fields.insert("tables".into(), option_value(result.tables));
    extra_fields.insert("keyValuePairs".into(), option_value(result.key_value_pairs));
    Ok(LiteLLMOcrResponse {
        pages,
        model: model.into(),
        document_annotation: None,
        usage_info: Some(json!({"pages_processed":pages_processed})),
        object: "ocr".into(),
        extra_fields,
        provider_native_response: None,
    })
}

fn normalize_page(page: AzureDocumentIntelligencePage) -> Result<Value, OcrResponseError> {
    let index = page
        .page_number
        .unwrap_or(1)
        .checked_sub(1)
        .ok_or(OcrResponseError::NumericRange("page.pageNumber"))?;
    let scale = if page.unit.as_deref().unwrap_or("inch") == "inch" {
        AZURE_DI_DEFAULT_DPI as f64
    } else {
        1.0
    };
    let width = pixel_dimension(
        page.width.unwrap_or(AZURE_DI_DEFAULT_WIDTH),
        scale,
        "page.width",
    )?;
    let height = pixel_dimension(
        page.height.unwrap_or(AZURE_DI_DEFAULT_HEIGHT),
        scale,
        "page.height",
    )?;
    let markdown = page
        .lines
        .iter()
        .map(|line| line.content.as_deref().unwrap_or_default())
        .collect::<Vec<_>>()
        .join("\n");
    Ok(json!({
        "index":index,
        "markdown":markdown,
        "images":null,
        "dimensions":{"width":width,"height":height,"dpi":AZURE_DI_DEFAULT_DPI}
    }))
}

fn pixel_dimension(value: f64, scale: f64, field: &'static str) -> Result<i64, OcrResponseError> {
    let value = value * scale;
    if !value.is_finite() || value < i64::MIN as f64 || value > i64::MAX as f64 {
        return Err(OcrResponseError::NumericRange(field));
    }
    Ok(value.trunc() as i64)
}

fn option_value<T: serde::Serialize>(value: Option<T>) -> Value {
    value
        .and_then(|value| serde_json::to_value(value).ok())
        .unwrap_or(Value::Null)
}
