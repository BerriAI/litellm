use base64::{Engine, engine::general_purpose::STANDARD};

use super::types::*;
use crate::constants::{AZURE_DI_DEFAULT_DPI, AZURE_DI_DEFAULT_HEIGHT, AZURE_DI_DEFAULT_WIDTH};
use crate::ocr::document::InlineDocument;
use crate::ocr::error::{OcrRequestError, OcrResponseError};
use crate::ocr::types::{OcrDocument, OcrPage, OcrPageDimensions, OcrResponseData, OcrUsageInfo};

fn pixel_dimension(value: f64, scale: f64, field: &'static str) -> Result<i64, OcrResponseError> {
    crate::ocr::wire::checked_truncated_i64(value * scale)
        .ok_or(OcrResponseError::NumericRange(field))
}

fn convert_dimensions(
    width: f64,
    height: f64,
    unit: &str,
) -> Result<OcrPageDimensions, OcrResponseError> {
    let scale = if unit == "inch" {
        AZURE_DI_DEFAULT_DPI as f64
    } else {
        1.0
    };
    Ok(OcrPageDimensions {
        dpi: Some(AZURE_DI_DEFAULT_DPI),
        width: Some(pixel_dimension(width, scale, "page.width")?),
        height: Some(pixel_dimension(height, scale, "page.height")?),
    })
}

fn page_markdown(lines: &[AzureDocumentIntelligenceLine]) -> String {
    lines
        .iter()
        .map(|line| line.content.as_deref().unwrap_or_default())
        .collect::<Vec<_>>()
        .join("\n")
}

fn transform_azure_page(page: AzureDocumentIntelligencePage) -> Result<OcrPage, OcrResponseError> {
    let index = page
        .page_number
        .unwrap_or(1)
        .checked_sub(1)
        .ok_or(OcrResponseError::NumericRange("page.pageNumber"))?;
    Ok(OcrPage {
        dimensions: Some(convert_dimensions(
            page.width.unwrap_or(AZURE_DI_DEFAULT_WIDTH),
            page.height.unwrap_or(AZURE_DI_DEFAULT_HEIGHT),
            page.unit.as_deref().unwrap_or("inch"),
        )?),
        ..OcrPage::text(index, page_markdown(&page.lines))
    })
}

pub(super) fn request(
    _model: &str,
    document: OcrDocument,
    _params: &DocumentIntelligenceParams,
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

pub(super) fn response(
    model: &str,
    response: AzureDocumentIntelligenceOperation,
    _params: &DocumentIntelligenceParams,
) -> Result<OcrResponseData, OcrResponseError> {
    if response.status != Some(OperationStatus::Succeeded) {
        return Err(OcrResponseError::OperationStatus(
            response
                .status
                .map(|s| s.to_string())
                .unwrap_or_else(|| "None".into()),
        ));
    }
    let result = response.analyze_result.unwrap_or_default();
    let pages = result
        .pages
        .into_iter()
        .map(transform_azure_page)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(OcrResponseData {
        usage_info: Some(OcrUsageInfo {
            pages_processed: Some(pages.len() as i64),
            ..Default::default()
        }),
        content: result.content,
        tables: result.tables,
        key_value_pairs: result.key_value_pairs,
        ..OcrResponseData::new(model.to_string(), pages)
    })
}
