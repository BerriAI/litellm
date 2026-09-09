use crate::ocr::document::InlineDocument;
use crate::ocr::error::OcrRequestError;
use crate::ocr::error::OcrResponseError;
use crate::ocr::error::PagesError;
use base64::{Engine, engine::general_purpose::STANDARD};
use std::collections::BTreeSet;

pub mod types;

use self::types::*;
use crate::constants::{AZURE_DI_DEFAULT_DPI, AZURE_DI_DEFAULT_HEIGHT, AZURE_DI_DEFAULT_WIDTH};
use crate::ocr::formats::OcrFormat;
use crate::ocr::types::{OcrDocument, OcrPage, OcrPageDimensions, OcrResponseData, OcrUsageInfo};

fn pages_token_is_valid(token: &str) -> bool {
    let mut parts = token.split('-');
    let start = parts.next().unwrap_or_default();
    if start.is_empty() || !start.chars().all(|ch| ch.is_ascii_digit()) {
        return false;
    }
    match parts.next() {
        None => true,
        Some(end) => {
            !end.is_empty() && end.chars().all(|ch| ch.is_ascii_digit()) && parts.next().is_none()
        }
    }
}

fn normalize_pages_param(pages: PagesInput) -> Result<Option<NormalizedPages>, PagesError> {
    let normalized = match pages {
        PagesInput::ZeroBasedIndices(indices) => {
            if indices.is_empty() {
                return Ok(None);
            }
            indices
                .into_iter()
                .map(|page| {
                    if page < 0 {
                        return Err(PagesError::NegativeIndex);
                    }
                    page.checked_add(1).ok_or(PagesError::IndexOutOfRange)
                })
                .collect::<Result<BTreeSet<_>, _>>()?
                .into_iter()
                .map(|page| page.to_string())
                .collect::<Vec<_>>()
                .join(",")
        }
        PagesInput::NativeTokens(tokens) => {
            if tokens.is_empty() {
                return Ok(None);
            }
            tokens
                .iter()
                .map(|token| token.trim())
                .collect::<Vec<_>>()
                .join(",")
        }
        PagesInput::NativeRange(range) => range
            .split(',')
            .map(str::trim)
            .collect::<Vec<_>>()
            .join(","),
    };
    if !normalized.split(',').all(pages_token_is_valid) {
        return Err(PagesError::InvalidNativeRange);
    }
    Ok(Some(NormalizedPages(normalized)))
}

fn normalize_features_param(
    features: FeaturesInput,
) -> Result<Option<NormalizedFeatures>, OcrRequestError> {
    let tokens = match features {
        FeaturesInput::Names(names) => names,
        FeaturesInput::CommaSeparated(names) => names.split(',').map(str::to_string).collect(),
    };
    if tokens.is_empty() {
        return Ok(None);
    }
    let normalized = tokens.iter().map(|token| token.trim()).collect::<Vec<_>>();
    if !normalized.iter().all(|token| {
        let Some((first, rest)) = token.as_bytes().split_first() else {
            return false;
        };
        first.is_ascii_alphabetic() && rest.iter().all(u8::is_ascii_alphanumeric)
    }) {
        return Err(OcrRequestError::Features);
    }
    Ok(Some(NormalizedFeatures(normalized.join(","))))
}

fn pixel_dimension(value: f64, scale: f64, field: &'static str) -> Result<i64, OcrResponseError> {
    let pixels = (value * scale).trunc();
    if !pixels.is_finite() || pixels < i64::MIN as f64 || pixels >= -(i64::MIN as f64) {
        return Err(OcrResponseError::NumericRange(field));
    }
    Ok(pixels as i64)
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

pub struct AzureDocumentIntelligenceOcrFormat;

impl OcrFormat for AzureDocumentIntelligenceOcrFormat {
    type InputParams = DocumentIntelligenceInputParams;
    type MappedParams = DocumentIntelligenceParams;
    type PreparedDocument = OcrDocument;
    type RequestBody = DocumentIntelligenceRequest;
    type ResponseBody = AzureDocumentIntelligenceOperation;

    #[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
    fn map_ocr_params(
        &self,
        params: Self::InputParams,
    ) -> Result<Self::MappedParams, OcrRequestError> {
        Ok(DocumentIntelligenceParams {
            pages: params
                .pages
                .map(normalize_pages_param)
                .transpose()?
                .flatten(),
            features: params
                .features
                .map(normalize_features_param)
                .transpose()?
                .flatten(),
            request_format: params.req_format.unwrap_or_default(),
        })
    }

    #[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
    fn transform_ocr_request(
        &self,
        _model: &str,
        document: OcrDocument,
        _params: &Self::MappedParams,
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

    #[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
    fn transform_ocr_response(
        &self,
        model: &str,
        response: Self::ResponseBody,
        _params: &Self::MappedParams,
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
}
