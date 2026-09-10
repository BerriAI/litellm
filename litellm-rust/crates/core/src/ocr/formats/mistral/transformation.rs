use super::types::{MistralOcrParams, MistralOcrRequest, MistralOcrResponse};
use crate::ocr::error::{OcrRequestError, OcrResponseError};
use crate::ocr::types::{OcrDocument, OcrResponseData};

pub(super) fn request(
    model: &str,
    document: OcrDocument,
    params: &MistralOcrParams,
) -> Result<MistralOcrRequest, OcrRequestError> {
    Ok(MistralOcrRequest {
        model: model.to_string(),
        document,
        params: params.clone(),
    })
}

pub(super) fn response(
    model: &str,
    response: MistralOcrResponse,
    _params: &MistralOcrParams,
) -> Result<OcrResponseData, OcrResponseError> {
    Ok(OcrResponseData {
        document_annotation: response.document_annotation,
        usage_info: response.usage_info,
        ..OcrResponseData::new(
            response.model.unwrap_or_else(|| model.to_string()),
            response.pages,
        )
    })
}
