use super::{MistralOcrParams, MistralOcrRequest, MistralOcrResponse};
use crate::ocr::error::{OcrRequestError, OcrResponseError};
use crate::ocr::types::{OcrDocument, OcrResponseData};

pub(crate) fn encode_request(
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

pub(crate) fn decode_response(
    model: &str,
    response: MistralOcrResponse,
    _params: &MistralOcrParams,
) -> Result<OcrResponseData, OcrResponseError> {
    Ok(OcrResponseData {
        pages: response.pages,
        model: response.model.unwrap_or_else(|| model.to_string()),
        document_annotation: response.document_annotation,
        usage_info: response.usage_info,
        object: "ocr".to_string(),
        extra_fields: response.extra_fields,
        provider_native_response: None,
    })
}
