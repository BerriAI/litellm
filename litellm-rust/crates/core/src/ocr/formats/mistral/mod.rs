pub mod types;

use self::types::{MistralOcrParams, MistralOcrRequest, MistralOcrResponse};
use crate::ocr::error::OcrRequestError;
use crate::ocr::error::OcrResponseError;
use crate::ocr::formats::OcrFormat;
use crate::ocr::types::{OcrDocument, OcrResponseData};

pub struct MistralOcrFormat;

impl OcrFormat for MistralOcrFormat {
    type InputParams = MistralOcrParams;
    type MappedParams = MistralOcrParams;
    type PreparedDocument = OcrDocument;
    type RequestBody = MistralOcrRequest;
    type ResponseBody = MistralOcrResponse;

    #[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
    fn map_ocr_params(
        &self,
        params: MistralOcrParams,
    ) -> Result<MistralOcrParams, OcrRequestError> {
        Ok(params)
    }

    #[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
    fn transform_ocr_request(
        &self,
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

    #[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
    fn transform_ocr_response(
        &self,
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
}
