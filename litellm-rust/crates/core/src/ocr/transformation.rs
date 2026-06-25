use crate::CoreResult;

use super::types::{
    MistralOcrOptionalParams, MistralOcrResponseData, OcrDocument, OcrRequestData, OcrResponseData,
};

pub trait OcrProviderConfig {
    fn supported_ocr_params(&self) -> &'static [&'static str];

    fn map_ocr_params(
        &self,
        non_default_params: &MistralOcrOptionalParams,
    ) -> MistralOcrOptionalParams {
        non_default_params.clone()
    }

    fn transform_ocr_request(
        &self,
        model: &str,
        document: OcrDocument,
        optional_params: MistralOcrOptionalParams,
    ) -> CoreResult<OcrRequestData>;

    fn transform_ocr_response(
        &self,
        model: &str,
        response_json: MistralOcrResponseData,
    ) -> CoreResult<OcrResponseData>;
}
