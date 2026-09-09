use crate::ocr::error::OcrRequestError;
use crate::ocr::error::OcrResponseError;

use super::types::OcrResponseData;
use crate::Error;
use serde::{Serialize, de::DeserializeOwned};

pub mod deepseek;
pub mod document_intelligence;
pub mod mistral;
pub mod reducto;

pub trait OcrFormat: Send + Sync + Sized + 'static {
    type InputParams: Clone + Serialize + DeserializeOwned + Send + Sync;
    type MappedParams: Clone + Serialize + Send + Sync + Into<Self::InputParams>;
    type PreparedDocument: Send;
    type RequestBody: Serialize + DeserializeOwned + Send + Sync;
    type ResponseBody: DeserializeOwned + Send;

    fn map_ocr_params(
        &self,
        params: Self::InputParams,
    ) -> Result<Self::MappedParams, OcrRequestError>;
    fn transform_ocr_request(
        &self,
        model: &str,
        document: Self::PreparedDocument,
        params: &Self::MappedParams,
    ) -> Result<Self::RequestBody, OcrRequestError>;
    fn transform_ocr_response(
        &self,
        model: &str,
        response: Self::ResponseBody,
        params: &Self::MappedParams,
    ) -> Result<OcrResponseData, OcrResponseError>;
}

pub(crate) fn request_error(path: &str) -> Error {
    OcrRequestError::RequestField { path: path.into() }.into()
}
