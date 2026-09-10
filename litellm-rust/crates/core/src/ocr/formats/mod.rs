use crate::ocr::error::OcrRequestError;
use crate::ocr::error::OcrResponseError;

use super::types::OcrResponseData;
use crate::Error;
use serde::{Serialize, de::DeserializeOwned};
use serde_json::{Map, Value};

mod macros;

pub(crate) use macros::ocr_format;

pub mod deepseek;
pub mod document_intelligence;
pub mod mistral;
pub mod reducto;

pub trait OcrFormat: Send + Sync + Sized + 'static {
    type InputParams: std::fmt::Debug + Clone + Serialize + DeserializeOwned + Send + Sync;
    type MappedParams: Clone + Serialize + Send + Sync;
    type PreparedDocument: Send;
    type RequestBody: Serialize + DeserializeOwned + Send + Sync;
    type ResponseBody: DeserializeOwned + Send;

    fn validate_input_params(_params: &Map<String, Value>) -> Result<(), OcrRequestError> {
        Ok(())
    }

    fn map_params(params: Self::InputParams) -> Result<Self::MappedParams, OcrRequestError>;
    fn transform_request(
        model: &str,
        document: Self::PreparedDocument,
        params: &Self::MappedParams,
    ) -> Result<Self::RequestBody, OcrRequestError>;
    fn transform_response(
        model: &str,
        response: Self::ResponseBody,
        params: &Self::MappedParams,
    ) -> Result<OcrResponseData, OcrResponseError>;
}

pub(crate) fn request_error(path: &str) -> Error {
    OcrRequestError::RequestField { path: path.into() }.into()
}
