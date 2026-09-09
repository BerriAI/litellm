use crate::auth::AuthError;
use crate::ocr::error::OcrError;
use crate::ocr::error::OcrRequestError;
use crate::ocr::error::OcrResponseError;
use std::future::Future;

use super::types::{OcrConnection, OcrDocument, OcrResponseData};
use super::wire::DecodedOcrResponse;
use crate::Error;
use serde::{Serialize, de::DeserializeOwned};

pub trait OcrProviderConfig: Send + Sync + Sized + 'static {
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
    fn complete_url(
        &self,
        connection: &OcrConnection,
        model: &str,
        params: &Self::MappedParams,
    ) -> Result<String, OcrError>;
    fn authenticate(
        &self,
        connection: &OcrConnection,
    ) -> impl Future<Output = Result<Vec<(String, String)>, AuthError>> + Send;
    fn prepare_document(
        &self,
        http_client: &reqwest::Client,
        document: OcrDocument,
        connection: &OcrConnection,
        headers: &[(String, String)],
    ) -> impl Future<Output = Result<Self::PreparedDocument, OcrError>> + Send;
    fn preserve_native_response(&self, _params: &Self::MappedParams) -> bool {
        false
    }
    fn guard_document_before_preparation(&self) -> bool {
        false
    }

    fn read_response(
        &self,
        _http_client: &reqwest::Client,
        response: reqwest::Response,
        _url: &str,
        _headers: &[(String, String)],
        _connection: &OcrConnection,
        params: &Self::MappedParams,
    ) -> impl Future<Output = Result<DecodedOcrResponse<Self::ResponseBody>, OcrError>> + Send {
        super::wire::read_json_response(response, self.preserve_native_response(params))
    }
}

pub(crate) fn request_error(path: &str) -> Error {
    OcrRequestError::RequestField { path: path.into() }.into()
}
