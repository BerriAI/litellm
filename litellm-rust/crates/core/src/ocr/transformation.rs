use crate::auth::AuthError;
use crate::ocr::error::OcrError;
use crate::ocr::error::OcrRequestError;
use crate::ocr::error::OcrResponseError;
use std::future::Future;

use super::hooks::OcrRequestBody;
use super::prepare::OcrProviderRequest;
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
        document: OcrDocument,
        connection: &OcrConnection,
        headers: &[(String, String)],
    ) -> impl Future<Output = Result<Self::PreparedDocument, OcrError>> + Send;
    fn params_for_hook(&self, params: Self::MappedParams) -> OcrProviderRequest;
    fn params_from_hook(
        &self,
        params: OcrProviderRequest,
    ) -> Result<Self::MappedParams, OcrRequestError>;
    fn body_for_hook(&self, body: Self::RequestBody) -> OcrRequestBody;
    fn body_from_hook(&self, body: OcrRequestBody) -> Result<Self::RequestBody, OcrRequestError>;
    fn provider_name(&self) -> &'static str;
    fn preserve_native_response(&self, _params: &Self::MappedParams) -> bool {
        false
    }
    fn guard_document_before_preparation(&self) -> bool {
        false
    }

    fn read_response(
        &self,
        response: reqwest::Response,
        _url: &str,
        _headers: &[(String, String)],
        _connection: &OcrConnection,
        params: &Self::MappedParams,
    ) -> impl Future<Output = Result<DecodedOcrResponse<Self::ResponseBody>, OcrError>> + Send {
        super::wire::read_json_response(response, self.preserve_native_response(params))
    }
}

#[macro_export]
macro_rules! ocr_provider_hooks {
    ($provider:ident, $body:ident) => {
        fn params_for_hook(
            &self,
            params: Self::MappedParams,
        ) -> $crate::ocr::prepare::OcrProviderRequest {
            $crate::ocr::prepare::OcrProviderRequest::$provider(params.into())
        }
        fn params_from_hook(
            &self,
            params: $crate::ocr::prepare::OcrProviderRequest,
        ) -> Result<Self::MappedParams, $crate::ocr::error::OcrRequestError> {
            match params {
                $crate::ocr::prepare::OcrProviderRequest::$provider(params) => {
                    self.map_ocr_params(params)
                }
                _ => Err($crate::ocr::error::OcrRequestError::RequestField {
                    path: "guardrail.optional_params.provider".into(),
                }),
            }
        }
        fn body_for_hook(&self, body: Self::RequestBody) -> $crate::ocr::hooks::OcrRequestBody {
            $crate::ocr::hooks::OcrRequestBody::$body(body)
        }
        fn body_from_hook(
            &self,
            body: $crate::ocr::hooks::OcrRequestBody,
        ) -> Result<Self::RequestBody, $crate::ocr::error::OcrRequestError> {
            match body {
                $crate::ocr::hooks::OcrRequestBody::$body(body) => Ok(body),
                _ => Err($crate::ocr::error::OcrRequestError::RequestField {
                    path: "guardrail.body".into(),
                }),
            }
        }
        fn provider_name(&self) -> &'static str {
            $crate::ocr::prepare::OcrProviderKind::$provider.provider_name()
        }
    };
}

pub(crate) fn request_error(path: &str) -> Error {
    OcrRequestError::RequestField { path: path.into() }.into()
}
