use std::future::Future;

use crate::auth::AuthError;

use super::OcrClient;
use super::error::OcrError;
use super::formats::OcrFormat;
use super::types::{OcrConnection, OcrDocument};
use super::wire::DecodedOcrResponse;

pub(crate) mod azure_ai;
mod azure_document_intelligence;
pub(crate) mod mistral;
pub(crate) mod reducto;
pub(crate) mod vertex_ai;

pub trait OcrBackend<F: OcrFormat>: Send + Sync + Sized + 'static {
    fn complete_url(
        &self,
        connection: &OcrConnection,
        model: &str,
        params: &F::MappedParams,
    ) -> Result<String, OcrError>;

    fn authenticate(
        &self,
        connection: &OcrConnection,
    ) -> impl Future<Output = Result<Vec<(String, String)>, AuthError>> + Send;

    fn prepare_document(
        &self,
        client: &OcrClient,
        document: OcrDocument,
        connection: &OcrConnection,
        headers: &[(String, String)],
    ) -> impl Future<Output = Result<F::PreparedDocument, OcrError>> + Send;

    fn preserve_native_response(&self, _params: &F::MappedParams) -> bool {
        false
    }

    fn guard_document_before_preparation(&self) -> bool {
        false
    }

    fn read_response(
        &self,
        _client: &OcrClient,
        response: reqwest::Response,
        _url: &str,
        _headers: &[(String, String)],
        _connection: &OcrConnection,
        params: &F::MappedParams,
    ) -> impl Future<Output = Result<DecodedOcrResponse<F::ResponseBody>, OcrError>> + Send {
        super::wire::read_json_response(response, self.preserve_native_response(params))
    }
}
