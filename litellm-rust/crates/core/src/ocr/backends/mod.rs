use std::future::Future;

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

pub struct PreparedOcrBackend {
    pub url: String,
    pub headers: Vec<(String, String)>,
}

pub trait OcrIntegration: Send + Sync + Sized + 'static {
    type Host: OcrHost;
    type Format: OcrFormat;
    type PreparedDocument: Into<<Self::Format as OcrFormat>::PreparedDocument> + Send;
    const FORMAT: Self::Format;

    fn provider_name(&self) -> &'static str {
        Self::Host::PROVIDER.as_str()
    }

    fn format(&self) -> Self::Format {
        Self::FORMAT
    }

    fn decode_input_params(
        &self,
        params: serde_json::Map<String, serde_json::Value>,
        prefix: &str,
    ) -> Result<InputParams<Self>, super::error::OcrRequestError> {
        super::registry::validate_request_format(
            &params,
            self.supports_native_request_format(),
            self.provider_name(),
        )?;
        Self::Format::validate_input_params(&params)?;
        super::wire::decode_request_value(serde_json::Value::Object(params), prefix)
    }

    fn prepare(
        &self,
        connection: &OcrConnection,
        config: &HostConfig<Self>,
        model: &str,
        params: &MappedParams<Self>,
        env_lookup: &(dyn Fn(&str) -> Option<String> + Sync),
    ) -> impl Future<Output = Result<PreparedOcrBackend, OcrError>> + Send;

    fn prepare_document(
        &self,
        client: &OcrClient,
        document: OcrDocument,
        connection: &OcrConnection,
        headers: &[(String, String)],
    ) -> impl Future<Output = Result<Self::PreparedDocument, OcrError>> + Send;

    fn validate_request_body(
        &self,
        _body: &<Self::Format as OcrFormat>::RequestBody,
    ) -> Result<(), super::error::OcrRequestError> {
        Ok(())
    }

    fn preserve_native_response(&self, _params: &MappedParams<Self>) -> bool {
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
        params: &MappedParams<Self>,
    ) -> impl Future<
        Output = Result<DecodedOcrResponse<<Self::Format as OcrFormat>::ResponseBody>, OcrError>,
    > + Send {
        super::client::read_json_response(response, self.preserve_native_response(params))
    }

    fn supports_native_request_format(&self) -> bool {
        false
    }
}

pub trait OcrHost: Send + Sync + 'static {
    type Config: Clone + std::fmt::Debug + Send + Sync + 'static;
    const PROVIDER: super::registry::OcrProvider;
}

pub type InputParams<I> = <<I as OcrIntegration>::Format as OcrFormat>::InputParams;
pub type MappedParams<I> = <<I as OcrIntegration>::Format as OcrFormat>::MappedParams;
pub type HostConfig<I> = <<I as OcrIntegration>::Host as OcrHost>::Config;
