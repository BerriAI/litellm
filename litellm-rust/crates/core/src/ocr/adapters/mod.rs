use std::future::Future;

use serde::{Serialize, de::DeserializeOwned};
use serde_json::{Map, Value};

use super::OcrClient;
use super::error::{OcrError, OcrRequestError, OcrResponseError};
use super::registry::OcrProvider;
use super::types::{OcrConnection, OcrDocument, OcrRequestFormat, OcrResponseData};
use super::wire::DecodedOcrResponse;
use crate::Error;

mod mistral;

pub(crate) use mistral::MistralAdapter;
#[cfg(test)]
pub(crate) use mistral::complete_url as complete_mistral_url;

pub(crate) struct PreparedAdapter {
    pub url: String,
    pub headers: Vec<(String, String)>,
}

pub(crate) trait OcrAdapter: Send + Sync + Sized + 'static {
    type Config: Clone + std::fmt::Debug + Send + Sync + 'static;
    type InputParams: std::fmt::Debug + Clone + Serialize + DeserializeOwned + Send + Sync;
    type Params: Clone + Serialize + Send + Sync;
    type RequestBody: Serialize + DeserializeOwned + Send + Sync;
    type ResponseBody: DeserializeOwned + Send;

    const PROVIDER: OcrProvider;

    fn decode_config(params: &Map<String, Value>) -> Result<Self::Config, Error>;

    fn decode_input_params(
        &self,
        params: Map<String, Value>,
        prefix: &str,
    ) -> Result<Self::InputParams, OcrRequestError> {
        validate_request_format(&params, Self::PROVIDER.as_str())?;
        super::wire::decode_request_value(Value::Object(params), prefix)
    }

    fn map_params(params: Self::InputParams) -> Result<Self::Params, OcrRequestError>;

    fn build_request(
        model: &str,
        document: OcrDocument,
        params: &Self::Params,
    ) -> Result<Self::RequestBody, OcrRequestError>;

    fn normalize_response(
        model: &str,
        response: Self::ResponseBody,
        params: &Self::Params,
    ) -> Result<OcrResponseData, OcrResponseError>;

    fn prepare(
        &self,
        connection: &OcrConnection,
        config: &Self::Config,
        model: &str,
        params: &Self::Params,
        env_lookup: &(dyn Fn(&str) -> Option<String> + Sync),
    ) -> impl Future<Output = Result<PreparedAdapter, OcrError>> + Send;

    fn validate_request_body(&self, _body: &Self::RequestBody) -> Result<(), OcrRequestError> {
        Ok(())
    }

    fn read_response(
        &self,
        _client: &OcrClient,
        response: reqwest::Response,
        _url: &str,
        _headers: &[(String, String)],
        _connection: &OcrConnection,
        _params: &Self::Params,
    ) -> impl Future<Output = Result<DecodedOcrResponse<Self::ResponseBody>, OcrError>> + Send {
        super::client::read_json_response(response, false)
    }
}

fn validate_request_format(
    params: &Map<String, Value>,
    provider: &'static str,
) -> Result<(), OcrRequestError> {
    let Some(format) = params.get("req_format") else {
        return Ok(());
    };
    let format: OcrRequestFormat =
        serde_json::from_value(format.clone()).map_err(|_| OcrRequestError::RequestFormat)?;
    if format == OcrRequestFormat::Native {
        return Err(OcrRequestError::NativeUnsupported(provider));
    }
    Ok(())
}

pub(crate) type InputParams<A> = <A as OcrAdapter>::InputParams;
pub(crate) type MappedParams<A> = <A as OcrAdapter>::Params;
pub(crate) type AdapterConfig<A> = <A as OcrAdapter>::Config;

macro_rules! for_each_ocr_adapter {
    ($callback:ident) => {
        $callback! {
            Mistral, $crate::ocr::adapters::MistralAdapter, $crate::ocr::adapters::MistralAdapter, Mistral;
        }
    };
}

pub(crate) use for_each_ocr_adapter;
