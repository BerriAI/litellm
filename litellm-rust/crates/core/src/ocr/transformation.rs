use crate::Error;
use crate::providers::auth::CredentialPlacement;
use crate::providers::auth::http::apply_credential;
use serde_json::{Map, Value};

use std::future::Future;
use std::pin::Pin;

use super::types::{OcrAuthInputs, OcrRequestData, OcrResponseData};

pub type OcrAuthFuture<'a> =
    Pin<Box<dyn Future<Output = Result<Vec<(String, String)>, Error>> + Send + 'a>>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OcrResponseHandling {
    Json,
    AzureDocumentIntelligencePoll,
}

pub trait OcrProviderConfig: Sync {
    fn supported_ocr_params(&self) -> &'static [&'static str];

    #[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
    fn map_ocr_params(&self, non_default_params: &Map<String, Value>) -> Map<String, Value> {
        let mut mapped_params = Map::new();
        for (param, value) in non_default_params {
            if self.supported_ocr_params().contains(&param.as_str()) {
                mapped_params.insert(param.clone(), value.clone());
            }
        }
        mapped_params
    }

    fn transform_ocr_request(
        &self,
        model: &str,
        document: Value,
        optional_params: Map<String, Value>,
    ) -> Result<OcrRequestData, Error>;

    fn transform_ocr_response(
        &self,
        model: &str,
        response_json: Value,
    ) -> Result<OcrResponseData, Error>;

    fn transform_ocr_response_with_params(
        &self,
        model: &str,
        response_json: Value,
        _optional_params: &Map<String, Value>,
    ) -> Result<OcrResponseData, Error> {
        self.transform_ocr_response(model, response_json)
    }

    fn complete_url(
        &self,
        api_base: Option<&str>,
        model: &str,
        optional_params: &Map<String, Value>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, Error>;

    fn resolve_api_key(
        &self,
        api_key: Option<&str>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, Error>;

    #[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
    fn validate_environment(
        &self,
        headers: Vec<(String, String)>,
        api_key: Option<&str>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<Vec<(String, String)>, Error> {
        let placement = self.credential_placement();
        if crate::http_utils::has_header(&headers, placement.header_name()) {
            return Ok(headers);
        }
        let api_key = self.resolve_api_key(api_key, env_lookup)?;
        apply_credential(headers, &api_key, placement).map_err(Error::from)
    }

    fn authenticate<'a>(
        &'a self,
        headers: Vec<(String, String)>,
        api_key: Option<&'a str>,
        auth_inputs: &'a OcrAuthInputs,
        env_lookup: &'a (dyn Fn(&str) -> Option<String> + Sync),
    ) -> OcrAuthFuture<'a> {
        Box::pin(async move {
            let _ = auth_inputs;
            self.validate_environment(headers, api_key, env_lookup)
        })
    }

    fn credential_placement(&self) -> CredentialPlacement {
        CredentialPlacement::Bearer
    }

    fn requires_data_uri_document(&self) -> bool {
        false
    }

    fn response_handling(&self) -> OcrResponseHandling {
        OcrResponseHandling::Json
    }
}
