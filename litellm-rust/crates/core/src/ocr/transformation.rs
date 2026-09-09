use std::future::Future;
use std::pin::Pin;

use reqwest::header::HeaderMap;
use serde_json::{Map, Value};

use crate::auth::CredentialPlacement;
use crate::auth::http::apply_credential;
use crate::{AuthError, Error};

use super::types::{OcrAuthInputs, OcrRequestData, OcrResponseData};

pub type OcrAuthFuture<'a> = Pin<Box<dyn Future<Output = Result<HeaderMap, Error>> + Send + 'a>>;
pub type OcrRequestSetupFuture<'a> =
    Pin<Box<dyn Future<Output = Result<(String, HeaderMap), Error>> + Send + 'a>>;

pub struct OcrRequestSetup<'a> {
    pub api_base: Option<&'a str>,
    pub model: &'a str,
    pub optional_params: &'a Map<String, Value>,
    pub headers: HeaderMap,
    pub api_key: Option<&'a str>,
    pub auth_inputs: &'a OcrAuthInputs,
    pub env_lookup: &'a (dyn Fn(&str) -> Option<String> + Sync),
}

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

    fn parse_auth_inputs(
        &self,
        _provider_params: &Map<String, Value>,
    ) -> Result<OcrAuthInputs, AuthError> {
        Ok(OcrAuthInputs::None)
    }

    #[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
    fn validate_environment(
        &self,
        headers: HeaderMap,
        api_key: Option<&str>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<HeaderMap, Error> {
        let placement = self.credential_placement();
        if headers.contains_key(placement.header_name()) {
            return Ok(headers);
        }
        let api_key = self.resolve_api_key(api_key, env_lookup)?;
        apply_credential(headers, &api_key, placement).map_err(Error::from)
    }

    fn authenticate<'a>(
        &'a self,
        headers: HeaderMap,
        api_key: Option<&'a str>,
        auth_inputs: &'a OcrAuthInputs,
        env_lookup: &'a (dyn Fn(&str) -> Option<String> + Sync),
    ) -> OcrAuthFuture<'a> {
        Box::pin(async move {
            let _ = auth_inputs;
            self.validate_environment(headers, api_key, env_lookup)
        })
    }

    fn prepare_url_and_headers<'a>(
        &'a self,
        request: OcrRequestSetup<'a>,
    ) -> OcrRequestSetupFuture<'a> {
        Box::pin(async move {
            let url = self.complete_url(
                request.api_base,
                request.model,
                request.optional_params,
                request.env_lookup,
            )?;
            let headers = self
                .authenticate(
                    request.headers,
                    request.api_key,
                    request.auth_inputs,
                    request.env_lookup,
                )
                .await?;
            Ok((url, headers))
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
