use crate::Error;
use serde_json::{Map, Value};

use super::types::{OcrRequestData, OcrResponseData};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OcrAuthStrategy {
    Bearer,
    Header(&'static str),
}

impl OcrAuthStrategy {
    pub fn header_name(self) -> &'static str {
        match self {
            Self::Bearer => "authorization",
            Self::Header(header_name) => header_name,
        }
    }
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
        let strategy = self.auth_strategy();
        if crate::http_utils::has_header(&headers, strategy.header_name()) {
            return Ok(headers);
        }
        let api_key = self.resolve_api_key(api_key, env_lookup)?;
        let auth_header = match strategy {
            OcrAuthStrategy::Bearer => ("Authorization".to_string(), format!("Bearer {api_key}")),
            OcrAuthStrategy::Header(name) => (name.to_string(), api_key),
        };
        Ok(std::iter::once(auth_header).chain(headers).collect())
    }

    fn auth_strategy(&self) -> OcrAuthStrategy {
        OcrAuthStrategy::Bearer
    }

    fn requires_data_uri_document(&self) -> bool {
        false
    }

    fn response_handling(&self) -> OcrResponseHandling {
        OcrResponseHandling::Json
    }
}
