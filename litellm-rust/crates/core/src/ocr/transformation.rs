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
    fn get_supported_ocr_params(&self) -> &'static [&'static str];

    #[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
    fn map_ocr_params(&self, non_default_params: &Map<String, Value>) -> Map<String, Value> {
        let supported_params = self.get_supported_ocr_params();
        non_default_params
            .iter()
            .filter(|(param, _)| supported_params.contains(&param.as_str()))
            .map(|(param, value)| (param.clone(), value.clone()))
            .collect()
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
