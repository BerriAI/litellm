use serde_json::{Map, Value};

use crate::CoreResult;

use super::types::{OcrRequestData, OcrResponseData};

pub const OCR_PUBLIC_PARAMS_RESERVED_BY_LITELLM: &[&str] = &["id"];

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OcrDocumentPreparation {
    None,
    DataUri,
    ReductoUpload,
}

pub trait OcrProviderConfig: Sync {
    fn supported_ocr_params(&self) -> &'static [&'static str];

    fn map_ocr_params(&self, non_default_params: &Map<String, Value>) -> Map<String, Value> {
        non_default_params
            .iter()
            .filter(|(param, value)| {
                self.supported_ocr_params().contains(&param.as_str())
                    && !(value.is_null()
                        && OCR_PUBLIC_PARAMS_RESERVED_BY_LITELLM.contains(&param.as_str()))
            })
            .map(|(param, value)| (param.clone(), value.clone()))
            .collect()
    }

    fn transform_ocr_request(
        &self,
        model: &str,
        document: Value,
        optional_params: Map<String, Value>,
    ) -> CoreResult<OcrRequestData>;

    fn transform_ocr_response(
        &self,
        model: &str,
        response_json: Value,
    ) -> CoreResult<OcrResponseData>;

    fn complete_url(
        &self,
        api_base: Option<&str>,
        model: &str,
        optional_params: &Map<String, Value>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> CoreResult<String>;

    fn resolve_api_key(
        &self,
        api_key: Option<&str>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> CoreResult<String>;

    fn auth_strategy(&self) -> OcrAuthStrategy {
        OcrAuthStrategy::Bearer
    }

    fn requires_data_uri_document(&self) -> bool {
        false
    }

    fn document_preparation(&self) -> OcrDocumentPreparation {
        if self.requires_data_uri_document() {
            OcrDocumentPreparation::DataUri
        } else {
            OcrDocumentPreparation::None
        }
    }

    fn response_handling(&self) -> OcrResponseHandling {
        OcrResponseHandling::Json
    }
}
