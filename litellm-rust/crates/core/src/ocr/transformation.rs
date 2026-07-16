use serde_json::{Map, Value};

use crate::CoreResult;

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

/// How the host must obtain the credential it sends upstream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OcrAuth {
    /// A provider key/token resolved from the request or environment and placed
    /// in the request header per [`OcrAuthStrategy`].
    ProviderKey,
    /// Google Vertex OAuth: the host mints/refreshes an access token from
    /// service-account JSON, ADC or `GOOGLE_APPLICATION_CREDENTIALS`, or accepts
    /// an explicitly supplied OAuth access token, and sends it as a Bearer.
    VertexOauth,
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

    /// Resolve the provider key/token for [`OcrAuth::ProviderKey`] providers.
    ///
    /// [`OcrAuth::VertexOauth`] providers never call this (the host mints the
    /// bearer), so they inherit the default that reports the wrong auth path was
    /// taken rather than carrying an unused implementation.
    fn resolve_api_key(
        &self,
        _api_key: Option<&str>,
        _env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> CoreResult<String> {
        Err(crate::error::CoreError::Auth(
            "provider does not use direct api-key auth".to_string(),
        ))
    }

    fn ocr_auth(&self) -> OcrAuth {
        OcrAuth::ProviderKey
    }

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
