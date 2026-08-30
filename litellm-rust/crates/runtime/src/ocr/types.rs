use std::time::Duration;

use litellm_core::call_lifecycle::{CallLifecycleContext, CallLifecycleRequest};
use litellm_core::ocr::transformation::OcrProviderTransformation;
use serde_json::{Map, Value};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OcrDeclineReason {
    NativeRequestFormat,
    UnsupportedProvider,
}

impl std::fmt::Display for OcrDeclineReason {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NativeRequestFormat => {
                formatter.write_str("native OCR response format requires Python")
            }
            Self::UnsupportedProvider => {
                formatter.write_str("OCR provider is not supported by the Rust runtime")
            }
        }
    }
}

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

pub(crate) trait OcrRuntimeConfig: Sync {
    fn transformation(&self) -> &'static dyn OcrProviderTransformation;

    fn complete_url(
        &self,
        api_base: Option<&str>,
        model: &str,
        optional_params: &Map<String, Value>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> litellm_core::CoreResult<String>;

    fn resolve_api_key(
        &self,
        api_key: Option<&str>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> litellm_core::CoreResult<String>;

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

pub struct OcrRequest<'a> {
    pub model: &'a str,
    pub document: Value,
    pub api_key: Option<&'a str>,
    pub api_base: Option<&'a str>,
    pub custom_llm_provider: Option<&'a str>,
    pub extra_headers: Option<Map<String, Value>>,
    pub optional_params: Map<String, Value>,
    pub timeout: Option<Duration>,
    pub litellm_call_id: Option<&'a str>,
}

pub struct PreparedOcrRequest {
    pub model: String,
    pub custom_llm_provider: String,
    pub litellm_call_id: String,
    pub document: Value,
    pub api_key: Option<String>,
    pub api_base: Option<String>,
    pub extra_headers: Option<Map<String, Value>>,
    pub optional_params: Map<String, Value>,
    pub timeout: Option<Duration>,
}

impl CallLifecycleRequest for PreparedOcrRequest {
    fn lifecycle_context(&self) -> CallLifecycleContext {
        CallLifecycleContext::new(
            "ocr",
            self.model.clone(),
            self.custom_llm_provider.clone(),
            self.litellm_call_id.clone(),
        )
    }
}

pub struct ProviderOcrRequest {
    pub model: String,
    pub(crate) config: &'static dyn OcrRuntimeConfig,
    pub url: String,
    pub body: Value,
    pub upstream_headers: Vec<(String, String)>,
    pub timeout: Option<Duration>,
}
