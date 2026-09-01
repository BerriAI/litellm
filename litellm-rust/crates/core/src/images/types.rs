use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use super::transformation::ImagesProviderConfig;

pub struct ImagesGenerationRequest<'a> {
    pub model: &'a str,
    pub prompt: String,
    pub n: Option<u32>,
    pub size: Option<String>,
    pub response_format: Option<String>,
    pub user: Option<String>,
    pub api_key: Option<&'a str>,
    pub api_base: Option<&'a str>,
    pub custom_llm_provider: Option<&'a str>,
    pub extra_headers: Option<Map<String, Value>>,
    pub timeout: Option<Duration>,
}

pub struct ImagesEditRequest<'a> {
    pub model: &'a str,
    pub image: Vec<u8>,
    pub mask: Option<Vec<u8>>,
    pub prompt: String,
    pub n: Option<u32>,
    pub size: Option<String>,
    pub response_format: Option<String>,
    pub user: Option<String>,
    pub api_key: Option<&'a str>,
    pub api_base: Option<&'a str>,
    pub custom_llm_provider: Option<&'a str>,
    pub extra_headers: Option<Map<String, Value>>,
    pub timeout: Option<Duration>,
}

pub(super) struct ProviderImagesGenerationRequest {
    pub(super) _provider: String,
    pub(super) model: String,
    pub(super) config: &'static dyn ImagesProviderConfig,
    pub(super) url: String,
    pub(super) body: Value,
    pub(super) upstream_headers: Vec<(String, String)>,
    pub(super) timeout: Option<Duration>,
}

pub(super) struct ProviderImagesEditRequest {
    pub(super) _provider: String,
    pub(super) model: String,
    pub(super) config: &'static dyn ImagesProviderConfig,
    pub(super) url: String,
    pub(super) body: Value,
    pub(super) image: Vec<u8>,
    pub(super) mask: Option<Vec<u8>>,
    pub(super) upstream_headers: Vec<(String, String)>,
    pub(super) timeout: Option<Duration>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ImagesData {
    pub url: Option<String>,
    pub b64_json: Option<String>,
    pub revised_prompt: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ImagesGenerationResponse {
    pub created: u64,
    pub data: Vec<ImagesData>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ImagesEditResponse {
    pub created: u64,
    pub data: Vec<ImagesData>,
}
