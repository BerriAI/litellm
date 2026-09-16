use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::{BaseCache, CacheKwargs, Error};

pub use crate::BaseCache as Cache;

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
pub enum CacheMode {
    #[default]
    #[serde(rename = "default_on")]
    DefaultOn,
    #[serde(rename = "default_off")]
    DefaultOff,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct CacheKeyField {
    pub name: String,
    pub value: Option<String>,
    pub api_parameter: bool,
    pub internal_parameter: bool,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct CacheKeyInput {
    pub fields: Vec<CacheKeyField>,
    pub preset: Option<String>,
    pub namespace: Option<String>,
    pub include_provider_parameters: bool,
}

#[derive(Default)]
pub struct CacheKeyContext {
    pub model_group: Option<String>,
    pub caching_groups: Vec<(Vec<String>, String)>,
    pub file_checksum: Option<String>,
    pub file_object_name: Option<String>,
    pub metadata_file_name: Option<String>,
    pub parameters_file_name: Option<String>,
}

impl CacheKeyContext {
    pub fn apply(self, input: &mut CacheKeyInput) {
        let group = self.model_group.as_ref().and_then(|model| {
            self.caching_groups
                .iter()
                .find(|(models, _)| models.contains(model))
        });
        for field in &mut input.fields {
            match field.name.as_str() {
                "model" => {
                    field.value = group
                        .map(|(_, formatted)| formatted.clone())
                        .or_else(|| self.model_group.clone())
                        .or_else(|| field.value.take())
                }
                "file" => {
                    field.value = self
                        .file_checksum
                        .clone()
                        .or_else(|| self.file_object_name.clone())
                        .or_else(|| self.metadata_file_name.clone())
                        .or_else(|| self.parameters_file_name.clone())
                }
                _ => {}
            }
        }
    }
}

pub fn get_cache_key(input: &CacheKeyInput) -> String {
    cache_key(input)
}

pub fn cache_key(input: &CacheKeyInput) -> String {
    if let Some(preset) = &input.preset {
        return preset.clone();
    }
    let mut digest = Sha256::new();
    for field in &input.fields {
        if (field.api_parameter || (input.include_provider_parameters && !field.internal_parameter))
            && let Some(value) = &field.value
        {
            digest.update(field.name.as_bytes());
            digest.update(b": ");
            digest.update(value.as_bytes());
        }
    }
    let hash = format!("{:x}", digest.finalize());
    input
        .namespace
        .as_deref()
        .filter(|namespace| !namespace.is_empty())
        .map_or(hash.clone(), |namespace| format!("{namespace}:{hash}"))
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize)]
pub struct CacheControls {
    pub supported_call_type: bool,
    pub configured: bool,
    pub native_backend: bool,
    pub default_on: bool,
    pub caching: Option<bool>,
    pub no_cache: bool,
    pub no_store: bool,
    #[serde(default)]
    pub use_cache: bool,
}

impl CacheControls {
    pub fn reads(self) -> bool {
        self.supported_call_type
            && self.configured
            && self.caching.unwrap_or(true)
            && !self.no_cache
            && (self.default_on || self.use_cache)
    }

    pub fn writes(self) -> bool {
        self.supported_call_type
            && self.configured
            && !self.no_store
            && (self.default_on || self.use_cache)
    }
}

pub fn should_use_cache(controls: CacheControls) -> bool {
    controls.reads() || controls.writes()
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CacheEntry {
    pub timestamp: f64,
    pub response: Value,
}

impl CacheEntry {
    pub fn fresh(&self, now: Duration, max_age: Option<Duration>) -> bool {
        self.timestamp.is_finite()
            && max_age.is_none_or(|age| now.as_secs_f64() - self.timestamp <= age.as_secs_f64())
    }
}

pub fn get_cache(
    cache: &dyn BaseCache<Value = CacheEntry>,
    key: &str,
    kwargs: &CacheKwargs,
) -> Result<Option<CacheEntry>, Error> {
    cache.get_cache(key, kwargs)
}

pub fn set_cache(
    cache: &dyn BaseCache<Value = CacheEntry>,
    key: &str,
    entry: CacheEntry,
    kwargs: CacheKwargs,
) -> Result<(), Error> {
    cache.set_cache(key, entry, kwargs)
}

pub type CacheBackend = Arc<dyn BaseCache<Value = CacheEntry>>;
