use std::collections::BTreeMap;

use crate::base_llm::translation::TranslationError;

pub enum ProviderBaseClass {
    OpenAiGpt,
    OpenAiLikeChat,
}

pub struct ProviderConstraints {
    pub temperature_min: Option<f64>,
    pub temperature_max: Option<f64>,
    pub temperature_min_with_n_gt_1: Option<f64>,
}

pub struct ProviderSpecialHandling {
    pub convert_content_list_to_string: bool,
    pub force_store_false: bool,
}

pub struct SimpleProviderConfig {
    pub slug: String,
    pub base_url: Option<String>,
    pub api_key_env: String,
    pub api_base_env: Option<String>,
    pub base_class: ProviderBaseClass,
    pub param_mappings: BTreeMap<String, String>,
    pub constraints: ProviderConstraints,
    pub special_handling: ProviderSpecialHandling,
    pub supported_endpoints: Box<[String]>,
}

pub struct JsonProviderRegistry {
    _providers: BTreeMap<String, SimpleProviderConfig>,
}

impl JsonProviderRegistry {
    pub fn load(_definitions: &[u8]) -> Result<Self, TranslationError> {
        todo!()
    }

    pub fn get(&self, _slug: &str) -> Option<&SimpleProviderConfig> {
        todo!()
    }

    pub fn exists(&self, _slug: &str) -> bool {
        todo!()
    }

    pub fn get_by_base_url(&self, _base_url: &str) -> Option<&SimpleProviderConfig> {
        todo!()
    }

    pub fn supports_responses_api(&self, _slug: &str) -> bool {
        todo!()
    }

    pub fn list_providers(&self) -> Box<[&str]> {
        todo!()
    }
}
