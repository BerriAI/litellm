use std::fmt;

use litellm_auth_types::SecretValue;
use serde::Deserialize;

use crate::{AdditionalFields, Flag, NumberOrString, Object};

#[derive(Clone, Deserialize)]
pub struct Model {
    pub model_name: String,
    pub litellm_params: LiteLlmParams,
    #[serde(default)]
    pub model_info: Object,
    pub blocked: Option<bool>,
    #[serde(flatten)]
    pub additional_fields: AdditionalFields,
}

impl fmt::Debug for Model {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Model")
            .field("model_name", &self.model_name)
            .field("litellm_params", &self.litellm_params)
            .field("model_info", &self.model_info)
            .field("blocked", &self.blocked)
            .field("additional_fields", &self.additional_fields.keys())
            .finish()
    }
}

#[derive(Clone, Deserialize)]
pub struct LiteLlmParams {
    pub model: String,
    pub api_key: Option<SecretValue>,
    pub api_base: Option<String>,
    pub api_version: Option<String>,
    pub custom_llm_provider: Option<String>,
    pub timeout: Option<NumberOrString>,
    pub stream_timeout: Option<NumberOrString>,
    pub max_retries: Option<NumberOrString>,
    pub tpm: Option<NumberOrString>,
    pub rpm: Option<NumberOrString>,
    pub itpm: Option<NumberOrString>,
    pub otpm: Option<NumberOrString>,
    pub max_parallel_requests: Option<u64>,
    pub organization: Option<serde_yaml_ng::Value>,
    pub drop_params: Option<Flag>,
    pub tags: Option<Box<[String]>>,
    pub tag_regex: Option<Box<[String]>>,
    pub max_budget: Option<f64>,
    pub budget_duration: Option<String>,
    pub default_api_key_tpm_limit: Option<u64>,
    pub default_api_key_rpm_limit: Option<u64>,
    pub use_in_pass_through: Option<bool>,
    pub use_chat_completions_api: Option<bool>,
    pub litellm_credential_name: Option<String>,
    pub provider_affinity_header: Option<String>,
    #[serde(flatten)]
    pub additional_fields: AdditionalFields,
}

impl LiteLlmParams {
    /// The additional fields read as a typed value, for the per-provider params this struct
    /// does not name (`aws_*`, `vertex_*`, ...). A field the type names but cannot hold is a
    /// config error, as it is for Python's `GenericLiteLLMParams`.
    pub fn additional<T: serde::de::DeserializeOwned>(&self) -> Result<T, crate::Error> {
        let mapping = self
            .additional_fields
            .iter()
            .map(|(name, value)| (serde_yaml_ng::Value::from(name.as_str()), value.clone()))
            .collect();
        Ok(serde_yaml_ng::from_value(serde_yaml_ng::Value::Mapping(
            mapping,
        ))?)
    }
}

impl fmt::Debug for LiteLlmParams {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("LiteLlmParams")
            .field("model", &self.model)
            .field("api_key", &self.api_key)
            .field("api_base", &self.api_base)
            .field("api_version", &self.api_version)
            .field("custom_llm_provider", &self.custom_llm_provider)
            .field("timeout", &self.timeout)
            .field("stream_timeout", &self.stream_timeout)
            .field("max_retries", &self.max_retries)
            .field("tpm", &self.tpm)
            .field("rpm", &self.rpm)
            .field("itpm", &self.itpm)
            .field("otpm", &self.otpm)
            .field("max_parallel_requests", &self.max_parallel_requests)
            .field("organization", &self.organization)
            .field("drop_params", &self.drop_params)
            .field("tags", &self.tags)
            .field("tag_regex", &self.tag_regex)
            .field("max_budget", &self.max_budget)
            .field("budget_duration", &self.budget_duration)
            .field("default_api_key_tpm_limit", &self.default_api_key_tpm_limit)
            .field("default_api_key_rpm_limit", &self.default_api_key_rpm_limit)
            .field("use_in_pass_through", &self.use_in_pass_through)
            .field("use_chat_completions_api", &self.use_chat_completions_api)
            .field("litellm_credential_name", &self.litellm_credential_name)
            .field("provider_affinity_header", &self.provider_affinity_header)
            .field("additional_fields", &self.additional_fields.keys())
            .finish()
    }
}
