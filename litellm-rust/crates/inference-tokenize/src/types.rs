use std::{collections::BTreeMap, time::Duration};

use litellm_auth::{SecretValue, Sourced};
use litellm_llms_types::formats::tokenize::TokenizeOutput;
use litellm_token_counter::raw_text::LocalTokenizerConfig;
use serde_json::Value;
use std::sync::Arc;

#[derive(Clone, Default)]
pub enum TokenizeBackend {
    #[default]
    Provider,
    Local {
        config: Sourced<Arc<LocalTokenizerConfig>>,
    },
}

#[derive(Clone, Default)]
pub struct TokenizeRequest {
    pub model: String,
    pub prompt: String,
    pub output: TokenizeOutput,
    pub backend: TokenizeBackend,
    pub custom_llm_provider: Option<String>,
    pub api_key: Option<Sourced<SecretValue>>,
    pub api_base: Option<Sourced<String>>,
    pub extra_headers: BTreeMap<String, Sourced<Value>>,
    pub deployment_params: BTreeMap<String, Sourced<Value>>,
    pub optional_params: BTreeMap<String, Sourced<Value>>,
    pub timeout: Option<Duration>,
}
