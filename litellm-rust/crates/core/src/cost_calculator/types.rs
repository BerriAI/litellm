use serde::Deserialize;
use std::collections::HashMap;

#[derive(Debug, Clone, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    #[serde(default)]
    pub total_tokens: u64,
    #[serde(default)]
    pub prompt_tokens_details: Option<PromptTokensDetails>,
    #[serde(default)]
    pub completion_tokens_details: Option<CompletionTokensDetails>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct PromptTokensDetails {
    #[serde(default)]
    pub cached_tokens: u64,
    #[serde(default)]
    pub cache_hit_tokens: u64,
    #[serde(default)]
    pub cache_creation_tokens: u64,
    #[serde(default)]
    pub text_tokens: u64,
    #[serde(default)]
    pub audio_tokens: u64,
    #[serde(default)]
    pub image_tokens: u64,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct CompletionTokensDetails {
    #[serde(default)]
    pub text_tokens: u64,
    #[serde(default)]
    pub audio_tokens: u64,
    #[serde(default)]
    pub reasoning_tokens: u64,
}

/// Zero-alloc cost request. All string fields are borrowed.
pub struct CostRequest<'a> {
    pub model: &'a str,
    pub usage: Usage,
    pub custom_llm_provider: Option<&'a str>,
    pub service_tier: Option<&'a str>,
}

#[derive(Debug, Clone)]
pub struct CostResponse {
    pub prompt_cost_usd: f64,
    pub completion_cost_usd: f64,
}

impl CostResponse {
    pub fn total_cost_usd(&self) -> f64 {
        self.prompt_cost_usd + self.completion_cost_usd
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelPricing {
    #[serde(default)]
    pub litellm_provider: Option<String>,
    #[serde(default)]
    pub input_cost_per_token: Option<f64>,
    #[serde(default)]
    pub output_cost_per_token: Option<f64>,
    #[serde(default, alias = "cache_read_input_token_cost")]
    pub cache_read_input_tokens_cost: Option<f64>,
    #[serde(default)]
    pub cache_creation_input_token_cost: Option<f64>,
    #[serde(default)]
    pub cache_creation_input_token_cost_above_1hr: Option<f64>,
    #[serde(default)]
    pub input_cost_per_token_priority: Option<f64>,
    #[serde(default)]
    pub output_cost_per_token_priority: Option<f64>,
    #[serde(default)]
    pub input_cost_per_token_flex: Option<f64>,
    #[serde(default)]
    pub output_cost_per_token_flex: Option<f64>,
    #[serde(default)]
    pub input_cost_per_token_above_128k_tokens: Option<f64>,
    #[serde(default)]
    pub output_cost_per_token_above_128k_tokens: Option<f64>,
    #[serde(default)]
    pub input_cost_per_token_above_200k_tokens: Option<f64>,
    #[serde(default)]
    pub output_cost_per_token_above_200k_tokens: Option<f64>,
    #[serde(default)]
    pub cache_creation_input_token_cost_above_200k_tokens: Option<f64>,
    #[serde(default, alias = "cache_read_input_token_cost_above_200k_tokens")]
    pub cache_read_input_tokens_cost_above_200k_tokens: Option<f64>,
    #[serde(default)]
    pub output_cost_per_reasoning_token: Option<f64>,
    #[serde(default)]
    pub input_cost_per_audio_token: Option<f64>,
    #[serde(default)]
    pub output_cost_per_audio_token: Option<f64>,
    #[serde(default)]
    pub input_cost_per_character: Option<f64>,
    #[serde(default)]
    pub output_cost_per_character: Option<f64>,
}

pub type PricingDatabase = HashMap<String, ModelPricing>;
