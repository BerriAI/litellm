pub trait Tally: Clone + Default + Send + Sync + 'static {
    #[must_use]
    fn merge(self, other: Self) -> Self;
}

#[derive(Clone, Copy, Debug, Default, PartialEq, PartialOrd)]
pub struct Cost(pub f64);

impl Tally for Cost {
    fn merge(self, other: Self) -> Self {
        Self(self.0 + other.0)
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct DailyTally {
    pub model_group: Option<String>,
    pub spend: f64,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub cache_read_input_tokens: u64,
    pub cache_creation_input_tokens: u64,
    pub compression_saved_tokens: u64,
    pub compression_savings_spend: f64,
    pub prompt_caching_savings_spend: f64,
    pub gateway_injected_caching_savings_spend: f64,
    pub autorouter_savings_spend: f64,
    pub api_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub total_response_time_ms: u64,
    pub timed_requests: u64,
}

impl Tally for DailyTally {
    fn merge(self, other: Self) -> Self {
        Self {
            model_group: self.model_group.or(other.model_group),
            spend: self.spend + other.spend,
            prompt_tokens: self.prompt_tokens + other.prompt_tokens,
            completion_tokens: self.completion_tokens + other.completion_tokens,
            cache_read_input_tokens: self.cache_read_input_tokens + other.cache_read_input_tokens,
            cache_creation_input_tokens: self.cache_creation_input_tokens
                + other.cache_creation_input_tokens,
            compression_saved_tokens: self.compression_saved_tokens
                + other.compression_saved_tokens,
            compression_savings_spend: self.compression_savings_spend
                + other.compression_savings_spend,
            prompt_caching_savings_spend: self.prompt_caching_savings_spend
                + other.prompt_caching_savings_spend,
            gateway_injected_caching_savings_spend: self.gateway_injected_caching_savings_spend
                + other.gateway_injected_caching_savings_spend,
            autorouter_savings_spend: self.autorouter_savings_spend
                + other.autorouter_savings_spend,
            api_requests: self.api_requests + other.api_requests,
            successful_requests: self.successful_requests + other.successful_requests,
            failed_requests: self.failed_requests + other.failed_requests,
            total_response_time_ms: self.total_response_time_ms + other.total_response_time_ms,
            timed_requests: self.timed_requests + other.timed_requests,
        }
    }
}
