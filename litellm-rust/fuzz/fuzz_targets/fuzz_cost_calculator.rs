#![no_main]

use libfuzzer_sys::fuzz_target;
use litellm_core::cost_calculator::{self, CostRequest};
use litellm_core::cost_calculator::types::Usage;

fuzz_target!(|data: &[u8]| {
    if data.len() < 24 {
        return;
    }

    // Extract token counts from fuzz input
    let prompt_tokens = u64::from_le_bytes(data[0..8].try_into().unwrap());
    let completion_tokens = u64::from_le_bytes(data[8..16].try_into().unwrap());
    let total_tokens = prompt_tokens.saturating_add(completion_tokens);

    // Cap tokens to prevent excessive computation
    let prompt_tokens = prompt_tokens.min(1_000_000);
    let completion_tokens = completion_tokens.min(1_000_000);

    // Pick a model from a fixed set based on input bytes
    let models = ["gpt-4o", "gpt-4o-mini", "claude-3-sonnet", "claude-3-haiku", "unknown-model"];
    let model_idx = data[16] as usize % models.len();
    let model = models[model_idx];

    let cost_request = CostRequest {
        model,
        usage: Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens,
            prompt_tokens_details: None,
            completion_tokens_details: None,
        },
        custom_llm_provider: None,
        service_tier: None,
    };

    // Cost calculation should never panic
    let _ = cost_calculator::calculate_cost(&cost_request);
});
