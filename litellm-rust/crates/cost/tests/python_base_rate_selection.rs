use litellm_cost::base_rate_selection::{
    TokenBaseRates, get_tiered_reasoning_rate, get_token_base_cost_without_off_peak,
};
use rstest::rstest;
use serde_json::{Value, json};

#[rstest]
#[case(json!({"input_cost_per_token": 2e-6, "output_cost_per_token": 4e-6}), 2e-6, 2e-6)]
#[case(json!({"input_cost_per_token": 2e-6, "output_cost_per_token": 4e-6, "cache_read_input_token_cost": 0, "cache_creation_input_token_cost": 0}), 0.0, 0.0)]
fn get_token_base_cost_without_off_peak_falls_back_only_for_missing_cache_rates(
    #[case] model_info: Value,
    #[case] expected_read: f64,
    #[case] expected_creation: f64,
) {
    let rates = get_token_base_cost_without_off_peak(&model_info, 100, None, false);
    assert_eq!(rates.input, 2e-6);
    assert_eq!(rates.output, 4e-6);
    assert_eq!(rates.cache_read, expected_read);
    assert_eq!(rates.cache_creation, expected_creation);
    assert_eq!(rates.cache_creation_above_1hr, expected_creation);
}

#[rstest]
#[case(Some("priority"), 6e-6, 8e-6)]
#[case(Some("fast"), 6e-6, 8e-6)]
#[case(Some("flex"), 2e-6, 4e-6)]
fn get_token_base_cost_without_off_peak_selects_tier_or_standard_fallback(
    #[case] tier: Option<&str>,
    #[case] expected_input: f64,
    #[case] expected_output: f64,
) {
    let model_info = json!({
        "input_cost_per_token": 2e-6,
        "output_cost_per_token": 4e-6,
        "input_cost_per_token_priority": 6e-6,
        "output_cost_per_token_priority": 8e-6
    });
    let rates = get_token_base_cost_without_off_peak(&model_info, 100, tier, false);
    assert_eq!(rates.input, expected_input);
    assert_eq!(rates.output, expected_output);
    assert_eq!(rates.cache_read, expected_input);
}

#[rstest]
#[case(128_000, false, 2e-6)]
#[case(128_000, true, 5e-6)]
#[case(128_001, false, 5e-6)]
#[case(256_001, false, 9e-6)]
fn get_token_base_cost_without_off_peak_resolves_highest_crossed_threshold(
    #[case] prompt_tokens: u64,
    #[case] inclusive: bool,
    #[case] expected_input: f64,
) {
    let model_info = json!({
        "input_cost_per_token": 2e-6,
        "output_cost_per_token": 4e-6,
        "input_cost_per_token_above_128k_tokens": 5e-6,
        "output_cost_per_token_above_128k_tokens": 7e-6,
        "cache_read_input_token_cost_above_128k_tokens": 1e-6,
        "input_cost_per_token_above_256k_tokens": 9e-6
    });
    let rates = get_token_base_cost_without_off_peak(&model_info, prompt_tokens, None, inclusive);
    assert_eq!(rates.input, expected_input);
    assert_eq!(
        rates.output,
        if expected_input == 5e-6 { 7e-6 } else { 4e-6 }
    );
    assert_eq!(
        rates.cache_read,
        if expected_input == 5e-6 {
            1e-6
        } else {
            expected_input
        }
    );
}

#[rstest]
fn get_token_base_cost_without_off_peak_uses_tier_specific_threshold_keys() {
    let model_info = json!({
        "input_cost_per_token": 2e-6,
        "output_cost_per_token": 4e-6,
        "input_cost_per_token_above_128k_tokens": 5e-6,
        "input_cost_per_token_above_128k_tokens_priority": 6e-6,
        "output_cost_per_token_above_128k_tokens": 7e-6,
        "cache_creation_input_token_cost_above_128k_tokens": 8e-6,
        "cache_creation_input_token_cost_above_1hr_above_128k_tokens": 9e-6
    });
    let rates = get_token_base_cost_without_off_peak(&model_info, 128_001, Some("priority"), false);
    assert_eq!(rates.input, 6e-6);
    assert_eq!(rates.output, 7e-6);
    assert_eq!(rates.cache_creation, 8e-6);
    assert_eq!(rates.cache_creation_above_1hr, 9e-6);
    assert_eq!(rates.cache_read, 6e-6);
}

#[rstest]
fn get_token_base_cost_without_off_peak_uses_one_tier_for_all_token_rates() {
    let model_info = json!({
        "input_cost_per_token": 1e-6,
        "output_cost_per_token": 4e-6,
        "tiered_pricing": [
            {"range": [32_000, 128_000], "input_cost_per_token": "5e-6", "cache_read_input_token_cost": "1e-6", "output_cost_per_reasoning_token": "9e-6"},
            {"range": [0, 32_000], "input_cost_per_token": "2e-6", "output_cost_per_token": "3e-6"}
        ]
    });
    let rates = get_token_base_cost_without_off_peak(&model_info, 32_001, Some("priority"), false);
    assert_eq!(
        rates,
        TokenBaseRates {
            input: 5e-6,
            output: 4e-6,
            cache_creation: 5e-6,
            cache_creation_above_1hr: 5e-6,
            cache_read: 1e-6,
        }
    );
    assert_eq!(get_tiered_reasoning_rate(&model_info, 32_001), Some(9e-6));
    assert_eq!(get_tiered_reasoning_rate(&model_info, 32_000), Some(3e-6));
}

#[rstest]
fn get_token_base_cost_without_off_peak_uses_image_output_rate_when_token_output_is_zero() {
    let model_info = json!({
        "input_cost_per_token": 2e-6,
        "output_cost_per_token": 0,
        "output_cost_per_image_token": 8e-6
    });
    assert_eq!(
        get_token_base_cost_without_off_peak(&model_info, 100, None, false).output,
        8e-6
    );
}
