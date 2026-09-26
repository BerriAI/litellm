#![allow(clippy::disallowed_types)]

// mirrors: test_litellm/litellm_core_utils/llm_cost_calc/test_llm_cost_calc_utils.py::test_get_token_base_cost_picks_highest_crossed_tier

use litellm_cost::base_rate_selection::{
    TokenBaseRates, get_tiered_reasoning_rate, get_token_base_cost_without_off_peak,
    uses_inclusive_token_thresholds,
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
fn threshold_detection_excludes_auto_and_keeps_plain_thresholds() {
    let model_info = json!({
        "input_cost_per_token": 2e-6,
        "output_cost_per_token": 4e-6,
        "input_cost_per_token_above_128k_tokens": 5e-6,
        "input_cost_per_token_above_128k_tokens_auto": 9e-6,
        "output_cost_per_token_above_128k_tokens": 7e-6
    });
    let rates = get_token_base_cost_without_off_peak(&model_info, 128_001, Some("auto"), false);
    assert_eq!(
        rates.input, 5e-6,
        "_auto-suffixed threshold keys are non-standard; the plain threshold rate applies"
    );
}

#[rstest]
fn non_standard_threshold_suffixes_match_python_service_tier_suffixes() {
    let mut python_suffixes: Vec<String> = litellm_cost::pricing::ServiceTier::SUFFIXES
        .into_iter()
        .map(|tier| format!("_{}", tier.as_str()))
        .collect();
    python_suffixes.push("_batches".to_string());
    assert_eq!(
        litellm_cost::base_rate_selection::NON_STANDARD_THRESHOLD_SUFFIXES.to_vec(),
        python_suffixes
    );
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

#[rstest]
fn inclusive_threshold_providers_stay_pinned_to_the_python_set() {
    assert!(uses_inclusive_token_thresholds(Some("xai")));
    for provider in [
        Some("openai"),
        Some("anthropic"),
        Some("bedrock"),
        Some("perplexity"),
        Some("dashscope"),
        None,
    ] {
        assert!(!uses_inclusive_token_thresholds(provider));
    }
}

#[rstest]
#[case::threshold_key_without_tokens_suffix(json!({"input_cost_per_token_above_50": 3e-6}), None, 3e-6, 4e-6)]
#[case::tiered_request_builds_the_tokens_key(json!({"input_cost_per_token_above_50": 3e-6, "input_cost_per_token_above_50_tokens_priority": 5e-6}), Some("priority"), 5e-6, 4e-6)]
#[case::unparseable_threshold_is_ignored(json!({"input_cost_per_token_above_abc_tokens": 3e-6}), None, 2e-6, 4e-6)]
fn get_token_base_cost_reads_threshold_keys_like_python_split(
    #[case] extra: Value,
    #[case] service_tier: Option<&str>,
    #[case] expected_input: f64,
    #[case] expected_output: f64,
) {
    let model_info = Value::Object(
        json!({"input_cost_per_token": 2e-6, "output_cost_per_token": 4e-6})
            .as_object()
            .unwrap()
            .clone()
            .into_iter()
            .chain(extra.as_object().unwrap().clone())
            .collect(),
    );
    let rates = get_token_base_cost_without_off_peak(&model_info, 100, service_tier, false);
    assert_eq!(
        (rates.input, rates.output),
        (expected_input, expected_output)
    );
}
