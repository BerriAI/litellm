use litellm_cost::tiered_pricing::{select_tier_for_input, tier_rate};
use rstest::rstest;
use serde_json::{Value, json};

#[rstest]
#[case(0, None)]
#[case(32_000, Some(0))]
#[case(32_001, Some(1))]
#[case(128_000, Some(1))]
#[case(128_001, Some(1))]
fn select_tier_for_input_uses_inclusive_upper_bound_and_last_tier_fallback(
    #[case] tokens: i64,
    #[case] expected: Option<u64>,
) {
    let tiers = vec![
        json!({"range": [32_000, 128_000], "tier": 1}),
        json!({"range": [0, 32_000], "tier": 0}),
    ];
    assert_eq!(
        select_tier_for_input(&tiers, tokens).and_then(|tier| tier["tier"].as_u64()),
        expected
    );
}

#[rstest]
fn select_tier_for_input_skips_invalid_ranges_and_chooses_first_matching_tier() {
    let tiers = vec![
        json!({"range": [0], "tier": 9}),
        json!({"range": [0, 100], "tier": 1}),
        json!({"range": [10, 200], "tier": 2}),
    ];
    assert_eq!(select_tier_for_input(&tiers, 50).unwrap()["tier"], 1);
    assert_eq!(select_tier_for_input(&tiers, 500).unwrap()["tier"], 2);
    assert_eq!(select_tier_for_input(&[json!({"range": [0]})], 10), None);
}

#[rstest]
#[case(json!({"input_cost_per_token": "4e-07"}), "input_cost_per_token", None, 4e-7)]
#[case(json!({"cache_read_input_token_cost": 0, "input_cost_per_token": "4e-07"}), "cache_read_input_token_cost", Some("input_cost_per_token"), 0.0)]
#[case(json!({"input_cost_per_token": "4e-07"}), "cache_read_input_token_cost", Some("input_cost_per_token"), 4e-7)]
#[case(json!({"input_cost_per_token": "bad"}), "input_cost_per_token", None, 0.0)]
fn tier_rate_preserves_zero_and_coerces_yaml_strings(
    #[case] tier: Value,
    #[case] key: &str,
    #[case] fallback: Option<&str>,
    #[case] expected: f64,
) {
    assert_eq!(tier_rate(&tier, key, fallback), expected);
}
