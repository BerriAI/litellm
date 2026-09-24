#![allow(clippy::disallowed_types)]

// mirrors: test_litellm/litellm_core_utils/llm_cost_calc/test_llm_cost_calc_utils.py::test_is_within_off_peak_window_same_day

use jiff::Timestamp;
use litellm_cost::base_rate_selection::get_token_base_cost;
use litellm_cost::off_peak::{
    TokenRates, apply_off_peak_pricing, is_off_peak, is_within_off_peak_window,
    open_off_peak_block, parse_off_peak_rate,
};
use rstest::rstest;
use serde_json::{Value, json};

fn instant(value: &str) -> Timestamp {
    value.parse().unwrap()
}

#[rstest]
#[case(json!("09:00-17:00"), "2026-01-01T09:00Z", true)]
#[case(json!("09:00-17:00"), "2026-01-01T17:00Z", false)]
#[case(json!("16:30-00:30"), "2026-01-01T00:15Z", true)]
#[case(json!("16:30-00:30"), "2026-01-01T00:30Z", false)]
#[case(json!("00:00-00:00"), "2026-01-01T12:00Z", true)]
#[case(json!(["bad", "13:00-16:00"]), "2026-01-01T14:00Z", true)]
#[case(json!("25:00-26:00"), "2026-01-01T18:00Z", false)]
#[case(json!("01:00-05:00"), "2026-01-01T09:00+08:00", true)]
fn is_within_off_peak_window_handles_boundaries_wrap_and_aware_time(
    #[case] windows: Value,
    #[case] at: &str,
    #[case] expected: bool,
) {
    assert_eq!(is_within_off_peak_window(&windows, instant(at)), expected);
}

#[rstest]
#[case("2026-08-24T01:30Z", false)]
#[case("2026-08-26T05:00Z", true)]
#[case("2026-08-29T02:00Z", true)]
#[case("2026-08-30T08:00Z", true)]
fn is_off_peak_applies_weekday_qualified_schedule(#[case] at: &str, #[case] expected: bool) {
    let block = json!({"windows": [
        {"hours_utc": ["00:00-01:00", "04:00-06:00", "10:00-00:00"], "weekdays": [1, 2, 3, 4, 5]},
        {"hours_utc": "00:00-00:00", "weekdays": [6, 7]}
    ]});
    assert_eq!(is_off_peak(&block, instant(at)), expected);
}

#[rstest]
#[case("2026-08-28T16:30Z", true)]
#[case("2026-08-29T16:30Z", false)]
fn is_off_peak_uses_configured_weekday_timezone(#[case] at: &str, #[case] expected: bool) {
    let block = json!({
        "weekday_timezone": "Asia/Shanghai",
        "windows": [{"hours_utc": "16:00-17:00", "weekdays": [6]}]
    });
    assert_eq!(is_off_peak(&block, instant(at)), expected);
}

#[rstest]
#[case(json!({"windows": [{"hours_utc": "00:00-00:00", "weekdays": ["Sat", "sunday"]}]}), "2026-08-29T12:00Z", true)]
#[case(json!({"weekday_timezone": "Not/AZone", "windows": [{"hours_utc": "16:00-17:00", "weekdays": [5]}]}), "2026-08-28T16:30Z", true)]
#[case(json!({"windows": [{"hours_utc": "00:00-00:00", "weekdays": [0, 8, "noday", true]}]}), "2026-08-29T12:00Z", false)]
#[case(json!({"windows": "00:00-00:00"}), "2026-08-29T12:00Z", false)]
#[case(json!({"hours_utc": "04:00-06:00", "windows": [{"hours_utc": "00:00-00:00", "weekdays": [7]}]}), "2026-08-28T05:00Z", true)]
fn is_off_peak_handles_names_invalid_rules_and_flat_union(
    #[case] block: Value,
    #[case] at: &str,
    #[case] expected: bool,
) {
    assert_eq!(is_off_peak(&block, instant(at)), expected);
}

#[rstest]
fn open_off_peak_block_requires_mapping_and_matching_window() {
    let at = instant("2026-01-01T18:00Z");
    let valid =
        json!({"off_peak_pricing": {"hours_utc": "16:30-00:30", "input_cost_per_token": 5e-7}});
    assert!(open_off_peak_block(&valid, at).is_some());
    for malformed in [json!("16:00-19:00"), json!(["16:00-19:00"]), json!(true)] {
        assert_eq!(
            open_off_peak_block(&json!({"off_peak_pricing": malformed}), at),
            None
        );
    }
}

#[rstest]
#[case(json!("5e-7"), Some(5e-7))]
#[case(json!(0), Some(0.0))]
#[case(json!(true), None)]
#[case(json!("bad"), None)]
#[case(json!(" 5e-7 "), Some(5e-7))]
#[case(json!(null), None)]
#[case(json!([1]), None)]
fn parse_off_peak_rate_accepts_numeric_strings_and_rejects_bools(
    #[case] raw: Value,
    #[case] expected: Option<f64>,
) {
    assert_eq!(parse_off_peak_rate(Some(&raw)), expected);
}

#[rstest]
#[case("2026-01-01T18:00Z", true)]
#[case("2026-01-01T12:00Z", false)]
fn apply_off_peak_pricing_preserves_unset_rates_and_replaces_reasoning(
    #[case] at: &str,
    #[case] inside: bool,
) {
    let model_info = json!({"off_peak_pricing": {
        "hours_utc": "16:30-00:30",
        "input_cost_per_token": "5e-7",
        "cache_creation_input_token_cost": true,
        "output_cost_per_reasoning_token": "4e-7"
    }});
    let standard = TokenRates {
        input_rate: 1e-6,
        output_rate: 2e-6,
        cache_read_rate: 1e-7,
        cache_creation_rate: 1.25e-6,
        reasoning_rate: Some(4e-6),
    };
    let rates = apply_off_peak_pricing(&model_info, instant(at), standard);
    assert_eq!(rates.input_rate, if inside { 5e-7 } else { 1e-6 });
    assert_eq!(rates.output_rate, standard.output_rate);
    assert_eq!(rates.cache_read_rate, standard.cache_read_rate);
    assert_eq!(rates.cache_creation_rate, standard.cache_creation_rate);
    assert_eq!(rates.reasoning_rate, Some(if inside { 4e-7 } else { 4e-6 }));
    assert_eq!(
        rates.billed_reasoning_rate(),
        if inside { 4e-7 } else { 4e-6 }
    );
}

#[rstest]
fn apply_off_peak_pricing_uses_output_rate_for_reasoning_without_dedicated_rate() {
    let model_info = json!({"off_peak_pricing": {
        "hours_utc": "00:00-00:00",
        "output_cost_per_token": 7e-7
    }});
    let standard = TokenRates {
        input_rate: 1e-6,
        output_rate: 2e-6,
        cache_read_rate: 1e-7,
        cache_creation_rate: 1.25e-6,
        reasoning_rate: None,
    };
    let rates = apply_off_peak_pricing(&model_info, instant("2026-01-01T12:00Z"), standard);
    assert_eq!(rates.reasoning_rate, None);
    assert_eq!(rates.billed_reasoning_rate(), 7e-7);
}

#[rstest]
fn get_token_base_cost_applies_off_peak_rates_over_threshold_rates() {
    let model_info = json!({
        "input_cost_per_token": 1e-6,
        "output_cost_per_token": 2e-6,
        "input_cost_per_token_above_200k_tokens": 3e-6,
        "output_cost_per_token_above_200k_tokens": 4e-6,
        "off_peak_pricing": {
            "hours_utc": "16:30-00:30",
            "input_cost_per_token": 5e-7,
            "output_cost_per_token": 1e-6
        }
    });
    let off_peak = get_token_base_cost(
        &model_info,
        250_000,
        None,
        false,
        instant("2026-01-01T18:00Z"),
    );
    let peak = get_token_base_cost(
        &model_info,
        250_000,
        None,
        false,
        instant("2026-01-01T12:00Z"),
    );
    assert_eq!(off_peak.input, 5e-7);
    assert_eq!(off_peak.output, 1e-6);
    assert_eq!(peak.input, 3e-6);
    assert_eq!(peak.output, 4e-6);
}

#[rstest]
fn get_token_base_cost_uses_off_peak_input_for_missing_cache_rates_and_one_hour() {
    let model_info = json!({
        "input_cost_per_token": 1e-6,
        "output_cost_per_token": 2e-6,
        "off_peak_pricing": {"hours_utc": "00:00-00:00", "input_cost_per_token": 5e-7}
    });
    let rates = get_token_base_cost(&model_info, 100, None, false, instant("2026-01-01T18:00Z"));
    assert_eq!(rates.input, 5e-7);
    assert_eq!(rates.cache_read, 5e-7);
    assert_eq!(rates.cache_creation, 5e-7);
    assert_eq!(rates.cache_creation_above_1hr, 5e-7);
}

#[rstest]
fn get_token_base_cost_keeps_explicit_one_hour_rate_during_off_peak() {
    let model_info = json!({
        "input_cost_per_token": 1e-6,
        "output_cost_per_token": 2e-6,
        "cache_creation_input_token_cost_above_1hr": 6e-6,
        "off_peak_pricing": {
            "hours_utc": "00:00-00:00",
            "cache_creation_input_token_cost": 5e-7
        }
    });
    let rates = get_token_base_cost(&model_info, 100, None, false, instant("2026-01-01T18:00Z"));
    assert_eq!(rates.cache_creation, 5e-7);
    assert_eq!(rates.cache_creation_above_1hr, 6e-6);
}
