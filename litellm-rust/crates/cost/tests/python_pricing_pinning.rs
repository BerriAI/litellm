#![allow(clippy::disallowed_types)]

// mirrors: crates/cost/tests/generate_python_fixtures.py
// (pinning + model_info_projection surfaces)

use serde_json::Value;

use litellm_cost::call_type::CallTypes;
use litellm_cost::pricing::{Metric, ServiceTier, tokenize};
use litellm_cost::provider::LlmProviders;
use litellm_cost::wire::{RawCatalogEntry, cost_per_unit};
use rstest::rstest;
use strum::VariantArray;

fn fixture() -> Value {
    serde_json::from_str(include_str!("python_fixtures.json")).expect("fixture parses")
}

fn pinning() -> Value {
    fixture()["pinning"].clone()
}

#[rstest]
fn tokenizer_agrees_with_parse_above_token_threshold_where_python_parses() {
    for row in pinning()["thresholds"].as_array().expect("threshold rows") {
        let key = row["key"].as_str().expect("key");
        if let Some(value) = row["threshold"].as_f64() {
            let rate_key = tokenize(key).expect("python-parsed keys tokenize");
            assert_eq!(rate_key.threshold, Some(value as u64), "key {key}");
        }
    }
    let no_threshold_keys = [
        "input_cost_per_video_per_second_above_15s_interval",
        "cache_creation_input_token_cost_above_1hr",
        "input_cost_per_token",
    ];
    for key in no_threshold_keys {
        assert_eq!(
            tokenize(key).expect("tokenizes").threshold,
            None,
            "no token threshold for {key}"
        );
    }
    assert_eq!(
        tokenize("cache_creation_input_token_cost_above_1hr_above_200k_tokens")
            .expect("tokenizes")
            .threshold,
        Some(200_000)
    );
}

#[rstest]
fn service_tier_suffixes_match_python() {
    let pinning = pinning();
    let pinned: Vec<&str> = pinning["service_tier_suffixes"]
        .as_array()
        .expect("suffix list")
        .iter()
        .map(|value| value.as_str().expect("suffix string"))
        .collect();
    let rust: Vec<String> = ServiceTier::SUFFIXES
        .into_iter()
        .map(|tier| format!("_{}", tier.as_str()))
        .collect();
    assert_eq!(rust, pinned);
}

#[rstest]
fn batch_tier_grammar_matches_python_regex_language() {
    let pattern = pinning()["batch_tier_key"]
        .as_str()
        .expect("batch pattern")
        .to_string();
    let regex = regex::Regex::new(&pattern).expect("python batch pattern compiles");
    for prefix in [
        "input_cost_per_token",
        "output_cost_per_token",
        "cache_read_input_token_cost",
        "cache_creation_input_token_cost",
    ] {
        for magnitude in ["1k", "128k", "500"] {
            let key = format!("{prefix}_above_{magnitude}_tokens_batches");
            let rate_key = tokenize(&key).unwrap_or_else(|_| panic!("{key} tokenizes"));
            assert!(regex.is_match(&key), "{key} matches python pattern");
            assert!(rate_key.batch);
            assert_eq!(
                rate_key.metric,
                prefix.parse::<Metric>().expect("known metric"),
                "{key}"
            );
        }
    }
    let rejected = [
        "input_cost_per_token_above_128.5k_tokens_batches",
        "input_cost_per_audio_token_above_128k_tokens_batches",
        "input_cost_per_token_above_128k_tokens",
    ];
    for key in rejected {
        assert!(!regex.is_match(key), "{key} outside python pattern");
    }
}

#[rstest]
fn cost_per_unit_replays_python_coercion_and_tier_fallback() {
    for row in pinning()["cost_per_unit"]
        .as_array()
        .expect("cost_per_unit rows")
    {
        let fields = row["model_info"].as_object().expect("model_info object");
        let cost_key = row["cost_key"].as_str().expect("cost key");
        let default = row["default"].as_f64();
        let expected = row["result"].as_f64();
        let actual = cost_per_unit(fields, cost_key, default);
        assert_eq!(actual, expected, "key {cost_key} default {default:?}");
    }
}

#[rstest]
fn model_info_projection_passes_rate_keys_the_tokenizer_recognizes() {
    let fixture = fixture();
    let projected = fixture["model_info_projection"]["projected"]
        .as_array()
        .expect("projected keys");
    let mut unrecognized: Vec<&str> = Vec::new();
    for key in projected.iter().filter_map(Value::as_str) {
        let known = tokenize(key).is_ok()
            || litellm_cost::wire::recognized_non_rate_key(key)
            || matches!(
                key,
                "off_peak_pricing"
                    | "tiered_pricing"
                    | "search_context_cost_per_query"
                    | "provider_specific_entry"
                    | "guardrail_cost_per_unit"
                    | "regional_processing_uplift_multiplier_eu"
                    | "regional_processing_uplift_multiplier_us"
                    | "regional_endpoint_uplift_multiplier"
            );
        if !known {
            unrecognized.push(key);
        }
    }
    assert!(
        unrecognized.is_empty(),
        "projected keys the tokenizer or allowlist must recognize: {unrecognized:?}"
    );
}

#[rstest]
fn registered_entry_parses_into_model_pricing_with_projected_rates() {
    let entry: RawCatalogEntry = serde_json::from_value(serde_json::json!({
        "input_cost_per_token": 1e-6,
        "output_cost_per_token": 2e-6,
        "input_cost_per_token_above_128k_tokens": 5e-7,
        "input_cost_per_token_above_128k_tokens_priority": 6e-7,
        "cache_read_input_token_cost": 1e-7,
    }))
    .expect("synthetic entry parses");
    let pricing = entry.pricing();
    assert_eq!(pricing.rate(Metric::InputPerToken).value(), Some(1e-6));
    assert_eq!(pricing.rate(Metric::CacheReadToken).value(), Some(1e-7));
    let threshold = &pricing.thresholds[0];
    assert_eq!(threshold.above_tokens, 128_000);
    assert_eq!(
        threshold
            .standard
            .get(&Metric::InputPerToken)
            .and_then(|r| r.value()),
        Some(5e-7)
    );
    assert_eq!(
        threshold
            .tiers
            .get(&(Metric::InputPerToken, ServiceTier::Priority))
            .and_then(|r| r.value()),
        Some(6e-7)
    );
}

#[rstest]
fn provider_and_call_type_enums_mirror_python() {
    let pinning = pinning();
    let python_providers: std::collections::BTreeSet<&str> = pinning["llm_providers"]
        .as_array()
        .expect("llm providers")
        .iter()
        .map(|value| value.as_str().expect("provider value"))
        .collect();
    let rust_providers: std::collections::BTreeSet<&str> = <LlmProviders as VariantArray>::VARIANTS
        .iter()
        .map(|provider| provider.as_str())
        .collect();
    assert_eq!(
        rust_providers, python_providers,
        "every Python LlmProviders value is a Rust variant and no extras exist"
    );
    for provider in &python_providers {
        assert!(provider.parse::<LlmProviders>().ok().is_some());
    }
    let python_calls: std::collections::BTreeSet<&str> = pinning["call_types"]
        .as_array()
        .expect("call types")
        .iter()
        .map(|value| value.as_str().expect("call type value"))
        .collect();
    let rust_calls: std::collections::BTreeSet<&str> = <CallTypes as VariantArray>::VARIANTS
        .iter()
        .map(|call| call.as_str())
        .collect();
    assert_eq!(
        rust_calls, python_calls,
        "every Python CallTypes value is a Rust variant and no extras exist"
    );
    for call in &python_calls {
        assert!(call.parse::<CallTypes>().ok().is_some());
    }
    assert_eq!(
        "_arealtime".parse::<CallTypes>().ok(),
        Some(CallTypes::arealtime)
    );
    assert_eq!(
        "_aresponses_websocket".parse::<CallTypes>().ok(),
        Some(CallTypes::aresponses_websocket)
    );
}
