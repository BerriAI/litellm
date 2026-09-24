#![allow(clippy::disallowed_types)]

use std::sync::LazyLock;

use litellm_cost::pricing::{Metric, Rate, ServiceTier, tokenize};
use litellm_cost::wire::{EMBEDDED_CATALOG, rate};
use rstest::rstest;

#[rstest]
fn every_embedded_key_is_recognized_by_the_tokenizer() {
    let mut offenders: Vec<String> = Vec::new();
    for (model, entry) in EMBEDDED_CATALOG.iter() {
        let pricing = entry.pricing();
        for key in pricing.extra.keys() {
            offenders.push(format!("{model}: {key}"));
        }
    }
    assert!(
        offenders.is_empty(),
        "unrecognized keys in the embedded catalog (a new key shape would ship silently unpriced): {}",
        offenders.join(", ")
    );
}

#[rstest]
fn embedded_catalog_yields_decorated_and_plain_rates() {
    let mut thresholds = 0_usize;
    let mut tiers = 0_usize;
    let mut batches = 0_usize;
    let mut plain = 0_usize;
    for entry in EMBEDDED_CATALOG.values() {
        let pricing = entry.pricing();
        thresholds += pricing.thresholds.len();
        tiers += pricing.tiers.len();
        batches += pricing.batches.len();
        plain += pricing.base.len();
    }
    assert!(thresholds > 0);
    assert!(tiers > 0);
    assert!(batches > 0);
    assert!(plain > 0);
}

#[rstest]
fn tokenizer_applies_the_rate_key_grammar() {
    let plain = tokenize("input_cost_per_token").expect("plain rate key");
    assert_eq!(plain.metric, Metric::InputPerToken);
    assert_eq!(plain.threshold, None);
    assert_eq!(plain.tier, None);
    assert!(!plain.batch);

    let threshold = tokenize("input_cost_per_token_above_128k_tokens").expect("threshold key");
    assert_eq!(threshold.metric, Metric::InputPerToken);
    assert_eq!(threshold.threshold, Some(128_000));

    let plain_threshold =
        tokenize("input_cost_per_token_above_512k_tokens").expect("threshold key");
    assert_eq!(plain_threshold.threshold, Some(512_000));

    let one_hour = tokenize("cache_creation_input_token_cost_above_1hr").expect("1hr is a metric");
    assert_eq!(one_hour.metric, Metric::CacheCreationToken1hr);
    assert_eq!(one_hour.threshold, None);

    let one_hour_threshold =
        tokenize("cache_creation_input_token_cost_above_1hr_above_200k_tokens")
            .expect("1hr metric with threshold");
    assert_eq!(one_hour_threshold.metric, Metric::CacheCreationToken1hr);
    assert_eq!(one_hour_threshold.threshold, Some(200_000));

    let interval = tokenize("input_cost_per_video_per_second_above_15s_interval")
        .expect("interval-suffixed metric stays whole");
    assert_eq!(interval.metric, Metric::InputVideoPerSecond15sInterval);
    assert_eq!(interval.threshold, None);

    let tiered = tokenize("input_cost_per_token_priority").expect("tier key");
    assert_eq!(tiered.tier, Some(ServiceTier::Priority));

    let auto = tokenize("input_cost_per_token_auto").expect("auto is a grammar tier");
    assert_eq!(auto.tier, Some(ServiceTier::Auto));

    let full = tokenize("cache_read_input_token_cost_above_272k_tokens_flex")
        .expect("threshold plus tier");
    assert_eq!(full.metric, Metric::CacheReadToken);
    assert_eq!(full.threshold, Some(272_000));
    assert_eq!(full.tier, Some(ServiceTier::Flex));

    let batch = tokenize("cache_read_input_token_cost_above_272k_tokens_batches")
        .expect("threshold plus batch");
    assert_eq!(batch.threshold, Some(272_000));
    assert!(batch.batch);
}

#[rstest]
fn tokenizer_rejects_unknown_metrics_and_malformed_decorations() {
    assert!(tokenize("input_cost_per_gadget").is_err());
    assert!(tokenize("input_cost_per_token_standard").is_err());
    assert!(tokenize("input_cost_per_token_above_128.5k_tokens").is_err());
    assert!(tokenize("input_cost_per_token_above_tokens").is_err());
    assert!(tokenize("max_tokens").is_err());
}

#[rstest]
fn rate_parser_mirrors_get_cost_per_unit_coercions() {
    use serde_json::json;
    assert_eq!(rate(None), Rate::Missing);
    assert_eq!(rate(Some(&json!(null))), Rate::Null);
    assert_eq!(rate(Some(&json!(true))), Rate::Value(1.0));
    assert_eq!(rate(Some(&json!(false))), Rate::Value(0.0));
    assert_eq!(rate(Some(&json!(3))), Rate::Value(3.0));
    assert_eq!(rate(Some(&json!(3.5))), Rate::Value(3.5));
    assert_eq!(rate(Some(&json!(" 0.5 "))), Rate::Value(0.5));
    assert_eq!(rate(Some(&json!("not a number"))), Rate::Invalid);
    assert_eq!(rate(Some(&json!([1]))), Rate::Invalid);
    assert_eq!(rate(Some(&json!({"cost": 1}))), Rate::Invalid);
}

#[rstest]
fn parsed_pricing_buckets_a_synthetic_overlay_entry() {
    let entry: litellm_cost::wire::RawCatalogEntry = serde_json::from_value(serde_json::json!({
        "input_cost_per_token": 3e-6,
        "output_cost_per_token": 15e-6,
        "input_cost_per_token_above_128k_tokens": 1.5e-6,
        "input_cost_per_token_priority": 6e-6,
        "input_cost_per_token_batches": 1.5e-6,
        "cache_creation_input_token_cost_above_1hr_above_200k_tokens": 1e-6,
        "off_peak_pricing": {
            "input_cost_per_token": 1e-6,
            "hours_utc": "22:00-06:00"
        },
        "tiered_pricing": [
            {"range": [0, 16000], "input_cost_per_token": 1e-6, "output_cost_per_token": 2e-6}
        ],
        "search_context_cost_per_query": {"search_context_size_medium": 3e-6},
        "provider_specific_entry": {"us": 1.1},
        "litellm_provider": "synthetic",
        "max_tokens": 8192,
        "supports_function_calling": true
    }))
    .expect("synthetic entry parses");
    let pricing = entry.pricing();
    assert_eq!(pricing.rate(Metric::InputPerToken), Rate::Value(3e-6));
    assert_eq!(
        pricing.tier_rate(Metric::InputPerToken, ServiceTier::Priority),
        Rate::Value(6e-6)
    );
    assert_eq!(
        pricing.batch_rate(Metric::InputPerToken),
        Rate::Value(1.5e-6)
    );
    let threshold = &pricing.thresholds[0];
    assert_eq!(threshold.above_tokens, 128_000);
    assert_eq!(
        threshold.standard.get(&Metric::InputPerToken),
        Some(&Rate::Value(1.5e-6))
    );
    let one_hour = &pricing.thresholds[1];
    assert_eq!(one_hour.above_tokens, 200_000);
    assert!(
        one_hour
            .standard
            .contains_key(&Metric::CacheCreationToken1hr)
    );
    let off_peak = pricing.off_peak.as_ref().expect("off peak parsed");
    assert_eq!(off_peak.input, Rate::Value(1e-6));
    assert_eq!(off_peak.hours_utc, vec!["22:00-06:00".to_string()]);
    let tier = &pricing.tiered_pricing.as_ref().expect("tiered parsed")[0];
    assert_eq!(tier.range, (0.0, Some(16_000.0)));
    assert_eq!(
        pricing
            .search_context_cost_per_query
            .expect("search context")
            .medium,
        Rate::Value(3e-6)
    );
    assert_eq!(
        pricing.provider_specific_entry.get("us"),
        Some(&Rate::Value(1.1))
    );
    assert_eq!(pricing.litellm_provider.as_deref(), Some("synthetic"));
    assert!(pricing.extra.is_empty(), "recognized non-rate keys drop");
    let _: LazyLock<()> = LazyLock::new(|| ());
}

#[rstest]
fn checked_rates_reject_negative_and_non_finite_where_python_bills_them() {
    use litellm_cost::error::CostError;
    use litellm_cost::pricing::Rate;
    assert_eq!(Rate::Value(3e-6).checked(), Ok(3e-6));
    assert_eq!(Rate::Value(0.0).checked(), Ok(0.0));
    assert_eq!(
        Rate::Value(-1.0).checked(),
        Err(CostError::InvalidRate),
        "divergence: Python bills negative rates as negative costs; Rust refuses them"
    );
    assert_eq!(
        Rate::Value(f64::NAN).checked(),
        Err(CostError::InvalidRate),
        "divergence: Python propagates NaN costs; Rust refuses them"
    );
    assert_eq!(
        Rate::Value(f64::INFINITY).checked(),
        Err(CostError::InvalidRate)
    );
    assert_eq!(Rate::Missing.checked(), Err(CostError::InvalidRate));
    assert_eq!(Rate::Null.checked(), Err(CostError::InvalidRate));
    assert_eq!(Rate::Invalid.checked(), Err(CostError::InvalidRate));
}
