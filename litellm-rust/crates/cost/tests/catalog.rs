use chrono::DateTime;
use litellm_cost::{
    PromptConvention, Usage,
    catalog::{CatalogCost, EstimateRequest, calculate, estimate},
};
use rstest::{fixture, rstest};
use serde_json::{Value, json};

#[fixture]
fn usage() -> Usage {
    Usage {
        prompt_tokens: 100,
        completion_tokens: 10,
        cache_read_tokens: 0,
        cache_write_tokens: 0,
        cache_write_5m_tokens: None,
        cache_write_1h_tokens: None,
        prompt_convention: PromptConvention::IncludesCache,
    }
}

#[rstest]
#[case::wrap_before_midnight(json!({"hours_utc":"16:30-00:30"}), "2026-01-05T23:00:00Z", 120.0)]
#[case::wrap_after_midnight(json!({"hours_utc":"16:30-00:30"}), "2026-01-06T00:29:59.999999999Z", 120.0)]
#[case::wrap_closed(json!({"hours_utc":"16:30-00:30"}), "2026-01-06T00:30:00Z", 240.0)]
#[case::multiple_windows(json!({"hours_utc":["16:30-00:30","04:00-06:00"]}), "2026-01-05T04:00:00Z", 120.0)]
#[case::full_day(json!({"hours_utc":"00:00-00:00"}), "2026-01-05T15:00:00Z", 120.0)]
#[case::bad_window_ignored(json!({"hours_utc":["bad","24:00-00:00"]}), "2026-01-05T04:00:00Z", 240.0)]
#[case::weekday_names(json!({"windows":[{"hours_utc":"00:00-00:00","weekdays":["Monday"]}]}), "2026-01-05T15:00:00Z", 120.0)]
#[case::weekday_numbers(json!({"windows":[{"hours_utc":"00:00-00:00","weekdays":[2]}]}), "2026-01-05T15:00:00Z", 240.0)]
#[case::timezone_rollover(json!({"weekday_timezone":"Asia/Shanghai","windows":[{"hours_utc":"16:00-18:00","weekdays":[2]}]}), "2026-01-05T16:00:00Z", 120.0)]
#[case::invalid_timezone_utc(json!({"weekday_timezone":"invalid","windows":[{"hours_utc":"16:00-18:00","weekdays":[2]}]}), "2026-01-05T16:00:00Z", 240.0)]
#[case::daylight_savings_calendar(json!({"weekday_timezone":"America/Los_Angeles","windows":[{"hours_utc":"06:00-08:00","weekdays":[7]}]}), "2026-07-06T06:59:59Z", 120.0)]
#[case::invalid_weekdays(json!({"windows":[{"hours_utc":"00:00-00:00","weekdays":[true,0,8,"bad"]}]}), "2026-01-05T15:00:00Z", 240.0)]
fn captured_time_selects_full_catalog_schedules(
    usage: Usage,
    #[case] schedule: Value,
    #[case] at: &str,
    #[case] expected: f64,
) {
    let mut block = schedule.as_object().unwrap().clone();
    block.insert("input_cost_per_token".into(), json!(1));
    block.insert("output_cost_per_token".into(), json!(2));
    let fields =
        json!({"input_cost_per_token":2,"output_cost_per_token":4,"off_peak_pricing":block});
    let request = EstimateRequest {
        usage,
        service_tier: None,
        threshold_is_inclusive: None,
        reasoning_tokens: Some(0),
        billed_at_ns: DateTime::parse_from_rfc3339(at)
            .unwrap()
            .timestamp_nanos_opt(),
    };
    assert_eq!(
        estimate(fields.as_object().unwrap(), &request).unwrap(),
        expected
    );
}

#[rstest]
#[case::balanced("balanced", "openai", 100, None, (300.0, 40.0))]
#[case::fast_alias("fast", "openai", 100, None, (500.0, 60.0))]
#[case::provider_inclusive("default", "xai", 100, None, (700.0, 80.0))]
#[case::provider_exclusive("default", "openai", 100, None, (100.0, 20.0))]
#[case::selected_threshold("balanced", "openai", 101, None, (909.0, 100.0))]
#[case::gateway_inclusive("default", "openai", 100, Some(true), (700.0, 80.0))]
#[case::gateway_exclusive("default", "xai", 100, Some(false), (100.0, 20.0))]
fn service_and_provider_context_select_catalog_rates(
    usage: Usage,
    #[case] tier: &str,
    #[case] provider: &str,
    #[case] input: u64,
    #[case] inclusive: Option<bool>,
    #[case] expected: (f64, f64),
) {
    let fields = json!({"litellm_provider":provider,"input_cost_per_token":1,"output_cost_per_token":2,"input_cost_per_token_balanced":3,"output_cost_per_token_balanced":4,"input_cost_per_token_priority":5,"output_cost_per_token_priority":6,"input_cost_per_token_above_100_tokens":7,"output_cost_per_token_above_100_tokens":8,"input_cost_per_token_above_100_tokens_balanced":9,"output_cost_per_token_above_100_tokens_balanced":10});
    let request = EstimateRequest {
        usage: Usage {
            prompt_tokens: input,
            ..usage
        },
        service_tier: Some(tier),
        threshold_is_inclusive: inclusive,
        reasoning_tokens: Some(0),
        billed_at_ns: None,
    };
    assert_eq!(
        estimate(fields.as_object().unwrap(), &request).unwrap(),
        expected.0 + expected.1
    );
    assert_eq!(
        calculate(fields.as_object().unwrap(), &request),
        Some(CatalogCost {
            input: expected.0,
            output: expected.1
        })
    );
}

#[rstest]
#[case::provider_default(json!({}), 90.0)]
#[case::explicit_zero(json!({"cache_read_input_token_cost":0}), 60.0)]
#[case::captured_off_peak_default(json!({"off_peak_pricing":{"hours_utc":"00:00-00:00","input_cost_per_token":2}}), 160.0)]
fn provider_cache_default_preserves_reported_counts(
    usage: Usage,
    #[case] overrides: Value,
    #[case] expected: f64,
) {
    let mut fields = json!({"litellm_provider":"fireworks_ai","input_cost_per_token":1,"output_cost_per_token":2}).as_object().unwrap().clone();
    fields.extend(overrides.as_object().unwrap().clone());
    let request = EstimateRequest {
        usage: Usage {
            cache_read_tokens: 60,
            ..usage
        },
        service_tier: None,
        threshold_is_inclusive: None,
        reasoning_tokens: Some(0),
        billed_at_ns: Some(0),
    };
    assert_eq!(estimate(&fields, &request).unwrap(), expected);
}

#[rstest]
#[case::missing_split(json!({}), 20, (None, None), None, None)]
#[case::equal_rates(json!({"cache_creation_input_token_cost_above_1hr":2}), 20, (None, None), None, Some(140.0))]
#[case::missing_hourly_rate(json!({"cache_creation_input_token_cost_above_1hr":null}), 20, (None, None), None, Some(140.0))]
#[case::invalid_hourly_rate(json!({"cache_creation_input_token_cost_above_1hr":"bad"}), 20, (None, None), None, None)]
#[case::partial_split(json!({}), 20, (Some(20), None), None, None)]
#[case::reported_five_minute(json!({"cache_creation_input_token_cost_above_1hr":"bad"}), 20, (Some(20), Some(0)), None, Some(140.0))]
#[case::reported_one_hour(json!({}), 20, (Some(0), Some(20)), None, Some(180.0))]
#[case::zero_writes(json!({"cache_creation_input_token_cost_above_1hr":"bad"}), 0, (None, None), None, Some(120.0))]
#[case::zero_rates(json!({"cache_creation_input_token_cost":0,"cache_creation_input_token_cost_above_1hr":0}), 20, (None, None), None, Some(100.0))]
#[case::selected_tier(json!({"tiered_pricing":[{"range":[0,200],"input_cost_per_token":1,"output_cost_per_token":2,"cache_creation_input_token_cost":2,"cache_creation_input_token_cost_above_1hr":4}]}), 20, (None, None), None, None)]
#[case::tier_fallbacks(json!({"tiered_pricing":[{"range":[0,200],"input_cost_per_token":1,"output_cost_per_token":2}]}), 20, (None, None), None, Some(120.0))]
#[case::selected_threshold(json!({"input_cost_per_token_above_50_tokens":1,"cache_creation_input_token_cost_above_50_tokens":3,"cache_creation_input_token_cost_above_1hr_above_50_tokens":3}), 20, (None, None), None, Some(160.0))]
#[case::selected_service_tier(json!({"cache_creation_input_token_cost_priority":4}), 20, (None, None), Some("priority"), Some(180.0))]
#[case::selected_off_peak(json!({"off_peak_pricing":{"hours_utc":"00:00-00:00","cache_creation_input_token_cost":4}}), 20, (None, None), None, Some(180.0))]
fn missing_cache_duration_is_only_safe_when_selected_prices_agree(
    usage: Usage,
    #[case] overrides: Value,
    #[case] writes: u64,
    #[case] durations: (Option<u64>, Option<u64>),
    #[case] service_tier: Option<&str>,
    #[case] expected: Option<f64>,
) {
    let mut fields = json!({"input_cost_per_token":1,"output_cost_per_token":2,"cache_creation_input_token_cost":2,"cache_creation_input_token_cost_above_1hr":4}).as_object().unwrap().clone();
    fields.extend(overrides.as_object().unwrap().clone());
    let request = EstimateRequest {
        usage: Usage {
            cache_write_tokens: writes,
            cache_write_5m_tokens: durations.0,
            cache_write_1h_tokens: durations.1,
            ..usage
        },
        service_tier,
        threshold_is_inclusive: None,
        reasoning_tokens: Some(0),
        billed_at_ns: Some(0),
    };
    assert_eq!(estimate(&fields, &request), expected);
}
