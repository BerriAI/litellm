use litellm_cost::usage_dispatch::get_usage_object;
use litellm_cost::zero_cost_diagnostic::{
    ZERO_COST_COUNTER_NAME, ZeroCostReason, diagnose_zero_cost, used_pricing_keys,
    zero_cost_warning,
};
use rstest::rstest;
use serde_json::{Value, json};

fn usage(value: Value) -> litellm_cost::responses_usage::ChatUsage {
    get_usage_object(&json!({"usage": value})).unwrap().unwrap()
}

#[rstest]
#[case(json!({"input_cost_per_second": 0.01}), false, Some((ZeroCostReason::MissingPricingKey, vec!["input_cost_per_token", "output_cost_per_token"])))]
#[case(json!({"input_cost_per_token": 0.01}), false, Some((ZeroCostReason::MissingPricingKey, vec!["output_cost_per_token"])))]
#[case(json!({"input_cost_per_token": 0.0, "output_cost_per_token": 0.0, "cache_read_input_token_cost": 0.01}), false, None)]
#[case(json!({"litellm_provider": "openai", "mode": "chat"}), false, None)]
#[case(json!({"tiered_pricing": [{"input_cost_per_token": 0.0, "output_cost_per_token": 0.0}]}), false, None)]
#[case(json!({"tiered_pricing": [{"input_cost_per_token": 0.01}]}), false, Some((ZeroCostReason::MissingPricingKey, vec!["input_cost_per_token", "output_cost_per_token"])))]
#[case(json!({"input_cost_per_token": 0.01, "output_cost_per_token": 0.02}), false, Some((ZeroCostReason::PricingNotApplied, vec![])))]
#[case(json!({"input_cost_per_token": 0.01, "output_cost_per_token": 0.02}), true, Some((ZeroCostReason::CostCalculationError, vec![])))]
fn diagnose_zero_cost_matches_python_rate_and_failure_precedence(
    #[case] entry: Value,
    #[case] calculation_failed: bool,
    #[case] expected: Option<(ZeroCostReason, Vec<&'static str>)>,
) {
    let actual = diagnose_zero_cost(
        &usage(json!({"prompt_tokens": 10, "completion_tokens": 20})),
        "deployment",
        &entry,
        calculation_failed,
    );
    assert_eq!(
        actual.map(|diagnostic| (diagnostic.reason, diagnostic.missing_pricing_keys)),
        expected
    );
}

#[rstest]
fn zero_usage_and_unpriced_nested_entries_stay_silent() {
    let empty = usage(json!({"prompt_tokens": 0, "completion_tokens": 0}));
    assert_eq!(used_pricing_keys(&empty), Vec::<&str>::new());
    assert_eq!(
        diagnose_zero_cost(
            &empty,
            "deployment",
            &json!({"input_cost_per_token": 0.01}),
            true
        ),
        None
    );
    assert_eq!(
        diagnose_zero_cost(
            &usage(json!({"prompt_tokens": 10})),
            "deployment",
            &json!({"tiered_pricing": [{"range": [0, 100], "input_cost_per_token": 0.0}]}),
            false,
        ),
        None
    );
}

#[rstest]
fn audio_usage_requires_audio_rates_and_warning_names_the_gap() {
    let usage = usage(json!({
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "prompt_tokens_details": {"audio_tokens": 10},
        "completion_tokens_details": {"audio_tokens": 5}
    }));
    assert_eq!(
        used_pricing_keys(&usage),
        vec![
            "input_cost_per_audio_token",
            "output_cost_per_token",
            "output_cost_per_audio_token"
        ]
    );
    let diagnostic = diagnose_zero_cost(
        &usage,
        "deployment",
        &json!({"input_cost_per_token": 0.01, "output_cost_per_token": 0.02}),
        false,
    )
    .unwrap();
    assert_eq!(
        diagnostic.missing_pricing_keys,
        vec!["input_cost_per_audio_token", "output_cost_per_audio_token"]
    );
    let warning = zero_cost_warning(
        &diagnostic,
        Some("group"),
        "model",
        Some("provider"),
        &usage,
    );
    assert!(warning.contains("model_group=group model=model provider=provider"));
    assert!(warning.contains("prompt_tokens=10 completion_tokens=20"));
    assert!(warning.contains("input_cost_per_audio_token, output_cost_per_audio_token"));
    assert!(warning.contains(&format!(
        "{ZERO_COST_COUNTER_NAME}{{reason=\"missing_pricing_key\"}}"
    )));
}
