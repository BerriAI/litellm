#![allow(clippy::disallowed_types)]

// mirrors: test_litellm/litellm_core_utils/llm_cost_calc/test_guardrail_cost.py::test_bedrock_guardrail_cost_prices_each_counter

use std::collections::BTreeMap;

use litellm_cost::guardrail_cost::{
    azure_prompt_shield_guardrail_cost, bedrock_guardrail_cost, bedrock_guardrail_cost_by_unit,
    billed_guardrail_cost_by_unit, cost_breakdown_with_guardrail, guardrail_cost_total,
    guardrail_information_cost,
};
use rstest::rstest;
use serde_json::{Value, json};

fn cost_map() -> BTreeMap<String, Value> {
    BTreeMap::from([
        (
            "bedrock/guardrails".to_owned(),
            json!({"guardrail_cost_per_unit": {
                "contentPolicyUnits": 0.00015,
                "topicPolicyUnits": 0.00015,
                "wordPolicyUnits": 0.0
            }}),
        ),
        (
            "bedrock/eu-west-1/guardrails".to_owned(),
            json!({"guardrail_cost_per_unit": {"contentPolicyUnits": 0.0002}}),
        ),
        (
            "bedrock/us-west-2/guardrails".to_owned(),
            json!({"guardrail_cost_per_unit": "malformed"}),
        ),
    ])
}

#[rstest]
#[case(Some("us-east-1"), 0.00015)]
#[case(Some("eu-west-1"), 0.0002)]
#[case(Some("us-west-2"), 0.00015)]
fn bedrock_guardrail_cost_selects_regional_or_fallback_price(
    #[case] region: Option<&str>,
    #[case] expected: f64,
) {
    let units = BTreeMap::from([("contentPolicyUnits".to_owned(), 1)]);
    assert!((bedrock_guardrail_cost(&units, region, &cost_map()) - expected).abs() < 1e-12);
}

#[rstest]
fn bedrock_guardrail_cost_by_unit_preserves_unknown_and_explicit_zero() {
    let units = BTreeMap::from([
        ("contentPolicyUnits".to_owned(), 2),
        ("topicPolicyUnits".to_owned(), 1),
        ("wordPolicyUnits".to_owned(), 5),
        ("someFutureCounter".to_owned(), 3),
    ]);
    let by_unit = bedrock_guardrail_cost_by_unit(&units, Some("us-east-1"), &cost_map()).unwrap();
    assert_eq!(
        by_unit.keys().collect::<Vec<_>>(),
        units.keys().collect::<Vec<_>>()
    );
    assert_eq!(by_unit["wordPolicyUnits"], Some(0.0));
    assert_eq!(by_unit["someFutureCounter"], None);
    assert!((guardrail_cost_total(Some(&by_unit)) - 0.00045).abs() < 1e-12);
    assert_eq!(
        guardrail_cost_total(Some(&by_unit)),
        bedrock_guardrail_cost(&units, Some("us-east-1"), &cost_map())
    );
}

#[rstest]
fn bedrock_guardrail_cost_unpriced_is_unknown_by_unit_and_zero_in_total() {
    let units = BTreeMap::from([("contentPolicyUnits".to_owned(), 1)]);
    assert_eq!(
        bedrock_guardrail_cost_by_unit(&units, None, &BTreeMap::new()),
        None
    );
    assert_eq!(bedrock_guardrail_cost(&units, None, &BTreeMap::new()), 0.0);
}

#[rstest]
#[case(json!({"guardrail_cost_by_unit": {"contentPolicyUnits": 0.15, "wordPolicyUnits": 0, "unknown": null}}), true)]
#[case(json!({"guardrail_cost_by_unit": {"contentPolicyUnits": 0.15}, "guardrail_cost_in_spend": null}), true)]
#[case(json!({"guardrail_cost_by_unit": {"contentPolicyUnits": 0.15}, "guardrail_cost_in_spend": false}), false)]
#[case(json!({"guardrail_cost_by_unit": {"contentPolicyUnits": -0.5}}), false)]
#[case(json!({"guardrail_cost_by_unit": {"contentPolicyUnits": "bad"}}), false)]
#[case(json!({"guardrail_cost_by_unit": {"contentPolicyUnits": 0.15}, "guardrail_cost_in_spend": "maybe"}), false)]
#[case(json!("not-an-entry"), false)]
fn billed_guardrail_cost_by_unit_validates_hook_stamp(
    #[case] raw: Value,
    #[case] expected_valid: bool,
) {
    assert_eq!(
        billed_guardrail_cost_by_unit(&raw).is_some(),
        expected_valid
    );
}

#[rstest]
#[case(json!([{"guardrail_cost": 0.0003}, {"guardrail_cost": null}, {}, {"guardrail_cost": 0.0001}]), 0.0004)]
#[case(json!([{"guardrail_cost": -0.005}, {"guardrail_cost": 0.5, "guardrail_cost_in_spend": false}, {"guardrail_cost": 0.0003}]), 0.0003)]
#[case(json!([{"guardrail_cost": 0.5, "guardrail_cost_in_spend": "maybe"}, {"guardrail_cost": 0.0003}]), 0.0003)]
#[case(json!([{"guardrail_cost": 0.5, "guardrail_cost_in_spend": null}, {"guardrail_cost": 0.0003}]), 0.5003)]
fn guardrail_information_cost_skips_bad_entries_without_losing_siblings(
    #[case] raw: Value,
    #[case] expected: f64,
) {
    assert!((guardrail_information_cost(&raw) - expected).abs() < 1e-12);
}

#[rstest]
#[case(Some("free"), Some(0.38), Some(0.0))]
#[case(Some("paid"), Some(0.38), Some(0.00114))]
#[case(None, None, None)]
fn azure_prompt_shield_guardrail_cost_prices_text_records(
    #[case] tier: Option<&str>,
    #[case] price: Option<f64>,
    #[case] expected: Option<f64>,
) {
    let units = BTreeMap::from([("text_records".to_owned(), 3)]);
    match (
        azure_prompt_shield_guardrail_cost(&units, tier, price),
        expected,
    ) {
        (Some(actual), Some(expected)) => assert!((actual - expected).abs() < 1e-12),
        (actual, expected) => assert_eq!(actual, expected),
    }
}

#[rstest]
fn cost_breakdown_with_guardrail_merges_existing_total_and_creates_new_breakdown() {
    let existing = BTreeMap::from([
        ("input_cost".to_owned(), 0.1),
        ("total_cost".to_owned(), 0.4),
    ]);
    assert_eq!(
        cost_breakdown_with_guardrail(Some(&existing), 0.0),
        Some(existing.clone())
    );
    let merged = cost_breakdown_with_guardrail(Some(&existing), 0.0003).unwrap();
    assert_eq!(merged["input_cost"], 0.1);
    assert!((merged["total_cost"] - 0.4003).abs() < 1e-12);
    assert_eq!(merged["guardrail_cost"], 0.0003);
    assert_eq!(
        cost_breakdown_with_guardrail(None, 0.0003),
        Some(BTreeMap::from([
            ("guardrail_cost".to_owned(), 0.0003),
            ("total_cost".to_owned(), 0.0003),
        ]))
    );
}

#[rstest]
#[case::string_in_spend(json!({"guardrail_cost": 0.5, "guardrail_cost_in_spend": "true"}), 0.5)]
#[case::integer_in_spend(json!({"guardrail_cost": 0.5, "guardrail_cost_in_spend": 1}), 0.5)]
#[case::string_off(json!({"guardrail_cost": 0.5, "guardrail_cost_in_spend": "off"}), 0.0)]
#[case::uninterpretable_in_spend(json!({"guardrail_cost": 0.5, "guardrail_cost_in_spend": "maybe"}), 0.0)]
#[case::padded_string_cost(json!({"guardrail_cost": " 0.5 "}), 0.5)]
#[case::boolean_cost(json!({"guardrail_cost": true}), 1.0)]
#[case::unparseable_cost(json!({"guardrail_cost": "free"}), 0.0)]
fn guardrail_information_cost_validates_entries_like_pydantic(
    #[case] entry: Value,
    #[case] expected: f64,
) {
    assert_eq!(guardrail_information_cost(&entry), expected);
}

#[rstest]
#[case::string_price(json!({"guardrail_cost_by_unit": {"text": "0.25"}}), Some(Some(0.25)))]
#[case::string_in_spend_false(json!({"guardrail_cost_by_unit": {"text": 0.25}, "guardrail_cost_in_spend": "false"}), None)]
#[case::negative_price_rejects_entry(json!({"guardrail_cost_by_unit": {"text": -1}}), None)]
fn billed_guardrail_cost_by_unit_validates_entries_like_pydantic(
    #[case] entry: Value,
    #[case] expected: Option<Option<f64>>,
) {
    assert_eq!(
        billed_guardrail_cost_by_unit(&entry).map(|units| units.get("text").copied().flatten()),
        expected
    );
}
