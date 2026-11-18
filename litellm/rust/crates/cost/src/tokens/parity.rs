//! Parity with main's `completion_cost` on the P1 providers' Claude rows. `fixtures/tokens_parity.json` was captured
//! once from main's Python (f631dbd13213, `LITELLM_LOCAL_MODEL_COST_MAP=True`): each case holds the row it priced,
//! the billed usage the adapter hands over, and the total, the stored cost breakdown and the prompt-caching savings
//! main computed. Every value must match to the last bit

use rstest::rstest;
use serde_json::Value;

use super::price;
use crate::adjust::MarginRule;
use crate::call::{InferenceGeo, Speed};
use crate::price::{SearchContextSize, ServiceTier};
use crate::query::{
    AdjustmentPolicy, Basis, CacheReads, CacheWrites, Header, Modalities, PriceSource, Provider,
    TokenUsage, TokensQuery, WebSearchUse,
};
use crate::receipt::{PriceOutcome, Receipt};
use crate::testing::{TestSource, basis, candidate, fixture_row, row_ref};

const FIXTURE: &str = include_str!("../../fixtures/tokens_parity.json");

/// The key of a `requested_model_info_override` row, beside the catalog row its candidate names
const GIVEN: &str = "requested_model_info_override";

fn case(name: &str) -> Value {
    let fixture: Value = serde_json::from_str(FIXTURE).unwrap();
    fixture["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|case| case["name"] == name)
        .unwrap_or_else(|| panic!("no fixture case {name}"))
        .clone()
}

fn count(value: &Value) -> u64 {
    value.as_u64().unwrap_or(0)
}

fn usage(written: &Value) -> TokenUsage {
    TokenUsage {
        prompt_tokens: count(&written["prompt_tokens"]),
        input: Modalities {
            text: count(&written["input"]["text"]),
            ..Modalities::default()
        },
        cache_read: CacheReads {
            text: count(&written["cache_read"]["text"]),
            ..CacheReads::default()
        },
        cache_write: CacheWrites {
            default: count(&written["cache_write"]["default"]),
            one_hour: count(&written["cache_write"]["one_hour"]),
            twelve_hours: 0,
        },
        output: Modalities {
            text: count(&written["output"]["text"]),
            ..Modalities::default()
        },
        reasoning: count(&written["reasoning"]),
        web_search: written.get("web_search").map(|search| WebSearchUse {
            requests: count(&search["requests"]),
            context: search.get("context").map(|size| match size.as_str() {
                Some("low") => SearchContextSize::Low,
                Some("high") => SearchContextSize::High,
                _ => SearchContextSize::Medium,
            }),
        }),
        speed: written.get("speed").map(|_| Speed::Fast),
        inference_geo: written
            .get("inference_geo")
            .and_then(Value::as_str)
            .map(InferenceGeo::new),
        ..TokenUsage::default()
    }
}

fn margin(rule: &Value) -> MarginRule {
    match rule {
        Value::Number(percent) => MarginRule::Percent(percent.as_f64().unwrap()),
        split => MarginRule::Split {
            percentage: split.get("percentage").and_then(Value::as_f64),
            fixed_amount: split.get("fixed_amount").and_then(Value::as_f64),
        },
    }
}

fn priced(case: &Value) -> Receipt {
    let provider: Provider = case["provider"].as_str().unwrap().parse().unwrap();
    let model = case["model"].as_str().unwrap();
    let discounts: Vec<(&str, f64)> = case["discount"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(key, fraction)| (key.as_str(), fraction.as_f64().unwrap()))
        .collect();
    let margins = case["margin"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(key, rule)| (key.clone(), margin(rule)))
        .collect();
    let (source, price_source) = match case.get("named_row") {
        Some(named) => (
            TestSource::with_row(model, fixture_row(named))
                .and_row(GIVEN, fixture_row(&case["row"])),
            PriceSource::Row {
                row: row_ref(GIVEN),
                breakdown_by_name: true,
            },
        ),
        None => (
            TestSource::with_row(model, fixture_row(&case["row"])),
            PriceSource::Catalog,
        ),
    };
    let source = source.with_discounts(&discounts).with_margins(margins);
    let query = TokensQuery {
        header: Header {
            candidates: vec![candidate(provider.clone(), model)],
            source: price_source,
            basis: Basis {
                service_tier: match case["service_tier"].as_str() {
                    Some("priority") => ServiceTier::Priority,
                    _ => ServiceTier::Standard,
                },
                vertex_location: case["vertex_location"].as_str().map(Into::into),
                ..basis()
            },
            adjustments: AdjustmentPolicy::Apply {
                provider: Some(provider.to_string().into()),
                model: None,
                region: None,
            },
            reported: None,
        },
        usage: usage(&case["usage"]),
        requested_model: None,
    };
    match price(&source, &query) {
        PriceOutcome::Priced(receipt) => *receipt,
        PriceOutcome::Unpriced(unpriced) => panic!("unpriced: {unpriced:?}"),
    }
}

fn number(value: &Value) -> f64 {
    value.as_f64().unwrap()
}

#[rstest]
#[case::anthropic_plain("anthropic_plain")]
#[case::anthropic_cache_read("anthropic_cache_read")]
#[case::anthropic_cache_write_5m("anthropic_cache_write_5m")]
#[case::anthropic_cache_write_1h("anthropic_cache_write_1h")]
#[case::anthropic_cache_mixed("anthropic_cache_mixed")]
#[case::anthropic_geo_us("anthropic_geo_us")]
#[case::anthropic_geo_global("anthropic_geo_global")]
#[case::anthropic_speed_fast("anthropic_speed_fast")]
#[case::anthropic_speed_fast_geo_us("anthropic_speed_fast_geo_us")]
#[case::anthropic_over_100k("anthropic_over_100k")]
#[case::anthropic_service_tier_priority("anthropic_service_tier_priority")]
#[case::anthropic_service_tier_without_tier_prices("anthropic_service_tier_without_tier_prices")]
#[case::anthropic_web_search("anthropic_web_search")]
#[case::anthropic_discount("anthropic_discount")]
#[case::anthropic_model_pattern_discount("anthropic_model_pattern_discount")]
#[case::anthropic_margin_percent("anthropic_margin_percent")]
#[case::anthropic_discount_and_fixed_margin("anthropic_discount_and_fixed_margin")]
#[case::anthropic_fixed_margin_only("anthropic_fixed_margin_only")]
#[case::bedrock_claude_platform_geo_us("bedrock_claude_platform_geo_us")]
#[case::bedrock_claude_platform_web_search_high("bedrock_claude_platform_web_search_high")]
#[case::bedrock_claude_platform_web_search_zero_requests(
    "bedrock_claude_platform_web_search_zero_requests"
)]
#[case::bedrock_over_200k("bedrock_over_200k")]
#[case::vertex_global("vertex_global")]
#[case::vertex_regional("vertex_regional")]
#[case::vertex_regional_over_100k_web_search("vertex_regional_over_100k_web_search")]
#[case::azure_ai_plain("azure_ai_plain")]
#[case::azure_ai_over_100k("azure_ai_over_100k")]
fn rust_prices_the_call_as_main_did(#[case] name: &str) {
    let case = case(name);
    let expected = &case["expected"];
    let stored = &expected["breakdown"];
    let rates = &stored["billed_token_rates"];

    let receipt = priced(&case);
    let billed = receipt.rates.unwrap();

    assert_eq!(receipt.total, number(&expected["total"]));
    assert_eq!(receipt.prompt.get(), number(&stored["input_cost"]));
    assert_eq!(receipt.completion.get(), number(&stored["output_cost"]));
    assert_eq!(
        receipt.tools.get(),
        number(&stored["cost_for_built_in_tools_cost_usd_dollar"])
    );
    assert_eq!(receipt.original(), number(&stored["original_cost"]));
    assert_eq!(
        receipt.discount.percent,
        number(&stored["discount_percent"])
    );
    assert_eq!(receipt.discount.amount, number(&stored["discount_amount"]));
    assert_eq!(receipt.margin.percent, number(&stored["margin_percent"]));
    assert_eq!(receipt.margin.fixed, number(&stored["margin_fixed_amount"]));
    assert_eq!(receipt.margin.total, number(&stored["margin_total_amount"]));
    assert_eq!(
        receipt.breakdown.cache_read,
        number(&stored["cache_read_cost"])
    );
    assert_eq!(
        receipt.breakdown.cache_creation,
        number(&stored["cache_creation_cost"])
    );
    assert_eq!(
        receipt.breakdown.reasoning,
        number(&stored["reasoning_cost"])
    );
    assert_eq!(
        [
            billed.input,
            billed.output,
            billed.cache_read,
            billed.cache_write,
            billed.cache_write_one_hour,
            billed.reasoning,
        ],
        [
            "input",
            "output",
            "cache_read",
            "cache_write",
            "cache_write_1h",
            "reasoning"
        ]
        .map(|key| number(&rates[key]))
    );
    assert_eq!(
        receipt.cache_savings,
        Some(number(&expected["cache_savings"]))
    );
    assert!(receipt.defaulted_to_zero.is_empty());
}
