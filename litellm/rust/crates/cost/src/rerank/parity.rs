//! Parity with main's `rerank_cost`. `fixtures/rerank_parity.json` was captured once from main's Python: each case
//! holds the model and provider, the cost fields `get_model_info` returned, the reported `billed_units` and main's
//! `(prompt, completion)`. The adapter answers a call no row prices with `(0.0, 0.0)`, as main did

use rstest::rstest;
use serde_json::Value;

use super::{RerankQuery, price};
use crate::receipt::PriceOutcome;
use crate::testing::{TestSource, expected_sides, header, units_row};

const FIXTURE: &str = include_str!("../../fixtures/rerank_parity.json");

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

#[rstest]
#[case::cohere_search_units("cohere_search_units")]
#[case::cohere_without_search_units("cohere_without_search_units")]
#[case::cohere_token_priced_row("cohere_token_priced_row")]
#[case::jina_tokens("jina_tokens")]
#[case::voyage_tokens("voyage_tokens")]
#[case::vertex_search_units("vertex_search_units")]
#[case::azure_ai_search_units("azure_ai_search_units")]
#[case::azure_ai_token_priced_row("azure_ai_token_priced_row")]
#[case::together_token_priced_row("together_token_priced_row")]
#[case::unmapped("unmapped")]
fn a_rerank_call_costs_what_main_charged(#[case] name: &str) {
    let case = case(name);
    let model = case["model"].as_str().unwrap();
    let source = match case["row"].is_null() {
        true => TestSource::default(),
        false => TestSource::with_row(model, units_row(&case["row"])),
    };
    let query = RerankQuery {
        header: header(case["provider"].as_str().unwrap(), model),
        search_units: case["units"]["search_units"].as_u64(),
        total_tokens: case["units"]["total_tokens"].as_u64(),
    };
    let sides = match price(&source, &query) {
        PriceOutcome::Priced(receipt) => (receipt.prompt.get(), receipt.completion.get()),
        PriceOutcome::Unpriced(_) => (0.0, 0.0),
    };
    assert_eq!(Some(sides), expected_sides(&case));
}
