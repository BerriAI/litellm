//! Parity with main's legacy budget hold. `fixtures/bound_parity.json` was captured once from main's Python
//! (f631dbd13213, `LITELLM_LOCAL_MODEL_COST_MAP=True`): `estimate_request_max_cost` and `estimate_request_input_cost`
//! on catalog rows without a router, and `_max_cost_for_cost_info` and `_input_cost_for_cost_info` on the group rates
//! a router reports

use rstest::rstest;
use serde_json::Value;

use super::*;
use crate::testing::{TestSource, fixture_row, row_ref};

const FIXTURE: &str = include_str!("../../fixtures/bound_parity.json");

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

/// What the adapter reads off the request body and route
fn caps(body: &Value, route: &str) -> RequestCaps {
    let requested = [
        &body["max_completion_tokens"],
        &body["max_tokens"],
        &body["max_output_tokens"],
        &body["inferenceConfig"]["maxTokens"],
    ];
    RequestCaps {
        requested_output: requested.into_iter().find_map(Value::as_i64),
        n: body["n"].as_i64(),
        best_of: body["best_of"].as_i64(),
        input_only: ["embeddings", "rerank", "moderations"]
            .iter()
            .any(|part| route.contains(part)),
    }
}

fn bound_for(case: &Value) -> Bound {
    let row = &case["row"];
    let (source, shape) = match case["shape"].as_str() {
        Some("group") => (
            TestSource::default(),
            CostShape {
                flat: FlatSource::Group(GroupRates {
                    input: row["input_cost_per_token"]
                        .as_f64()
                        .and_then(UnitPrice::new),
                    output: row["output_cost_per_token"]
                        .as_f64()
                        .and_then(UnitPrice::new),
                    max_input_tokens: row["max_input_tokens"].as_i64(),
                    max_output_tokens: row["max_output_tokens"].as_i64(),
                }),
                tiers: None,
            },
        ),
        _ => (
            TestSource::with_row("row", fixture_row(row)),
            CostShape {
                flat: FlatSource::Row {
                    row: row_ref("row"),
                },
                tiers: None,
            },
        ),
    };
    bound(
        &source,
        &BoundQuery {
            models: vec![BoundModel {
                counted_input: case["counted"].as_i64(),
                candidates: vec![shape],
            }],
            request: caps(&case["body"], case["route"].as_str().unwrap()),
            policy: BoundPolicy::Legacy,
        },
    )
}

#[rstest]
#[case::anthropic_requested_output("anthropic_requested_output")]
#[case::anthropic_default_output("anthropic_default_output")]
#[case::anthropic_output_clamped_to_row("anthropic_output_clamped_to_row")]
#[case::bedrock_n_and_best_of("bedrock_n_and_best_of")]
#[case::bedrock_inference_config("bedrock_inference_config")]
#[case::vertex_max_completion_tokens("vertex_max_completion_tokens")]
#[case::azure_ai_input_only_route("azure_ai_input_only_route")]
#[case::tiered_low("tiered_low")]
#[case::tiered_high("tiered_high")]
#[case::tiered_empty_prompt("tiered_empty_prompt")]
#[case::image_generation("image_generation")]
#[case::group_uncounted_reads_max_input("group_uncounted_reads_max_input")]
#[case::group_missing_output_rate("group_missing_output_rate")]
#[case::group_zero_max_output_reads_default("group_zero_max_output_reads_default")]
fn rust_holds_what_main_held(#[case] name: &str) {
    let case = case(name);
    let bound = bound_for(&case);

    assert_eq!(bound.max, case["expected"]["max"].as_f64());
    assert_eq!(bound.input_floor, case["expected"]["input"].as_f64());
}
