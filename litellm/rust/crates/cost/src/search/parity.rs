//! Parity with main's `search_provider_cost_per_query`. `fixtures/search_parity.json` was captured once from main's
//! Python: each case holds the row the request's mode picked, the request parameters and main's
//! `(prompt, completion)`. The requests and additional results each case bills are what the adapter counts from
//! those parameters

use rstest::rstest;

use super::{SearchQuery, price};
use crate::testing::{TestSource, expected_sides, fixture_case, header, priced, units_row};

const FIXTURE: &str = include_str!("../../fixtures/search_parity.json");

#[rstest]
#[case::tavily_basic("tavily_basic", 1, 0)]
#[case::tavily_advanced_depth("tavily_advanced_depth", 3, 0)]
#[case::linkup_deep("linkup_deep", 2, 0)]
#[case::serper_queries("serper_queries", 7, 0)]
#[case::parallel_default_results("parallel_default_results", 1, 15)]
#[case::parallel_fast_usage("parallel_fast_usage", 3, 12)]
#[case::parallel_usage_without_extra_results("parallel_usage_without_extra_results", 2, 0)]
fn a_search_costs_what_main_charged(
    #[case] name: &str,
    #[case] queries: u64,
    #[case] additional_results: u64,
) {
    let case = fixture_case(FIXTURE, name);
    let model = case["model"].as_str().unwrap();
    let source = TestSource::with_row(model, units_row(&case["row"]));
    let query = SearchQuery {
        header: header(case["provider"].as_str().unwrap(), model),
        queries,
        additional_results,
    };
    let receipt = priced(price(&source, &query));
    assert_eq!(
        Some((receipt.prompt.get(), receipt.completion.get())),
        expected_sides(&case)
    );
}
