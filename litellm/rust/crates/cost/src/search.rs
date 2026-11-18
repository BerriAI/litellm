//! The `search` kind, a port of `search_provider_cost_per_query` (`search/cost_calculator.py`), the Parallel AI
//! `parallel_ai_search_cost` and the search branch of `completion_cost`: queries at the first table's
//! `input_cost_per_query`, with discount and margin when the query asks for them. Parallel AI bills each request at
//! its mode's row and each result past the ones a request includes at a fixed rate the cost map does not list. A row
//! that prices queries by the results the request asks for (`max_results_range` tiers) is `NotPorted`

use crate::money::UnitPrice;
use crate::price::{Channel, Feature, InputUnit, Meter};
use crate::priced::{Quantity, Side};
use crate::query::{Header, Kind, Provider};
use crate::receipt::{PriceOutcome, Unpriced};
use crate::row::{LoweredRow, RowSource};
use crate::units::{self, Billed};

const QUERY: Meter = Meter::Input(InputUnit::Query { over_results: None });
const ADDITIONAL_RESULT: Meter = Meter::Feature(Feature::AdditionalResult);
const PARALLEL_AI: &str = "parallel_ai";
/// `PARALLEL_AI_ADDITIONAL_RESULT_COST` (`llms/parallel_ai/search/cost_calculator.py`)
const PARALLEL_AI_ADDITIONAL_RESULT: f64 = 0.001;

/// One search call. The header's `Rows` are the caller's configured per-query price, read ahead of the catalog row
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize))]
pub struct SearchQuery {
    #[cfg_attr(feature = "serde", serde(flatten))]
    pub header: Header,
    /// Queries, or for Parallel AI the requests its usage reports
    pub queries: u64,
    /// Parallel AI results past the ones a request includes
    #[cfg_attr(feature = "serde", serde(default))]
    pub additional_results: u64,
}

pub(crate) fn price<S: RowSource>(source: &S, query: &SearchQuery) -> PriceOutcome {
    let layers = units::layers(source, &query.header);
    let Some((first, _)) = layers.first() else {
        return units::no_row(&query.header);
    };
    let priced = layers.iter().find_map(|(at, row)| {
        units::listed(*row, &QUERY, Channel::Online).map(|rate| (at, *row, rate))
    });
    if layers.iter().any(|(_, row)| row.facts().prices_by_results) {
        return Unpriced::NotPorted(Kind::Search).into();
    }
    let parallel = query
        .header
        .candidates
        .first()
        .is_some_and(|candidate| candidate.provider == Provider::Other(PARALLEL_AI.to_owned()));
    let additional = UnitPrice::new(PARALLEL_AI_ADDITIONAL_RESULT)
        .filter(|_| parallel)
        .map(|rate| {
            units::line(
                ADDITIONAL_RESULT,
                Quantity::Count(query.additional_results),
                rate,
                Side::Prompt,
            )
        });
    let Some((at, row, rate)) = priced else {
        return Unpriced::MissingRates {
            row: first.clone(),
            missing: vec![QUERY],
        }
        .into();
    };
    let lines = [
        Some(units::line(
            QUERY,
            Quantity::Count(query.queries),
            rate,
            Side::Prompt,
        )),
        additional,
    ]
    .into_iter()
    .flatten()
    .collect();
    let Some(candidate) = query.header.candidates.first() else {
        return units::no_row(&query.header);
    };
    let named = units::Named {
        candidate,
        at: at.clone(),
        row,
    };
    units::receipt(
        source,
        &query.header,
        Some(&named),
        Billed::of(lines, Vec::new()),
    )
    .into()
}

#[cfg(test)]
mod parity {
    //! Parity with main's `search_provider_cost_per_query`. `fixtures/search_parity.json` was captured once from main's
    //! Python: each case holds the row the request's mode picked, the request parameters and main's
    //! `(prompt, completion)`. The requests and additional results each case bills are what the adapter counts from
    //! those parameters

    use rstest::rstest;

    use super::{SearchQuery, price};
    use crate::testing::{TestSource, expected_sides, fixture_case, header, priced, units_row};

    const FIXTURE: &str = include_str!("../fixtures/search_parity.json");

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
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;
    use crate::adjust::MarginRule;
    use crate::price::ServiceTier;
    use crate::query::{AdjustmentPolicy, PriceSource};
    use crate::row::RowFacts;
    use crate::testing::{BASE, TestRow, TestSource, header, priced, prices, row_ref};

    fn rated(rate: f64) -> TestRow {
        TestRow {
            prices: prices(&[(QUERY, ServiceTier::Standard, BASE, rate)]),
            ..TestRow::default()
        }
    }

    fn query(provider: &str, model: &str, queries: u64, additional_results: u64) -> SearchQuery {
        SearchQuery {
            header: header(provider, model),
            queries,
            additional_results,
        }
    }

    #[rstest]
    fn queries_bill_at_the_query_rate() {
        let source = TestSource::with_row("tavily/search", rated(0.008));
        let receipt = priced(price(&source, &query("tavily", "tavily/search", 3, 9)));
        assert_eq!(
            (receipt.prompt.get(), receipt.completion.get()),
            (0.008 * 3.0, 0.0)
        );
    }

    #[rstest]
    fn parallel_ai_adds_its_additional_results() {
        let source = TestSource::with_row("parallel_ai/search", rated(0.005));
        let receipt = priced(price(
            &source,
            &query(PARALLEL_AI, "parallel_ai/search", 2, 15),
        ));
        assert_eq!(receipt.prompt.get(), 0.005 * 2.0 + 0.001 * 15.0);
    }

    #[rstest]
    fn a_configured_query_price_wins_over_the_catalog() {
        let source =
            TestSource::with_row("tavily/search", rated(0.008)).and_row("configured", rated(0.5));
        let configured = SearchQuery {
            header: Header {
                source: PriceSource::Rows {
                    rows: vec![row_ref("configured")],
                },
                ..header("tavily", "tavily/search")
            },
            ..query("tavily", "tavily/search", 2, 0)
        };
        assert_eq!(priced(price(&source, &configured)).total, 1.0);
    }

    #[rstest]
    fn discount_and_margin_apply_when_asked() {
        let source = TestSource::with_row("tavily/search", rated(0.01))
            .with_discounts(&[("tavily", 0.5)])
            .with_margins(vec![("tavily".to_owned(), MarginRule::Percent(0.1))]);
        let adjusted = SearchQuery {
            header: Header {
                adjustments: AdjustmentPolicy::Apply {
                    provider: Some("tavily".into()),
                    model: None,
                    region: None,
                },
                ..header("tavily", "tavily/search")
            },
            ..query("tavily", "tavily/search", 10, 0)
        };
        let receipt = priced(price(&source, &adjusted));
        assert_eq!(receipt.original(), 0.01 * 10.0);
        assert_eq!(
            (receipt.discount.percent, receipt.margin.percent),
            (0.5, 0.1)
        );
        assert!(receipt.total < receipt.original());
    }

    /// Main reads a row's result tiers ahead of its flat query rate
    #[rstest]
    #[case::tiers_only(TestRow::default())]
    #[case::tiers_beside_a_flat_rate(rated(0.01))]
    fn a_row_priced_by_requested_results_is_not_ported(#[case] row: TestRow) {
        let tiered = TestRow {
            facts: RowFacts {
                prices_by_results: true,
                ..RowFacts::default()
            },
            ..row
        };
        let source = TestSource::with_row("exa_ai/search", tiered);
        assert_eq!(
            price(&source, &query("exa_ai", "exa_ai/search", 1, 0)),
            Unpriced::NotPorted(Kind::Search).into()
        );
    }
}
