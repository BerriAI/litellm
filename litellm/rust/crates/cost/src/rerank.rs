//! The `rerank` kind, a port of `rerank_cost` (`cost_calculator.py`) and each rerank config's
//! `calculate_rerank_cost`: search units at `input_cost_per_query`, except the providers whose config bills the
//! total tokens at `input_cost_per_token` (Jina AI, Scaleway built on it, and Voyage). A unit count the response did
//! not report bills nothing

use std::collections::BTreeMap;

use crate::price::TokenKind;
use crate::query::{Header, Provider};
use crate::receipt::PriceOutcome;
use crate::row::RowSource;
use crate::units;
use crate::usage::{Search, Tokens, Usage};

const TOKEN_BILLED: [&str; 3] = ["jina_ai", "scaleway", "voyage"];

/// One rerank call and the `billed_units` its response reported
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize))]
pub struct RerankQuery {
    #[cfg_attr(feature = "serde", serde(flatten))]
    pub header: Header,
    #[cfg_attr(feature = "serde", serde(default))]
    pub search_units: Option<u64>,
    #[cfg_attr(feature = "serde", serde(default))]
    pub total_tokens: Option<u64>,
}

pub(crate) fn price<S: RowSource>(source: &S, query: &RerankQuery) -> PriceOutcome {
    let Some(named) = units::named(source, &query.header) else {
        return units::no_row(&query.header);
    };
    let usage = match &named.candidate.provider {
        Provider::Other(provider) if TOKEN_BILLED.contains(&provider.as_str()) => Usage {
            tokens: Tokens {
                input: BTreeMap::from([(TokenKind::Text, query.total_tokens.unwrap_or(0))]),
                ..Tokens::default()
            },
            ..Usage::default()
        },
        _ => Usage {
            search: Search {
                queries: query.search_units.unwrap_or(0),
                ..Search::default()
            },
            ..Usage::default()
        },
    };
    units::bill(
        source,
        &query.header,
        &named,
        &usage,
        &units::call(&query.header.basis),
    )
}

#[cfg(test)]
mod parity {
    //! Parity with main's `rerank_cost`. `fixtures/rerank_parity.json` was captured once from main's Python: each case
    //! holds the model and provider, the cost fields `get_model_info` returned, the reported `billed_units` and main's
    //! `(prompt, completion)`. The adapter answers a call no row prices with `(0.0, 0.0)`, as main did

    use rstest::rstest;
    use serde_json::Value;

    use super::{RerankQuery, price};
    use crate::receipt::PriceOutcome;
    use crate::testing::{TestSource, expected_sides, fixture_case, header, units_row};

    const FIXTURE: &str = include_str!("../fixtures/rerank_parity.json");

    fn case(name: &str) -> Value {
        fixture_case(FIXTURE, name)
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
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;
    use crate::price::{InputUnit, Meter, ServiceTier};
    use crate::receipt::Unpriced;
    use crate::testing::{BASE, TestRow, TestSource, header, priced, prices};

    const QUERY: Meter = Meter::Input(InputUnit::Query { over_results: None });

    fn rerank_row() -> TestRow {
        TestRow {
            prices: prices(&[
                (QUERY, ServiceTier::Standard, BASE, 0.002),
                (Meter::TEXT_INPUT, ServiceTier::Standard, BASE, 0.00001),
            ]),
            ..TestRow::default()
        }
    }

    fn query(provider: &str, search_units: Option<u64>, total_tokens: Option<u64>) -> RerankQuery {
        RerankQuery {
            header: header(provider, "rerank-model"),
            search_units,
            total_tokens,
        }
    }

    #[rstest]
    #[case::cohere_bills_search_units("cohere", 0.002 * 3.0)]
    #[case::jina_bills_tokens("jina_ai", 0.00001 * 500.0)]
    #[case::scaleway_bills_tokens_like_jina("scaleway", 0.00001 * 500.0)]
    #[case::voyage_bills_tokens("voyage", 0.00001 * 500.0)]
    fn each_provider_bills_the_unit_its_config_reads(
        #[case] provider: &str,
        #[case] expected: f64,
    ) {
        let source = TestSource::with_row("rerank-model", rerank_row());
        let receipt = priced(price(&source, &query(provider, Some(3), Some(500))));

        assert_eq!(receipt.total, expected);
        assert_eq!(receipt.prompt.get(), expected);
        assert_eq!(receipt.completion.get(), 0.0);
    }

    #[rstest]
    fn unreported_units_bill_nothing() {
        let source = TestSource::with_row("rerank-model", rerank_row());
        assert_eq!(
            priced(price(&source, &query("cohere", None, Some(500)))).total,
            0.0
        );
    }

    #[rstest]
    fn a_row_without_the_unit_rate_is_missing_it() {
        let source = TestSource::with_row(
            "rerank-model",
            TestRow {
                prices: prices(&[(Meter::TEXT_INPUT, ServiceTier::Standard, BASE, 1.0)]),
                ..TestRow::default()
            },
        );
        let PriceOutcome::Unpriced(Unpriced::MissingRates { missing, .. }) =
            price(&source, &query("cohere", Some(2), None))
        else {
            panic!("a row without a query rate is unpriced");
        };
        assert_eq!(missing, vec![QUERY]);
    }

    #[rstest]
    fn an_unknown_model_has_no_row() {
        let source = TestSource::with_row("other", rerank_row());
        assert!(matches!(
            price(&source, &query("cohere", Some(1), None)),
            PriceOutcome::Unpriced(Unpriced::NoRow { .. })
        ));
    }
}
