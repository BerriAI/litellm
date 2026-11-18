//! The `vector_store_search` kind, a port of the vector store configs' `calculate_vector_store_cost`: each search is
//! one query at the row's `input_cost_per_query`. The caller names the row (`vertex_ai/search_api`), and a search no
//! config prices names none

use crate::query::Header;
use crate::receipt::PriceOutcome;
use crate::row::RowSource;
use crate::units;
use crate::usage::{Search, Usage};

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize))]
pub struct VectorStoreSearchQuery {
    #[cfg_attr(feature = "serde", serde(flatten))]
    pub header: Header,
    pub queries: u64,
}

pub(crate) fn price<S: RowSource>(source: &S, query: &VectorStoreSearchQuery) -> PriceOutcome {
    let Some(named) = units::named(source, &query.header) else {
        return units::no_row(&query.header);
    };
    let usage = Usage {
        search: Search {
            queries: query.queries,
            ..Search::default()
        },
        ..Usage::default()
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
mod tests {
    use rstest::rstest;

    use super::*;
    use crate::price::{InputUnit, Meter, ServiceTier};
    use crate::receipt::Unpriced;
    use crate::testing::{BASE, TestRow, TestSource, header, priced, prices};

    #[rstest]
    fn a_search_bills_its_queries_at_the_query_rate() {
        let source = TestSource::with_row(
            "vertex_ai/search_api",
            TestRow {
                prices: prices(&[(
                    Meter::Input(InputUnit::Query { over_results: None }),
                    ServiceTier::Standard,
                    BASE,
                    0.0015,
                )]),
                ..TestRow::default()
            },
        );
        let receipt = priced(price(
            &source,
            &VectorStoreSearchQuery {
                header: header("vertex_ai", "vertex_ai/search_api"),
                queries: 1,
            },
        ));
        assert_eq!(
            (receipt.prompt.get(), receipt.completion.get()),
            (0.0015, 0.0)
        );
    }

    #[rstest]
    fn a_search_no_row_prices_has_no_row() {
        let outcome = price(
            &TestSource::default(),
            &VectorStoreSearchQuery {
                header: header("vertex_ai", "vertex_ai/search_api"),
                queries: 1,
            },
        );
        assert!(matches!(
            outcome,
            PriceOutcome::Unpriced(Unpriced::NoRow { .. })
        ));
    }
}
