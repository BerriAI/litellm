//! The `mcp` kind, a port of `MCPCostCalculator.calculate_mcp_tool_call_cost` and `mcp_tool_cost_per_query`
//! (`proxy/mcp_server/cost_calculator.py`) after the hook-set `response_cost`, which the adapter reads first. A tool
//! the server prices on its own bills that price, any other tool the server's default price. A server with neither
//! is unpriced and the adapter bills it 0.0, as main did. Main applies no discount or margin to a tool call

use std::collections::BTreeMap;

use crate::money::UnitPrice;
use crate::price::{InputUnit, Meter};
use crate::priced::{Quantity, Side};
use crate::query::Header;
use crate::receipt::PriceOutcome;
use crate::row::RowSource;
use crate::units::{self, Billed};

const QUERY: Meter = Meter::Input(InputUnit::Query { over_results: None });

/// One tool call and its server's `mcp_server_cost_info`
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize))]
pub struct McpQuery {
    #[cfg_attr(feature = "serde", serde(flatten))]
    pub header: Header,
    pub tool: Box<str>,
    /// `tool_name_to_cost_per_query`
    #[cfg_attr(feature = "serde", serde(default))]
    pub tool_prices: BTreeMap<Box<str>, UnitPrice>,
    /// `default_cost_per_query`
    #[cfg_attr(feature = "serde", serde(default))]
    pub default_price: Option<UnitPrice>,
}

pub(crate) fn price<S: RowSource>(source: &S, query: &McpQuery) -> PriceOutcome {
    let Some(rate) = query
        .tool_prices
        .get(&query.tool)
        .or(query.default_price.as_ref())
    else {
        return units::no_row(&query.header);
    };
    let line = units::line(QUERY, Quantity::Count(1), *rate, Side::Prompt);
    units::receipt(
        source,
        &query.header,
        None,
        Billed::of(vec![line], Vec::new()),
    )
    .into()
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;
    use crate::receipt::Unpriced;
    use crate::testing::{TestSource, header, priced};

    fn query(tool: &str, default_price: Option<f64>) -> McpQuery {
        McpQuery {
            header: header("mcp", "search"),
            tool: tool.into(),
            tool_prices: BTreeMap::from([("search_web".into(), UnitPrice::new(0.05).unwrap())]),
            default_price: default_price.and_then(UnitPrice::new),
        }
    }

    #[rstest]
    #[case::listed_tool_beats_the_default("search_web", Some(0.01), 0.05)]
    #[case::listed_tool_without_a_default("search_web", None, 0.05)]
    #[case::other_tool_bills_the_default("generate_code", Some(0.01), 0.01)]
    #[case::a_zero_default_is_free("generate_code", Some(0.0), 0.0)]
    fn a_tool_call_bills_its_tool_price_or_the_default(
        #[case] tool: &str,
        #[case] default_price: Option<f64>,
        #[case] expected: f64,
    ) {
        let receipt = priced(price(&TestSource::default(), &query(tool, default_price)));
        assert_eq!(receipt.total, expected);
    }

    #[rstest]
    fn a_server_without_a_price_for_the_tool_is_unpriced() {
        assert!(matches!(
            price(&TestSource::default(), &query("generate_code", None)),
            PriceOutcome::Unpriced(Unpriced::NoRow { .. })
        ));
    }
}
