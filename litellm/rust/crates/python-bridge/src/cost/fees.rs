//! The fee kinds Python prices through `NativeModelCostMap.price_fee`: an A2A agent call, an MCP tool call and a
//! guardrail run. Each document carries the prices the caller configured or the catalog keys main read, and the usage
//! the call reported. Main's precedence between an agent's prices lives here, the hook-set `response_cost` that beats
//! them all stays in Python

use std::collections::BTreeMap;

use litellm_cost::money::UnitPrice;
use litellm_cost::query::{
    A2aQuery, AdjustmentPolicy, AgentUsage, Basis, FlatRates, GuardrailQuery, GuardrailUse, Header,
    McpQuery, PriceQuery, PriceSource, PromptShield,
};
use litellm_host_python::from_py_argument;
use pyo3::prelude::*;
use serde::Deserialize;

use super::kinds::outcome;
use crate::model_cost::CostSnapshot;

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum FeeDocument {
    A2a {
        #[serde(default)]
        cost_per_query: Option<UnitPrice>,
        #[serde(default)]
        input_cost_per_token: Option<UnitPrice>,
        #[serde(default)]
        output_cost_per_token: Option<UnitPrice>,
        #[serde(default)]
        usage: Option<AgentUsage>,
    },
    Mcp {
        tool: Box<str>,
        #[serde(default)]
        tool_prices: BTreeMap<Box<str>, UnitPrice>,
        #[serde(default)]
        default_price: Option<UnitPrice>,
    },
    Guardrail {
        /// The catalog keys main reads the pricing from, in order
        keys: Vec<String>,
        units: Vec<GuardrailUse>,
    },
    PromptShield {
        units: Vec<GuardrailUse>,
        free_tier: bool,
        #[serde(default)]
        per_thousand: Option<UnitPrice>,
    },
}

/// Prices one fee from the facts in `document`, a receipt or why it is unpriced
pub(crate) fn price_fee(
    py: Python<'_>,
    snapshot: &CostSnapshot,
    document: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let document: FeeDocument = from_py_argument(document)?;
    let header = |source| Header {
        candidates: Vec::new(),
        source,
        basis: Basis {
            service_tier: Default::default(),
            data_residency: None,
            vertex_location: None,
            at: jiff::Timestamp::now(),
        },
        adjustments: AdjustmentPolicy::None,
        reported: None,
    };
    let (rows, query) = match document {
        FeeDocument::A2a {
            cost_per_query,
            input_cost_per_token,
            output_cost_per_token,
            usage,
        } => (
            snapshot.pricing_rows([], []),
            PriceQuery::A2a(Box::new(A2aQuery {
                header: header(agent_prices(
                    cost_per_query,
                    input_cost_per_token,
                    output_cost_per_token,
                )),
                usage,
            })),
        ),
        FeeDocument::Mcp {
            tool,
            tool_prices,
            default_price,
        } => (
            snapshot.pricing_rows([], []),
            PriceQuery::Mcp(Box::new(McpQuery {
                header: header(PriceSource::Catalog),
                tool,
                tool_prices,
                default_price,
            })),
        ),
        FeeDocument::Guardrail { keys, units } => {
            let found: Vec<_> = keys.iter().filter_map(|key| snapshot.row_at(key)).collect();
            let rows = found
                .iter()
                .map(litellm_model_catalog::priced_ref)
                .collect();
            (
                snapshot.pricing_rows([], found),
                PriceQuery::Guardrail(Box::new(GuardrailQuery {
                    header: header(PriceSource::Rows { rows }),
                    units,
                    prompt_shield: None,
                })),
            )
        }
        FeeDocument::PromptShield {
            units,
            free_tier,
            per_thousand,
        } => (
            snapshot.pricing_rows([], []),
            PriceQuery::Guardrail(Box::new(GuardrailQuery {
                header: header(PriceSource::Catalog),
                units,
                prompt_shield: Some(PromptShield {
                    free_tier,
                    per_thousand,
                }),
            })),
        ),
    };
    outcome(py, litellm_cost::price(&rows, &query))
}

/// `calculate_a2a_cost`: `cost_per_query` first, then the token prices, a missing direction billed at 0.0
fn agent_prices(
    per_query: Option<UnitPrice>,
    input: Option<UnitPrice>,
    output: Option<UnitPrice>,
) -> PriceSource {
    let free = UnitPrice::new(0.0).expect("zero is a price");
    match (per_query, input, output) {
        (Some(rate), _, _) => PriceSource::PerQuery { rate },
        (None, None, None) => PriceSource::Catalog,
        (None, input, output) => PriceSource::FlatPerToken(FlatRates {
            input: input.unwrap_or(free),
            output: output.unwrap_or(free),
            cache_read: None,
            cache_write: None,
        }),
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    fn rate(usd: f64) -> Option<UnitPrice> {
        UnitPrice::new(usd)
    }

    #[rstest]
    #[case::query_price_wins(rate(0.5), rate(0.1), None, PriceSource::PerQuery { rate: rate(0.5).unwrap() })]
    #[case::no_price(None, None, None, PriceSource::Catalog)]
    #[case::input_only(None, rate(0.1), None, PriceSource::FlatPerToken(FlatRates {
        input: rate(0.1).unwrap(),
        output: rate(0.0).unwrap(),
        cache_read: None,
        cache_write: None,
    }))]
    fn an_agent_is_priced_by_query_then_by_token(
        #[case] per_query: Option<UnitPrice>,
        #[case] input: Option<UnitPrice>,
        #[case] output: Option<UnitPrice>,
        #[case] expected: PriceSource,
    ) {
        assert_eq!(agent_prices(per_query, input, output), expected);
    }
}
