//! The `a2a` kind, a port of `A2ACostCalculator.calculate_a2a_cost` (`a2a_protocol/cost_calculator.py`) after the
//! hook-set `response_cost`, which the adapter reads first. An agent's configured `cost_per_query` bills the call
//! once, else its configured token prices bill the usage the call reported. An agent with neither price is unpriced
//! and the adapter bills it 0.0, as main did. Main applies no discount or margin to an agent call

use crate::price::{InputUnit, Meter};
use crate::priced::{Line, Quantity, Side};
use crate::query::{FlatRates, Header, PriceSource};
use crate::receipt::{PriceOutcome, Unpriced};
use crate::row::RowSource;
use crate::units::{self, Billed};

const REQUEST: Meter = Meter::Input(InputUnit::Request);

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize))]
pub struct A2aQuery {
    /// `PerQuery` for `cost_per_query`, `FlatPerToken` for the agent's token prices
    #[cfg_attr(feature = "serde", serde(flatten))]
    pub header: Header,
    /// `None` when the call logged no usage
    #[cfg_attr(feature = "serde", serde(default))]
    pub usage: Option<AgentUsage>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize))]
pub struct AgentUsage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
}

pub(crate) fn price<S: RowSource>(source: &S, query: &A2aQuery) -> PriceOutcome {
    let lines = match (&query.header.source, query.usage) {
        (PriceSource::PerQuery { rate }, _) => {
            vec![units::line(
                REQUEST,
                Quantity::Count(1),
                *rate,
                Side::Prompt,
            )]
        }
        (PriceSource::FlatPerToken(rates), Some(usage)) => tokens(rates, usage),
        (PriceSource::FlatPerToken(_), None) => {
            return Unpriced::MissingUsage {
                row: None,
                meter: Meter::TEXT_INPUT,
            }
            .into();
        }
        _ => return units::no_row(&query.header),
    };
    units::receipt(source, &query.header, None, Billed::of(lines, Vec::new())).into()
}

fn tokens(rates: &FlatRates, usage: AgentUsage) -> Vec<Line> {
    vec![
        units::line(
            Meter::TEXT_INPUT,
            Quantity::Count(usage.prompt_tokens),
            rates.input,
            Side::Prompt,
        ),
        units::line(
            Meter::TEXT_OUTPUT,
            Quantity::Count(usage.completion_tokens),
            rates.output,
            Side::Completion,
        ),
    ]
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;
    use crate::money::UnitPrice;
    use crate::testing::{TestSource, header, priced};

    fn query(source: PriceSource, usage: Option<AgentUsage>) -> A2aQuery {
        A2aQuery {
            header: Header {
                source,
                ..header("a2a_agent", "agent")
            },
            usage,
        }
    }

    fn rate(usd: f64) -> UnitPrice {
        UnitPrice::new(usd).unwrap()
    }

    fn per_token(input: f64, output: f64) -> PriceSource {
        PriceSource::FlatPerToken(FlatRates {
            input: rate(input),
            output: rate(output),
            cache_read: None,
            cache_write: None,
        })
    }

    const USAGE: AgentUsage = AgentUsage {
        prompt_tokens: 120,
        completion_tokens: 30,
    };

    #[rstest]
    #[case::with_usage(Some(USAGE))]
    #[case::without_usage(None)]
    fn a_query_price_bills_the_call_once_whatever_it_used(#[case] usage: Option<AgentUsage>) {
        let receipt = priced(price(
            &TestSource::default(),
            &query(PriceSource::PerQuery { rate: rate(0.25) }, usage),
        ));
        assert_eq!(receipt.total, 0.25);
    }

    #[rstest]
    fn token_prices_bill_each_direction_at_its_rate() {
        let receipt = priced(price(
            &TestSource::default(),
            &query(per_token(0.001, 0.002), Some(USAGE)),
        ));
        assert_eq!(
            (receipt.prompt.get(), receipt.completion.get()),
            (120.0 * 0.001, 30.0 * 0.002)
        );
        assert_eq!(receipt.total, 120.0 * 0.001 + 30.0 * 0.002);
    }

    #[rstest]
    fn token_prices_without_usage_are_unpriced() {
        assert_eq!(
            price(
                &TestSource::default(),
                &query(per_token(0.001, 0.002), None)
            ),
            Unpriced::MissingUsage {
                row: None,
                meter: Meter::TEXT_INPUT
            }
            .into()
        );
    }

    #[rstest]
    fn an_agent_without_a_price_is_unpriced() {
        assert!(matches!(
            price(
                &TestSource::default(),
                &query(PriceSource::Catalog, Some(USAGE))
            ),
            PriceOutcome::Unpriced(Unpriced::NoRow { .. })
        ));
    }
}
