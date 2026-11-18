//! A Decisions call (`/v1/decisions`, `/v1/systemone`), main's `decisions_request_price` and the `_decisions` row
//! fields `cost_per_token` lays over the plain ones (`litellm_core_utils/decisions_pricing.py`)

use crate::money::UnitPrice;
use crate::price::{Band, Channel, InputUnit, Meter, Price, Prices, ServiceTier, When};
use crate::priced::{Line, Quantity, Side};
use crate::row::LoweredRow;

use super::branch::Sides;

/// `QUERY_PRICED_DECISIONS_PROVIDERS`: providers whose Decisions calls bill a flat price per request
const QUERY_PRICED: [&str; 1] = ["parallel_ai"];

const QUERY: Meter = Meter::Input(InputUnit::Query { over_results: None });

/// Whether the row lists any `_decisions` price
pub(super) fn prices_decisions(prices: &Prices) -> bool {
    prices
        .iter()
        .any(|(price, _)| price.channel == Channel::Decisions)
}

fn base(prices: &Prices, meter: Meter, channel: Channel) -> Option<UnitPrice> {
    prices.get(&Price {
        meter,
        channel,
        when: When {
            tier: ServiceTier::Standard,
            band: Band::Base,
        },
    })
}

/// `decisions_request_price`: a query-priced provider bills the request at its query rate, unless the row bills tokens
/// and the query rate is not a `_decisions` one
pub(super) fn request_price(row: &impl LoweredRow, prices: &Prices) -> Option<UnitPrice> {
    let provider = row.facts().provider.as_deref()?;
    if !QUERY_PRICED.contains(&provider) {
        return None;
    }
    let view = prices.decisions_view();
    let rate = base(&view, QUERY, Channel::Online)?;
    let bills_tokens = [Meter::TEXT_INPUT, Meter::TEXT_OUTPUT]
        .into_iter()
        .any(|meter| base(&view, meter, Channel::Online).is_some_and(|rate| rate.get() != 0.0));
    let own = base(prices, QUERY, Channel::Decisions).is_some();
    (!bills_tokens || own).then_some(rate)
}

/// One request at `rate`, the prompt side as main returned it
pub(super) fn request(rate: UnitPrice) -> Sides {
    let amount = rate * Quantity::Count(1);
    Sides {
        lines: vec![Line {
            meter: QUERY,
            quantity: Quantity::Count(1),
            unit_price: rate,
            amount,
            fallback_from: None,
            side: Side::Prompt,
        }],
        defaulted_to_zero: Vec::new(),
        prompt: amount,
        completion: crate::money::Usd::ZERO,
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;
    use crate::price::Billing;
    use crate::row::RowFacts;
    use crate::testing::{BASE, TestRow};

    fn prices_with(rates: &[(Price, f64)]) -> Prices {
        Prices::new(
            rates
                .iter()
                .map(|(price, rate)| (price.clone(), UnitPrice::new(*rate).unwrap()))
                .collect(),
            None,
            Billing::default(),
        )
    }

    fn decisions(meter: Meter) -> Price {
        Price {
            meter,
            channel: Channel::Decisions,
            when: When::default(),
        }
    }

    fn online(meter: Meter) -> Price {
        Price::online(meter, When::default())
    }

    fn row(provider: &str, rates: &[(Price, f64)]) -> TestRow {
        TestRow {
            prices: prices_with(rates),
            facts: RowFacts {
                provider: Some(provider.into()),
                ..RowFacts::default()
            },
            ..TestRow::default()
        }
    }

    #[rstest]
    #[case::query_rate_alone(row("parallel_ai", &[(online(QUERY), 0.004)]), Some(0.004))]
    #[case::own_decisions_query_rate_beats_token_rates(
        row("parallel_ai", &[(online(Meter::TEXT_INPUT), 1e-6), (decisions(QUERY), 0.002)]),
        Some(0.002)
    )]
    #[case::plain_query_rate_loses_to_token_rates(
        row("parallel_ai", &[(online(Meter::TEXT_INPUT), 1e-6), (online(QUERY), 0.004)]),
        None
    )]
    #[case::decisions_zeroes_the_token_rate(
        row("parallel_ai", &[(online(Meter::TEXT_INPUT), 1e-6), (decisions(Meter::TEXT_INPUT), 0.0), (online(QUERY), 0.004)]),
        Some(0.004)
    )]
    #[case::other_provider(row("openai", &[(online(QUERY), 0.004)]), None)]
    fn a_query_priced_provider_bills_the_request(
        #[case] row: TestRow,
        #[case] expected: Option<f64>,
    ) {
        let prices = row.prices.clone();
        assert_eq!(request_price(&row, &prices).map(UnitPrice::get), expected);
    }

    #[rstest]
    fn the_decisions_view_lays_decisions_prices_over_the_online_ones_and_keeps_the_rest() {
        let priority = Price::online(
            Meter::TEXT_OUTPUT,
            When {
                tier: ServiceTier::Priority,
                band: BASE,
            },
        );
        let prices = prices_with(&[
            (online(Meter::TEXT_INPUT), 1e-7),
            (online(Meter::TEXT_OUTPUT), 5e-7),
            (decisions(Meter::TEXT_OUTPUT), 0.0),
            (priority.clone(), 1e-6),
        ]);
        let view = prices.decisions_view();
        assert_eq!(
            view.get(&online(Meter::TEXT_INPUT)).map(UnitPrice::get),
            Some(1e-7)
        );
        assert_eq!(
            view.get(&online(Meter::TEXT_OUTPUT)).map(UnitPrice::get),
            Some(0.0)
        );
        assert_eq!(view.get(&priority).map(UnitPrice::get), Some(1e-6));
        assert!(prices_decisions(&prices));
        assert!(!prices_decisions(&prices_with(&[(
            online(Meter::TEXT_INPUT),
            1e-7
        )])));
    }
}
