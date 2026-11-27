//! `rates`: a row's rates and one priced, free or unpriced status, for the auth checks that decide whether a model
//! is free or unpriced before a call (`proxy/auth/auth_checks.py` `is_model_cost_zero`, `model_has_no_cost_mapping`
//! and their helpers) and the explicit-free check on a $0 bill (`llm_cost_calc/zero_cost_diagnostic.py`
//! `is_explicitly_free_usage`). Those predicates read raw keys, so the card hands back the rates as listed plus the
//! key facts lowering cannot recover, and the adapter rebuilds each predicate from them unchanged

use crate::money::UnitPrice;
use crate::price::{Band, Channel, Meter, Price, Prices, ServiceTier, When};
use crate::row::{Declared, LoweredRow, RowRef, RowSource};

#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize))]
pub struct RatesQuery {
    pub row: RowRef,
    /// The prompt size that picks a `tiered_pricing` tier, `None` to read the flat rates only
    #[cfg_attr(feature = "serde", serde(default))]
    pub prompt_tokens: Option<u64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RateCard {
    pub row: RowRef,
    /// Every rate the row lists, in every channel, tier and band
    pub listed: Vec<(Price, UnitPrice)>,
    /// The standard rates of the tier the prompt picked, empty when the row has no tiers or the tier prices no input
    pub tier: Vec<(Meter, UnitPrice)>,
    pub declared: Declared,
    pub pricing: Pricing,
}

/// Whether the row bills anything. `price` bills an `ExplicitlyFree` row at 0.0, and an `Unpriced` row at 0.0 with
/// every meter it used in `defaulted_to_zero`
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pricing {
    /// A rate above zero, or reserved PTU capacity billed as a flat cost
    Priced,
    /// Rates listed, every one 0.0
    ExplicitlyFree,
    /// No rate listed
    Unpriced,
}

/// `None` when the source has no row at `query.row`
pub fn rates(source: &impl RowSource, query: &RatesQuery) -> Option<RateCard> {
    let row = source.row(&query.row)?;
    let prices = row.prices();
    let flat = prices.flat();
    let tier = prices
        .tiers()
        .zip(query.prompt_tokens)
        .map(|(tiers, prompt)| selected_tier(tiers, prompt))
        .unwrap_or_default();
    let listed: Vec<(Price, UnitPrice)> = flat
        .iter()
        .chain(prices.tiers().into_iter().flat_map(Prices::iter))
        .map(|(price, rate)| (price.clone(), rate))
        .collect();
    let declared = row.facts().declared;
    let pricing = if declared.ptu_flat_cost || listed.iter().any(|(_, rate)| rate.get() > 0.0) {
        Pricing::Priced
    } else if listed.is_empty() {
        Pricing::Unpriced
    } else {
        Pricing::ExplicitlyFree
    };
    Some(RateCard {
        row: query.row.clone(),
        listed,
        tier,
        declared,
        pricing,
    })
}

impl RateCard {
    /// The rate a plain key reads: the picked tier's, else the standard online rate below every band
    pub fn rate(&self, meter: &Meter) -> Option<UnitPrice> {
        let flat = Price::online(meter.clone(), When::default());
        let tiered = self
            .tier
            .iter()
            .find(|(tiered, _)| tiered == meter)
            .map(|(_, rate)| *rate);
        tiered.or_else(|| {
            self.listed
                .iter()
                .find(|(price, _)| *price == flat)
                .map(|(_, rate)| *rate)
        })
    }
}

/// `_pricing_for_usage`: the standard online rates of the tier `select_tier_for_input` picks, when it prices input
fn selected_tier(tiers: &Prices, prompt: u64) -> Vec<(Meter, UnitPrice)> {
    let reached = |band: &Band| match band {
        Band::Base => prompt > 0,
        Band::OverPromptTokens(floor) => prompt > floor.get(),
    };
    let Some(band) = tiers
        .iter()
        .map(|(price, _)| price.when.band)
        .filter(reached)
        .max()
    else {
        return Vec::new();
    };
    let rates: Vec<(Meter, UnitPrice)> = tiers
        .iter()
        .filter(|(price, _)| {
            price.channel == Channel::Online
                && price.when.tier == ServiceTier::Standard
                && price.when.band == band
        })
        .map(|(price, rate)| (price.meter.clone(), rate))
        .collect();
    if rates.iter().any(|(meter, _)| *meter == Meter::TEXT_INPUT) {
        rates
    } else {
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;
    use crate::price::ServiceTier::Standard;
    use crate::row::RowFacts;
    use crate::testing::{BASE, TestRow, TestSource, over, prices, row_ref, text_row};

    fn card(row: TestRow, prompt_tokens: Option<u64>) -> RateCard {
        rates(
            &TestSource::with_row("row", row),
            &RatesQuery {
                row: row_ref("row"),
                prompt_tokens,
            },
        )
        .unwrap()
    }

    #[rstest]
    #[case::priced(text_row(0.0, 1.0), false, Pricing::Priced)]
    #[case::free(text_row(0.0, 0.0), false, Pricing::ExplicitlyFree)]
    #[case::ptu_is_not_free(text_row(0.0, 0.0), true, Pricing::Priced)]
    #[case::nothing_listed(TestRow::default(), false, Pricing::Unpriced)]
    #[case::a_paid_image_price_is_not_free(
        TestRow {
            prices: prices(&[
                (Meter::TEXT_INPUT, Standard, BASE, 0.0),
                (Meter::Output(crate::price::OutputUnit::Image(None)), Standard, BASE, 0.04),
            ]),
            ..TestRow::default()
        },
        false,
        Pricing::Priced
    )]
    fn pricing_is_free_only_when_every_listed_rate_is_zero(
        #[case] row: TestRow,
        #[case] ptu: bool,
        #[case] expected: Pricing,
    ) {
        let row = TestRow {
            facts: RowFacts {
                declared: Declared {
                    ptu_flat_cost: ptu,
                    ..Declared::default()
                },
                ..RowFacts::default()
            },
            ..row
        };
        assert_eq!(card(row, None).pricing, expected);
    }

    #[rstest]
    #[case::no_prompt_reads_flat(None, Some(1.0))]
    #[case::empty_prompt_picks_no_tier(Some(0), Some(1.0))]
    #[case::low_tier(Some(10), Some(2.0))]
    #[case::high_tier(Some(300), Some(3.0))]
    fn a_tiered_row_reads_the_tier_the_prompt_picks(
        #[case] prompt: Option<u64>,
        #[case] expected: Option<f64>,
    ) {
        let row = TestRow {
            tiers: Some(prices(&[
                (Meter::TEXT_INPUT, Standard, BASE, 2.0),
                (Meter::TEXT_INPUT, Standard, over(256), 3.0),
            ])),
            ..text_row(1.0, 5.0)
        };
        let card = card(row, prompt);
        assert_eq!(card.rate(&Meter::TEXT_INPUT).map(UnitPrice::get), expected);
        assert_eq!(
            card.rate(&Meter::TEXT_OUTPUT).map(UnitPrice::get),
            Some(5.0)
        );
    }

    #[rstest]
    fn a_band_is_not_a_plain_key() {
        let row = TestRow {
            prices: prices(&[(Meter::TEXT_INPUT, Standard, over(200_000), 2.0)]),
            ..TestRow::default()
        };
        assert_eq!(card(row, Some(300_000)).rate(&Meter::TEXT_INPUT), None);
    }

    #[rstest]
    fn a_missing_row_has_no_card() {
        assert_eq!(
            rates(
                &TestSource::default(),
                &RatesQuery {
                    row: row_ref("missing"),
                    prompt_tokens: None
                }
            ),
            None
        );
    }
}
