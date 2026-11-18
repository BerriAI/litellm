//! The `guardrail` kind, a port of `bedrock_guardrail_cost_by_unit` and `azure_prompt_shield_guardrail_cost`
//! (`llm_cost_calc/guardrail_cost.py`). Each usage counter is one part of the receipt. A Bedrock guardrail reads the
//! first of the header's rows that lists `guardrail_cost_per_unit` (main tries the region's row, then the default
//! one), and a counter that row has no price for stays an unpriced part, which the totals leave out but the rollup
//! never reads as free. Azure Prompt Shield bills its text records per thousand at the guardrail's configured price
//! and nothing on the free tier. Main applies no discount or margin to a guardrail

use crate::money::{UnitPrice, Usd};
use crate::price::{Channel, GuardrailUnit, Meter};
use crate::priced::{Line, Quantity, Side};
use crate::query::{Header, PriceSource};
use crate::receipt::{Part, PartOutcome, PriceOutcome, Receipt};
use crate::row::{LoweredRow, RowRef, RowSource};
use crate::units::{self, Billed};

/// `AZURE_PROMPT_SHIELD_TEXT_RECORD_UNIT`
const TEXT_RECORDS: &str = "text_records";

/// The counters one guardrail run reported, in the order its usage lists them
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize))]
pub struct GuardrailQuery {
    /// `Rows` lists the pricing rows in the order main tries them
    #[cfg_attr(feature = "serde", serde(flatten))]
    pub header: Header,
    pub units: Vec<GuardrailUse>,
    /// Set for Azure Prompt Shield, which a guardrail's config prices instead of a row
    #[cfg_attr(feature = "serde", serde(default))]
    pub prompt_shield: Option<PromptShield>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize))]
pub struct GuardrailUse {
    pub counter: Box<str>,
    pub units: u64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize))]
pub struct PromptShield {
    /// `cost_tier: free`
    pub free_tier: bool,
    /// `price_per_1000_text_records`
    #[cfg_attr(feature = "serde", serde(default))]
    pub per_thousand: Option<UnitPrice>,
}

pub(crate) fn price<S: RowSource>(source: &S, query: &GuardrailQuery) -> PriceOutcome {
    match query.prompt_shield {
        Some(shield) => prompt_shield(source, query, shield),
        None => by_counter(source, query),
    }
}

fn by_counter<S: RowSource>(source: &S, query: &GuardrailQuery) -> PriceOutcome {
    let rows: &[RowRef] = match &query.header.source {
        PriceSource::Rows { rows } => rows,
        _ => &[],
    };
    let Some((at, row)) = rows.iter().find_map(|at| {
        source
            .row(at)
            .filter(|row| row.facts().guardrail_units)
            .map(|row| (at, row))
    }) else {
        return units::no_row(&query.header);
    };
    let priced: Vec<(Part, Option<Line>)> = query
        .units
        .iter()
        .map(|used| {
            let meter = Meter::Guardrail(GuardrailUnit::new(&used.counter));
            let line = units::listed(row, &meter, Channel::Online)
                .map(|rate| units::line(meter, Quantity::Count(used.units), rate, Side::Prompt));
            let outcome = match &line {
                Some(line) => PartOutcome::Priced {
                    row: Some(at.clone()),
                    amount: line.amount,
                },
                None => PartOutcome::Unpriced,
            };
            (
                Part {
                    label: used.counter.clone(),
                    outcome,
                },
                line,
            )
        })
        .collect();
    let (parts, lines): (Vec<Part>, Vec<Option<Line>>) = priced.into_iter().unzip();
    Receipt {
        row: Some(at.clone()),
        parts,
        ..units::receipt(
            source,
            &query.header,
            None,
            Billed::of(lines.into_iter().flatten().collect(), Vec::new()),
        )
    }
    .into()
}

fn prompt_shield<S: RowSource>(
    source: &S,
    query: &GuardrailQuery,
    shield: PromptShield,
) -> PriceOutcome {
    let records = query
        .units
        .iter()
        .find(|used| &*used.counter == TEXT_RECORDS)
        .map_or(0, |used| used.units);
    let line = match (shield.free_tier, shield.per_thousand) {
        (true, _) => None,
        (false, None) => return units::no_row(&query.header),
        (false, Some(rate)) => Some(Line {
            meter: Meter::Guardrail(GuardrailUnit::new(TEXT_RECORDS)),
            quantity: Quantity::Count(records),
            unit_price: rate,
            amount: Usd::new(records as f64 * rate.get() / 1000.0).unwrap_or_default(),
            fallback_from: None,
            side: Side::Prompt,
        }),
    };
    let part = Part {
        label: TEXT_RECORDS.into(),
        outcome: PartOutcome::Priced {
            row: None,
            amount: line.as_ref().map_or(Usd::ZERO, |line| line.amount),
        },
    };
    Receipt {
        parts: vec![part],
        ..units::receipt(
            source,
            &query.header,
            None,
            Billed::of(line.into_iter().collect(), Vec::new()),
        )
    }
    .into()
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;
    use crate::price::ServiceTier;
    use crate::receipt::Unpriced;
    use crate::row::RowFacts;
    use crate::testing::{BASE, TestRow, TestSource, header, priced, prices, row_ref};

    const DEFAULT: &str = "bedrock/guardrails";
    const REGIONAL: &str = "bedrock/eu-west-1/guardrails";

    fn guardrail_row(rates: &[(&str, f64)]) -> TestRow {
        TestRow {
            prices: prices(
                &rates
                    .iter()
                    .map(|(unit, rate)| {
                        (
                            Meter::Guardrail(GuardrailUnit::new(unit)),
                            ServiceTier::Standard,
                            BASE,
                            *rate,
                        )
                    })
                    .collect::<Vec<_>>(),
            ),
            facts: RowFacts {
                guardrail_units: true,
                ..RowFacts::default()
            },
            ..TestRow::default()
        }
    }

    fn source() -> TestSource {
        TestSource::with_row(
            DEFAULT,
            guardrail_row(&[("contentPolicyUnits", 0.00015), ("wordPolicyUnits", 0.0)]),
        )
        .and_row(REGIONAL, guardrail_row(&[("contentPolicyUnits", 0.0002)]))
        .and_row("bedrock/us-west-2/guardrails", TestRow::default())
    }

    fn query(rows: &[&str], units: &[(&str, u64)]) -> GuardrailQuery {
        GuardrailQuery {
            header: Header {
                source: PriceSource::Rows {
                    rows: rows.iter().map(|key| row_ref(key)).collect(),
                },
                ..header("bedrock", "guardrails")
            },
            units: units
                .iter()
                .map(|(counter, units)| GuardrailUse {
                    counter: (*counter).into(),
                    units: *units,
                })
                .collect(),
            prompt_shield: None,
        }
    }

    fn costs(receipt: &Receipt) -> Vec<(&str, Option<f64>)> {
        receipt
            .parts
            .iter()
            .map(|part| (&*part.label, part.amount().map(Usd::get)))
            .collect()
    }

    #[rstest]
    fn each_counter_is_a_part_and_an_unlisted_one_stays_unpriced() {
        let receipt = priced(price(
            &source(),
            &query(
                &[DEFAULT],
                &[
                    ("contentPolicyUnits", 2),
                    ("wordPolicyUnits", 5),
                    ("someFutureCounter", 3),
                ],
            ),
        ));
        assert_eq!(
            costs(&receipt),
            vec![
                ("contentPolicyUnits", Some(2.0 * 0.00015)),
                ("wordPolicyUnits", Some(0.0)),
                ("someFutureCounter", None),
            ]
        );
        assert_eq!(receipt.total, 2.0 * 0.00015);
    }

    #[rstest]
    #[case::the_region_row_wins(&[REGIONAL, DEFAULT], 0.0002)]
    #[case::a_row_without_guardrail_prices_is_skipped(&["bedrock/us-west-2/guardrails", DEFAULT], 0.00015)]
    #[case::a_missing_row_is_skipped(&["bedrock/ap-south-1/guardrails", DEFAULT], 0.00015)]
    fn the_first_row_that_prices_guardrails_prices_every_counter(
        #[case] rows: &[&str],
        #[case] expected: f64,
    ) {
        let receipt = priced(price(&source(), &query(rows, &[("contentPolicyUnits", 1)])));
        assert_eq!(receipt.total, expected);
    }

    #[rstest]
    fn no_row_that_prices_guardrails_is_unpriced() {
        assert!(matches!(
            price(
                &source(),
                &query(
                    &["bedrock/us-west-2/guardrails"],
                    &[("contentPolicyUnits", 1)]
                )
            ),
            PriceOutcome::Unpriced(Unpriced::NoRow { .. })
        ));
    }

    fn shield(
        free_tier: bool,
        per_thousand: Option<f64>,
        records: &[(&str, u64)],
    ) -> GuardrailQuery {
        GuardrailQuery {
            prompt_shield: Some(PromptShield {
                free_tier,
                per_thousand: per_thousand.and_then(UnitPrice::new),
            }),
            ..query(&[], records)
        }
    }

    #[rstest]
    #[case::paid_tier(false, Some(0.38), &[("text_records", 2500)], 2500.0 * 0.38 / 1000.0)]
    #[case::free_tier(true, Some(0.38), &[("text_records", 50)], 0.0)]
    #[case::no_records(false, Some(0.38), &[], 0.0)]
    fn prompt_shield_bills_text_records_per_thousand(
        #[case] free_tier: bool,
        #[case] per_thousand: Option<f64>,
        #[case] records: &[(&str, u64)],
        #[case] expected: f64,
    ) {
        let receipt = priced(price(
            &TestSource::default(),
            &shield(free_tier, per_thousand, records),
        ));
        assert_eq!(receipt.total, expected);
        assert_eq!(costs(&receipt), vec![("text_records", Some(expected))]);
    }

    #[rstest]
    fn prompt_shield_without_a_price_is_unpriced() {
        assert!(matches!(
            price(
                &TestSource::default(),
                &shield(false, None, &[("text_records", 50)])
            ),
            PriceOutcome::Unpriced(Unpriced::NoRow { .. })
        ));
    }
}
