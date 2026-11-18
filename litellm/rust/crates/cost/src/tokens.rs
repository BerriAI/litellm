//! The `tokens` kind: a chat, Messages or Responses call billed by its prompt and completion tokens, a port of the
//! token path of `completion_cost` (`cost_calculator.py`) for the P1 providers. The first candidate with a row
//! prices the call, as main's candidate loop moves on only when pricing raises. A rate the row lacks is billed at
//! 0.0 and listed in `defaulted_to_zero`, as main's `get_model_info` defaults it. Providers and cases not ported
//! yet answer `NotPorted`, so the caller prices them the old way

mod breakdown;
mod custom;
mod rule;

use crate::adjust::{Discount, Margin};
use crate::calculate;
use crate::call::CallContext;
use crate::money::Usd;
use crate::priced::Outcome;
use crate::query::{Candidate, Kind, PriceSource, TokensQuery};
use crate::quote::Quote;
use crate::receipt::{PriceOutcome, Receipt, Unpriced};
use crate::row::{LoweredRow, RowRef, RowSource};

use rule::{Lookup, Rule};

pub(crate) fn price<S: RowSource>(source: &S, query: &TokensQuery) -> PriceOutcome {
    let header = &query.header;
    if header.reported.is_some() {
        return Unpriced::NotPorted(Kind::Tokens).into();
    }
    let no_row = || {
        Unpriced::NoRow {
            tried: header.candidates.clone(),
        }
        .into()
    };
    match &header.source {
        PriceSource::FlatPerToken(rates) => custom::price(source, query, rates),
        PriceSource::PerSecond { .. } | PriceSource::Rows { .. } => {
            Unpriced::NotPorted(Kind::Tokens).into()
        }
        PriceSource::Row {
            row: at,
            breakdown_by_name,
        } => source
            .row(at)
            .and_then(|row| {
                let lookup = Lookup::Given {
                    breakdown_by_name: *breakdown_by_name,
                };
                header
                    .candidates
                    .iter()
                    .find_map(|candidate| attempt(source, query, candidate, at, row, lookup))
            })
            .unwrap_or_else(no_row),
        PriceSource::Catalog => header
            .candidates
            .iter()
            .find_map(|candidate| {
                let at = source.resolve(candidate)?;
                let row = source.row(&at)?;
                attempt(source, query, candidate, &at, row, Lookup::Catalog)
            })
            .unwrap_or_else(no_row),
    }
}

/// Prices the call under one candidate's row. `None` where main's pricing raises, so the next candidate is tried
fn attempt<S: RowSource>(
    source: &S,
    query: &TokensQuery,
    candidate: &Candidate,
    at: &RowRef,
    row: &S::Row,
    lookup: Lookup,
) -> Option<PriceOutcome> {
    let rule = match Rule::new(candidate, at, row, lookup, query) {
        Ok(rule) => rule,
        Err(kind) => return Some(Unpriced::NotPorted(kind).into()),
    };
    let basis = &query.header.basis;
    let call = rule.call(basis, &query.usage, &row.facts().provider_entry)?;
    let usage = query.usage.billed();
    let prices = row.prices().for_prompt(usage.prompt_tokens);
    let quote = prices.quote(&call, usage.prompt_tokens);
    let (lines, defaulted_to_zero, prompt, completion) =
        match calculate(prices, row.adjustments(), &usage, &call) {
            Outcome::Priced(priced) => (
                priced.lines,
                priced.missing,
                priced.prompt,
                priced.completion,
            ),
            Outcome::Unpriced { missing } => (Vec::new(), missing, Usd::ZERO, Usd::ZERO),
        };
    let named = source
        .resolve(candidate)
        .and_then(|named| source.row(&named));
    let tools = match named.map(|named| {
        rule.web_search_fee(&quote_of(named, &call, usage.prompt_tokens), &query.usage)
    }) {
        Some(Err(kind)) => return Some(Unpriced::NotPorted(kind).into()),
        Some(Ok(fee)) => fee,
        None => Usd::ZERO,
    };
    let breakdown_row = match lookup {
        Lookup::Given {
            breakdown_by_name: true,
        } => named,
        Lookup::Catalog | Lookup::Given { .. } => Some(row),
    };
    let rates = breakdown_row.map(|counted| {
        let multiplier = breakdown::multiplier(
            counted.adjustments(),
            basis,
            &query.usage,
            &counted.facts().provider_entry,
        );
        breakdown::billed_rates(
            &quote_of(counted, &call, usage.prompt_tokens),
            query.usage.cache_read.explicit,
        )
        .scaled(multiplier)
    });
    let receipt = Receipt {
        candidate: Some(candidate.clone()),
        row: Some(at.clone()),
        lines,
        defaulted_to_zero,
        breakdown: rates
            .map(|rates| breakdown::breakdown(&rates, &query.usage, false))
            .unwrap_or_default(),
        rates,
        prompt,
        completion,
        tools,
        additional: Usd::ZERO,
        discount: Discount::default(),
        margin: Margin::default(),
        cache_savings: Some(breakdown::cache_savings(
            &quote,
            &query.usage,
            breakdown::savings_uplift(row.adjustments(), basis),
        )),
        basis: basis.clone(),
        total: 0.0,
    };
    Some(
        receipt
            .settled(source, &query.header.adjustments, Some(&candidate.model))
            .into(),
    )
}

fn quote_of<'a>(row: &'a impl LoweredRow, call: &CallContext, prompt_tokens: u64) -> Quote<'a> {
    row.prices()
        .for_prompt(prompt_tokens)
        .quote(call, prompt_tokens)
}

#[cfg(test)]
mod parity;

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;
    use crate::call::{InferenceGeo, Speed};
    use crate::money::Multiplier;
    use crate::price::{Adjustments, Meter, Uplift};
    use crate::query::{AdjustmentPolicy, Provider, TokenUsage};
    use crate::row::{Provenance, ProviderEntry, RowFacts};
    use crate::testing::{TestRow, TestSource, candidate, row_ref, text_row, tokens_query, usage};

    fn priced(outcome: PriceOutcome) -> Receipt {
        match outcome {
            PriceOutcome::Priced(receipt) => *receipt,
            PriceOutcome::Unpriced(unpriced) => panic!("unpriced: {unpriced:?}"),
        }
    }

    fn uplifted_row() -> TestRow {
        TestRow {
            adjustments: Adjustments::new([
                (Uplift::Speed(Speed::Fast), Multiplier::new(2.0).unwrap()),
                (
                    Uplift::InferenceGeo(InferenceGeo::new("us")),
                    Multiplier::new(1.5).unwrap(),
                ),
            ]),
            ..text_row(1.0, 4.0)
        }
    }

    fn fast_us() -> TokenUsage {
        TokenUsage {
            speed: Some(Speed::Fast),
            inference_geo: Some(InferenceGeo::new("US")),
            ..usage(100, 10)
        }
    }

    #[rstest]
    #[case::anthropic_applies_speed_and_geo(Provider::Anthropic, ProviderEntry::Readable, 420.0)]
    #[case::bedrock_applies_geo_only(Provider::Bedrock, ProviderEntry::Readable, 210.0)]
    #[case::vertex_applies_neither(Provider::VertexAi, ProviderEntry::Readable, 140.0)]
    #[case::azure_ai_applies_neither(Provider::AzureAi, ProviderEntry::Readable, 140.0)]
    #[case::anthropic_skips_both_when_one_value_is_unreadable(
        Provider::Anthropic,
        ProviderEntry::Unreadable([Uplift::Speed(Speed::Fast)].into()),
        140.0
    )]
    fn speed_and_geo_apply_by_provider(
        #[case] provider: Provider,
        #[case] entry: ProviderEntry,
        #[case] expected: f64,
    ) {
        let row = TestRow {
            facts: RowFacts {
                provider_entry: entry,
                ..RowFacts::default()
            },
            ..uplifted_row()
        };
        let source = TestSource::with_row("claude-opus-4-8", row);
        let receipt = priced(price(
            &source,
            &tokens_query(provider, "claude-opus-4-8", fast_us()),
        ));

        assert_eq!(receipt.total, expected);
        assert_eq!(
            receipt.total,
            receipt
                .lines
                .iter()
                .map(|line| line.amount.get())
                .sum::<f64>()
        );
    }

    #[rstest]
    fn bedrock_with_an_unreadable_geo_value_moves_to_the_next_candidate() {
        let broken = TestRow {
            facts: RowFacts {
                provider_entry: ProviderEntry::NotAMapping,
                ..RowFacts::default()
            },
            ..uplifted_row()
        };
        let source = TestSource::with_row("broken", broken).and_row("fallback", text_row(2.0, 0.0));
        let query = TokensQuery {
            header: crate::query::Header {
                candidates: vec![
                    candidate(Provider::Bedrock, "broken"),
                    candidate(Provider::Bedrock, "fallback"),
                ],
                ..tokens_query(Provider::Bedrock, "broken", fast_us()).header
            },
            ..tokens_query(Provider::Bedrock, "broken", fast_us())
        };
        let receipt = priced(price(&source, &query));

        assert_eq!(receipt.row, Some(row_ref("fallback")));
        assert_eq!(receipt.total, 200.0);
    }

    #[rstest]
    fn bedrock_geo_is_skipped_for_a_registered_row() {
        let source = TestSource::with_row("claude", uplifted_row())
            .with_provenance("claude", Provenance::Registered);
        let receipt = priced(price(
            &source,
            &tokens_query(Provider::Bedrock, "claude", fast_us()),
        ));

        assert_eq!(receipt.total, 140.0);
    }

    #[rstest]
    fn the_breakdown_rates_carry_every_uplift_whatever_the_provider() {
        let source = TestSource::with_row("claude", uplifted_row());
        let receipt = priced(price(
            &source,
            &tokens_query(Provider::AzureAi, "claude", fast_us()),
        ));

        assert_eq!(receipt.total, 140.0);
        let rates = receipt.rates.unwrap();
        assert_eq!((rates.input, rates.output), (3.0, 12.0));
    }

    #[rstest]
    fn the_first_candidate_with_a_row_prices_even_when_it_lacks_rates() {
        let source =
            TestSource::with_row("second", text_row(1.0, 1.0)).and_row("first", TestRow::default());
        let query = TokensQuery {
            header: crate::query::Header {
                candidates: vec![
                    candidate(Provider::Anthropic, "missing"),
                    candidate(Provider::Anthropic, "first"),
                    candidate(Provider::Anthropic, "second"),
                ],
                ..tokens_query(Provider::Anthropic, "first", usage(10, 10)).header
            },
            ..tokens_query(Provider::Anthropic, "first", usage(10, 10))
        };
        let receipt = priced(price(&source, &query));

        assert_eq!(receipt.row, Some(row_ref("first")));
        assert_eq!(receipt.total, 0.0);
        assert_eq!(
            receipt.defaulted_to_zero,
            [Meter::TEXT_INPUT, Meter::TEXT_OUTPUT]
        );
    }

    #[rstest]
    fn no_candidate_with_a_row_is_unpriced() {
        let source = TestSource::default();
        let query = tokens_query(Provider::Anthropic, "missing", usage(10, 10));

        assert_eq!(
            price(&source, &query),
            Unpriced::NoRow {
                tried: vec![candidate(Provider::Anthropic, "missing")]
            }
            .into()
        );
    }

    #[rstest]
    #[case::other_provider(Provider::Other("openai".to_owned()), "gpt-5")]
    #[case::vertex_gemini(Provider::VertexAi, "gemini-3-pro")]
    #[case::azure_model_router(Provider::AzureAi, "model-router")]
    fn a_provider_or_model_not_ported_yet_is_left_to_python(
        #[case] provider: Provider,
        #[case] model: &str,
    ) {
        let source = TestSource::with_row(model, text_row(1.0, 1.0));
        assert_eq!(
            price(&source, &tokens_query(provider, model, usage(10, 10))),
            Unpriced::NotPorted(Kind::Tokens).into()
        );
    }

    #[rstest]
    fn a_reported_cost_is_left_to_python() {
        let source = TestSource::with_row("claude", text_row(1.0, 1.0));
        let query = tokens_query(Provider::Anthropic, "claude", usage(10, 10));
        let reported = TokensQuery {
            header: crate::query::Header {
                reported: Some(crate::query::ReportedCost {
                    source: "usage.cost".into(),
                    usd: 1.0,
                }),
                ..query.header.clone()
            },
            ..query
        };
        assert_eq!(
            price(&source, &reported),
            Unpriced::NotPorted(Kind::Tokens).into()
        );
    }

    #[rstest]
    fn without_an_adjustment_policy_the_total_is_the_original_cost() {
        let source = TestSource::with_row("claude", text_row(1.0, 1.0))
            .with_discounts(&[("anthropic", 0.5)]);
        let query = tokens_query(Provider::Anthropic, "claude", usage(10, 10));
        let unadjusted = TokensQuery {
            header: crate::query::Header {
                adjustments: AdjustmentPolicy::None,
                ..query.header.clone()
            },
            ..query.clone()
        };

        assert_eq!(priced(price(&source, &unadjusted)).total, 20.0);
        assert_eq!(priced(price(&source, &query)).total, 10.0);
    }

    #[rstest]
    #[case::breakdown_off_the_given_row(false, true, Some(3.0))]
    #[case::breakdown_off_the_named_row(true, true, Some(1.0))]
    #[case::no_named_row_no_breakdown(true, false, None)]
    fn a_given_row_prices_the_total(
        #[case] breakdown_by_name: bool,
        #[case] named_row: bool,
        #[case] breakdown_input_rate: Option<f64>,
    ) {
        let given = TestSource::with_row("override", text_row(3.0, 3.0));
        let source = if named_row {
            given.and_row("claude", text_row(1.0, 1.0))
        } else {
            given
        };
        let query = tokens_query(Provider::Anthropic, "claude", usage(10, 0));
        let query = TokensQuery {
            header: crate::query::Header {
                source: PriceSource::Row {
                    row: row_ref("override"),
                    breakdown_by_name,
                },
                ..query.header
            },
            ..query
        };
        let receipt = priced(price(&source, &query));

        assert_eq!(receipt.total, 30.0);
        assert_eq!(receipt.rates.map(|rates| rates.input), breakdown_input_rate);
    }
}
