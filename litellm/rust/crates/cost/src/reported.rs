//! A cost the provider, or a gateway in front of it, reported for a call, billed by the fixed policy of the source
//! that stated it: whether the discount and margin config applies, and how the total is apportioned between the
//! prompt and the completion. The source's eligibility (which responses carry a usable figure) stays with the caller

use crate::adjust::{Discount, Margin};
use crate::money::Usd;
use crate::query::{AdjustmentPolicy, Header, Kind, ReportedCost};
use crate::receipt::{Breakdown, PriceOutcome, Receipt, Unpriced};
use crate::row::RowSource;

/// Where a reported cost came from, which fixes how it is billed
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize))]
pub enum ReportSource {
    /// `usage.cost` of a provider billed from its own usage fields (xAI, Perplexity), which its token branch bills
    #[cfg_attr(feature = "serde", serde(rename = "usage.cost"))]
    UsageCost,
    /// The `llm_provider-x-litellm-response-cost` header a provider integration stamped, billed as reported
    #[cfg_attr(
        feature = "serde",
        serde(rename = "llm_provider-x-litellm-response-cost")
    )]
    ResponseCostHeader,
    /// The charge an OpenAI-compatible upstream reports in `usage.cost`, discounted and marked up like a price
    /// (`billed_usage_cost`)
    #[cfg_attr(feature = "serde", serde(rename = "upstream_usage_cost"))]
    UpstreamUsageCost,
}

impl ReportSource {
    /// Whether the configured discount and margin apply to the reported figure
    fn adjusted(self) -> bool {
        match self {
            Self::UpstreamUsageCost => true,
            Self::ResponseCostHeader | Self::UsageCost => false,
        }
    }
}

/// A reported cost to bill, beside what the catalog would charge the same usage on each side, which apportions it
#[derive(Clone, Debug, PartialEq)]
pub struct ReportedQuery {
    pub header: Header,
    pub reported: ReportedCost,
    /// The catalog's prompt and completion price of the call's usage, `None` when the caller priced none
    pub split_by: Option<PricedSides>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize))]
pub struct PricedSides {
    pub prompt: f64,
    pub completion: f64,
}

pub(crate) fn price<S: RowSource>(source: &S, query: &ReportedQuery) -> PriceOutcome {
    let reported = &query.reported;
    let Some(usd) = Usd::new(reported.usd).filter(|_| reported.source != ReportSource::UsageCost)
    else {
        return Unpriced::NotPorted(Kind::Reported).into();
    };
    let policy = match reported.source.adjusted() {
        true => query.header.adjustments.clone(),
        false => AdjustmentPolicy::None,
    };
    let settled = Receipt {
        candidate: query.header.candidates.first().cloned(),
        row: None,
        lines: Vec::new(),
        defaulted_to_zero: Vec::new(),
        rates: None,
        prompt: Usd::ZERO,
        completion: Usd::ZERO,
        tools: Usd::ZERO,
        additional: Usd::ZERO,
        discount: Discount::default(),
        margin: Margin::default(),
        breakdown: Breakdown::default(),
        parts: Vec::new(),
        cache_savings: None,
        basis: query.header.basis.clone(),
        reported: Some(usd),
        total: 0.0,
    }
    .settled(source, &policy, None);
    let (prompt, completion) = split(settled.total, query.split_by);
    Receipt {
        prompt,
        completion,
        ..settled
    }
    .into()
}

/// `split_reported_cost_by_cost_map_ratio`: `total` apportioned in the ratio the catalog prices the two sides, all
/// of it on the completion when the catalog prices neither
fn split(total: f64, by: Option<PricedSides>) -> (Usd, Usd) {
    let all_to_completion = (Usd::ZERO, Usd::new(total).unwrap_or(Usd::ZERO));
    let Some(PricedSides { prompt, completion }) = by else {
        return all_to_completion;
    };
    let priced = prompt + completion;
    if priced <= 0.0 {
        return all_to_completion;
    }
    let input = total * (prompt / priced);
    (
        Usd::new(input).unwrap_or(Usd::ZERO),
        Usd::new(total - input).unwrap_or(Usd::ZERO),
    )
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;
    use crate::testing::{TestSource, basis};

    fn query(source: ReportSource, usd: f64, split_by: Option<PricedSides>) -> ReportedQuery {
        ReportedQuery {
            header: Header {
                candidates: Vec::new(),
                source: crate::query::PriceSource::Rows { rows: Vec::new() },
                basis: basis(),
                adjustments: AdjustmentPolicy::Apply {
                    provider: Some("openai".into()),
                    model: None,
                    region: None,
                },
                reported: None,
            },
            reported: ReportedCost { source, usd },
            split_by,
        }
    }

    fn receipt(query: &ReportedQuery) -> Receipt {
        let source = TestSource::default().with_discounts(&[("openai", 0.5)]);
        match price(&source, query) {
            PriceOutcome::Priced(receipt) => *receipt,
            PriceOutcome::Unpriced(unpriced) => panic!("unpriced: {unpriced:?}"),
        }
    }

    #[rstest]
    #[case::header_billed_as_reported(ReportSource::ResponseCostHeader, 8.0)]
    #[case::upstream_charge_discounted(ReportSource::UpstreamUsageCost, 4.0)]
    fn each_source_bills_by_its_own_policy(#[case] source: ReportSource, #[case] total: f64) {
        let receipt = receipt(&query(source, 8.0, None));

        assert_eq!(receipt.original(), 8.0);
        assert_eq!(receipt.total, total);
    }

    #[rstest]
    #[case::by_the_catalog_ratio(Some(PricedSides { prompt: 1.0, completion: 3.0 }), 1.0)]
    #[case::all_to_completion_when_unpriced(Some(PricedSides { prompt: 0.0, completion: 0.0 }), 0.0)]
    #[case::all_to_completion_without_a_price(None, 0.0)]
    fn the_billed_total_is_split_between_prompt_and_completion(
        #[case] split_by: Option<PricedSides>,
        #[case] prompt: f64,
    ) {
        let receipt = receipt(&query(ReportSource::UpstreamUsageCost, 8.0, split_by));

        assert_eq!(receipt.prompt.get(), prompt);
        assert_eq!(
            receipt.prompt.get() + receipt.completion.get(),
            receipt.total
        );
    }

    #[rstest]
    #[case::stated_usage_cost(ReportSource::UsageCost, 1.0)]
    #[case::negative(ReportSource::ResponseCostHeader, -1.0)]
    fn a_figure_this_kind_does_not_bill_is_left_to_the_caller(
        #[case] source: ReportSource,
        #[case] usd: f64,
    ) {
        let source_rows = TestSource::default();

        assert!(matches!(
            price(&source_rows, &query(source, usd, None)),
            PriceOutcome::Unpriced(Unpriced::NotPorted(Kind::Reported))
        ));
    }
}

#[cfg(test)]
mod parity {
    //! Parity with main's `split_reported_cost_by_cost_map_ratio` and `_adjusted_provider_billed_cost`.
    //! `fixtures/reported_parity.json` was captured once from main's Python (532df6ef85): each case holds the reported
    //! cost, the catalog's price of each side, the discount and margin config, and main's split and adjusted total for
    //! a cost stated in the response cost header and in an upstream's `usage.cost`. Every value must match to the
    //! last bit

    use rstest::rstest;
    use serde_json::Value;

    use super::*;
    use crate::adjust::MarginRule;
    use crate::testing::{TestSource, basis, fixture_case};

    const FIXTURE: &str = include_str!("../fixtures/reported_parity.json");

    fn margin(rule: &Value) -> MarginRule {
        match rule {
            Value::Number(percent) => MarginRule::Percent(percent.as_f64().unwrap()),
            split => MarginRule::Split {
                percentage: split["percentage"].as_f64(),
                fixed_amount: split["fixed_amount"].as_f64(),
            },
        }
    }

    fn billed(case: &Value, source: ReportSource) -> Receipt {
        let entries = |config: &Value| config.as_object().cloned().unwrap_or_default();
        let discounts: Vec<(String, f64)> = entries(&case["discount"])
            .into_iter()
            .map(|(key, fraction)| (key, fraction.as_f64().unwrap()))
            .collect();
        let discounts: Vec<(&str, f64)> = discounts
            .iter()
            .map(|(key, fraction)| (key.as_str(), *fraction))
            .collect();
        let margins = entries(&case["margin"])
            .into_iter()
            .map(|(key, rule)| (key, margin(&rule)))
            .collect();
        let rows = TestSource::default()
            .with_discounts(&discounts)
            .with_margins(margins);
        let query = ReportedQuery {
            header: Header {
                candidates: Vec::new(),
                source: crate::query::PriceSource::Rows { rows: Vec::new() },
                basis: basis(),
                adjustments: AdjustmentPolicy::Apply {
                    provider: Some("openrouter".into()),
                    model: Some("m".into()),
                    region: None,
                },
                reported: None,
            },
            reported: ReportedCost {
                source,
                usd: case["usd"].as_f64().unwrap(),
            },
            split_by: Some(PricedSides {
                prompt: case["priced"]["prompt"].as_f64().unwrap(),
                completion: case["priced"]["completion"].as_f64().unwrap(),
            }),
        };
        match price(&rows, &query) {
            PriceOutcome::Priced(receipt) => *receipt,
            PriceOutcome::Unpriced(unpriced) => panic!("unpriced: {unpriced:?}"),
        }
    }

    fn number(case: &Value, block: &str, field: &str) -> f64 {
        case[block][field].as_f64().unwrap()
    }

    #[rstest]
    #[case::header_split_by_ratio("header_split_by_ratio")]
    #[case::header_all_input("header_all_input")]
    #[case::header_unpriced_all_output("header_unpriced_all_output")]
    #[case::upstream_discounted_and_split("upstream_discounted_and_split")]
    #[case::upstream_margin_split("upstream_margin_split")]
    #[case::upstream_discount_margin_unpriced("upstream_discount_margin_unpriced")]
    fn each_source_matches_main(#[case] name: &str) {
        let case = fixture_case(FIXTURE, name);
        let header = billed(&case, ReportSource::ResponseCostHeader);
        let upstream = billed(&case, ReportSource::UpstreamUsageCost);

        assert_eq!(
            (header.prompt.get(), header.completion.get()),
            (
                number(&case, "header", "prompt"),
                number(&case, "header", "completion")
            )
        );
        assert_eq!(
            (
                upstream.total,
                upstream.prompt.get(),
                upstream.completion.get(),
                upstream.original()
            ),
            (
                number(&case, "upstream", "total"),
                number(&case, "upstream", "prompt"),
                number(&case, "upstream", "completion"),
                number(&case, "upstream", "original"),
            )
        );
    }
}
