//! The `tokens` kind: a chat, Messages or Responses call billed by its prompt and completion tokens, a port of
//! `cost_per_token` (`cost_calculator.py`) and the token path of `completion_cost`. The first candidate with a row
//! prices the call, as main's candidate loop moves on only when pricing raises. A rate the row lacks is billed at
//! 0.0 and listed in `defaulted_to_zero`, as main's `get_model_info` defaults it. Cases not ported yet answer
//! `NotPorted`, so the caller prices them the old way

mod branch;
mod breakdown;
mod custom;
mod decisions;
mod rule;

use crate::adjust::{Discount, Margin};
use crate::calculate;
use crate::call::CallContext;
use crate::money::{UnitPrice, Usd};
use crate::price::{Channel, Meter};
use crate::priced::{Line, Outcome, Quantity, Side};
use crate::query::{
    Basis, BreakdownRow, Candidate, FeeBilling, Kind, PriceSource, ReportedCost, RouterFee,
    TokensQuery,
};
use crate::quote::Quote;
use crate::receipt::{BilledRates, Breakdown, Part, PartOutcome, PriceOutcome, Receipt, Unpriced};
use crate::row::{LoweredRow, RowRef, RowSource};
use crate::units;

use branch::Sides;
use rule::{Lookup, Rule, Stated};

pub(crate) fn price<S: RowSource>(source: &S, query: &TokensQuery) -> PriceOutcome {
    let header = &query.header;
    if let Some(reported) = &header.reported {
        return match rule::stated(query) {
            Some(stated) => stated_cost(source, query, stated, reported),
            None => Unpriced::NotPorted(Kind::Tokens).into(),
        };
    }
    if let Some(candidate) = header
        .candidates
        .first()
        .filter(|_| rule::free_without_row(query))
    {
        return rowless(source, query, candidate, Sides::default());
    }
    let no_row = || {
        Unpriced::NoRow {
            tried: header.candidates.clone(),
        }
        .into()
    };
    match &header.source {
        PriceSource::FlatPerToken(rates) => custom::price(source, query, rates),
        PriceSource::PerSecond { rate } => match header.candidates.first() {
            Some(candidate) => rowless(source, query, candidate, per_second(*rate, query)),
            None => no_row(),
        },
        PriceSource::Rows { .. } | PriceSource::PerQuery { .. } => {
            Unpriced::NotPorted(Kind::Tokens).into()
        }
        PriceSource::Row { row: at } => source
            .row(at)
            .and_then(|row| {
                header
                    .candidates
                    .iter()
                    .find_map(|candidate| attempt(source, query, candidate, at, row, Lookup::Given))
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

/// `custom_cost_per_second` times the response time, all of it on the completion side
fn per_second(rate: UnitPrice, query: &TokensQuery) -> Sides {
    let elapsed_ms = query.elapsed.unwrap_or_default().response_ms;
    let amount = Usd::new(rate.get() * elapsed_ms / 1000.0).unwrap_or(Usd::ZERO);
    Sides {
        lines: vec![Line {
            meter: Meter::WallClockSecond,
            quantity: Quantity::Measured(elapsed_ms / 1000.0),
            unit_price: rate,
            amount,
            fallback_from: None,
            side: Side::Completion,
        }],
        defaulted_to_zero: Vec::new(),
        prompt: Usd::ZERO,
        completion: amount,
    }
}

/// A cost the provider stated: Perplexity's is the completion side whatever the row, xAI's is split in the ratio the
/// token math gives, all on the completion side when no row prices it
fn stated_cost<S: RowSource>(
    source: &S,
    query: &TokensQuery,
    stated: Stated,
    reported: &ReportedCost,
) -> PriceOutcome {
    let Some(candidate) = query.header.candidates.first() else {
        return Unpriced::NoRow { tried: Vec::new() }.into();
    };
    let Some(usd) = Usd::new(reported.usd) else {
        return Unpriced::NotPorted(Kind::Tokens).into();
    };
    let all_to_completion = Sides {
        completion: usd,
        ..Sides::default()
    };
    let unstated = TokensQuery {
        header: crate::query::Header {
            reported: None,
            ..query.header.clone()
        },
        ..query.clone()
    };
    let token_math = match stated {
        Stated::AllToCompletion => return rowless(source, query, candidate, all_to_completion),
        Stated::Proportional => price(source, &unstated),
    };
    let sides = match token_math {
        PriceOutcome::Priced(priced) => {
            let total = priced.prompt.get() + priced.completion.get();
            match total > 0.0 {
                true => {
                    let prompt = reported.usd * priced.prompt.get() / total;
                    Sides {
                        prompt: Usd::new(prompt).unwrap_or(Usd::ZERO),
                        completion: Usd::new(reported.usd - prompt).unwrap_or(Usd::ZERO),
                        ..Sides::default()
                    }
                }
                false => all_to_completion,
            }
        }
        PriceOutcome::Unpriced(Unpriced::NoRow { .. }) => all_to_completion,
        unpriced @ PriceOutcome::Unpriced(_) => return unpriced,
    };
    rowless(source, query, candidate, sides)
}

/// A receipt no row priced: a stated cost, custom pricing per second or a free branch. Only a breakdown row the
/// caller looked up gives it billed rates
fn rowless<S: RowSource>(
    source: &S,
    query: &TokensQuery,
    candidate: &Candidate,
    sides: Sides,
) -> PriceOutcome {
    let counted = match &query.breakdown {
        BreakdownRow::Row { row } => source.row(row),
        BreakdownRow::Priced | BreakdownRow::Named | BreakdownRow::Absent => None,
    };
    let (rates, breakdown) = stored_breakdown(counted, query);
    let extras = extras(source, query);
    let receipt = Receipt {
        candidate: Some(candidate.clone()),
        row: None,
        lines: sides.lines,
        defaulted_to_zero: sides.defaulted_to_zero,
        rates,
        prompt: sides.prompt + extras.prompt_fee,
        completion: sides.completion,
        tools: extras.tools,
        additional: extras.additional,
        discount: Discount::default(),
        margin: Margin::default(),
        breakdown,
        parts: extras.parts,
        cache_savings: None,
        reported: None,
        basis: query.header.basis.clone(),
        total: 0.0,
    };
    receipt
        .settled(source, &query.header.adjustments, Some(&candidate.model))
        .into()
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
    let listed = row.prices().for_prompt(query.usage.prompt_tokens);
    let request = query
        .decisions
        .then(|| decisions::request_price(row, listed))
        .flatten();
    let decided =
        (query.decisions && decisions::prices_decisions(listed)).then(|| listed.decisions_view());
    let rule = match (request, &decided) {
        (Some(_), _) | (None, Some(_)) => Rule::decisions(),
        (None, None) => match Rule::new(candidate, at, row, lookup, query) {
            Ok(rule) => rule,
            Err(kind) => return Some(Unpriced::NotPorted(kind).into()),
        },
    };
    let usage = rule::billed_usage(candidate, query).billed();
    let basis = &query.header.basis;
    let call = rule.call(basis, &query.usage, &row.facts().provider_entry)?;
    let prices = decided.as_ref().unwrap_or(listed);
    let calculated = || match calculate(prices, row.adjustments(), &usage, &call) {
        Outcome::Priced(priced) => Sides {
            lines: priced.lines,
            defaulted_to_zero: priced.missing,
            prompt: priced.prompt,
            completion: priced.completion,
        },
        Outcome::Unpriced { missing } => Sides {
            defaulted_to_zero: missing,
            ..Sides::default()
        },
    };
    let sides = match (request, rule) {
        (Some(rate), _) => decisions::request(rate),
        (None, Rule::Formula(formula)) => branch::sides(formula, row, query, &call, calculated),
        (None, _) => calculated(),
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
    let counted = match &query.breakdown {
        BreakdownRow::Priced => Some(row),
        BreakdownRow::Named => named,
        BreakdownRow::Row { row: at } => source.row(at),
        BreakdownRow::Absent => None,
    };
    let (rates, breakdown) = stored_breakdown(counted, query);
    let extras = extras(source, query);
    let receipt = Receipt {
        candidate: Some(candidate.clone()),
        row: Some(at.clone()),
        lines: sides.lines,
        defaulted_to_zero: sides.defaulted_to_zero,
        breakdown,
        rates,
        prompt: sides.prompt + extras.prompt_fee,
        completion: sides.completion,
        tools: tools + extras.tools,
        additional: extras.additional,
        discount: Discount::default(),
        margin: Margin::default(),
        parts: extras.parts,
        cache_savings: Some(breakdown::cache_savings(
            &prices.quote(&basis_call(basis), usage.prompt_tokens),
            &query.usage,
            breakdown::savings_uplift(row.adjustments(), basis),
        )),
        reported: None,
        basis: basis.clone(),
        total: 0.0,
    };
    Some(
        receipt
            .settled(source, &query.header.adjustments, Some(&candidate.model))
            .into(),
    )
}

/// The call facts main's breakdown and savings quote a row with: the call's tier and endpoint, without the speed and
/// geo the provider branches read, whose uplifts they multiply in apart (`call_context` in `llm_cost_calc/utils.py`)
fn basis_call(basis: &Basis) -> CallContext {
    CallContext {
        tier: basis.service_tier,
        channel: Channel::Online,
        residency: basis.residency(),
        fedramp_endpoint: basis.fedramp(),
        non_global_endpoint: basis.non_global_endpoint(),
        speed: None,
        inference_geo: None,
        at: basis.at,
        overlay: None,
    }
}

/// `get_token_type_cost_breakdown` on `counted`: the billed rates with every uplift the call names, and the cache and
/// reasoning lines at them. No row bills no rates and a zero breakdown
fn stored_breakdown<R: LoweredRow>(
    counted: Option<&R>,
    query: &TokensQuery,
) -> (Option<BilledRates>, Breakdown) {
    let basis = &query.header.basis;
    let prompt_tokens = query.usage.billed().prompt_tokens;
    let rates = counted.map(|counted| {
        let multiplier = breakdown::multiplier(
            counted.adjustments(),
            basis,
            &query.usage,
            &counted.facts().provider_entry,
        );
        breakdown::billed_rates(
            &quote_of(counted, &basis_call(basis), prompt_tokens),
            query.usage.cache_read.explicit,
        )
        .scaled(multiplier)
    });
    let stored = rates
        .map(|rates| breakdown::breakdown(&rates, &query.usage, false))
        .unwrap_or_default();
    (rates, stored)
}

/// What a call bills beside its tokens and the web search fee
struct Extras {
    tools: Usd,
    prompt_fee: Usd,
    additional: Usd,
    parts: Vec<Part>,
}

/// The label main's stored breakdown lists the Model Router fee under in `additional_costs`
pub const ROUTER_FEE: &str = "Azure Model Router Flat Cost";

/// The built-in tool fees and the Azure AI Model Router fee. Main bills the fee only when it is positive
/// (`calculate_azure_model_router_flat_cost`), and a router row that is gone bills none
fn extras<S: RowSource>(source: &S, query: &TokensQuery) -> Extras {
    let tools = crate::tools::fees(source, &query.tools, &query.header);
    let fee = query.router_fee.as_ref().and_then(|fee| {
        let rate = units::listed(source.row(&fee.row)?, &Meter::TEXT_INPUT, Channel::Online)?;
        let amount = rate * Quantity::Count(query.usage.billed().prompt_tokens);
        (amount.get() > 0.0).then_some((fee, amount))
    });
    match fee {
        Some((
            RouterFee {
                billed: FeeBilling::Prompt,
                ..
            },
            amount,
        )) => Extras {
            tools,
            prompt_fee: amount,
            additional: Usd::ZERO,
            parts: Vec::new(),
        },
        Some((
            RouterFee {
                row,
                billed: FeeBilling::Additional,
            },
            amount,
        )) => Extras {
            tools,
            prompt_fee: Usd::ZERO,
            additional: amount,
            parts: vec![Part {
                label: ROUTER_FEE.into(),
                outcome: PartOutcome::Priced {
                    row: Some(row.clone()),
                    amount,
                },
            }],
        },
        None => Extras {
            tools,
            prompt_fee: Usd::ZERO,
            additional: Usd::ZERO,
            parts: Vec::new(),
        },
    }
}

fn quote_of<'a>(row: &'a impl LoweredRow, call: &CallContext, prompt_tokens: u64) -> Quote<'a> {
    row.prices()
        .for_prompt(prompt_tokens)
        .quote(call, prompt_tokens)
}

#[cfg(test)]
mod parity {
    //! Parity with main's `completion_cost` on the P1 providers' Claude rows. `fixtures/tokens_parity.json` was captured
    //! once from main's Python (f631dbd13213, `LITELLM_LOCAL_MODEL_COST_MAP=True`): each case holds the row it priced,
    //! the billed usage the adapter hands over, and the total, the stored cost breakdown and the prompt-caching savings
    //! main computed. Every value must match to the last bit

    use rstest::rstest;
    use serde_json::Value;

    use super::price;
    use crate::adjust::MarginRule;
    use crate::call::{InferenceGeo, Speed};
    use crate::price::{SearchContextSize, ServiceTier};
    use crate::query::{
        AdjustmentPolicy, Basis, BreakdownRow, CacheReads, CacheWrites, Header, Modalities,
        PriceSource, Provider, TokenUsage, TokensQuery, WebSearchUse,
    };
    use crate::receipt::{PriceOutcome, Receipt};
    use crate::testing::{TestSource, basis, candidate, fixture_row, row_ref};

    const FIXTURE: &str = include_str!("../fixtures/tokens_parity.json");

    /// The key of a `requested_model_info_override` row, beside the catalog row its candidate names
    const GIVEN: &str = "requested_model_info_override";

    fn case(name: &str) -> Value {
        let fixture: Value = serde_json::from_str(FIXTURE).unwrap();
        fixture["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|case| case["name"] == name)
            .unwrap_or_else(|| panic!("no fixture case {name}"))
            .clone()
    }

    fn count(value: &Value) -> u64 {
        value.as_u64().unwrap_or(0)
    }

    fn usage(written: &Value) -> TokenUsage {
        TokenUsage {
            prompt_tokens: count(&written["prompt_tokens"]),
            input: Modalities {
                text: count(&written["input"]["text"]),
                ..Modalities::default()
            },
            cache_read: CacheReads {
                text: count(&written["cache_read"]["text"]),
                ..CacheReads::default()
            },
            cache_write: CacheWrites {
                default: count(&written["cache_write"]["default"]),
                one_hour: count(&written["cache_write"]["one_hour"]),
                twelve_hours: 0,
            },
            output: Modalities {
                text: count(&written["output"]["text"]),
                ..Modalities::default()
            },
            reasoning: count(&written["reasoning"]),
            web_search: written.get("web_search").map(|search| WebSearchUse {
                requests: count(&search["requests"]),
                context: search.get("context").map(|size| match size.as_str() {
                    Some("low") => SearchContextSize::Low,
                    Some("high") => SearchContextSize::High,
                    _ => SearchContextSize::Medium,
                }),
            }),
            speed: written.get("speed").map(|_| Speed::Fast),
            inference_geo: written
                .get("inference_geo")
                .and_then(Value::as_str)
                .map(InferenceGeo::new),
            ..TokenUsage::default()
        }
    }

    fn margin(rule: &Value) -> MarginRule {
        match rule {
            Value::Number(percent) => MarginRule::Percent(percent.as_f64().unwrap()),
            split => MarginRule::Split {
                percentage: split.get("percentage").and_then(Value::as_f64),
                fixed_amount: split.get("fixed_amount").and_then(Value::as_f64),
            },
        }
    }

    fn priced(case: &Value) -> Receipt {
        let provider: Provider = case["provider"].as_str().unwrap().parse().unwrap();
        let model = case["model"].as_str().unwrap();
        let discounts: Vec<(&str, f64)> = case["discount"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(key, fraction)| (key.as_str(), fraction.as_f64().unwrap()))
            .collect();
        let margins = case["margin"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(key, rule)| (key.clone(), margin(rule)))
            .collect();
        let (source, price_source, breakdown) = match case.get("named_row") {
            Some(named) => (
                TestSource::with_row(model, fixture_row(named))
                    .and_row(GIVEN, fixture_row(&case["row"])),
                PriceSource::Row {
                    row: row_ref(GIVEN),
                },
                BreakdownRow::Named,
            ),
            None => (
                TestSource::with_row(model, fixture_row(&case["row"])),
                PriceSource::Catalog,
                BreakdownRow::Priced,
            ),
        };
        let source = source.with_discounts(&discounts).with_margins(margins);
        let query = TokensQuery::new(
            Header {
                candidates: vec![candidate(provider.clone(), model)],
                source: price_source,
                basis: Basis {
                    service_tier: match case["service_tier"].as_str() {
                        Some("priority") => ServiceTier::Priority,
                        _ => ServiceTier::Standard,
                    },
                    vertex_location: case["vertex_location"].as_str().map(Into::into),
                    ..basis()
                },
                adjustments: AdjustmentPolicy::Apply {
                    provider: Some(provider.to_string().into()),
                    model: None,
                    region: None,
                },
                reported: None,
            },
            usage(&case["usage"]),
        );
        let query = TokensQuery { breakdown, ..query };
        match price(&source, &query) {
            PriceOutcome::Priced(receipt) => *receipt,
            PriceOutcome::Unpriced(unpriced) => panic!("unpriced: {unpriced:?}"),
        }
    }

    fn number(value: &Value) -> f64 {
        value.as_f64().unwrap()
    }

    #[rstest]
    #[case::anthropic_plain("anthropic_plain")]
    #[case::anthropic_cache_read("anthropic_cache_read")]
    #[case::anthropic_cache_write_5m("anthropic_cache_write_5m")]
    #[case::anthropic_cache_write_1h("anthropic_cache_write_1h")]
    #[case::anthropic_cache_mixed("anthropic_cache_mixed")]
    #[case::anthropic_geo_us("anthropic_geo_us")]
    #[case::anthropic_geo_global("anthropic_geo_global")]
    #[case::anthropic_speed_fast("anthropic_speed_fast")]
    #[case::anthropic_speed_fast_geo_us("anthropic_speed_fast_geo_us")]
    #[case::anthropic_over_100k("anthropic_over_100k")]
    #[case::anthropic_service_tier_priority("anthropic_service_tier_priority")]
    #[case::anthropic_service_tier_without_tier_prices(
        "anthropic_service_tier_without_tier_prices"
    )]
    #[case::anthropic_web_search("anthropic_web_search")]
    #[case::anthropic_discount("anthropic_discount")]
    #[case::anthropic_model_pattern_discount("anthropic_model_pattern_discount")]
    #[case::anthropic_margin_percent("anthropic_margin_percent")]
    #[case::anthropic_discount_and_fixed_margin("anthropic_discount_and_fixed_margin")]
    #[case::anthropic_fixed_margin_only("anthropic_fixed_margin_only")]
    #[case::bedrock_claude_platform_geo_us("bedrock_claude_platform_geo_us")]
    #[case::bedrock_claude_platform_web_search_high("bedrock_claude_platform_web_search_high")]
    #[case::bedrock_claude_platform_web_search_zero_requests(
        "bedrock_claude_platform_web_search_zero_requests"
    )]
    #[case::bedrock_over_200k("bedrock_over_200k")]
    #[case::vertex_global("vertex_global")]
    #[case::vertex_regional("vertex_regional")]
    #[case::vertex_regional_over_100k_web_search("vertex_regional_over_100k_web_search")]
    #[case::azure_ai_plain("azure_ai_plain")]
    #[case::azure_ai_over_100k("azure_ai_over_100k")]
    fn rust_prices_the_call_as_main_did(#[case] name: &str) {
        let case = case(name);
        let expected = &case["expected"];
        let stored = &expected["breakdown"];
        let rates = &stored["billed_token_rates"];

        let receipt = priced(&case);
        let billed = receipt.rates.unwrap();

        assert_eq!(receipt.total, number(&expected["total"]));
        assert_eq!(receipt.prompt.get(), number(&stored["input_cost"]));
        assert_eq!(receipt.completion.get(), number(&stored["output_cost"]));
        assert_eq!(
            receipt.tools.get(),
            number(&stored["cost_for_built_in_tools_cost_usd_dollar"])
        );
        assert_eq!(receipt.original(), number(&stored["original_cost"]));
        assert_eq!(
            receipt.discount.percent,
            number(&stored["discount_percent"])
        );
        assert_eq!(receipt.discount.amount, number(&stored["discount_amount"]));
        assert_eq!(receipt.margin.percent, number(&stored["margin_percent"]));
        assert_eq!(receipt.margin.fixed, number(&stored["margin_fixed_amount"]));
        assert_eq!(receipt.margin.total, number(&stored["margin_total_amount"]));
        assert_eq!(
            receipt.breakdown.cache_read,
            number(&stored["cache_read_cost"])
        );
        assert_eq!(
            receipt.breakdown.cache_creation,
            number(&stored["cache_creation_cost"])
        );
        assert_eq!(
            receipt.breakdown.reasoning,
            number(&stored["reasoning_cost"])
        );
        assert_eq!(
            [
                billed.input,
                billed.output,
                billed.cache_read,
                billed.cache_write,
                billed.cache_write_one_hour,
                billed.reasoning,
            ],
            [
                "input",
                "output",
                "cache_read",
                "cache_write",
                "cache_write_1h",
                "reasoning"
            ]
            .map(|key| number(&rates[key]))
        );
        assert_eq!(
            receipt.cache_savings,
            Some(number(&expected["cache_savings"]))
        );
        assert!(receipt.defaulted_to_zero.is_empty());
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;
    use crate::call::{InferenceGeo, Speed};
    use crate::money::Multiplier;
    use crate::price::{Adjustments, Meter, ServiceTier, Uplift};
    use crate::query::{AdjustmentPolicy, Provider, TokenRules, TokenUsage};
    use crate::row::{Provenance, ProviderEntry, RowFacts};
    use crate::testing::{
        BASE, TestRow, TestSource, candidate, prices, row_ref, text_row, tokens_query, usage,
    };

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

    fn other_query(provider: &str, usage: TokenUsage) -> TokensQuery {
        tokens_query(Provider::Other(provider.to_owned()), "model", usage)
    }

    #[rstest]
    #[case::billed_on_the_prompt(FeeBilling::Prompt, 30.0 + 5.0, 0.0, None)]
    #[case::billed_beside_the_tokens(FeeBilling::Additional, 30.0, 5.0, Some(5.0))]
    fn the_router_fee_bills_the_router_rate_on_every_prompt_token(
        #[case] billed: FeeBilling,
        #[case] prompt: f64,
        #[case] additional: f64,
        #[case] part: Option<f64>,
    ) {
        let source =
            TestSource::with_row("model", text_row(3.0, 1.0)).and_row("router", text_row(0.5, 0.0));
        let query = TokensQuery {
            router_fee: Some(RouterFee {
                row: row_ref("router"),
                billed,
            }),
            ..other_query("azure_ai", usage(10, 2))
        };
        let receipt = priced(price(&source, &query));

        assert_eq!(receipt.prompt.get(), prompt);
        assert_eq!(receipt.additional.get(), additional);
        assert_eq!(
            receipt
                .parts
                .iter()
                .find(|part| &*part.label == ROUTER_FEE)
                .and_then(Part::amount)
                .map(Usd::get),
            part
        );
        assert_eq!(receipt.total, 30.0 + 2.0 + 5.0);
    }

    #[rstest]
    #[case::free_router_row(text_row(0.0, 0.0))]
    #[case::router_row_without_an_input_rate(TestRow::default())]
    fn a_router_row_with_no_positive_rate_bills_no_fee(#[case] router: TestRow) {
        let source = TestSource::with_row("model", text_row(3.0, 1.0)).and_row("router", router);
        let query = TokensQuery {
            router_fee: Some(RouterFee {
                row: row_ref("router"),
                billed: FeeBilling::Additional,
            }),
            ..other_query("azure", usage(10, 2))
        };
        let receipt = priced(price(&source, &query));

        assert!(receipt.parts.is_empty());
        assert_eq!(receipt.total, 32.0);
    }

    #[rstest]
    fn tool_fees_are_discounted_with_the_tokens() {
        let source =
            TestSource::with_row("model", text_row(3.0, 1.0)).with_discounts(&[("openai", 0.5)]);
        let query = TokensQuery {
            tools: crate::tools::ToolUse {
                file_search: Some(crate::tools::FileSearch {
                    requested: true,
                    provider: Some("openai".into()),
                    row: None,
                    storage_gb: None,
                    days: None,
                    per_call: UnitPrice::new(8.0).unwrap(),
                    azure_gigabyte_day: UnitPrice::ZERO,
                }),
                ..crate::tools::ToolUse::default()
            },
            ..other_query("openai", usage(10, 2))
        };
        let receipt = priced(price(&source, &query));

        assert_eq!(receipt.tools.get(), 8.0);
        assert_eq!(receipt.original(), 40.0);
        assert_eq!(receipt.total, 20.0);
    }

    fn reasoning(completion_tokens: u64, total_tokens: u64) -> TokensQuery {
        let query = other_query(
            "xai",
            TokenUsage {
                reasoning: 6,
                ..usage(10, completion_tokens)
            },
        );
        TokensQuery {
            raw: crate::query::RawCounts {
                prompt_tokens: 10,
                completion_tokens,
                total_tokens,
                reasoning_tokens: 6,
                ..Default::default()
            },
            ..query
        }
    }

    #[rstest]
    #[case::reported_beside_the_completion(reasoning(4, 20), 10.0)]
    #[case::already_in_the_completion(reasoning(4, 14), 4.0)]
    fn xai_bills_reasoning_as_completion_text(
        #[case] query: TokensQuery,
        #[case] completion_tokens: f64,
    ) {
        let row = TestRow {
            prices: prices(&[
                (Meter::TEXT_INPUT, ServiceTier::Standard, BASE, 1.0),
                (Meter::TEXT_OUTPUT, ServiceTier::Standard, BASE, 2.0),
                (Meter::REASONING, ServiceTier::Standard, BASE, 100.0),
            ]),
            ..TestRow::default()
        };
        let receipt = priced(price(&TestSource::with_row("model", row), &query));

        assert_eq!(receipt.completion.get(), completion_tokens * 2.0);
        assert_eq!(receipt.breakdown.reasoning, 600.0);
    }

    #[rstest]
    fn the_stored_breakdown_reads_the_tier_a_branch_that_ignores_it_bills_at_standard() {
        let row = TestRow {
            prices: prices(&[
                (Meter::TEXT_INPUT, ServiceTier::Standard, BASE, 1.0),
                (Meter::TEXT_INPUT, ServiceTier::Priority, BASE, 2.0),
                (Meter::TEXT_OUTPUT, ServiceTier::Standard, BASE, 1.0),
            ]),
            ..TestRow::default()
        };
        let query = other_query("deepseek", usage(10, 0));
        let query = TokensQuery {
            header: crate::query::Header {
                basis: crate::query::Basis {
                    service_tier: ServiceTier::Priority,
                    ..query.header.basis.clone()
                },
                ..query.header
            },
            ..query
        };
        let receipt = priced(price(&TestSource::with_row("model", row), &query));

        assert_eq!(receipt.total, 10.0);
        assert_eq!(receipt.rates.map(|rates| rates.input), Some(2.0));
    }

    #[rstest]
    #[case::generic(TokenRules::Generic, 11.0)]
    #[case::generic_with_the_tier_alone(TokenRules::GenericTier, 10.0)]
    fn generic_with_the_tier_alone_leaves_the_residency_to_the_breakdown(
        #[case] rules: TokenRules,
        #[case] total: f64,
    ) {
        let row = TestRow {
            adjustments: Adjustments::new([(
                Uplift::Residency(crate::price::Residency::Eu),
                Multiplier::new(1.1).unwrap(),
            )]),
            ..text_row(1.0, 0.0)
        };
        let query = other_query("azure_ai", usage(10, 0));
        let query = TokensQuery {
            header: crate::query::Header {
                basis: crate::query::Basis {
                    data_residency: Some(crate::query::DataResidency::Eu),
                    ..query.header.basis.clone()
                },
                ..query.header
            },
            rules,
            ..query
        };
        let receipt = priced(price(&TestSource::with_row("model", row), &query));

        assert_eq!(receipt.total, total);
        assert_eq!(receipt.rates.map(|rates| rates.input), Some(1.1));
    }

    #[rstest]
    #[case::a_row_looked_up_apart(true, Some(4.0))]
    #[case::no_row(false, None)]
    fn custom_pricing_per_second_reads_billed_rates_off_the_breakdown_row(
        #[case] looked_up: bool,
        #[case] input_rate: Option<f64>,
    ) {
        let query = other_query("openai", usage(10, 0));
        let query = TokensQuery {
            header: crate::query::Header {
                source: PriceSource::PerSecond {
                    rate: UnitPrice::new(0.5).unwrap(),
                },
                ..query.header
            },
            elapsed: Some(crate::query::Elapsed {
                response_ms: 3000.0,
                audio_seconds: 0.0,
            }),
            breakdown: if looked_up {
                BreakdownRow::Row {
                    row: row_ref("catalog"),
                }
            } else {
                BreakdownRow::Absent
            },
            ..query
        };
        let receipt = priced(price(
            &TestSource::with_row("catalog", text_row(4.0, 1.0)),
            &query,
        ));

        assert_eq!(receipt.total, 1.5);
        assert_eq!(receipt.rates.map(|rates| rates.input), input_rate);
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
                    source: crate::query::ReportSource::UsageCost,
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
    #[case::breakdown_off_the_given_row(BreakdownRow::Priced, true, Some(3.0))]
    #[case::breakdown_off_the_named_row(BreakdownRow::Named, true, Some(1.0))]
    #[case::no_named_row_no_breakdown(BreakdownRow::Named, false, None)]
    #[case::breakdown_off_a_row_looked_up_apart(BreakdownRow::Row { row: row_ref("looked-up") }, false, Some(5.0))]
    #[case::no_row_looked_up_no_breakdown(BreakdownRow::Absent, true, None)]
    fn a_given_row_prices_the_total(
        #[case] breakdown: BreakdownRow,
        #[case] named_row: bool,
        #[case] breakdown_input_rate: Option<f64>,
    ) {
        let given = TestSource::with_row("override", text_row(3.0, 3.0))
            .and_row("looked-up", text_row(5.0, 5.0));
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
                },
                ..query.header
            },
            breakdown,
            ..query
        };
        let receipt = priced(price(&source, &query));

        assert_eq!(receipt.total, 30.0);
        assert_eq!(receipt.rates.map(|rates| rates.input), breakdown_input_rate);
    }

    fn other(provider: &str) -> Provider {
        Provider::Other(provider.to_owned())
    }

    fn stated(query: TokensQuery, usd: f64) -> TokensQuery {
        TokensQuery {
            header: crate::query::Header {
                reported: Some(crate::query::ReportedCost {
                    source: crate::query::ReportSource::UsageCost,
                    usd,
                }),
                ..query.header
            },
            ..query
        }
    }

    fn sides(outcome: PriceOutcome) -> (f64, f64) {
        let receipt = priced(outcome);
        (receipt.prompt.get(), receipt.completion.get())
    }

    #[rstest]
    fn lemonade_bills_nothing_for_a_model_without_a_row() {
        let query = tokens_query(other("lemonade"), "local-7b", usage(100, 10));
        assert_eq!(sides(price(&TestSource::default(), &query)), (0.0, 0.0));
    }

    #[rstest]
    #[case::priced_row(true)]
    #[case::no_row(false)]
    fn perplexity_bills_its_stated_cost_on_the_completion_side(#[case] with_row: bool) {
        let source = if with_row {
            TestSource::with_row("sonar", text_row(1.0, 1.0))
        } else {
            TestSource::default()
        };
        let query = stated(
            tokens_query(other("perplexity"), "sonar", usage(10, 10)),
            0.5,
        );
        assert_eq!(sides(price(&source, &query)), (0.0, 0.5));
    }

    #[rstest]
    #[case::split_by_token_math(Some(text_row(1.0, 3.0)), (1.0, 3.0))]
    #[case::all_to_completion_without_a_row(None, (0.0, 4.0))]
    #[case::all_to_completion_when_tokens_price_nothing(Some(text_row(0.0, 0.0)), (0.0, 4.0))]
    fn xai_splits_its_stated_cost_in_the_ratio_of_the_token_math(
        #[case] row: Option<TestRow>,
        #[case] expected: (f64, f64),
    ) {
        let source = row.map_or_else(TestSource::default, |row| TestSource::with_row("grok", row));
        let query = TokensQuery {
            raw: crate::query::RawCounts {
                prompt_tokens: 10,
                completion_tokens: 10,
                total_tokens: 20,
                ..Default::default()
            },
            ..stated(tokens_query(other("xai"), "grok", usage(10, 10)), 4.0)
        };
        assert_eq!(sides(price(&source, &query)), expected);
    }

    #[rstest]
    #[case::left_to_python_without_the_time(None, None)]
    #[case::billed_by_the_response_time(Some(2500.0), Some(1.25))]
    fn a_row_billed_by_the_second_needs_the_elapsed_time(
        #[case] response_ms: Option<f64>,
        #[case] expected: Option<f64>,
    ) {
        let row = TestRow {
            prices: crate::testing::prices(&[(
                Meter::WallClockSecond,
                crate::price::ServiceTier::Standard,
                crate::testing::BASE,
                0.5,
            )]),
            ..TestRow::default()
        };
        let source = TestSource::with_row("provisioned", row);
        let query = TokensQuery {
            elapsed: response_ms.map(|response_ms| crate::query::Elapsed {
                response_ms,
                audio_seconds: 0.0,
            }),
            ..tokens_query(other("groq"), "provisioned", usage(10, 10))
        };
        let outcome = price(&source, &query);
        match expected {
            Some(prompt) => assert_eq!(sides(outcome), (prompt, 0.0)),
            None => assert_eq!(outcome, Unpriced::NotPorted(Kind::Tokens).into()),
        }
    }

    #[rstest]
    #[case::named_branch_reads_no_residency("deepseek", 10.0)]
    #[case::openai_reads_residency("openai", 15.0)]
    #[case::generic_branch_reads_residency("mistral", 15.0)]
    #[case::azure_reads_no_residency("azure", 10.0)]
    fn each_branch_reads_the_residency_main_passed_it(
        #[case] provider: &str,
        #[case] expected: f64,
    ) {
        let row = TestRow {
            adjustments: Adjustments::new([(
                Uplift::Residency(crate::price::Residency::Eu),
                Multiplier::new(1.5).unwrap(),
            )]),
            ..text_row(1.0, 0.0)
        };
        let source = TestSource::with_row("model", row);
        let query = tokens_query(other(provider), "model", usage(10, 0));
        let query = TokensQuery {
            header: crate::query::Header {
                basis: crate::query::Basis {
                    data_residency: Some(crate::query::DataResidency::Eu),
                    ..query.header.basis.clone()
                },
                ..query.header
            },
            ..query
        };
        assert_eq!(sides(price(&source, &query)).0, expected);
    }

    #[rstest]
    fn a_generic_branch_row_without_token_prices_bills_nothing() {
        let row = TestRow {
            prices: crate::testing::prices(&[(
                Meter::Output(crate::price::OutputUnit::Image(None)),
                crate::price::ServiceTier::Standard,
                crate::testing::BASE,
                0.04,
            )]),
            ..TestRow::default()
        };
        let source = TestSource::with_row("image-model", row);
        let query = tokens_query(other("recraft"), "image-model", usage(10, 10));
        assert_eq!(sides(price(&source, &query)), (0.0, 0.0));
    }
}
