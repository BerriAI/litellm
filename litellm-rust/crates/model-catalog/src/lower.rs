use std::collections::BTreeMap;
use std::collections::btree_map::Entry;
use std::num::NonZeroU64;

use litellm_cost::money::UnitPrice;
use litellm_cost::price::{
    Adjustments, Band, CacheReadMode, CacheTtl, GuardrailUnit, InputUnit, Meter, OffPeak,
    OffPeakRates, OutputUnit, Price, Prices, Residency, SearchContextSize, TokenKind, Uplift, When,
};
use litellm_cost::schedule::{Schedule, Window};

use crate::error::{EntryProblem, PriceProblem};
use crate::model::Model;
use crate::wire::{EntryWire, FlatPrice, OffPeakBlock, TierRow};

impl TryFrom<&EntryWire> for Prices {
    type Error = Vec<PriceProblem>;

    fn try_from(wire: &EntryWire) -> Result<Self, Self::Error> {
        let mut problems = Vec::new();
        let sourced = flat_prices(&wire.prices)
            .chain(tier_prices(wire, &mut problems))
            .chain(search_context_prices(wire))
            .chain(guardrail_prices(wire));
        let rates = dedupe(sourced, &mut problems);
        let off_peak = wire
            .off_peak
            .as_ref()
            .and_then(|block| off_peak(block, &mut problems));
        if problems.is_empty() {
            Ok(Prices::new(rates, off_peak))
        } else {
            Err(problems)
        }
    }
}

impl From<&EntryWire> for Adjustments {
    fn from(wire: &EntryWire) -> Self {
        Adjustments::new(
            [
                (Uplift::NonGlobalEndpoint, wire.regional_endpoint_uplift),
                (Uplift::Residency(Residency::Eu), wire.residency_uplift_eu),
                (Uplift::Residency(Residency::Us), wire.residency_uplift_us),
            ]
            .into_iter()
            .filter_map(|(uplift, factor)| Some((uplift, factor?))),
        )
    }
}

impl TryFrom<EntryWire> for Model {
    type Error = Vec<EntryProblem>;

    fn try_from(wire: EntryWire) -> Result<Self, Self::Error> {
        let prices = Prices::try_from(&wire);
        let adjustments = Adjustments::from(&wire);
        let EntryWire {
            attributes,
            capabilities,
            unknown,
            problems,
            ..
        } = wire;
        let missing_provider = attributes
            .litellm_provider
            .is_none()
            .then_some(EntryProblem::MissingProvider);
        match prices {
            Ok(prices) if problems.is_empty() && missing_provider.is_none() => Ok(Model {
                attributes,
                prices,
                adjustments,
                capabilities,
                unrecognized: unknown.into_iter().map(|(key, _)| key).collect(),
            }),
            prices => Err(problems
                .into_iter()
                .chain(missing_provider)
                .chain(prices.err().into_iter().flatten().map(EntryProblem::Price))
                .collect()),
        }
    }
}

struct Sourced {
    price: Price,
    rate: UnitPrice,
    source: String,
}

fn flat_prices(prices: &[FlatPrice]) -> impl Iterator<Item = Sourced> + '_ {
    prices.iter().map(|flat| {
        let (channel, when) = flat.flat.channel_and_when();
        Sourced {
            price: Price {
                meter: flat.flat.meter.clone(),
                channel,
                when,
            },
            rate: flat.price,
            source: flat.key.clone(),
        }
    })
}

fn dedupe(
    sourced: impl Iterator<Item = Sourced>,
    problems: &mut Vec<PriceProblem>,
) -> BTreeMap<Price, UnitPrice> {
    let mut seen: BTreeMap<Price, (UnitPrice, String)> = BTreeMap::new();
    for Sourced {
        price,
        rate,
        source,
    } in sourced
    {
        match seen.entry(price) {
            Entry::Occupied(first) => problems.push(PriceProblem::Duplicate {
                first: first.get().1.clone(),
                second: source,
            }),
            Entry::Vacant(slot) => {
                slot.insert((rate, source));
            }
        }
    }
    seen.into_iter()
        .map(|(price, (rate, _))| (price, rate))
        .collect()
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum TierBasis {
    PromptTokens,
    Results,
}

fn tier_basis(row: &TierRow) -> Result<(TierBasis, [u64; 2]), PriceProblem> {
    match (row.range, row.max_results_range) {
        (Some(range), None) => Ok((TierBasis::PromptTokens, range)),
        (None, Some(range)) => Ok((TierBasis::Results, range)),
        _ => Err(PriceProblem::TierRange {
            row: row.path.clone(),
        }),
    }
}

fn tier_prices(wire: &EntryWire, problems: &mut Vec<PriceProblem>) -> Vec<Sourced> {
    let Some(rows) = wire.tier_rows.as_deref().filter(|rows| !rows.is_empty()) else {
        return Vec::new();
    };
    let based: Vec<_> = rows
        .iter()
        .filter_map(|row| match tier_basis(row) {
            Ok((basis, range)) => Some((basis, range, row)),
            Err(problem) => {
                problems.push(problem);
                None
            }
        })
        .collect();
    let Some(&(basis, ..)) = based.first() else {
        return Vec::new();
    };
    if based.iter().any(|(other, ..)| *other != basis) {
        problems.push(PriceProblem::MixedTierBasis);
        return Vec::new();
    }
    if basis == TierBasis::PromptTokens {
        problems.extend(
            wire.prices
                .iter()
                .filter(|flat| flat.flat.band != Band::Base)
                .map(|flat| PriceProblem::BandsFromTwoSources {
                    key: flat.key.clone(),
                }),
        );
    }
    let mut ordered = based;
    ordered.sort_by_key(|(_, [start, _], _)| *start);
    problems.extend(ladder_problems(basis, &ordered));
    ordered
        .iter()
        .flat_map(|(basis, [start, _], row)| {
            row.prices.iter().map(move |flat| (*basis, *start, flat))
        })
        .filter_map(
            |(basis, start, flat)| match tier_price(basis, start, flat) {
                Ok(sourced) => Some(sourced),
                Err(problem) => {
                    problems.push(problem);
                    None
                }
            },
        )
        .collect()
}

fn ladder_problems(
    basis: TierBasis,
    ordered: &[(TierBasis, [u64; 2], &TierRow)],
) -> Vec<PriceProblem> {
    let step = match basis {
        TierBasis::PromptTokens => 0,
        TierBasis::Results => 1,
    };
    let reversed = ordered
        .iter()
        .filter(|(_, [start, end], _)| end < start)
        .map(|(.., row)| PriceProblem::TierReversed {
            row: row.path.clone(),
        });
    let first = ordered
        .first()
        .filter(|(_, [start, _], _)| *start > step)
        .map(|(_, [start, _], row)| PriceProblem::TierGap {
            row: row.path.clone(),
            start: *start,
            expected: 0,
        });
    let gaps = ordered.windows(2).filter_map(|pair| {
        let (_, [_, end], _) = pair[0];
        let (_, [start, _], row) = pair[1];
        (end.checked_add(step) != Some(start)).then(|| PriceProblem::TierGap {
            row: row.path.clone(),
            start,
            expected: end,
        })
    });
    reversed.chain(first).chain(gaps).collect()
}

fn tier_price(basis: TierBasis, start: u64, flat: &FlatPrice) -> Result<Sourced, PriceProblem> {
    if flat.flat.has_suffix() {
        return Err(PriceProblem::SuffixInsideBlock {
            key: flat.key.clone(),
            block: "tiered_pricing",
        });
    }
    let (meter, band) = match (basis, &flat.flat.meter) {
        (TierBasis::PromptTokens, meter) => (meter.clone(), Band::over_prompt_tokens(start)),
        (TierBasis::Results, Meter::Input(InputUnit::Query { over_results: None })) => (
            Meter::Input(InputUnit::Query {
                over_results: NonZeroU64::new(start.saturating_sub(1)),
            }),
            Band::Base,
        ),
        (TierBasis::Results, _) => {
            return Err(PriceProblem::NotPricedByResults {
                key: flat.key.clone(),
            });
        }
    };
    Ok(Sourced {
        price: Price::online(
            meter,
            When {
                band,
                ..When::default()
            },
        ),
        rate: flat.price,
        source: flat.key.clone(),
    })
}

fn search_context_prices(wire: &EntryWire) -> impl Iterator<Item = Sourced> + '_ {
    wire.search_context.iter().flat_map(|costs| {
        [
            (SearchContextSize::Low, "low", costs.search_context_size_low),
            (
                SearchContextSize::Medium,
                "medium",
                costs.search_context_size_medium,
            ),
            (
                SearchContextSize::High,
                "high",
                costs.search_context_size_high,
            ),
        ]
        .into_iter()
        .filter_map(|(size, name, rate)| {
            Some(Sourced {
                price: Price::online(Meter::WebSearch(size), When::default()),
                rate: rate?,
                source: format!("search_context_cost_per_query.search_context_size_{name}"),
            })
        })
    })
}

fn guardrail_prices(wire: &EntryWire) -> impl Iterator<Item = Sourced> + '_ {
    wire.guardrail.iter().flatten().map(|(unit, rate)| Sourced {
        price: Price::online(Meter::Guardrail(GuardrailUnit::new(unit)), When::default()),
        rate: *rate,
        source: format!("guardrail_cost_per_unit.{unit}"),
    })
}

fn off_peak(block: &OffPeakBlock, problems: &mut Vec<PriceProblem>) -> Option<OffPeak> {
    let rates = off_peak_rates(&block.prices, problems);
    let schedule = schedule(block, problems)?;
    let off_peak = OffPeak::new(schedule, rates);
    if off_peak.is_none() {
        problems.push(PriceProblem::NoOffPeakPrices);
    }
    off_peak
}

fn off_peak_rates(prices: &[FlatPrice], problems: &mut Vec<PriceProblem>) -> OffPeakRates {
    let mut rates = OffPeakRates::default();
    for flat in prices {
        if flat.flat.has_suffix() {
            problems.push(PriceProblem::SuffixInsideBlock {
                key: flat.key.clone(),
                block: "off_peak_pricing",
            });
            continue;
        }
        let slot = match flat.flat.meter {
            Meter::Input(InputUnit::Token(TokenKind::Text)) => &mut rates.input,
            Meter::Output(OutputUnit::Token(TokenKind::Text)) => &mut rates.output,
            Meter::Output(OutputUnit::ReasoningToken) => &mut rates.reasoning,
            Meter::CacheRead {
                kind: TokenKind::Text,
                mode: CacheReadMode::Implicit,
            } => &mut rates.cache_read,
            Meter::CacheRead {
                kind: TokenKind::Text,
                mode: CacheReadMode::Explicit,
            } => &mut rates.explicit_cache_read,
            Meter::CacheWrite {
                kind: TokenKind::Text,
                ttl: CacheTtl::Default,
            } => &mut rates.cache_write,
            _ => {
                problems.push(PriceProblem::NotAnOffPeakRate {
                    key: flat.key.clone(),
                });
                continue;
            }
        };
        *slot = Some(flat.price);
    }
    rates
}

fn schedule(block: &OffPeakBlock, problems: &mut Vec<PriceProblem>) -> Option<Schedule> {
    let empty = |key: String| PriceProblem::EmptyList { key };
    let every_day = block
        .hours_utc
        .iter()
        .map(|hours| ("off_peak_pricing.hours_utc".to_owned(), hours, None));
    let windows = block
        .windows
        .iter()
        .flatten()
        .enumerate()
        .map(|(index, window)| {
            (
                format!("off_peak_pricing.windows[{index}].hours_utc"),
                &window.hours_utc,
                Some((index, window.weekdays.as_ref())),
            )
        });
    let windows: Vec<Window> = every_day
        .chain(windows)
        .map(|(key, hours, weekdays)| {
            if hours.windows().is_empty() {
                problems.push(empty(key));
            }
            if let Some((index, Some(days))) = weekdays
                && days.is_empty()
            {
                problems.push(empty(format!("off_peak_pricing.windows[{index}].weekdays")));
            }
            Window {
                hours: hours.windows().to_vec(),
                weekdays: weekdays.and_then(|(_, days)| days.cloned()),
            }
        })
        .collect();
    if block.windows.as_ref().is_some_and(Vec::is_empty) {
        problems.push(empty("off_peak_pricing.windows".to_owned()));
    }
    if block.hours_utc.is_none() && block.windows.is_none() {
        problems.push(PriceProblem::NoOffPeakHours);
        return None;
    }
    Some(Schedule {
        windows,
        weekday_timezone: block.weekday_timezone.clone(),
    })
}

#[cfg(test)]
mod tests {
    use litellm_cost::money::Multiplier;
    use litellm_cost::price::{Channel, ServiceTier};
    use litellm_cost::schedule::{UtcWindow, Weekday};
    use rstest::rstest;
    use serde_json::{Value, json};

    use super::*;

    const TEXT_INPUT: Meter = Meter::Input(InputUnit::Token(TokenKind::Text));

    fn wire(entry: Value) -> EntryWire {
        let Value::Object(fields) = entry else {
            panic!("entry must be an object")
        };
        let wire = EntryWire::read(&fields);
        assert!(wire.problems.is_empty(), "{:?}", wire.problems);
        wire
    }

    fn prices(entry: Value) -> Result<Prices, Vec<PriceProblem>> {
        Prices::try_from(&wire(entry))
    }

    fn online(meter: Meter, tier: ServiceTier, band: Band) -> Price {
        Price::online(meter, When { tier, band })
    }

    #[rstest]
    #[case::flat_key(json!({"input_cost_per_token": 1e-6}), online(TEXT_INPUT, ServiceTier::Standard, Band::Base))]
    #[case::band_and_tier(
        json!({"input_cost_per_token_above_272k_tokens_priority": 1e-6}),
        online(TEXT_INPUT, ServiceTier::Priority, Band::over_prompt_tokens(272_000))
    )]
    #[case::batch(
        json!({"input_cost_per_token_batches": 1e-6}),
        Price { meter: TEXT_INPUT, channel: Channel::Batch, when: When::default() }
    )]
    #[case::first_token_tier_is_the_base_band(
        json!({"tiered_pricing": [{"range": [0.0, 256000.0], "input_cost_per_token": 1e-6}]}),
        online(TEXT_INPUT, ServiceTier::Standard, Band::Base)
    )]
    #[case::result_tier_stays_in_its_meter(
        json!({"tiered_pricing": [
            {"max_results_range": [0, 25], "input_cost_per_query": 2e-6},
            {"max_results_range": [26, 100], "input_cost_per_query": 1e-6}
        ]}),
        online(
            Meter::Input(InputUnit::Query { over_results: NonZeroU64::new(25) }),
            ServiceTier::Standard,
            Band::Base
        )
    )]
    #[case::search_context(
        json!({"search_context_cost_per_query": {"search_context_size_low": 1e-6}}),
        online(Meter::WebSearch(SearchContextSize::Low), ServiceTier::Standard, Band::Base)
    )]
    #[case::guardrail(
        json!({"guardrail_cost_per_unit": {"contentPolicyUnits": 1e-6}}),
        online(Meter::Guardrail(GuardrailUnit::new("contentPolicyUnits")), ServiceTier::Standard, Band::Base)
    )]
    fn every_spelling_lowers_to_one_price(#[case] entry: Value, #[case] expected: Price) {
        assert_eq!(prices(entry).unwrap().get(&expected), UnitPrice::new(1e-6));
    }

    #[rstest]
    fn token_tiers_lower_to_prompt_bands_at_each_start() {
        let prices = prices(json!({"tiered_pricing": [
            {"range": [0, 256000], "input_cost_per_token": 1e-6},
            {"range": [256000, 1000000], "input_cost_per_token": 2e-6}
        ]}))
        .unwrap();
        assert_eq!(
            prices.band(ServiceTier::Standard, 256_001, false),
            Band::over_prompt_tokens(256_000)
        );
        assert_eq!(
            prices
                .rate(
                    &TEXT_INPUT,
                    ServiceTier::Standard,
                    Band::over_prompt_tokens(256_000)
                )
                .map(UnitPrice::get),
            Some(2e-6)
        );
    }

    #[rstest]
    #[case::two_spellings_of_one_price(
        json!({
            "input_cost_per_token": 1e-6,
            "tiered_pricing": [{"range": [0, 10], "input_cost_per_token": 2e-6}]
        }),
        PriceProblem::Duplicate {
            first: "input_cost_per_token".into(),
            second: "tiered_pricing[0].input_cost_per_token".into(),
        }
    )]
    #[case::zero_band_is_the_base_price(
        json!({"input_cost_per_token": 1e-6, "input_cost_per_token_above_0k_tokens": 2e-6}),
        PriceProblem::Duplicate {
            first: "input_cost_per_token".into(),
            second: "input_cost_per_token_above_0k_tokens".into(),
        }
    )]
    #[case::tier_without_range(
        json!({"tiered_pricing": [{"input_cost_per_token": 1e-6}]}),
        PriceProblem::TierRange { row: "tiered_pricing[0]".into() }
    )]
    #[case::tier_with_suffix(
        json!({"tiered_pricing": [{"range": [0, 10], "input_cost_per_token_priority": 1e-6}]}),
        PriceProblem::SuffixInsideBlock {
            key: "tiered_pricing[0].input_cost_per_token_priority".into(),
            block: "tiered_pricing",
        }
    )]
    #[case::reversed_range(
        json!({"tiered_pricing": [{"range": [0, 10], "input_cost_per_token": 1e-6}, {"range": [10, 5], "input_cost_per_token": 1e-6}]}),
        PriceProblem::TierReversed { row: "tiered_pricing[1]".into() }
    )]
    #[case::gap_between_tiers(
        json!({"tiered_pricing": [
            {"range": [0, 10], "input_cost_per_token": 1e-6},
            {"range": [20, 30], "input_cost_per_token": 2e-6}
        ]}),
        PriceProblem::TierGap { row: "tiered_pricing[1]".into(), start: 20, expected: 10 }
    )]
    #[case::ladder_not_from_zero(
        json!({"tiered_pricing": [{"range": [5, 10], "input_cost_per_token": 1e-6}]}),
        PriceProblem::TierGap { row: "tiered_pricing[0]".into(), start: 5, expected: 0 }
    )]
    #[case::result_ladder_overlaps(
        json!({"tiered_pricing": [
            {"max_results_range": [1, 10], "input_cost_per_query": 1e-6},
            {"max_results_range": [10, 20], "input_cost_per_query": 2e-6}
        ]}),
        PriceProblem::TierGap { row: "tiered_pricing[1]".into(), start: 10, expected: 10 }
    )]
    #[case::mixed_basis(
        json!({"tiered_pricing": [
            {"range": [0, 10], "input_cost_per_token": 1e-6},
            {"max_results_range": [11, 20], "input_cost_per_query": 2e-6}
        ]}),
        PriceProblem::MixedTierBasis
    )]
    #[case::result_tier_prices_a_token(
        json!({"tiered_pricing": [{"max_results_range": [0, 10], "input_cost_per_token": 1e-6}]}),
        PriceProblem::NotPricedByResults { key: "tiered_pricing[0].input_cost_per_token".into() }
    )]
    #[case::two_band_sources(
        json!({
            "output_cost_per_token_above_200k_tokens": 1e-6,
            "tiered_pricing": [{"range": [0, 10], "input_cost_per_token": 1e-6}]
        }),
        PriceProblem::BandsFromTwoSources { key: "output_cost_per_token_above_200k_tokens".into() }
    )]
    #[case::off_peak_without_hours(
        json!({"off_peak_pricing": {"input_cost_per_token": 1e-6}}),
        PriceProblem::NoOffPeakHours
    )]
    #[case::off_peak_without_prices(
        json!({"off_peak_pricing": {"hours_utc": "00:00-01:00"}}),
        PriceProblem::NoOffPeakPrices
    )]
    #[case::off_peak_empty_windows(
        json!({"off_peak_pricing": {"windows": [], "input_cost_per_token": 1e-6}}),
        PriceProblem::EmptyList { key: "off_peak_pricing.windows".into() }
    )]
    #[case::off_peak_empty_hours(
        json!({"off_peak_pricing": {"hours_utc": [], "input_cost_per_token": 1e-6}}),
        PriceProblem::EmptyList { key: "off_peak_pricing.hours_utc".into() }
    )]
    #[case::off_peak_empty_weekdays(
        json!({"off_peak_pricing": {
            "windows": [{"hours_utc": "00:00-01:00", "weekdays": []}],
            "input_cost_per_token": 1e-6
        }}),
        PriceProblem::EmptyList { key: "off_peak_pricing.windows[0].weekdays".into() }
    )]
    #[case::off_peak_image_price(
        json!({"off_peak_pricing": {"hours_utc": "00:00-01:00", "output_cost_per_image": 1}}),
        PriceProblem::NotAnOffPeakRate { key: "off_peak_pricing.output_cost_per_image".into() }
    )]
    #[case::off_peak_band(
        json!({"off_peak_pricing": {"hours_utc": "00:00-01:00", "input_cost_per_token_above_200k_tokens": 1}}),
        PriceProblem::SuffixInsideBlock {
            key: "off_peak_pricing.input_cost_per_token_above_200k_tokens".into(),
            block: "off_peak_pricing",
        }
    )]
    fn rules_across_keys_name_where_each_price_came_from(
        #[case] entry: Value,
        #[case] expected: PriceProblem,
    ) {
        let problems = prices(entry).unwrap_err();
        assert!(problems.contains(&expected), "{problems:?}");
    }

    #[rstest]
    fn off_peak_hours_and_windows_form_one_schedule_with_its_rates() {
        let prices = prices(json!({"off_peak_pricing": {
            "hours_utc": ["00:00-01:00"],
            "windows": [{"hours_utc": "10:00-12:00", "weekdays": [6, " sun "]}],
            "weekday_timezone": "Asia/Shanghai",
            "output_cost_per_token": 1e-6,
            "explicit_cache_read_input_token_cost": 2e-6
        }}))
        .unwrap();
        let off_peak = prices.off_peak().unwrap();
        let window = |text: &str| text.parse::<UtcWindow>().unwrap();

        assert_eq!(
            off_peak.schedule(),
            &Schedule {
                windows: vec![
                    Window {
                        hours: vec![window("00:00-01:00")],
                        weekdays: None,
                    },
                    Window {
                        hours: vec![window("10:00-12:00")],
                        weekdays: Some(vec![Weekday::Saturday, Weekday::Sunday]),
                    },
                ],
                weekday_timezone: Some("Asia/Shanghai".into()),
            }
        );
        assert_eq!(
            off_peak.rates(),
            &OffPeakRates {
                output: UnitPrice::new(1e-6),
                explicit_cache_read: UnitPrice::new(2e-6),
                ..OffPeakRates::default()
            }
        );
        assert_eq!(prices.iter().count(), 0);
    }

    #[rstest]
    fn uplifts_are_adjustments_not_prices() {
        let wire = wire(json!({
            "regional_endpoint_uplift_multiplier": 1.1,
            "regional_processing_uplift_multiplier_eu": "1.2"
        }));
        let adjustments = Adjustments::from(&wire);

        assert_eq!(Prices::try_from(&wire).unwrap().iter().count(), 0);
        assert_eq!(
            adjustments
                .get(Uplift::NonGlobalEndpoint)
                .map(Multiplier::get),
            Some(1.1)
        );
        assert_eq!(
            adjustments
                .get(Uplift::Residency(Residency::Eu))
                .map(Multiplier::get),
            Some(1.2)
        );
        assert_eq!(adjustments.get(Uplift::Residency(Residency::Us)), None);
    }

    #[rstest]
    fn a_model_reports_field_provider_and_price_problems_together() {
        let Value::Object(fields) = json!({
            "mode": "chatt",
            "tiered_pricing": [{"input_cost_per_token": 1e-6}]
        }) else {
            unreachable!()
        };
        let problems: Vec<String> = Model::from_fields(&fields)
            .unwrap_err()
            .iter()
            .map(ToString::to_string)
            .collect();

        assert_eq!(problems.len(), 3, "{problems:?}");
        assert!(problems[0].starts_with("mode: unknown variant `chatt`"));
        assert_eq!(problems[1], "litellm_provider is required");
        assert_eq!(
            problems[2],
            "tiered_pricing[0]: a tier needs exactly one of range and max_results_range"
        );
    }

    #[rstest]
    fn a_null_value_is_absent() {
        let Value::Object(fields) = json!({
            "litellm_provider": "openai",
            "input_cost_per_token": null,
            "mode": null
        }) else {
            unreachable!()
        };
        let model = Model::from_fields(&fields).unwrap();

        assert_eq!(model.prices.iter().count(), 0);
        assert_eq!(model.attributes.mode, None);
        assert!(model.unrecognized.is_empty());
    }
}
