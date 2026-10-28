use std::collections::BTreeMap;
use std::collections::btree_map::Entry;
use std::num::NonZeroU64;

use litellm_cost::money::UnitPrice;
use litellm_cost::price::{
    Adjustments, Band, BandEdge, Billing, CacheReadFallback, CacheReadMode, CacheTtl,
    GuardrailUnit, InputUnit, Meter, OffPeak, OffPeakRates, OutputUnit, Price, Prices, Residency,
    SearchContextSize, TextUnit, TierScope, TokenKind, Uplift, WallClock, WebSearchUnit, When,
};
use litellm_cost::schedule::{Schedule, Window};

use crate::error::{Error, Located, pointer_child, pointer_for_key};
use crate::model::Model;
use crate::vocabulary::Mode;
use crate::wire::{EntryWire, FlatPrice, OffPeakBlock, TierRow, WebSearchBillingUnit};

pub fn fatal(error: &Error) -> bool {
    match error {
        Error::Json(_) | Error::Empty | Error::BelowMinimum { .. } | Error::Shrunk { .. } => true,
        Error::Invalid(_)
        | Error::NotAnObject
        | Error::UnknownBlockKey
        | Error::MissingProvider
        | Error::UnknownTimezone { .. }
        | Error::AliasShadowsModel { .. }
        | Error::AliasTaken { .. }
        | Error::DuplicatePrice { .. }
        | Error::SuffixInsideBlock { .. }
        | Error::TierRange
        | Error::TierReversed
        | Error::TierGap { .. }
        | Error::MixedTierBasis
        | Error::NotPricedByResults
        | Error::BandsFromTwoSources
        | Error::NotAnOffPeakRate
        | Error::NoOffPeakPrices
        | Error::NoOffPeakHours
        | Error::EmptyList
        | Error::InvalidDate => false,
    }
}

pub fn rejects_row(error: &Error) -> bool {
    match error {
        Error::MissingProvider => false,
        Error::Json(_) | Error::Empty | Error::BelowMinimum { .. } | Error::Shrunk { .. } => false,
        Error::NotAnObject
        | Error::Invalid(_)
        | Error::UnknownBlockKey
        | Error::UnknownTimezone { .. }
        | Error::AliasShadowsModel { .. }
        | Error::AliasTaken { .. }
        | Error::DuplicatePrice { .. }
        | Error::SuffixInsideBlock { .. }
        | Error::TierRange
        | Error::TierReversed
        | Error::TierGap { .. }
        | Error::MixedTierBasis
        | Error::NotPricedByResults
        | Error::BandsFromTwoSources
        | Error::NotAnOffPeakRate
        | Error::NoOffPeakPrices
        | Error::NoOffPeakHours
        | Error::EmptyList
        | Error::InvalidDate => true,
    }
}

impl TryFrom<&EntryWire> for Prices {
    type Error = Vec<Located>;

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
            let billing = billing(wire, &rates);
            Ok(Prices::new(rates, off_peak, billing))
        } else {
            Err(problems)
        }
    }
}

fn billing(wire: &EntryWire, rates: &BTreeMap<Price, UnitPrice>) -> Billing {
    let provider = wire.attributes.litellm_provider.as_deref();
    let prices_meter = |priced: fn(&Meter) -> bool| rates.keys().any(|price| priced(&price.meter));
    let prices_tokens = prices_meter(|meter| {
        matches!(
            meter,
            Meter::Input(InputUnit::Token(_)) | Meter::Output(OutputUnit::Token(_))
        )
    });
    let prices_characters = prices_meter(|meter| {
        matches!(
            meter,
            Meter::Input(InputUnit::Character) | Meter::Output(OutputUnit::Character)
        )
    });
    let prices_cache_read = prices_meter(|meter| {
        matches!(
            meter,
            Meter::CacheRead {
                kind: TokenKind::Text,
                ..
            }
        )
    });
    Billing {
        text_unit: if prices_characters && !prices_tokens {
            TextUnit::Character
        } else {
            TextUnit::Token
        },
        band_edge: if provider == Some("xai") {
            BandEdge::Inclusive
        } else {
            BandEdge::Exclusive
        },
        cache_read_fallback: if provider == Some("fireworks_ai") && !prices_cache_read {
            CacheReadFallback::HalfInputRate
        } else {
            CacheReadFallback::InputRate
        },
        wall_clock: match wire.attributes.mode {
            None | Some(Mode::Chat | Mode::Completion | Mode::Embedding | Mode::Responses) => {
                WallClock::ResponseTime
            }
            Some(Mode::AudioTranscription) => WallClock::AudioFile,
            Some(_) => WallClock::Never,
        },
        web_search: match wire.attributes.web_search_billing_unit {
            Some(WebSearchBillingUnit::PerQuery) => WebSearchUnit::PerQuery,
            Some(WebSearchBillingUnit::PerPrompt) | None => WebSearchUnit::PerPrompt,
        },
        tier_scope: if provider == Some("dashscope") && wire.tier_rows.is_some() {
            TierScope::WholeCall
        } else {
            TierScope::PerMeter
        },
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
    type Error = Vec<Located>;

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
            .then_some(Located::new("", Error::MissingProvider));
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
                .chain(prices.err().into_iter().flatten())
                .collect()),
        }
    }
}

struct Sourced {
    price: Price,
    rate: UnitPrice,
    source: String,
    path: String,
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
            path: flat.path.clone(),
        }
    })
}

fn dedupe(
    sourced: impl Iterator<Item = Sourced>,
    problems: &mut Vec<Located>,
) -> BTreeMap<Price, UnitPrice> {
    let mut seen: BTreeMap<Price, (UnitPrice, String, String)> = BTreeMap::new();
    for Sourced {
        price,
        rate,
        source,
        path,
    } in sourced
    {
        match seen.entry(price) {
            Entry::Occupied(first) if first.get().0 == rate => {}
            Entry::Occupied(first) => problems.push(Located::new(
                path,
                Error::DuplicatePrice {
                    first: first.get().1.clone(),
                    second: source.clone(),
                },
            )),
            Entry::Vacant(slot) => {
                slot.insert((rate, source, path));
            }
        }
    }
    seen.into_iter()
        .map(|(price, (rate, _, _))| (price, rate))
        .collect()
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum TierBasis {
    PromptTokens,
    Results,
}

fn tier_basis(row: &TierRow) -> Result<(TierBasis, [u64; 2]), Located> {
    match (row.range, row.max_results_range) {
        (Some(range), None) => Ok((TierBasis::PromptTokens, range)),
        (None, Some(range)) => Ok((TierBasis::Results, range)),
        _ => Err(Located::new(row.path.clone(), Error::TierRange)),
    }
}

fn tier_prices(wire: &EntryWire, problems: &mut Vec<Located>) -> Vec<Sourced> {
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
        problems.push(Located::from_key("tiered_pricing", Error::MixedTierBasis));
        return Vec::new();
    }
    if basis == TierBasis::PromptTokens {
        problems.extend(
            wire.prices
                .iter()
                .filter(|flat| flat.flat.band != Band::Base)
                .map(|flat| Located::new(flat.path.clone(), Error::BandsFromTwoSources)),
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

fn ladder_problems(basis: TierBasis, ordered: &[(TierBasis, [u64; 2], &TierRow)]) -> Vec<Located> {
    let step = match basis {
        TierBasis::PromptTokens => 0,
        TierBasis::Results => 1,
    };
    let reversed = ordered
        .iter()
        .filter(|(_, [start, end], _)| end < start)
        .map(|(.., row)| Located::new(row.path.clone(), Error::TierReversed));
    let first = ordered
        .first()
        .filter(|(_, [start, _], _)| *start > step)
        .map(|(_, [start, _], row)| {
            Located::new(
                row.path.clone(),
                Error::TierGap {
                    start: *start,
                    expected: 0,
                },
            )
        });
    let gaps = ordered.windows(2).filter_map(|pair| {
        let (_, [_, end], _) = pair[0];
        let (_, [start, _], row) = pair[1];
        (end.checked_add(step) != Some(start)).then(|| {
            Located::new(
                row.path.clone(),
                Error::TierGap {
                    start,
                    expected: end,
                },
            )
        })
    });
    reversed.chain(first).chain(gaps).collect()
}

fn tier_price(basis: TierBasis, start: u64, flat: &FlatPrice) -> Result<Sourced, Located> {
    if flat.flat.has_suffix() {
        return Err(Located::new(
            flat.path.clone(),
            Error::SuffixInsideBlock {
                block: "tiered_pricing",
            },
        ));
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
            return Err(Located::new(flat.path.clone(), Error::NotPricedByResults));
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
        path: flat.path.clone(),
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
                path: pointer_child(
                    &pointer_for_key("search_context_cost_per_query"),
                    &format!("search_context_size_{name}"),
                ),
            })
        })
    })
}

fn guardrail_prices(wire: &EntryWire) -> impl Iterator<Item = Sourced> + '_ {
    wire.guardrail.iter().flatten().map(|(unit, rate)| Sourced {
        price: Price::online(Meter::Guardrail(GuardrailUnit::new(unit)), When::default()),
        rate: *rate,
        source: format!("guardrail_cost_per_unit.{unit}"),
        path: pointer_child(&pointer_for_key("guardrail_cost_per_unit"), unit),
    })
}

fn off_peak(block: &OffPeakBlock, problems: &mut Vec<Located>) -> Option<OffPeak> {
    let rates = off_peak_rates(&block.prices, problems);
    let schedule = schedule(block, problems)?;
    let off_peak = OffPeak::new(schedule, rates);
    if off_peak.is_none() {
        problems.push(Located::from_key(
            "off_peak_pricing",
            Error::NoOffPeakPrices,
        ));
    }
    off_peak
}

fn off_peak_rates(prices: &[FlatPrice], problems: &mut Vec<Located>) -> OffPeakRates {
    let mut rates = OffPeakRates::default();
    for flat in prices {
        if flat.flat.has_suffix() {
            problems.push(Located::new(
                flat.path.clone(),
                Error::SuffixInsideBlock {
                    block: "off_peak_pricing",
                },
            ));
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
                problems.push(Located::new(flat.path.clone(), Error::NotAnOffPeakRate));
                continue;
            }
        };
        *slot = Some(flat.price);
    }
    rates
}

fn schedule(block: &OffPeakBlock, problems: &mut Vec<Located>) -> Option<Schedule> {
    let empty = |key: String| Located::from_path(&key, Error::EmptyList);
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
        problems.push(Located::from_key("off_peak_pricing", Error::NoOffPeakHours));
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

    fn prices(entry: Value) -> Result<Prices, Vec<Located>> {
        Prices::try_from(&wire(entry))
    }

    fn variant_name(error: &Error) -> &'static str {
        match error {
            Error::Json(_) => "Json",
            Error::Empty => "Empty",
            Error::BelowMinimum { .. } => "BelowMinimum",
            Error::Shrunk { .. } => "Shrunk",
            Error::NotAnObject => "NotAnObject",
            Error::Invalid(_) => "Invalid",
            Error::UnknownBlockKey => "UnknownBlockKey",
            Error::MissingProvider => "MissingProvider",
            Error::UnknownTimezone { .. } => "UnknownTimezone",
            Error::AliasShadowsModel { .. } => "AliasShadowsModel",
            Error::AliasTaken { .. } => "AliasTaken",
            Error::DuplicatePrice { .. } => "DuplicatePrice",
            Error::SuffixInsideBlock { .. } => "SuffixInsideBlock",
            Error::TierRange => "TierRange",
            Error::TierReversed => "TierReversed",
            Error::TierGap { .. } => "TierGap",
            Error::MixedTierBasis => "MixedTierBasis",
            Error::NotPricedByResults => "NotPricedByResults",
            Error::BandsFromTwoSources => "BandsFromTwoSources",
            Error::NotAnOffPeakRate => "NotAnOffPeakRate",
            Error::NoOffPeakPrices => "NoOffPeakPrices",
            Error::NoOffPeakHours => "NoOffPeakHours",
            Error::EmptyList => "EmptyList",
            Error::InvalidDate => "InvalidDate",
        }
    }

    #[rstest]
    #[case::json(Error::Json(serde_json::from_str::<Value>("{").unwrap_err()), true, false)]
    #[case::empty(Error::Empty, true, false)]
    #[case::below_minimum(Error::BelowMinimum { actual: 1, minimum: 2 }, true, false)]
    #[case::shrunk(Error::Shrunk { actual: 1, reference: 3, ratio: 0.5 }, true, false)]
    #[case::not_an_object(Error::NotAnObject, false, true)]
    #[case::invalid(Error::Invalid(serde_json::from_str::<Value>("x").unwrap_err()), false, true)]
    #[case::unknown_block_key(Error::UnknownBlockKey, false, true)]
    #[case::missing_provider(Error::MissingProvider, false, false)]
    #[case::unknown_timezone(Error::UnknownTimezone { zone: "invalid".into() }, false, true)]
    #[case::alias_shadows_model(Error::AliasShadowsModel { alias: "model".into() }, false, true)]
    #[case::alias_taken(Error::AliasTaken { alias: "alias".into(), taken_by: "owner".into() }, false, true)]
    #[case::duplicate_price(Error::DuplicatePrice { first: "a".into(), second: "b".into() }, false, true)]
    #[case::suffix_inside_block(Error::SuffixInsideBlock { block: "tiered_pricing" }, false, true)]
    #[case::tier_range(Error::TierRange, false, true)]
    #[case::tier_reversed(Error::TierReversed, false, true)]
    #[case::tier_gap(Error::TierGap { start: 1, expected: 0 }, false, true)]
    #[case::mixed_tier_basis(Error::MixedTierBasis, false, true)]
    #[case::not_priced_by_results(Error::NotPricedByResults, false, true)]
    #[case::bands_from_two_sources(Error::BandsFromTwoSources, false, true)]
    #[case::not_an_off_peak_rate(Error::NotAnOffPeakRate, false, true)]
    #[case::no_off_peak_prices(Error::NoOffPeakPrices, false, true)]
    #[case::no_off_peak_hours(Error::NoOffPeakHours, false, true)]
    #[case::empty_list(Error::EmptyList, false, true)]
    #[case::invalid_date(Error::InvalidDate, false, true)]
    fn policy_classifies_each_error(
        #[case] error: Error,
        #[case] is_fatal: bool,
        #[case] rejects: bool,
    ) {
        assert_eq!(fatal(&error), is_fatal);
        assert_eq!(rejects_row(&error), rejects);
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
    #[case::character_band_is_a_prompt_token_band(
        json!({"input_cost_per_character_above_128k_tokens": 1e-6}),
        online(Meter::Input(InputUnit::Character), ServiceTier::Standard, Band::over_prompt_tokens(128_000))
    )]
    #[case::guardrail(
        json!({"guardrail_cost_per_unit": {"contentPolicyUnits": 1e-6}}),
        online(Meter::Guardrail(GuardrailUnit::new("contentPolicyUnits")), ServiceTier::Standard, Band::Base)
    )]
    fn every_spelling_lowers_to_one_price(#[case] entry: Value, #[case] expected: Price) {
        assert_eq!(prices(entry).unwrap().get(&expected), UnitPrice::new(1e-6));
    }

    #[rstest]
    fn two_spellings_that_agree_are_one_price() {
        let prices = prices(json!({
            "input_cost_per_token": 1e-6,
            "tiered_pricing": [
                {"range": [0, 10], "input_cost_per_token": 1e-6},
                {"range": [10, 20], "input_cost_per_token": 2e-6}
            ]
        }))
        .unwrap();
        assert_eq!(
            prices.get(&Price::online(TEXT_INPUT, When::default())),
            UnitPrice::new(1e-6)
        );
    }

    #[rstest]
    fn token_tiers_lower_to_prompt_bands_at_each_start() {
        let prices = prices(json!({"tiered_pricing": [
            {"range": [0, 256000], "input_cost_per_token": 1e-6},
            {"range": [256000, 1000000], "input_cost_per_token": 2e-6}
        ]}))
        .unwrap();
        assert_eq!(
            prices.band(ServiceTier::Standard, 256_001),
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
        "/tiered_pricing/0/input_cost_per_token",
        "DuplicatePrice"
    )]
    #[case::zero_band_is_the_base_price(
        json!({"input_cost_per_token": 1e-6, "input_cost_per_token_above_0k_tokens": 2e-6}),
        "/input_cost_per_token_above_0k_tokens",
        "DuplicatePrice"
    )]
    #[case::tier_without_range(
        json!({"tiered_pricing": [{"input_cost_per_token": 1e-6}]}),
        "/tiered_pricing/0",
        "TierRange"
    )]
    #[case::tier_with_suffix(
        json!({"tiered_pricing": [{"range": [0, 10], "input_cost_per_token_priority": 1e-6}]}),
        "/tiered_pricing/0/input_cost_per_token_priority",
        "SuffixInsideBlock"
    )]
    #[case::reversed_range(
        json!({"tiered_pricing": [{"range": [0, 10], "input_cost_per_token": 1e-6}, {"range": [10, 5], "input_cost_per_token": 1e-6}]}),
        "/tiered_pricing/1",
        "TierReversed"
    )]
    #[case::gap_between_tiers(
        json!({"tiered_pricing": [
            {"range": [0, 10], "input_cost_per_token": 1e-6},
            {"range": [20, 30], "input_cost_per_token": 2e-6}
        ]}),
        "/tiered_pricing/1",
        "TierGap"
    )]
    #[case::ladder_not_from_zero(
        json!({"tiered_pricing": [{"range": [5, 10], "input_cost_per_token": 1e-6}]}),
        "/tiered_pricing/0",
        "TierGap"
    )]
    #[case::result_ladder_overlaps(
        json!({"tiered_pricing": [
            {"max_results_range": [1, 10], "input_cost_per_query": 1e-6},
            {"max_results_range": [10, 20], "input_cost_per_query": 2e-6}
        ]}),
        "/tiered_pricing/1",
        "TierGap"
    )]
    #[case::mixed_basis(
        json!({"tiered_pricing": [
            {"range": [0, 10], "input_cost_per_token": 1e-6},
            {"max_results_range": [11, 20], "input_cost_per_query": 2e-6}
        ]}),
        "/tiered_pricing",
        "MixedTierBasis"
    )]
    #[case::result_tier_prices_a_token(
        json!({"tiered_pricing": [{"max_results_range": [0, 10], "input_cost_per_token": 1e-6}]}),
        "/tiered_pricing/0/input_cost_per_token",
        "NotPricedByResults"
    )]
    #[case::two_band_sources(
        json!({
            "output_cost_per_token_above_200k_tokens": 1e-6,
            "tiered_pricing": [{"range": [0, 10], "input_cost_per_token": 1e-6}]
        }),
        "/output_cost_per_token_above_200k_tokens",
        "BandsFromTwoSources"
    )]
    #[case::off_peak_without_hours(
        json!({"off_peak_pricing": {"input_cost_per_token": 1e-6}}),
        "/off_peak_pricing",
        "NoOffPeakHours"
    )]
    #[case::off_peak_without_prices(
        json!({"off_peak_pricing": {"hours_utc": "00:00-01:00"}}),
        "/off_peak_pricing",
        "NoOffPeakPrices"
    )]
    #[case::off_peak_empty_windows(
        json!({"off_peak_pricing": {"windows": [], "input_cost_per_token": 1e-6}}),
        "/off_peak_pricing/windows",
        "EmptyList"
    )]
    #[case::off_peak_empty_hours(
        json!({"off_peak_pricing": {"hours_utc": [], "input_cost_per_token": 1e-6}}),
        "/off_peak_pricing/hours_utc",
        "EmptyList"
    )]
    #[case::off_peak_empty_weekdays(
        json!({"off_peak_pricing": {
            "windows": [{"hours_utc": "00:00-01:00", "weekdays": []}],
            "input_cost_per_token": 1e-6
        }}),
        "/off_peak_pricing/windows/0/weekdays",
        "EmptyList"
    )]
    #[case::off_peak_image_price(
        json!({"off_peak_pricing": {"hours_utc": "00:00-01:00", "output_cost_per_image": 1}}),
        "/off_peak_pricing/output_cost_per_image",
        "NotAnOffPeakRate"
    )]
    #[case::off_peak_band(
        json!({"off_peak_pricing": {"hours_utc": "00:00-01:00", "input_cost_per_token_above_200k_tokens": 1}}),
        "/off_peak_pricing/input_cost_per_token_above_200k_tokens",
        "SuffixInsideBlock"
    )]
    fn rules_across_keys_name_where_each_price_came_from(
        #[case] entry: Value,
        #[case] expected_path: &str,
        #[case] expected_variant: &str,
    ) {
        let problems = prices(entry).unwrap_err();
        assert!(
            problems.iter().any(|problem| problem.path == expected_path
                && variant_name(&problem.error) == expected_variant),
            "{problems:?}"
        );
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
                .get(&Uplift::NonGlobalEndpoint)
                .map(Multiplier::get),
            Some(1.1)
        );
        assert_eq!(
            adjustments
                .get(&Uplift::Residency(Residency::Eu))
                .map(Multiplier::get),
            Some(1.2)
        );
        assert_eq!(adjustments.get(&Uplift::Residency(Residency::Us)), None);
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

    #[rstest]
    #[case::tokens_win_over_characters(
        json!({"litellm_provider": "vertex_ai-language-models", "input_cost_per_character": 1e-6, "input_cost_per_token": 1e-6}),
        Billing::default()
    )]
    #[case::characters_only(
        json!({"litellm_provider": "vertex_ai-language-models", "input_cost_per_character": 1e-6}),
        Billing { text_unit: TextUnit::Character, ..Billing::default() }
    )]
    #[case::xai_band_is_inclusive(
        json!({"litellm_provider": "xai", "input_cost_per_token": 1e-6}),
        Billing { band_edge: BandEdge::Inclusive, ..Billing::default() }
    )]
    #[case::fireworks_without_cache_read_borrows_half(
        json!({"litellm_provider": "fireworks_ai", "input_cost_per_token": 1e-6}),
        Billing { cache_read_fallback: CacheReadFallback::HalfInputRate, ..Billing::default() }
    )]
    #[case::fireworks_with_cache_read_keeps_it(
        json!({"litellm_provider": "fireworks_ai", "input_cost_per_token": 1e-6, "cache_read_input_token_cost": 1e-7}),
        Billing::default()
    )]
    #[case::transcription_prices_the_audio_file(
        json!({"litellm_provider": "openai", "mode": "audio_transcription", "input_cost_per_second": 1e-4}),
        Billing { wall_clock: WallClock::AudioFile, ..Billing::default() }
    )]
    #[case::provisioned_chat_prices_response_time(
        json!({"litellm_provider": "bedrock", "mode": "chat", "cost_per_second": 1e-2}),
        Billing::default()
    )]
    #[case::image_generation_is_never_wall_clock(
        json!({"litellm_provider": "openai", "mode": "image_generation"}),
        Billing { wall_clock: WallClock::Never, ..Billing::default() }
    )]
    #[case::gemini_per_query_search(
        json!({"litellm_provider": "gemini", "web_search_billing_unit": "per_query"}),
        Billing { web_search: WebSearchUnit::PerQuery, ..Billing::default() }
    )]
    #[case::dashscope_tiers_bill_the_whole_call(
        json!({"litellm_provider": "dashscope", "tiered_pricing": [{"range": [0, 32000], "input_cost_per_token": 1e-6}]}),
        Billing { tier_scope: TierScope::WholeCall, ..Billing::default() }
    )]
    fn billing_is_lowered_from_the_row_and_its_provider(
        #[case] entry: Value,
        #[case] expected: Billing,
    ) {
        assert_eq!(prices(entry).unwrap().billing(), expected);
    }
}
