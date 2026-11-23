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

use crate::error::{Error, Repr, pointer_child, pointer_for_key};
use crate::model::Model;
use crate::vocabulary::Mode;
use crate::wire::{
    EntryWire, FlatKey, FlatPrice, OffPeakBlock, TierRow, TierSuffix, WebSearchBillingUnit,
};

impl TryFrom<&EntryWire> for Prices {
    type Error = Vec<Error>;

    fn try_from(wire: &EntryWire) -> Result<Self, Self::Error> {
        let (prices, problems) = lowered(wire, &wire.prices, Tiers::Priced);
        if problems.is_empty() {
            Ok(prices)
        } else {
            Err(problems)
        }
    }
}

/// The prices main bills a row at. Unlike `Prices::try_from` this never fails, because a
/// deployment's `model_info` merged over a catalog row can hold what the catalog checks reject
#[derive(Clone, Debug, PartialEq)]
pub enum BilledPrices {
    /// One price list, whatever the prompt size
    Flat(Prices),
    /// Token tiers price input, so main bills the whole call from the tier the prompt picks
    Tiered {
        /// The tiers as prompt bands, beside the flat prices they do not replace
        tiers: Prices,
        /// The flat prices alone. An empty prompt picks no tier and reads these
        empty_prompt: Prices,
    },
}

impl BilledPrices {
    pub fn for_prompt(&self, prompt_tokens: u64) -> &Prices {
        match self {
            Self::Flat(prices) => prices,
            Self::Tiered { empty_prompt, .. } if prompt_tokens == 0 => empty_prompt,
            Self::Tiered { tiers, .. } => tiers,
        }
    }
}

/// Lowers `wire` the way main reads it
/// - A row whose token tiers price input is `Tiered`. The flat text token prices the tiers replace
///   are dropped from `tiers`, and a tier's missing cache and reasoning rates are that tier's
///   input and output rate
/// - Result tiers are left out, the search calculator prices them
/// - A modality input token price main could not parse is 0, where a missing one borrows the text rate
/// - Any other contradiction keeps the price read first
pub fn billed_prices(wire: &EntryWire) -> BilledPrices {
    let flat: Vec<FlatPrice> = wire
        .prices
        .iter()
        .cloned()
        .chain(unparsable_modality_prices(wire))
        .collect();
    let empty_prompt = lowered(wire, &flat, Tiers::Unread).0;
    if !tiers_price(wire, &TEXT_INPUT) {
        return BilledPrices::Flat(empty_prompt);
    }
    let beside_tiers: Vec<FlatPrice> = flat
        .into_iter()
        .filter(|flat| !replaced_by_tiers(wire, &flat.flat))
        .collect();
    BilledPrices::Tiered {
        tiers: lowered(wire, &beside_tiers, Tiers::WholeCall).0,
        empty_prompt,
    }
}

fn unparsable_modality_prices(wire: &EntryWire) -> impl Iterator<Item = FlatPrice> + '_ {
    wire.problems.iter().filter_map(|problem| {
        let key = problem.key().filter(|_| problem.field().is_none())?;
        let flat = FlatKey::parse(&key)?;
        matches!(
            flat.meter,
            Meter::Input(InputUnit::Token(
                TokenKind::Audio | TokenKind::Image | TokenKind::Video
            ))
        )
        .then_some(FlatPrice {
            key,
            path: problem.path().to_owned(),
            flat,
            price: UnitPrice::new(0.0)?,
        })
    })
}

fn tiers_price(wire: &EntryWire, meter: &Meter) -> bool {
    wire.tier_rows
        .iter()
        .flatten()
        .any(|row| row.range.is_some() && row.prices.iter().any(|price| price.flat.meter == *meter))
}

/// Whether main ignores a flat price because the row has token tiers. A tier supplies input and
/// the cache rates, which fall back to the tier's input. It supplies output and reasoning only
/// when a tier prices output, otherwise the standard flat output price stands in
fn replaced_by_tiers(wire: &EntryWire, flat: &FlatKey) -> bool {
    let text_output = Meter::Output(OutputUnit::Token(TokenKind::Text));
    if matches!(flat.tier, Some(TierSuffix::Batches | TierSuffix::Decisions)) {
        return false;
    }
    match &flat.meter {
        Meter::Input(InputUnit::Token(TokenKind::Text))
        | Meter::CacheRead {
            kind: TokenKind::Text,
            ..
        }
        | Meter::CacheWrite {
            kind: TokenKind::Text,
            ..
        } => true,
        Meter::Output(OutputUnit::Token(TokenKind::Text)) => {
            tiers_price(wire, &text_output) || flat.has_suffix()
        }
        Meter::Output(OutputUnit::ReasoningToken) => {
            tiers_price(wire, &text_output) || tiers_price(wire, &flat.meter)
        }
        _ => false,
    }
}

const TEXT_INPUT: Meter = Meter::Input(InputUnit::Token(TokenKind::Text));

/// How `lowered` reads `tiered_pricing`
#[derive(Clone, Copy, PartialEq, Eq)]
enum Tiers {
    /// Every tier is a price, the catalog's reading
    Priced,
    /// Token tiers are the prices of the whole call, main's reading of a prompt that picks a tier
    WholeCall,
    /// No tier is read, main's reading of an empty prompt or of tiers that price no input
    Unread,
}

fn lowered(wire: &EntryWire, flat: &[FlatPrice], tiers: Tiers) -> (Prices, Vec<Error>) {
    let mut problems = Vec::new();
    let tiered = match tiers {
        Tiers::Priced => tier_prices(wire, &mut problems),
        Tiers::WholeCall => with_tier_fallbacks(tier_prices(wire, &mut problems)),
        Tiers::Unread => Vec::new(),
    };
    let sourced = flat_prices(flat)
        .chain(tiered)
        .chain(search_context_prices(wire))
        .chain(guardrail_prices(wire));
    let rates = dedupe(sourced, &mut problems);
    let off_peak = wire
        .off_peak
        .as_ref()
        .and_then(|block| off_peak(block, &mut problems));
    let billing = billing(wire, &rates);
    (Prices::new(rates, off_peak, billing), problems)
}

/// Main fills a tier's missing cache-read and cache-write rates with that tier's input rate and its
/// missing reasoning rate with that tier's output rate, before any off-peak swap
fn with_tier_fallbacks(tiered: Vec<Sourced>) -> Vec<Sourced> {
    const TEXT_OUTPUT: Meter = Meter::Output(OutputUnit::Token(TokenKind::Text));
    const FROM_INPUT: [Meter; 2] = [
        Meter::CacheRead {
            kind: TokenKind::Text,
            mode: CacheReadMode::Implicit,
        },
        Meter::CacheWrite {
            kind: TokenKind::Text,
            ttl: CacheTtl::Default,
        },
    ];
    const FROM_OUTPUT: [Meter; 1] = [Meter::Output(OutputUnit::ReasoningToken)];
    let filled: Vec<Sourced> = tiered
        .iter()
        .flat_map(|lender| {
            let borrowers: &[Meter] = if lender.price.meter == TEXT_INPUT {
                &FROM_INPUT
            } else if lender.price.meter == TEXT_OUTPUT {
                &FROM_OUTPUT
            } else {
                &[]
            };
            borrowers
                .iter()
                .filter(|meter| {
                    !tiered.iter().any(|priced| {
                        priced.price.meter == **meter && priced.price.when == lender.price.when
                    })
                })
                .map(|meter| Sourced {
                    price: Price::online(meter.clone(), lender.price.when),
                    rate: lender.rate,
                    source: lender.source.clone(),
                    path: lender.path.clone(),
                })
        })
        .collect();
    tiered.into_iter().chain(filled).collect()
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
    let prices_cache_read = rates.contains_key(&Price::online(
        Meter::CacheRead {
            kind: TokenKind::Text,
            mode: CacheReadMode::Implicit,
        },
        When::default(),
    ));
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
                (Uplift::FedrampEndpoint, wire.fedramp_uplift),
            ]
            .into_iter()
            .filter_map(|(uplift, factor)| Some((uplift, factor?))),
        )
    }
}

impl TryFrom<EntryWire> for Model {
    type Error = Vec<Error>;

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
        let missing_provider = attributes.litellm_provider.is_none().then_some(Error::new(
            pointer_for_key("litellm_provider"),
            Repr::MissingProvider,
        ));
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
    problems: &mut Vec<Error>,
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
            Entry::Occupied(first) => problems.push(Error::new(
                path,
                Repr::DuplicatePrice {
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

fn tier_basis(row: &TierRow) -> Result<(TierBasis, [u64; 2]), Error> {
    match (row.range, row.max_results_range) {
        (Some(range), None) => Ok((TierBasis::PromptTokens, range)),
        (None, Some(range)) => Ok((TierBasis::Results, range)),
        _ => Err(Error::new(row.path.clone(), Repr::TierRange)),
    }
}

fn tier_prices(wire: &EntryWire, problems: &mut Vec<Error>) -> Vec<Sourced> {
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
        problems.push(Error::from_key("tiered_pricing", Repr::MixedTierBasis));
        return Vec::new();
    }
    if basis == TierBasis::PromptTokens {
        problems.extend(
            wire.prices
                .iter()
                .filter(|flat| flat.flat.band != Band::Base)
                .map(|flat| Error::new(flat.path.clone(), Repr::BandsFromTwoSources)),
        );
    }
    let mut ordered = based;
    ordered.sort_by_key(|(_, [start, _], _)| *start);
    problems.extend(ladder_problems(basis, &ordered));
    let floors = std::iter::once(0).chain(ordered.iter().map(|(_, [_, end], _)| *end));
    ordered
        .iter()
        .zip(floors)
        .flat_map(|((basis, [start, _], row), floor)| {
            let over = match basis {
                TierBasis::PromptTokens => floor,
                TierBasis::Results => start.saturating_sub(1),
            };
            row.prices.iter().map(move |flat| (*basis, over, flat))
        })
        .filter_map(|(basis, over, flat)| match tier_price(basis, over, flat) {
            Ok(sourced) => Some(sourced),
            Err(problem) => {
                problems.push(problem);
                None
            }
        })
        .collect()
}

fn ladder_problems(basis: TierBasis, ordered: &[(TierBasis, [u64; 2], &TierRow)]) -> Vec<Error> {
    let step = match basis {
        TierBasis::PromptTokens => 0,
        TierBasis::Results => 1,
    };
    let reversed = ordered
        .iter()
        .filter(|(_, [start, end], _)| end < start)
        .map(|(.., row)| Error::new(row.path.clone(), Repr::TierReversed));
    let first = ordered
        .first()
        .filter(|(_, [start, _], _)| *start > step)
        .map(|(_, [start, _], row)| {
            Error::new(
                row.path.clone(),
                Repr::TierGap {
                    start: *start,
                    expected: 0,
                },
            )
        });
    let gaps = ordered.windows(2).filter_map(|pair| {
        let (_, [_, end], _) = pair[0];
        let (_, [start, _], row) = pair[1];
        (end.checked_add(step) != Some(start)).then(|| {
            Error::new(
                row.path.clone(),
                Repr::TierGap {
                    start,
                    expected: end,
                },
            )
        })
    });
    reversed.chain(first).chain(gaps).collect()
}

/// `over` is where the tier starts: the end of the token tier before it, so a ladder written with a
/// one token gap still bills the next tier from its first token, or one less than a result tier's start
fn tier_price(basis: TierBasis, over: u64, flat: &FlatPrice) -> Result<Sourced, Error> {
    if flat.flat.has_suffix() {
        return Err(Error::new(
            flat.path.clone(),
            Repr::SuffixInsideBlock {
                block: "tiered_pricing",
            },
        ));
    }
    let (meter, band) = match (basis, &flat.flat.meter) {
        (TierBasis::PromptTokens, meter) => (meter.clone(), Band::over_prompt_tokens(over)),
        (TierBasis::Results, Meter::Input(InputUnit::Query { over_results: None })) => (
            Meter::Input(InputUnit::Query {
                over_results: NonZeroU64::new(over),
            }),
            Band::Base,
        ),
        (TierBasis::Results, _) => {
            return Err(Error::new(flat.path.clone(), Repr::NotPricedByResults));
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

fn off_peak(block: &OffPeakBlock, problems: &mut Vec<Error>) -> Option<OffPeak> {
    let rates = off_peak_rates(&block.prices, problems);
    let schedule = schedule(block, problems)?;
    let off_peak = OffPeak::new(schedule, rates);
    if off_peak.is_none() {
        problems.push(Error::from_key("off_peak_pricing", Repr::NoOffPeakPrices));
    }
    off_peak
}

fn off_peak_rates(prices: &[FlatPrice], problems: &mut Vec<Error>) -> OffPeakRates {
    let mut rates = OffPeakRates::default();
    for flat in prices {
        if flat.flat.has_suffix() {
            problems.push(Error::new(
                flat.path.clone(),
                Repr::SuffixInsideBlock {
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
                problems.push(Error::new(flat.path.clone(), Repr::NotAnOffPeakRate));
                continue;
            }
        };
        *slot = Some(flat.price);
    }
    rates
}

fn schedule(block: &OffPeakBlock, problems: &mut Vec<Error>) -> Option<Schedule> {
    let empty = |key: String| Error::from_path(&key, Repr::EmptyList);
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
        problems.push(Error::from_key("off_peak_pricing", Repr::NoOffPeakHours));
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

    fn wire(entry: Value) -> EntryWire {
        let Value::Object(fields) = entry else {
            panic!("entry must be an object")
        };
        let wire = EntryWire::read(&fields);
        assert!(wire.problems.is_empty(), "{:?}", wire.problems);
        wire
    }

    fn prices(entry: Value) -> Result<Prices, Vec<Error>> {
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
            prices
                .iter()
                .map(|(price, rate)| (price.when.band, rate.get()))
                .collect::<Vec<_>>(),
            [
                (Band::Base, 1e-6),
                (Band::over_prompt_tokens(256_000), 2e-6)
            ]
        );
    }

    #[rstest]
    #[case::two_spellings_of_one_price(
        json!({
            "input_cost_per_token": 1e-6,
            "tiered_pricing": [{"range": [0, 10], "input_cost_per_token": 2e-6}]
        }),
        "/tiered_pricing/0/input_cost_per_token",
        "duplicate_price"
    )]
    #[case::zero_band_is_the_base_price(
        json!({"input_cost_per_token": 1e-6, "input_cost_per_token_above_0k_tokens": 2e-6}),
        "/input_cost_per_token_above_0k_tokens",
        "duplicate_price"
    )]
    #[case::tier_without_range(
        json!({"tiered_pricing": [{"input_cost_per_token": 1e-6}]}),
        "/tiered_pricing/0",
        "tier_range"
    )]
    #[case::tier_with_suffix(
        json!({"tiered_pricing": [{"range": [0, 10], "input_cost_per_token_priority": 1e-6}]}),
        "/tiered_pricing/0/input_cost_per_token_priority",
        "suffix_inside_block"
    )]
    #[case::reversed_range(
        json!({"tiered_pricing": [{"range": [0, 10], "input_cost_per_token": 1e-6}, {"range": [10, 5], "input_cost_per_token": 1e-6}]}),
        "/tiered_pricing/1",
        "tier_reversed"
    )]
    #[case::gap_between_tiers(
        json!({"tiered_pricing": [
            {"range": [0, 10], "input_cost_per_token": 1e-6},
            {"range": [20, 30], "input_cost_per_token": 2e-6}
        ]}),
        "/tiered_pricing/1",
        "tier_gap"
    )]
    #[case::ladder_not_from_zero(
        json!({"tiered_pricing": [{"range": [5, 10], "input_cost_per_token": 1e-6}]}),
        "/tiered_pricing/0",
        "tier_gap"
    )]
    #[case::result_ladder_overlaps(
        json!({"tiered_pricing": [
            {"max_results_range": [1, 10], "input_cost_per_query": 1e-6},
            {"max_results_range": [10, 20], "input_cost_per_query": 2e-6}
        ]}),
        "/tiered_pricing/1",
        "tier_gap"
    )]
    #[case::mixed_basis(
        json!({"tiered_pricing": [
            {"range": [0, 10], "input_cost_per_token": 1e-6},
            {"max_results_range": [11, 20], "input_cost_per_query": 2e-6}
        ]}),
        "/tiered_pricing",
        "mixed_tier_basis"
    )]
    #[case::result_tier_prices_a_token(
        json!({"tiered_pricing": [{"max_results_range": [0, 10], "input_cost_per_token": 1e-6}]}),
        "/tiered_pricing/0/input_cost_per_token",
        "not_priced_by_results"
    )]
    #[case::two_band_sources(
        json!({
            "output_cost_per_token_above_200k_tokens": 1e-6,
            "tiered_pricing": [{"range": [0, 10], "input_cost_per_token": 1e-6}]
        }),
        "/output_cost_per_token_above_200k_tokens",
        "bands_from_two_sources"
    )]
    #[case::off_peak_without_hours(
        json!({"off_peak_pricing": {"input_cost_per_token": 1e-6}}),
        "/off_peak_pricing",
        "no_off_peak_hours"
    )]
    #[case::off_peak_without_prices(
        json!({"off_peak_pricing": {"hours_utc": "00:00-01:00"}}),
        "/off_peak_pricing",
        "no_off_peak_prices"
    )]
    #[case::off_peak_empty_windows(
        json!({"off_peak_pricing": {"windows": [], "input_cost_per_token": 1e-6}}),
        "/off_peak_pricing/windows",
        "empty_list"
    )]
    #[case::off_peak_empty_hours(
        json!({"off_peak_pricing": {"hours_utc": [], "input_cost_per_token": 1e-6}}),
        "/off_peak_pricing/hours_utc",
        "empty_list"
    )]
    #[case::off_peak_empty_weekdays(
        json!({"off_peak_pricing": {
            "windows": [{"hours_utc": "00:00-01:00", "weekdays": []}],
            "input_cost_per_token": 1e-6
        }}),
        "/off_peak_pricing/windows/0/weekdays",
        "empty_list"
    )]
    #[case::off_peak_image_price(
        json!({"off_peak_pricing": {"hours_utc": "00:00-01:00", "output_cost_per_image": 1}}),
        "/off_peak_pricing/output_cost_per_image",
        "not_an_off_peak_rate"
    )]
    #[case::off_peak_band(
        json!({"off_peak_pricing": {"hours_utc": "00:00-01:00", "input_cost_per_token_above_200k_tokens": 1}}),
        "/off_peak_pricing/input_cost_per_token_above_200k_tokens",
        "suffix_inside_block"
    )]
    fn rules_across_keys_name_where_each_price_came_from(
        #[case] entry: Value,
        #[case] expected_path: &str,
        #[case] expected_variant: &str,
    ) {
        let problems = prices(entry).unwrap_err();
        assert!(
            problems
                .iter()
                .any(|problem| problem.path() == expected_path
                    && <&str>::from(problem.kind()) == expected_variant),
            "{problems:?}"
        );
    }

    fn billed(entry: Value) -> Vec<(Meter, ServiceTier, Band, f64)> {
        let Value::Object(fields) = entry else {
            panic!("entry must be an object")
        };
        billed_prices(&EntryWire::read(&fields))
            .for_prompt(1)
            .iter()
            .map(|(price, rate)| {
                (
                    price.meter.clone(),
                    price.when.tier,
                    price.when.band,
                    rate.get(),
                )
            })
            .collect()
    }

    const TEXT_OUTPUT: Meter = Meter::Output(OutputUnit::Token(TokenKind::Text));
    const REASONING: Meter = Meter::Output(OutputUnit::ReasoningToken);
    const CACHE_READ: Meter = Meter::CacheRead {
        kind: TokenKind::Text,
        mode: CacheReadMode::Implicit,
    };
    const CACHE_WRITE: Meter = Meter::CacheWrite {
        kind: TokenKind::Text,
        ttl: CacheTtl::Default,
    };

    #[rstest]
    fn tiers_over_a_flat_priced_row_bill_text_tokens_from_the_tiers_alone() {
        use ServiceTier::Standard;
        let second = Band::over_prompt_tokens(256_000);
        let audio = Meter::Input(InputUnit::Token(TokenKind::Audio));

        assert_eq!(
            billed(json!({
                "input_cost_per_token": 9e-6,
                "input_cost_per_token_priority": 9e-6,
                "input_cost_per_token_above_200k_tokens": 9e-6,
                "output_cost_per_token": 9e-6,
                "output_cost_per_reasoning_token": 9e-6,
                "cache_read_input_token_cost": 9e-6,
                "cache_creation_input_token_cost_above_1hr": 9e-6,
                "input_cost_per_audio_token": 7e-6,
                "tiered_pricing": [
                    {"range": [0, 256000], "input_cost_per_token": 1e-6, "output_cost_per_token": 2e-6},
                    {"range": [256000, 1000000], "input_cost_per_token": 3e-6, "output_cost_per_token": 4e-6}
                ]
            })),
            [
                (TEXT_INPUT, Standard, Band::Base, 1e-6),
                (TEXT_INPUT, Standard, second, 3e-6),
                (audio, Standard, Band::Base, 7e-6),
                (TEXT_OUTPUT, Standard, Band::Base, 2e-6),
                (TEXT_OUTPUT, Standard, second, 4e-6),
                (REASONING, Standard, Band::Base, 2e-6),
                (REASONING, Standard, second, 4e-6),
                (CACHE_READ, Standard, Band::Base, 1e-6),
                (CACHE_READ, Standard, second, 3e-6),
                (CACHE_WRITE, Standard, Band::Base, 1e-6),
                (CACHE_WRITE, Standard, second, 3e-6),
            ]
        );
    }

    #[rstest]
    fn tiers_that_price_no_output_keep_the_standard_flat_output_and_reasoning() {
        use ServiceTier::Standard;

        assert_eq!(
            billed(json!({
                "input_cost_per_token": 9e-6,
                "output_cost_per_token": 5e-6,
                "output_cost_per_token_priority": 9e-6,
                "output_cost_per_token_above_200k_tokens": 9e-6,
                "output_cost_per_reasoning_token": 6e-6,
                "tiered_pricing": [{"range": [0, 256000], "input_cost_per_token": 1e-6}]
            })),
            [
                (TEXT_INPUT, Standard, Band::Base, 1e-6),
                (TEXT_OUTPUT, Standard, Band::Base, 5e-6),
                (REASONING, Standard, Band::Base, 6e-6),
                (CACHE_READ, Standard, Band::Base, 1e-6),
                (CACHE_WRITE, Standard, Band::Base, 1e-6),
            ]
        );
    }

    #[rstest]
    #[case::batch_prices(json!({
        "input_cost_per_token_batches": 5e-6,
        "tiered_pricing": [{"range": [0, 10], "input_cost_per_token": 1e-6}]
    }), 4)]
    #[case::result_tiers_do_not_replace_token_prices(json!({
        "input_cost_per_token": 5e-6,
        "cache_read_input_token_cost": 1e-6,
        "tiered_pricing": [{"max_results_range": [0, 25], "input_cost_per_query": 1e-3}]
    }), 2)]
    #[case::a_ladder_the_catalog_rejects_still_prices(json!({
        "tiered_pricing": [
            {"range": [0, 128000], "input_cost_per_token": 1e-6},
            {"range": [128001, 1000000], "input_cost_per_token": 2e-6}
        ]
    }), 6)]
    #[case::two_spellings_keep_the_first(json!({
        "input_cost_per_token": 1e-6,
        "input_cost_per_token_above_0k_tokens": 2e-6,
        "cache_read_input_token_cost": 1e-7
    }), 2)]
    fn billing_keeps_what_the_catalog_checks_would_reject(
        #[case] entry: Value,
        #[case] expected_prices: usize,
    ) {
        assert_eq!(billed(entry).len(), expected_prices);
    }

    fn billed_wire(entry: Value) -> BilledPrices {
        let Value::Object(fields) = entry else {
            panic!("entry must be an object")
        };
        billed_prices(&EntryWire::read(&fields))
    }

    fn rates(prices: &Prices) -> Vec<(Meter, Band, f64)> {
        prices
            .iter()
            .map(|(price, rate)| (price.meter.clone(), price.when.band, rate.get()))
            .collect()
    }

    #[rstest]
    fn an_empty_prompt_reads_the_flat_prices_of_a_tiered_row() {
        let billed = billed_wire(json!({
            "litellm_provider": "fireworks_ai",
            "input_cost_per_token": 9e-6,
            "output_cost_per_token": 8e-6,
            "tiered_pricing": [
                {"range": [0, 256000], "input_cost_per_token": 1e-6, "output_cost_per_token": 2e-6}
            ]
        }));

        assert_eq!(
            rates(billed.for_prompt(0)),
            [
                (TEXT_INPUT, Band::Base, 9e-6),
                (TEXT_OUTPUT, Band::Base, 8e-6)
            ]
        );
        assert_eq!(
            rates(billed.for_prompt(1)),
            [
                (TEXT_INPUT, Band::Base, 1e-6),
                (TEXT_OUTPUT, Band::Base, 2e-6),
                (REASONING, Band::Base, 2e-6),
                (CACHE_READ, Band::Base, 1e-6),
                (CACHE_WRITE, Band::Base, 1e-6),
            ]
        );
        assert_eq!(
            billed.for_prompt(0).billing(),
            Billing {
                cache_read_fallback: CacheReadFallback::HalfInputRate,
                ..Billing::default()
            }
        );
        assert_eq!(billed.for_prompt(1).billing(), Billing::default());
    }

    #[rstest]
    fn a_tier_only_row_has_no_price_for_an_empty_prompt() {
        let billed = billed_wire(json!({
            "litellm_provider": "dashscope",
            "tiered_pricing": [{"range": [0, 256000], "input_cost_per_token": 1e-6}]
        }));

        assert_eq!(rates(billed.for_prompt(0)), []);
    }

    #[rstest]
    fn a_row_without_token_tiers_bills_every_prompt_from_its_flat_prices() {
        let billed = billed_wire(json!({
            "input_cost_per_token": 9e-6,
            "tiered_pricing": [{"range": [0, 256000], "output_cost_per_token": 2e-6}]
        }));

        assert_eq!(billed.for_prompt(0), billed.for_prompt(1));
        assert_eq!(
            rates(billed.for_prompt(1)),
            [(TEXT_INPUT, Band::Base, 9e-6)]
        );
        assert_eq!(billed.for_prompt(1).billing(), Billing::default());
    }

    #[rstest]
    fn billing_leaves_result_tiers_to_the_search_calculator() {
        let billed = billed_wire(json!({"tiered_pricing": [
            {"max_results_range": [0, 25], "input_cost_per_query": 5e-3},
            {"max_results_range": [26, 100], "input_cost_per_query": 2.5e-2}
        ]}));

        assert_eq!(rates(billed.for_prompt(1)), []);
    }

    #[rstest]
    #[case::one_token_gap(json!([0, 128000]), json!([128001, 1000000]), 128_000)]
    #[case::contiguous(json!([0, 128000]), json!([128000, 1000000]), 128_000)]
    #[case::first_tier_not_from_zero(json!([5, 128000]), json!([128000, 1000000]), 128_000)]
    fn a_token_tier_starts_where_the_one_before_it_ends(
        #[case] first: Value,
        #[case] second: Value,
        #[case] expected_floor: u64,
    ) {
        let billed = billed_wire(json!({"tiered_pricing": [
            {"range": first, "input_cost_per_token": 1e-6},
            {"range": second, "input_cost_per_token": 2e-6}
        ]}));

        let input: Vec<_> = rates(billed.for_prompt(1))
            .into_iter()
            .filter(|(meter, ..)| *meter == TEXT_INPUT)
            .collect();
        assert_eq!(
            input,
            [
                (TEXT_INPUT, Band::Base, 1e-6),
                (TEXT_INPUT, Band::over_prompt_tokens(expected_floor), 2e-6)
            ]
        );
    }

    #[rstest]
    #[case::audio(
        "input_cost_per_audio_token",
        Some(Meter::Input(InputUnit::Token(TokenKind::Audio)))
    )]
    #[case::image_priority(
        "input_cost_per_image_token_priority",
        Some(Meter::Input(InputUnit::Token(TokenKind::Image)))
    )]
    #[case::text_input("input_cost_per_token", None)]
    #[case::cache_read("cache_read_input_token_cost", None)]
    #[case::output_audio("output_cost_per_audio_token", None)]
    #[case::character("input_cost_per_character", None)]
    fn only_a_modality_input_token_price_main_could_not_parse_is_free(
        #[case] key: &str,
        #[case] free: Option<Meter>,
    ) {
        let billed = billed_wire(json!({"output_cost_per_token": 2e-6, key: "invalid"}));
        let priced_at_zero: Vec<Meter> = billed
            .for_prompt(1)
            .iter()
            .filter(|(_, rate)| rate.get() == 0.0)
            .map(|(price, _)| price.meter.clone())
            .collect();

        assert_eq!(priced_at_zero, free.into_iter().collect::<Vec<_>>());
    }

    #[rstest]
    fn billing_drops_an_off_peak_block_with_no_hours() {
        let Value::Object(fields) = json!({
            "input_cost_per_token": 1e-6,
            "off_peak_pricing": {"input_cost_per_token": 5e-7}
        }) else {
            unreachable!()
        };
        let billed = billed_prices(&EntryWire::read(&fields));
        let prices = billed.for_prompt(1);

        assert_eq!(prices.off_peak(), None);
        assert_eq!(
            prices.get(&Price::online(TEXT_INPUT, When::default())),
            UnitPrice::new(1e-6)
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
            "regional_processing_uplift_multiplier_eu": "1.2",
            "fedramp_uplift_multiplier": 1.1
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
        assert_eq!(
            adjustments
                .get(&Uplift::FedrampEndpoint)
                .map(Multiplier::get),
            Some(1.1)
        );
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
            .map(|problem| litellm_error::Report::new(problem).to_string())
            .collect();

        assert_eq!(problems.len(), 3, "{problems:?}");
        assert!(problems[0].starts_with("mode: invalid value: unknown variant `chatt`"));
        assert_eq!(problems[1], "litellm_provider: required");
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
    #[case::fireworks_with_only_a_tier_cache_read_still_borrows_half(
        json!({"litellm_provider": "fireworks_ai", "input_cost_per_token": 1e-6, "cache_read_input_token_cost_priority": 1e-7}),
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
