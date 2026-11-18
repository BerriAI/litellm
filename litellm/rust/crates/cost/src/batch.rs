//! The `batch_line` kind, a port of `batch_cost_calculator` (`cost_calculator.py`) with `_batch_prompt_cost`,
//! `_half_price_prompt_cost`, `_cached_modality_rate` and `get_batch_cost_rates` (`llm_cost_calc/utils.py`): one line
//! of a batch output file billed at its row's `_batches` rates. A row with no batch input rate bills the prompt at half
//! its online rates, and a direction with no batch rate at half its online rate. Each rate is read the way main read
//! its key: a batch rate at the highest prompt band the line crossed, a modality batch rate at a band either price
//! list names, an online rate at its own bands. Residency, FedRAMP and regional endpoint uplifts multiply both sides.
//! Main applies no discount or margin to a batch line

use std::num::NonZeroU64;

use crate::money::{UnitPrice, Usd};
use crate::price::{
    Band, CacheReadMode, CacheTtl, Channel, InputUnit, Meter, OutputUnit, Price, Prices,
    ServiceTier, TokenKind, Uplift, When,
};
use crate::priced::{Line, Quantity, Side};
use crate::query::Header;
use crate::receipt::PriceOutcome;
use crate::row::{LoweredRow, RowSource};
use crate::units::{self, Billed};

/// `_INCLUSIVE_THRESHOLD_PROVIDERS`: xAI bills a band from its floor
const INCLUSIVE_BAND_PROVIDER: &str = "xai";

/// One output line of a batch, priced under the header's row
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize))]
pub struct BatchLineQuery {
    #[cfg_attr(feature = "serde", serde(flatten))]
    pub header: Header,
    pub usage: BatchUsage,
}

/// The line's token counts as `parse_prompt_tokens_details` reads them off its usage
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize), serde(default))]
pub struct BatchUsage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    /// `completion_tokens_details.image_tokens`
    pub completion_image_tokens: u64,
    pub cache_hit_tokens: u64,
    pub cache_hit_audio_tokens: u64,
    pub cached_image_tokens: u64,
    pub cache_creation_tokens: u64,
    /// `cache_creation_token_details`, `None` when the usage reports no write lifetimes
    pub cache_creation_by_ttl: Option<WritesByTtl>,
    pub audio_tokens: u64,
    pub image_tokens: u64,
    pub video_tokens: u64,
}

/// Cache writes by lifetime, each `None` when the usage leaves it out
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize), serde(default))]
pub struct WritesByTtl {
    pub five_minutes: Option<u64>,
    pub thirty_minutes: Option<u64>,
    pub one_hour: Option<u64>,
    pub twelve_hours: Option<u64>,
}

const fn cache_read(kind: TokenKind) -> Meter {
    Meter::CacheRead {
        kind,
        mode: CacheReadMode::Implicit,
    }
}

const fn cache_write(ttl: CacheTtl) -> Meter {
    Meter::CacheWrite {
        kind: TokenKind::Text,
        ttl,
    }
}

const fn input(kind: TokenKind) -> Meter {
    Meter::Input(InputUnit::Token(kind))
}

const IMAGE_OUTPUT: Meter = Meter::Output(OutputUnit::Token(TokenKind::Image));

pub(crate) fn price<S: RowSource>(source: &S, query: &BatchLineQuery) -> PriceOutcome {
    let Some(named) = units::named(source, &query.header) else {
        return units::no_row(&query.header);
    };
    let usage = &query.usage;
    let rates = Rates {
        prices: named.row.prices().flat(),
        prompt_tokens: usage.prompt_tokens,
        inclusive: named.candidate.provider.to_string() == INCLUSIVE_BAND_PROVIDER,
    };
    let prompt_terms = match (
        rates.batch(&Meter::TEXT_INPUT),
        rates.flat(&Meter::TEXT_INPUT, Channel::Online),
    ) {
        (Some(input_rate), _) => batch_prompt(&rates, usage, input_rate),
        (None, Some(input_rate)) => half_price_prompt(&rates, usage, input_rate),
        (None, None) => Vec::new(),
    };
    let text_rate = batch_or_half(
        rates.batch(&Meter::TEXT_OUTPUT),
        rates.flat(&Meter::TEXT_OUTPUT, Channel::Online),
        0.0,
    );
    let image_rate = batch_or_half(
        rates.flat(&IMAGE_OUTPUT, Channel::Batch),
        rates.flat(&IMAGE_OUTPUT, Channel::Online),
        text_rate,
    );
    let image_tokens = usage.completion_image_tokens.min(usage.completion_tokens);
    let completion_terms = vec![
        Term::of(
            Meter::TEXT_OUTPUT,
            usage.completion_tokens - image_tokens,
            text_rate,
        ),
        Term::of(IMAGE_OUTPUT, image_tokens, image_rate),
    ];
    let row = named.row;
    let uplift = |found: Option<Uplift>| {
        found
            .and_then(|uplift| row.adjustments().get(&uplift))
            .map_or(1.0, |multiplier| multiplier.get())
    };
    let basis = &query.header.basis;
    let regional = uplift(basis.residency().map(Uplift::Residency))
        * uplift(basis.fedramp().then_some(Uplift::FedrampEndpoint));
    let endpoint = uplift(
        basis
            .non_global_endpoint()
            .then_some(Uplift::NonGlobalEndpoint),
    );
    let settle = |total: f64| match regional == 1.0 {
        true => total * endpoint,
        false => total * regional * endpoint,
    };
    let side_total = |terms: &[Term]| terms.iter().fold(0.0, |sum, term| sum + term.amount);
    let prompt = settle(side_total(&prompt_terms));
    let completion = settle(side_total(&completion_terms));
    let lines = prompt_terms
        .into_iter()
        .map(|term| term.line(Side::Prompt))
        .chain(
            completion_terms
                .into_iter()
                .map(|term| term.line(Side::Completion)),
        )
        .collect();
    units::receipt(
        source,
        &query.header,
        Some(&named),
        Billed {
            lines,
            missing: Vec::new(),
            prompt: Usd::new(prompt).unwrap_or_default(),
            completion: Usd::new(completion).unwrap_or_default(),
        },
    )
    .into()
}

/// One summand of main's side total: tokens times a rate, or times a rate and halved
struct Term {
    meter: Meter,
    tokens: u64,
    rate: f64,
    amount: f64,
}

impl Term {
    fn of(meter: Meter, tokens: u64, rate: f64) -> Self {
        Self {
            meter,
            tokens,
            rate,
            amount: tokens as f64 * rate,
        }
    }

    /// `tokens * rate / 2`, multiplied before it is halved as main does
    fn halved(meter: Meter, tokens: u64, rate: f64) -> Self {
        Self {
            amount: tokens as f64 * rate / 2.0,
            ..Self::of(meter, tokens, rate)
        }
    }

    fn line(self, side: Side) -> Line {
        let unit_price =
            UnitPrice::new(self.rate).unwrap_or(UnitPrice::new(0.0).expect("zero is a price"));
        Line {
            meter: self.meter,
            quantity: Quantity::Count(self.tokens),
            unit_price,
            amount: Usd::new(self.amount).unwrap_or_default(),
            fallback_from: None,
            side,
        }
    }
}

/// The rates of one row as main's batch helpers read its keys
struct Rates<'a> {
    prices: &'a Prices,
    prompt_tokens: u64,
    inclusive: bool,
}

impl Rates<'_> {
    fn at(&self, meter: &Meter, channel: Channel, band: Band) -> Option<f64> {
        self.prices
            .get(&Price {
                meter: meter.clone(),
                channel,
                when: When {
                    tier: ServiceTier::Standard,
                    band,
                },
            })
            .map(UnitPrice::get)
    }

    fn flat(&self, meter: &Meter, channel: Channel) -> Option<f64> {
        self.at(meter, channel, Band::Base)
    }

    fn bands<'b>(
        &'b self,
        meter: &'b Meter,
        channel: Channel,
    ) -> impl Iterator<Item = NonZeroU64> + 'b {
        self.prices
            .iter()
            .filter_map(move |(price, _)| match price.when {
                When {
                    tier: ServiceTier::Standard,
                    band: Band::OverPromptTokens(floor),
                } if price.channel == channel && &price.meter == meter => Some(floor),
                _ => None,
            })
    }

    /// `_batch_rate_for_prefix`: the batch rate of the highest batch band the prompt crossed, else the flat one
    fn batch(&self, meter: &Meter) -> Option<f64> {
        let crossed = self
            .bands(meter, Channel::Batch)
            .filter(|floor| crosses(self.prompt_tokens, *floor, self.inclusive))
            .max();
        crossed
            .and_then(|floor| self.at(meter, Channel::Batch, Band::OverPromptTokens(floor)))
            .or_else(|| self.flat(meter, Channel::Batch))
    }

    /// `_optional_batch_rate`: the band comes from the online and the batch price lists alike, and a band the batch
    /// list does not price reads the flat batch rate
    fn modality_batch(&self, meter: &Meter, prompt_tokens: u64) -> Option<f64> {
        let crossed = self
            .bands(meter, Channel::Online)
            .chain(self.bands(meter, Channel::Batch))
            .filter(|floor| crosses(prompt_tokens, *floor, false))
            .max();
        crossed
            .and_then(|floor| self.at(meter, Channel::Batch, Band::OverPromptTokens(floor)))
            .or_else(|| self.flat(meter, Channel::Batch))
    }

    /// `_standard_rate`: the online rate of the highest online band the prompt crossed, else the flat one
    fn standard(&self, meter: &Meter) -> Option<f64> {
        let crossed = self
            .bands(meter, Channel::Online)
            .filter(|floor| crosses(self.prompt_tokens, *floor, false))
            .max();
        match crossed {
            Some(floor) => self.at(meter, Channel::Online, Band::OverPromptTokens(floor)),
            None => self.flat(meter, Channel::Online),
        }
    }
}

/// `_prompt_exceeds_threshold`
fn crosses(prompt_tokens: u64, floor: NonZeroU64, inclusive: bool) -> bool {
    prompt_tokens > floor.get() || (inclusive && prompt_tokens == floor.get())
}

/// `_batch_or_half`
fn batch_or_half(batch: Option<f64>, standard: Option<f64>, default: f64) -> f64 {
    batch.or(standard.map(|rate| rate / 2.0)).unwrap_or(default)
}

/// `_batch_prompt_cost`: the prompt at the row's batch input rate
fn batch_prompt(rates: &Rates<'_>, usage: &BatchUsage, input_rate: f64) -> Vec<Term> {
    let text_read = cache_read(TokenKind::Text);
    let batch_read = rates.batch(&text_read);
    let read_rate = batch_read.or_else(|| rates.standard(&text_read));
    let modality_read = |kind: TokenKind, batch: Option<f64>| {
        batch
            .or(batch_read)
            .or_else(|| rates.standard(&cache_read(kind)).or(read_rate))
    };
    let audio_read = modality_read(
        TokenKind::Audio,
        rates.modality_batch(&cache_read(TokenKind::Audio), rates.prompt_tokens),
    );
    let image_read = modality_read(TokenKind::Image, rates.batch(&cache_read(TokenKind::Image)));
    let cached_audio = audio_read.map_or(0, |_| usage.cache_hit_audio_tokens);
    let cached_image = image_read.map_or(0, |_| usage.cached_image_tokens);
    let cached_text = read_rate.map_or(0, |_| cached_text_tokens(usage));
    let write_rate = rates.batch(&cache_write(CacheTtl::Default));
    let ttl_rates = [
        (
            CacheTtl::OneHour,
            rates.batch(&cache_write(CacheTtl::OneHour)),
        ),
        (
            CacheTtl::TwelveHours,
            rates.batch(&cache_write(CacheTtl::TwelveHours)),
        ),
    ];
    let bills_writes = write_rate.is_some() || ttl_rates.iter().any(|(_, rate)| rate.is_some());
    let written = if bills_writes {
        usage.cache_creation_tokens
    } else {
        0
    };
    let uncached = usage
        .prompt_tokens
        .saturating_sub(usage.audio_tokens + usage.image_tokens + usage.video_tokens)
        .saturating_sub(cached_text + cached_audio + cached_image + written);
    let modality_rate = |kind: TokenKind, prompt_tokens: u64| {
        rates
            .modality_batch(&input(kind), prompt_tokens)
            .unwrap_or(input_rate)
    };
    let writes = bills_writes.then(|| {
        cache_writes(
            written,
            usage.cache_creation_by_ttl,
            write_rate.unwrap_or(input_rate),
            ttl_rates,
        )
    });
    [
        Term::of(Meter::TEXT_INPUT, uncached, input_rate),
        Term::of(
            input(TokenKind::Audio),
            usage.audio_tokens,
            modality_rate(TokenKind::Audio, rates.prompt_tokens),
        ),
        Term::of(
            input(TokenKind::Image),
            usage.image_tokens,
            modality_rate(TokenKind::Image, 0),
        ),
        Term::of(
            input(TokenKind::Video),
            usage.video_tokens,
            modality_rate(TokenKind::Video, 0),
        ),
        Term::of(text_read, cached_text, read_rate.unwrap_or(0.0)),
        Term::of(
            cache_read(TokenKind::Audio),
            cached_audio,
            audio_read.unwrap_or(0.0),
        ),
        Term::of(
            cache_read(TokenKind::Image),
            cached_image,
            image_read.unwrap_or(0.0),
        ),
        Term {
            meter: cache_write(CacheTtl::Default),
            tokens: written,
            rate: write_rate.unwrap_or(input_rate),
            amount: writes.unwrap_or(0.0),
        },
    ]
    .into_iter()
    .collect()
}

/// `_half_price_prompt_cost`: the prompt of a row with no batch input rate, at half its online rates
fn half_price_prompt(rates: &Rates<'_>, usage: &BatchUsage, input_rate: f64) -> Vec<Term> {
    let half_read = rates
        .flat(&cache_read(TokenKind::Text), Channel::Online)
        .unwrap_or(0.0)
        / 2.0;
    let audio_read = batch_or_half(
        rates.modality_batch(&cache_read(TokenKind::Audio), rates.prompt_tokens),
        rates.standard(&cache_read(TokenKind::Audio)),
        half_read,
    );
    let image_read = batch_or_half(
        rates.batch(&cache_read(TokenKind::Image)),
        rates.standard(&cache_read(TokenKind::Image)),
        half_read,
    );
    let write_rate = rates
        .flat(&cache_write(CacheTtl::Default), Channel::Online)
        .unwrap_or(input_rate);
    let uncached = usage
        .prompt_tokens
        .saturating_sub(usage.cache_hit_tokens)
        .saturating_sub(usage.cache_creation_tokens);
    vec![
        Term::halved(Meter::TEXT_INPUT, uncached, input_rate),
        Term::of(
            cache_read(TokenKind::Text),
            cached_text_tokens(usage),
            half_read,
        ),
        Term::of(
            cache_read(TokenKind::Audio),
            usage.cache_hit_audio_tokens,
            audio_read,
        ),
        Term::of(
            cache_read(TokenKind::Image),
            usage.cached_image_tokens,
            image_read,
        ),
        Term::halved(
            cache_write(CacheTtl::Default),
            usage.cache_creation_tokens,
            write_rate,
        ),
    ]
}

fn cached_text_tokens(usage: &BatchUsage) -> u64 {
    usage
        .cache_hit_tokens
        .saturating_sub(usage.cache_hit_audio_tokens + usage.cached_image_tokens)
}

/// `calculate_cache_writing_cost`: each reported lifetime at its own rate, the rest at the base write rate
fn cache_writes(
    written: u64,
    by_ttl: Option<WritesByTtl>,
    base: f64,
    ttl_rates: [(CacheTtl, Option<f64>); 2],
) -> f64 {
    let Some(by_ttl) = by_ttl else {
        return written as f64 * base;
    };
    let ttl_rate = |ttl: CacheTtl| {
        ttl_rates
            .iter()
            .find(|(priced, _)| *priced == ttl)
            .and_then(|(_, rate)| *rate)
            .unwrap_or(base)
    };
    let counts = [
        (by_ttl.five_minutes, base),
        (by_ttl.thirty_minutes, base),
        (by_ttl.one_hour, ttl_rate(CacheTtl::OneHour)),
        (by_ttl.twelve_hours, ttl_rate(CacheTtl::TwelveHours)),
    ];
    let attributed: u64 = counts.iter().map(|(tokens, _)| tokens.unwrap_or(0)).sum();
    let unattributed = written.saturating_sub(attributed);
    unattributed as f64 * base
        + counts.iter().fold(0.0, |sum, (tokens, rate)| {
            sum + tokens.unwrap_or(0) as f64 * rate
        })
}
