//! The provider branches of main's `cost_per_token` that bill a row by their own formula instead of
//! `generic_cost_per_token`: a row billed by the second (`_per_second_pricing_cost`), Vertex AI's legacy 128k bands and
//! character pricing (`llms/vertex_ai/cost_calculator.py`), Perplexity (`llms/perplexity/cost_calculator.py`) and
//! Dashscope (`llms/dashscope/cost_calculator.py`). Each side is summed in the order main summed it

use std::num::NonZeroU64;

use crate::call::CallContext;
use crate::money::{Multiplier, UnitPrice, Usd};
use crate::price::{
    Band, CacheReadMode, CacheTtl, Channel, Feature, InputUnit, Meter, OutputUnit, Price, Prices,
    SearchContextSize, ServiceTier, TokenKind, Uplift, WallClock, When,
};
use crate::priced::{Line, Quantity, Side};
use crate::query::{Basis, Elapsed, TokensQuery, WritesByTtl};
use crate::row::LoweredRow;

/// A formula `cost_per_token` bills instead of the generic one
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Formula {
    PerSecond,
    /// `_handle_128k_pricing`
    VertexLegacy,
    /// `cost_per_character`
    VertexCharacters,
    Perplexity,
    Dashscope,
    /// Lemonade, and a row the generic branch finds without a token or tier price
    Free,
}

/// What one formula billed on each side
#[derive(Clone, Debug, Default, PartialEq)]
pub(super) struct Sides {
    pub(super) lines: Vec<Line>,
    pub(super) defaulted_to_zero: Vec<Meter>,
    pub(super) prompt: Usd,
    pub(super) completion: Usd,
}

/// `is_above_128k` and the band `_handle_128k_pricing` reads
const LEGACY_EDGE: u64 = 128_000;
const OVER_128K: Band = Band::OverPromptTokens(NonZeroU64::new(LEGACY_EDGE).unwrap());

/// `models_without_dynamic_pricing`: Vertex AI models whose characters never bill the above-128k rate
const WITHOUT_DYNAMIC_PRICING: [&str; 3] = ["gemini-1.0-pro", "gemini-pro", "gemini-2"];

impl Sides {
    fn of(lines: Vec<Line>) -> Self {
        let side = |side: Side| {
            lines
                .iter()
                .filter(|line| line.side == side)
                .map(|line| line.amount)
                .sum()
        };
        Self {
            prompt: side(Side::Prompt),
            completion: side(Side::Completion),
            lines,
            defaulted_to_zero: Vec::new(),
        }
    }

    fn uplifted(self, uplift: Option<Multiplier>) -> Self {
        let Some(uplift) = uplift else {
            return self;
        };
        Self {
            lines: self
                .lines
                .into_iter()
                .map(|line| Line {
                    amount: line.amount * uplift,
                    ..line
                })
                .collect(),
            prompt: self.prompt * uplift,
            completion: self.completion * uplift,
            defaulted_to_zero: self.defaulted_to_zero,
        }
    }

    /// The prompt side of `self` and the completion side of `other`
    fn joined(self, other: Self) -> Self {
        let lines: Vec<Line> = self
            .lines
            .into_iter()
            .filter(|line| line.side == Side::Prompt)
            .chain(
                other
                    .lines
                    .into_iter()
                    .filter(|line| line.side == Side::Completion),
            )
            .collect();
        Self {
            lines,
            defaulted_to_zero: self
                .defaulted_to_zero
                .into_iter()
                .chain(other.defaulted_to_zero)
                .collect(),
            prompt: self.prompt,
            completion: other.completion,
        }
    }
}

/// The sides `formula` bills. `tokens` is what the row bills by `generic_cost_per_token` for this call
pub(super) fn sides(
    formula: Formula,
    row: &impl LoweredRow,
    query: &TokensQuery,
    call: &CallContext,
    tokens: impl FnOnce() -> Sides,
) -> Sides {
    let uplift = || vertex_uplift(row, &query.header.basis);
    match formula {
        Formula::PerSecond => per_second(row, query.elapsed.unwrap_or_default()),
        Formula::Free => Sides::default(),
        Formula::Perplexity => perplexity(row, query, call),
        Formula::Dashscope => dashscope(row, query, call),
        Formula::VertexLegacy => legacy_128k(row, query).uplifted(uplift()),
        Formula::VertexCharacters => {
            let by_token = if prices_above_128k(row) {
                legacy_128k(row, query)
            } else {
                tokens()
            };
            characters(row, query, by_token).uplifted(uplift())
        }
    }
}

fn line(meter: Meter, quantity: Quantity, rate: UnitPrice, side: Side) -> Line {
    Line {
        meter,
        quantity,
        unit_price: rate,
        amount: rate * quantity,
        fallback_from: None,
        side,
    }
}

fn count(meter: Meter, tokens: u64, rate: UnitPrice, side: Side) -> Line {
    line(meter, Quantity::Count(tokens), rate, side)
}

fn listed(prices: &Prices, meter: Meter, band: Band) -> Option<UnitPrice> {
    prices.get(&Price {
        meter,
        channel: Channel::Online,
        when: When {
            tier: ServiceTier::Standard,
            band,
        },
    })
}

fn flat(row: &impl LoweredRow, meter: Meter) -> Option<UnitPrice> {
    listed(row.prices().flat(), meter, Band::Base)
}

/// `get_vertex_regional_endpoint_uplift`
fn vertex_uplift(row: &impl LoweredRow, basis: &Basis) -> Option<Multiplier> {
    basis
        .non_global_endpoint()
        .then(|| row.adjustments().get(&Uplift::NonGlobalEndpoint))
        .flatten()
}

/// Whether the row lists an `_above_128k_tokens` token price, which sends Vertex AI to `_handle_128k_pricing`
pub(super) fn prices_above_128k(row: &impl LoweredRow) -> bool {
    let flat = row.prices().flat();
    [Meter::TEXT_INPUT, Meter::TEXT_OUTPUT]
        .into_iter()
        .any(|meter| listed(flat, meter, OVER_128K).is_some())
}

/// `_has_token_or_tiered_pricing`
pub(super) fn prices_tokens(row: &impl LoweredRow) -> bool {
    let positive = |meter: Meter| flat(row, meter).is_some_and(|rate| rate.get() > 0.0);
    positive(Meter::TEXT_INPUT)
        || positive(Meter::TEXT_OUTPUT)
        || row.prices().tiers().is_some()
        || row.facts().prices_by_results
}

/// The rate `_per_second_pricing_cost_from_info` bills, `None` for a row it leaves to the token branches
pub(super) fn per_second_rate(row: &impl LoweredRow) -> Option<(Meter, UnitPrice)> {
    if prices_tokens(row) || row.prices().flat().billing().wall_clock == WallClock::Never {
        return None;
    }
    [
        Meter::WallClockSecond,
        Meter::Input(InputUnit::Second),
        Meter::Output(OutputUnit::Second(None)),
    ]
    .into_iter()
    .find_map(|meter| Some((meter.clone(), flat(row, meter)?)))
}

/// The rate times the audio file's length for a transcription row that read one, else the response time
fn per_second(row: &impl LoweredRow, elapsed: Elapsed) -> Sides {
    let Some((meter, rate)) = per_second_rate(row) else {
        return Sides::default();
    };
    let audio = row.prices().flat().billing().wall_clock == WallClock::AudioFile;
    let seconds = if audio && elapsed.audio_seconds > 0.0 {
        elapsed.audio_seconds
    } else {
        elapsed.response_ms / 1000.0
    };
    Sides::of(vec![line(
        meter,
        Quantity::Measured(seconds),
        rate,
        Side::Prompt,
    )])
}

/// `_handle_128k_pricing`: the prompt band by prompt tokens and the output band by completion tokens, every prompt
/// token at the input rate whatever the cache
fn legacy_128k(row: &impl LoweredRow, query: &TokensQuery) -> Sides {
    let flat = row.prices().flat();
    let side = |meter: Meter, tokens: u64, side: Side| {
        let above = listed(flat, meter.clone(), OVER_128K).filter(|_| tokens > LEGACY_EDGE);
        let rate = above
            .or_else(|| listed(flat, meter.clone(), Band::Base))
            .unwrap_or(UnitPrice::ZERO);
        count(meter, tokens, rate, side)
    };
    Sides::of(vec![
        side(Meter::TEXT_INPUT, query.usage.prompt_tokens, Side::Prompt),
        side(
            Meter::TEXT_OUTPUT,
            query.raw.completion_tokens,
            Side::Completion,
        ),
    ])
}

/// `cost_per_character`: each counted side at its character rate, the above-128k rate past 128k tokens at four
/// characters a token, else what the token branch bills that side
fn characters(row: &impl LoweredRow, query: &TokensQuery, by_token: Sides) -> Sides {
    let counted = query.characters.unwrap_or_default();
    let flat = row.prices().flat();
    let model = query
        .header
        .candidates
        .first()
        .map_or("", |candidate| &*candidate.model);
    let dynamic = !WITHOUT_DYNAMIC_PRICING.contains(&model);
    let above = |characters: u64| dynamic && characters.saturating_mul(4) > LEGACY_EDGE;
    let rate = |meter: Meter, characters: u64| match above(characters) {
        true => listed(flat, meter, OVER_128K),
        false => listed(flat, meter, Band::Base),
    };
    let prompt = counted.prompt.and_then(|characters| {
        let meter = Meter::Input(InputUnit::Character);
        rate(meter.clone(), characters).map(|rate| count(meter, characters, rate, Side::Prompt))
    });
    let completion = counted.completion.and_then(|characters| {
        let meter = Meter::Output(OutputUnit::Character);
        let billed = match above(characters) {
            true => query.raw.completion_tokens,
            false => characters,
        };
        rate(meter.clone(), characters).map(|rate| count(meter, billed, rate, Side::Completion))
    });
    let prompt_side = match prompt {
        Some(line) => Sides::of(vec![line]),
        None => by_token.clone(),
    };
    let completion_side = match completion {
        Some(line) => Sides::of(vec![line]),
        None => by_token,
    };
    prompt_side.joined(completion_side)
}

/// The rate a quote bills `meter` at, 0.0 for one the row cannot price, as `NativeQuote` reads it
fn quoted(quote: &crate::quote::Quote<'_>, meter: &Meter) -> UnitPrice {
    quote
        .rate(meter)
        .map_or(UnitPrice::ZERO, |rate| rate.unit_price)
}

/// Perplexity's own math: cached tokens at the cache read rate, citation tokens, reasoning tokens split out only when
/// the row prices them, and every search query at the low context price on the completion side
fn perplexity(row: &impl LoweredRow, query: &TokensQuery, call: &CallContext) -> Sides {
    let prices = row.prices().for_prompt(0);
    let quote = prices.quote(call, 0);
    let raw = &query.raw;
    let prompt_tokens = query.usage.prompt_tokens;
    let cached = prompt_tokens.min(raw.cached_tokens);
    let input = quoted(&quote, &Meter::TEXT_INPUT);
    let output = quoted(&quote, &Meter::TEXT_OUTPUT);
    let citation = flat(row, Meter::Feature(Feature::CitationToken))
        .filter(|_| raw.citation_tokens > 0)
        .map(|rate| {
            count(
                Meter::Feature(Feature::CitationToken),
                raw.citation_tokens,
                rate,
                Side::Prompt,
            )
        });
    let reasoning_rate = flat(row, Meter::REASONING).filter(|_| raw.stated_reasoning_tokens > 0);
    let completion = match reasoning_rate {
        Some(rate) => vec![
            count(
                Meter::TEXT_OUTPUT,
                raw.completion_tokens
                    .saturating_sub(raw.stated_reasoning_tokens),
                output,
                Side::Completion,
            ),
            count(
                Meter::REASONING,
                raw.stated_reasoning_tokens,
                rate,
                Side::Completion,
            ),
        ],
        None => vec![count(
            Meter::TEXT_OUTPUT,
            raw.completion_tokens,
            output,
            Side::Completion,
        )],
    };
    let search_rate = row.facts().legacy_search_query_rate.or_else(|| {
        let low = Meter::WebSearch(SearchContextSize::Low);
        let listed_any = [
            SearchContextSize::Low,
            SearchContextSize::Medium,
            SearchContextSize::High,
        ]
        .into_iter()
        .any(|size| flat(row, Meter::WebSearch(size)).is_some());
        listed_any.then(|| flat(row, low).unwrap_or(UnitPrice::ZERO))
    });
    let search = search_rate.filter(|_| raw.search_queries > 0).map(|rate| {
        count(
            Meter::WebSearch(SearchContextSize::Low),
            raw.search_queries,
            rate,
            Side::Completion,
        )
    });
    Sides::of(
        [
            count(
                Meter::TEXT_INPUT,
                prompt_tokens - cached,
                input,
                Side::Prompt,
            ),
            count(
                Meter::CACHE_READ,
                cached,
                quoted(&quote, &Meter::CACHE_READ),
                Side::Prompt,
            ),
        ]
        .into_iter()
        .chain(citation)
        .chain(completion)
        .chain(search)
        .collect(),
    )
}

/// Dashscope bills the whole call at the tier its input picks: uncached text, cache reads at the explicit or implicit
/// rate, cache writes by lifetime where only the 1h one has a rate of its own, then output and reasoning
fn dashscope(row: &impl LoweredRow, query: &TokensQuery, call: &CallContext) -> Sides {
    let raw = &query.raw;
    let text = query
        .usage
        .prompt_tokens
        .saturating_sub(raw.cached_tokens)
        .saturating_sub(raw.cache_creation_tokens);
    let total_input = text + raw.cached_tokens + raw.cache_creation_tokens;
    let prices = row.prices().for_prompt(total_input);
    let quote = prices.quote(call, total_input);
    let read = match query.usage.cache_read.explicit {
        true => Meter::EXPLICIT_CACHE_READ,
        false => Meter::CACHE_READ,
    };
    let write = quoted(&quote, &Meter::CACHE_WRITE);
    let write_one_hour = quoted(
        &quote,
        &Meter::CacheWrite {
            kind: TokenKind::Text,
            ttl: CacheTtl::OneHour,
        },
    );
    let completion = raw.completion_tokens.saturating_sub(raw.reasoning_tokens);
    let writes = cache_writes(
        raw.cache_creation_tokens,
        raw.cache_writes,
        write,
        write_one_hour,
    );
    Sides::of(vec![
        count(
            Meter::TEXT_INPUT,
            text,
            quoted(&quote, &Meter::TEXT_INPUT),
            Side::Prompt,
        ),
        count(
            Meter::CacheRead {
                kind: TokenKind::Text,
                mode: match query.usage.cache_read.explicit {
                    true => CacheReadMode::Explicit,
                    false => CacheReadMode::Implicit,
                },
            },
            raw.cached_tokens,
            quoted(&quote, &read),
            Side::Prompt,
        ),
        Line {
            meter: Meter::CACHE_WRITE,
            quantity: Quantity::Count(raw.cache_creation_tokens),
            unit_price: write,
            amount: writes,
            fallback_from: None,
            side: Side::Prompt,
        },
        count(
            Meter::TEXT_OUTPUT,
            completion,
            quoted(&quote, &Meter::TEXT_OUTPUT),
            Side::Completion,
        ),
        count(
            Meter::REASONING,
            raw.reasoning_tokens,
            quoted(&quote, &Meter::REASONING),
            Side::Completion,
        ),
    ])
}

/// `calculate_cache_writing_cost` with only a 1h rate of its own: every other lifetime at the base write rate
fn cache_writes(
    written: u64,
    by_ttl: Option<WritesByTtl>,
    base: UnitPrice,
    one_hour: UnitPrice,
) -> Usd {
    let Some(by_ttl) = by_ttl else {
        return base * Quantity::Count(written);
    };
    let listed = [
        (by_ttl.five_minutes, base),
        (by_ttl.thirty_minutes, base),
        (by_ttl.one_hour, one_hour),
        (by_ttl.twelve_hours, base),
    ];
    let reported: u64 = listed.iter().map(|(tokens, _)| tokens.unwrap_or(0)).sum();
    let by_lifetime: Usd = listed
        .into_iter()
        .map(|(tokens, rate)| rate * Quantity::Count(tokens.unwrap_or(0)))
        .sum();
    base * Quantity::Count(written.saturating_sub(reported)) + by_lifetime
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    fn rate(value: f64) -> UnitPrice {
        UnitPrice::new(value).unwrap()
    }

    #[rstest]
    #[case::no_lifetimes(None, 30.0)]
    #[case::lifetimes_with_unattributed(
        Some(WritesByTtl { five_minutes: Some(2), one_hour: Some(3), twelve_hours: Some(1), ..WritesByTtl::default() }),
        12.0 + 6.0 + 6.0 + 3.0
    )]
    #[case::lifetimes_over_the_written_count(
        Some(WritesByTtl { one_hour: Some(20), ..WritesByTtl::default() }),
        40.0
    )]
    fn writes_bill_the_1h_lifetime_at_its_rate_and_the_rest_at_base(
        #[case] by_ttl: Option<WritesByTtl>,
        #[case] expected: f64,
    ) {
        assert_eq!(
            cache_writes(10, by_ttl, rate(3.0), rate(2.0)).get(),
            expected
        );
    }
}
