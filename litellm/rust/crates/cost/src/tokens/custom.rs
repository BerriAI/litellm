//! `custom_cost_per_token`, main's `_cost_per_token_custom_pricing_helper`: flat input, output and cache rates with
//! no tiers, bands or uplifts. Uncached input is the prompt less every cache read and write, and a missing cache rate
//! is the input rate

use crate::money::{UnitPrice, Usd};
use crate::price::{CacheTtl, Meter, TokenKind};
use crate::priced::{Line, Quantity, Side};
use crate::query::{FlatRates, Kind, TokensQuery};
use crate::receipt::{BilledRates, PriceOutcome, Receipt, Unpriced};
use crate::row::RowSource;

use super::breakdown;

pub(super) fn price<S: RowSource>(
    source: &S,
    query: &TokensQuery,
    rates: &FlatRates,
) -> PriceOutcome {
    let Some(candidate) = query.header.candidates.first() else {
        return Unpriced::NoRow { tried: Vec::new() }.into();
    };
    if query.usage.web_search.is_some() {
        return Unpriced::NotPorted(Kind::Tokens).into();
    }
    let usage = &query.usage;
    let cache_read = rates.cache_read.unwrap_or(rates.input);
    let cache_write = rates.cache_write.unwrap_or(rates.input);
    let cached = usage.cache_read_total();
    let written = usage.cache_write_total();
    let uncached = usage
        .prompt_tokens
        .saturating_sub(cached)
        .saturating_sub(written);
    let line = |meter: Meter, tokens: u64, rate: UnitPrice, side: Side| Line {
        meter,
        quantity: Quantity::Count(tokens),
        unit_price: rate,
        amount: rate * Quantity::Count(tokens),
        fallback_from: None,
        side,
    };
    let lines = vec![
        line(Meter::TEXT_INPUT, uncached, rates.input, Side::Prompt),
        line(Meter::CACHE_READ, cached, cache_read, Side::Prompt),
        line(
            Meter::CacheWrite {
                kind: TokenKind::Text,
                ttl: CacheTtl::Default,
            },
            written,
            cache_write,
            Side::Prompt,
        ),
        line(
            Meter::TEXT_OUTPUT,
            usage.completion_total(),
            rates.output,
            Side::Completion,
        ),
    ];
    let prompt: Usd = lines
        .iter()
        .filter(|line| line.side == Side::Prompt)
        .map(|line| line.amount)
        .sum();
    let completion: Usd = lines
        .iter()
        .filter(|line| line.side == Side::Completion)
        .map(|line| line.amount)
        .sum();
    let billed = BilledRates {
        input: rates.input.get(),
        output: rates.output.get(),
        cache_read: cache_read.get(),
        cache_read_audio: cache_read.get(),
        cache_read_image: None,
        cache_write: cache_write.get(),
        cache_write_one_hour: cache_write.get(),
        cache_write_twelve_hours: cache_write.get(),
        reasoning: rates.output.get(),
    };
    let receipt = Receipt {
        candidate: Some(candidate.clone()),
        row: None,
        lines: lines
            .into_iter()
            .filter(|line| line.quantity != Quantity::Count(0))
            .collect(),
        defaulted_to_zero: Vec::new(),
        breakdown: breakdown::breakdown(&billed, usage, true),
        rates: Some(billed),
        prompt,
        completion,
        tools: Usd::ZERO,
        additional: Usd::ZERO,
        discount: Default::default(),
        margin: Default::default(),
        parts: Vec::new(),
        cache_savings: None,
        basis: query.header.basis.clone(),
        total: 0.0,
    };
    receipt
        .settled(source, &query.header.adjustments, Some(&candidate.model))
        .into()
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;
    use crate::query::{CacheReads, CacheWrites, PriceSource, Provider, TokenUsage};
    use crate::testing::{TestSource, tokens_query, usage};

    fn flat(cache_read: Option<f64>, cache_write: Option<f64>) -> FlatRates {
        FlatRates {
            input: UnitPrice::new(1.0).unwrap(),
            output: UnitPrice::new(3.0).unwrap(),
            cache_read: cache_read.and_then(UnitPrice::new),
            cache_write: cache_write.and_then(UnitPrice::new),
        }
    }

    fn cached() -> TokenUsage {
        TokenUsage {
            prompt_tokens: 100,
            cache_read: CacheReads {
                text: 50,
                ..CacheReads::default()
            },
            cache_write: CacheWrites {
                one_hour: 20,
                ..CacheWrites::default()
            },
            reasoning: 5,
            ..usage(30, 10)
        }
    }

    #[rstest]
    #[case::own_cache_rates(flat(Some(0.1), Some(2.0)), (75.0, 45.0))]
    #[case::cache_rates_default_to_input(flat(None, None), (100.0, 45.0))]
    fn custom_pricing_bills_flat_rates(#[case] rates: FlatRates, #[case] expected: (f64, f64)) {
        let query = tokens_query(Provider::Anthropic, "custom", cached());
        let query = TokensQuery {
            header: crate::query::Header {
                source: PriceSource::FlatPerToken(rates),
                ..query.header
            },
            ..query
        };
        let PriceOutcome::Priced(receipt) = price(&TestSource::default(), &query, &rates) else {
            panic!("custom pricing always prices");
        };
        assert_eq!((receipt.prompt.get(), receipt.completion.get()), expected);
        assert_eq!(receipt.total, expected.0 + expected.1);
        assert_eq!(receipt.breakdown.reasoning, 15.0);
    }
}
