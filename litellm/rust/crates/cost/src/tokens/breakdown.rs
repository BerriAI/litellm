//! The values main stores beside a token call's total: the billed per-token rates (`get_billed_token_rates`), the
//! cache-read, cache-write and reasoning lines (`get_token_type_cost_breakdown`) and the prompt-caching savings
//! (`calculate_prompt_caching_savings`), all in `llm_cost_calc/utils.py`. The rates multiply every uplift the call
//! names whatever the provider, as main's breakdown does, while the totals only carry their provider's

use crate::call::Speed;
use crate::price::{Adjustments, CacheReadMode, CacheTtl, Meter, TokenKind, Uplift};
use crate::query::{Basis, TokenUsage};
use crate::quote::Quote;
use crate::receipt::{BilledRates, Breakdown};
use crate::row::ProviderEntry;

use super::rule::billed_geo;

/// The product of the residency, FedRAMP, regional endpoint, geo and speed multipliers, in main's order. A speed or
/// geo value main cannot read counts as no uplift
pub(super) fn multiplier(
    adjustments: &Adjustments,
    basis: &Basis,
    usage: &TokenUsage,
    entry: &ProviderEntry,
) -> f64 {
    let readable = |uplift: Uplift| (!entry.rejects(&uplift)).then_some(uplift);
    [
        basis.residency().map(Uplift::Residency),
        basis.fedramp().then_some(Uplift::FedrampEndpoint),
        basis
            .non_global_endpoint()
            .then_some(Uplift::NonGlobalEndpoint),
        billed_geo(usage)
            .map(Uplift::InferenceGeo)
            .and_then(readable),
        usage
            .speed
            .filter(|speed| *speed == Speed::Fast)
            .map(Uplift::Speed)
            .and_then(readable),
    ]
    .into_iter()
    .fold(1.0, |product, uplift| product * factor(adjustments, uplift))
}

/// The residency, FedRAMP and regional endpoint multipliers the savings carry
pub(super) fn savings_uplift(adjustments: &Adjustments, basis: &Basis) -> f64 {
    [
        basis.residency().map(Uplift::Residency),
        basis.fedramp().then_some(Uplift::FedrampEndpoint),
        basis
            .non_global_endpoint()
            .then_some(Uplift::NonGlobalEndpoint),
    ]
    .into_iter()
    .fold(1.0, |product, uplift| product * factor(adjustments, uplift))
}

fn factor(adjustments: &Adjustments, uplift: Option<Uplift>) -> f64 {
    uplift
        .and_then(|uplift| adjustments.get(&uplift))
        .map_or(1.0, |multiplier| multiplier.get())
}

/// The rates `quote` bills each token meter at, 0.0 where the row has none
pub(super) fn billed_rates(quote: &Quote<'_>, explicit_reads: bool) -> BilledRates {
    let rate = |meter: &Meter| quote.rate(meter).map(|rate| rate.unit_price.get());
    let billed = |meter: &Meter| rate(meter).unwrap_or(0.0);
    let own_cached = |kind| {
        quote
            .rate(&Meter::CacheRead {
                kind,
                mode: CacheReadMode::Implicit,
            })
            .filter(|rate| rate.fallback_from.is_none())
            .map(|rate| rate.unit_price.get())
    };
    let write = |ttl| {
        billed(&Meter::CacheWrite {
            kind: TokenKind::Text,
            ttl,
        })
    };
    let cache_read = billed(if explicit_reads {
        &Meter::EXPLICIT_CACHE_READ
    } else {
        &Meter::CACHE_READ
    });
    BilledRates {
        input: billed(&Meter::TEXT_INPUT),
        output: billed(&Meter::TEXT_OUTPUT),
        cache_read,
        cache_read_audio: own_cached(TokenKind::Audio).unwrap_or(cache_read),
        cache_read_image: own_cached(TokenKind::Image),
        cache_write: write(CacheTtl::Default),
        cache_write_one_hour: write(CacheTtl::OneHour),
        cache_write_twelve_hours: write(CacheTtl::TwelveHours),
        reasoning: billed(&Meter::REASONING),
    }
}

impl BilledRates {
    /// `BilledTokenRates.scaled`
    pub fn scaled(self, multiplier: f64) -> Self {
        Self {
            input: self.input * multiplier,
            output: self.output * multiplier,
            cache_read: self.cache_read * multiplier,
            cache_read_audio: self.cache_read_audio * multiplier,
            cache_read_image: self.cache_read_image.map(|rate| rate * multiplier),
            cache_write: self.cache_write * multiplier,
            cache_write_one_hour: self.cache_write_one_hour * multiplier,
            cache_write_twelve_hours: self.cache_write_twelve_hours * multiplier,
            reasoning: self.reasoning * multiplier,
        }
    }
}

/// `get_token_type_cost_breakdown`. Custom pricing bills every write at the flat write rate
pub(super) fn breakdown(rates: &BilledRates, usage: &TokenUsage, flat_writes: bool) -> Breakdown {
    let reads = usage.cache_read;
    let writes = usage.cache_write;
    let cache_creation = if flat_writes {
        usage.cache_write_total() as f64 * rates.cache_write
    } else {
        writes.default as f64 * rates.cache_write
            + writes.one_hour as f64 * rates.cache_write_one_hour
            + writes.twelve_hours as f64 * rates.cache_write_twelve_hours
    };
    Breakdown {
        cache_read: reads.text as f64 * rates.cache_read
            + reads.audio as f64 * rates.cache_read_audio
            + reads.image as f64 * rates.cache_read_image.unwrap_or(rates.cache_read),
        cache_creation,
        reasoning: usage.reasoning as f64 * rates.reasoning,
    }
}

/// `calculate_prompt_caching_savings`: what the reads saved over the input rate, minus what the writes cost over it.
/// A write rate of 0.0 claims no premium, and the result carries the residency, FedRAMP and regional uplifts only
pub(super) fn cache_savings(quote: &Quote<'_>, usage: &TokenUsage, uplift: f64) -> f64 {
    let rates = billed_rates(quote, usage.cache_read.explicit);
    let input = rates.input;
    let write = if rates.cache_write == 0.0 {
        input
    } else {
        rates.cache_write
    };
    let read_discount = usage.cache_read_total() as f64 * (input - rates.cache_read).max(0.0);
    let writes = usage.cache_write;
    let write_premium = writes.default as f64 * (write - input)
        + writes.one_hour as f64 * (rates.cache_write_one_hour - input)
        + writes.twelve_hours as f64 * (rates.cache_write_twelve_hours - input);
    (read_discount - write_premium) * uplift
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;
    use crate::call::InferenceGeo;
    use crate::money::Multiplier;
    use crate::price::ServiceTier::Standard;
    use crate::query::{CacheReads, CacheWrites, DataResidency};
    use crate::testing::{BASE, CACHE_WRITE_HOUR, basis, call, prices};

    fn caching(read: u64, default: u64, one_hour: u64) -> TokenUsage {
        TokenUsage {
            cache_read: CacheReads {
                text: read,
                ..CacheReads::default()
            },
            cache_write: CacheWrites {
                default,
                one_hour,
                twelve_hours: 0,
            },
            ..TokenUsage::default()
        }
    }

    #[rstest]
    #[case::reads_save_the_input_rate_gap(caching(100, 0, 0), 90.0)]
    #[case::writes_cost_the_premium(caching(0, 10, 0), -2.5)]
    #[case::hour_writes_cost_their_own_premium(caching(0, 0, 10), -10.0)]
    #[case::both(caching(100, 10, 10), 77.5)]
    fn savings_are_read_discount_minus_write_premium(
        #[case] usage: TokenUsage,
        #[case] expected: f64,
    ) {
        let prices = prices(&[
            (Meter::TEXT_INPUT, Standard, BASE, 1.0),
            (Meter::CACHE_READ, Standard, BASE, 0.1),
            (Meter::CACHE_WRITE, Standard, BASE, 1.25),
            (CACHE_WRITE_HOUR, Standard, BASE, 2.0),
        ]);
        assert_eq!(
            cache_savings(&prices.quote(&call(), 0), &usage, 1.0),
            expected
        );
    }

    #[rstest]
    fn a_free_write_rate_claims_no_premium() {
        let prices = prices(&[
            (Meter::TEXT_INPUT, Standard, BASE, 1.0),
            (Meter::CACHE_WRITE, Standard, BASE, 0.0),
        ]);
        assert_eq!(
            cache_savings(&prices.quote(&call(), 0), &caching(0, 10, 0), 1.0),
            0.0
        );
    }

    #[rstest]
    #[case::none(None, None, 1.0)]
    #[case::geo(Some("us"), None, 1.5)]
    #[case::speed(None, Some(Speed::Fast), 2.0)]
    #[case::global_geo(Some("global"), Some(Speed::Fast), 2.0)]
    #[case::both(Some("US"), Some(Speed::Fast), 3.0)]
    fn the_breakdown_multiplier_takes_every_named_uplift(
        #[case] geo: Option<&str>,
        #[case] speed: Option<Speed>,
        #[case] expected: f64,
    ) {
        let adjustments = Adjustments::new([
            (Uplift::Speed(Speed::Fast), Multiplier::new(2.0).unwrap()),
            (
                Uplift::InferenceGeo(InferenceGeo::new("us")),
                Multiplier::new(1.5).unwrap(),
            ),
            (
                Uplift::Residency(crate::price::Residency::Eu),
                Multiplier::new(1.1).unwrap(),
            ),
        ]);
        let usage = TokenUsage {
            inference_geo: geo.map(InferenceGeo::new),
            speed,
            ..TokenUsage::default()
        };
        assert_eq!(
            multiplier(&adjustments, &basis(), &usage, &ProviderEntry::Readable),
            expected
        );
        let eu = Basis {
            data_residency: Some(DataResidency::Eu),
            ..basis()
        };
        assert_eq!(savings_uplift(&adjustments, &eu), 1.1);
    }

    #[rstest]
    fn the_breakdown_bills_reads_writes_and_reasoning_at_the_billed_rates() {
        let rates = BilledRates {
            cache_read: 0.5,
            cache_read_audio: 0.25,
            cache_write: 2.0,
            cache_write_one_hour: 3.0,
            reasoning: 4.0,
            ..BilledRates::default()
        };
        let usage = TokenUsage {
            cache_read: CacheReads {
                text: 10,
                audio: 4,
                image: 2,
                explicit: false,
            },
            cache_write: CacheWrites {
                default: 1,
                one_hour: 2,
                twelve_hours: 0,
            },
            reasoning: 5,
            ..TokenUsage::default()
        };
        assert_eq!(
            breakdown(&rates, &usage, false),
            Breakdown {
                cache_read: 7.0,
                cache_creation: 8.0,
                reasoning: 20.0
            }
        );
        assert_eq!(breakdown(&rates, &usage, true).cache_creation, 6.0);
    }
}
