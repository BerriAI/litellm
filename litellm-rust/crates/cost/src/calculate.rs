use crate::call::CallContext;
use crate::money::{Multiplier, UnitPrice};
use crate::price::{
    Adjustments, Band, CacheReadFallback, CacheReadMode, CacheTtl, Channel, Feature, InputUnit,
    Meter, OffPeakRates, OutputUnit, Price, Prices, ServiceTier, TierScope, TokenKind, Uplift,
    When,
};
use crate::priced::{Line, Outcome, Priced, Quantity, Side};
use crate::usage::{DurationMeter, Usage};

const TEXT_INPUT: Meter = Meter::Input(InputUnit::Token(TokenKind::Text));
const TEXT_OUTPUT: Meter = Meter::Output(OutputUnit::Token(TokenKind::Text));
const IMAGE_OUTPUT: Meter = Meter::Output(OutputUnit::Token(TokenKind::Image));
const REASONING: Meter = Meter::Output(OutputUnit::ReasoningToken);
const CACHE_READ: Meter = Meter::CacheRead {
    kind: TokenKind::Text,
    mode: CacheReadMode::Implicit,
};
const CACHE_WRITE: Meter = Meter::CacheWrite {
    kind: TokenKind::Text,
    ttl: CacheTtl::Default,
};

/// Prices one call's token usage (decisions 21 and 30). Every meter with usage is a `Line`, one that borrowed another
/// meter's rate names it in `fallback_from`, and one with no price and no fallback is listed in `missing`
pub fn calculate(
    prices: &Prices,
    adjustments: &Adjustments,
    usage: &Usage,
    call: &CallContext,
) -> Outcome {
    let rates = Rates::new(prices, usage, call);
    let uplifts = uplifts(adjustments, call);
    let tokens = &usage.tokens;
    let billed: Vec<(Meter, Option<Rate>, u64, Side)> = tokens
        .input
        .iter()
        .map(|(kind, count)| {
            (
                Meter::Input(InputUnit::Token(*kind)),
                rates.input(*kind),
                *count,
                Side::Prompt,
            )
        })
        .chain(tokens.cache_read.iter().map(|((kind, mode), count)| {
            let meter = Meter::CacheRead {
                kind: *kind,
                mode: *mode,
            };
            (meter, rates.cache_read(*kind, *mode), *count, Side::Prompt)
        }))
        .chain(tokens.cache_write.iter().map(|((kind, ttl), count)| {
            let meter = Meter::CacheWrite {
                kind: *kind,
                ttl: *ttl,
            };
            (meter, rates.cache_write(*ttl), *count, Side::Prompt)
        }))
        .chain(tokens.output.iter().map(|(kind, count)| {
            let meter = Meter::Output(OutputUnit::Token(*kind));
            (meter, rates.output(*kind), *count, Side::Completion)
        }))
        .chain([(
            REASONING,
            rates.reasoning(),
            tokens.reasoning,
            Side::Completion,
        )])
        .filter(|(.., count, _)| *count > 0)
        .collect();
    let lines: Vec<Line> = billed
        .iter()
        .filter_map(|(meter, rate, count, side)| {
            Some(rate.clone()?.line(meter.clone(), *count, *side, &uplifts))
        })
        .collect();
    let missing: Vec<Meter> = billed
        .into_iter()
        .filter_map(|(meter, rate, ..)| rate.is_none().then_some(meter))
        .chain(without_an_arm(usage))
        .collect();
    if lines.is_empty() && !missing.is_empty() {
        return Outcome::Unpriced { missing };
    }
    let total = lines.iter().map(|line| line.amount).sum();
    Outcome::Priced(Priced {
        lines,
        missing,
        total,
    })
}

/// One resolved rate, and the meter it came from when that is not the billed meter's own
#[derive(Clone, Debug)]
struct Rate {
    price: UnitPrice,
    borrowed: Option<Meter>,
}

impl Rate {
    fn own(price: UnitPrice) -> Self {
        Self {
            price,
            borrowed: None,
        }
    }

    fn borrowed_from(self, meter: &Meter) -> Self {
        Self {
            borrowed: Some(self.borrowed.unwrap_or_else(|| meter.clone())),
            ..self
        }
    }

    fn halved(self) -> Option<Self> {
        Some(Self {
            price: UnitPrice::new(self.price.get() * 0.5)?,
            ..self
        })
    }

    fn line(self, meter: Meter, count: u64, side: Side, uplifts: &[Multiplier]) -> Line {
        let quantity = Quantity::Count(count);
        Line {
            quantity,
            unit_price: self.price,
            amount: uplifts
                .iter()
                .fold(self.price * quantity, |amount, factor| amount * *factor),
            fallback_from: self.borrowed,
            meter,
            side,
        }
    }
}

/// The rates one call resolves, main's `_get_token_base_cost` and its modality, cache and reasoning rules
struct Rates<'a> {
    prices: &'a Prices,
    overlay: Option<&'a Prices>,
    channel: Channel,
    tier: ServiceTier,
    band: Band,
    whole_call: bool,
    off_peak: Option<OffPeakRates>,
    text_input: Option<Rate>,
    text_output: Option<Rate>,
}

impl<'a> Rates<'a> {
    /// A tier table bills the whole call at the tier the prompt picks, unless that tier sets no input price. Then
    /// the row is priced as if it had no tiers, as main's `_select_priced_tier` does
    fn new(prices: &'a Prices, usage: &Usage, call: &'a CallContext) -> Self {
        let band = prices.band(call.tier, usage.prompt_tokens);
        let tier_input = Price::online(
            TEXT_INPUT,
            When {
                tier: ServiceTier::Standard,
                band,
            },
        );
        let whole_call = prices.billing().tier_scope == TierScope::WholeCall
            && prices.get(&tier_input).is_some();
        let lookup = Self {
            prices,
            overlay: call.overlay.as_ref(),
            channel: call.channel,
            tier: call.tier,
            band: match (prices.billing().tier_scope, whole_call) {
                (TierScope::WholeCall, false) => Band::Base,
                _ => band,
            },
            whole_call,
            off_peak: prices
                .off_peak()
                .filter(|off_peak| off_peak.schedule().is_open(call.at))
                .map(|off_peak| *off_peak.rates()),
            text_input: None,
            text_output: None,
        };
        let text_input = lookup
            .off_peak(|rates| rates.input)
            .or_else(|| lookup.banded(&TEXT_INPUT));
        let text_output = lookup
            .off_peak(|rates| rates.output)
            .or_else(|| lookup.output_or_image_rate());
        Self {
            text_input,
            text_output,
            ..lookup
        }
    }

    fn input(&self, kind: TokenKind) -> Option<Rate> {
        match kind {
            TokenKind::Text => self.text_input.clone(),
            _ => self
                .banded(&Meter::Input(InputUnit::Token(kind)))
                .or_else(|| self.borrowed_input()),
        }
    }

    fn output(&self, kind: TokenKind) -> Option<Rate> {
        match kind {
            TokenKind::Text => self.text_output.clone(),
            _ => self
                .banded(&Meter::Output(OutputUnit::Token(kind)))
                .or_else(|| self.borrowed_output()),
        }
    }

    fn cache_read(&self, kind: TokenKind, mode: CacheReadMode) -> Option<Rate> {
        let borrowed_implicit = || {
            self.implicit_cache_read()
                .map(|rate| rate.borrowed_from(&CACHE_READ))
        };
        match (kind, mode) {
            (TokenKind::Text, CacheReadMode::Implicit) => self.implicit_cache_read(),
            (TokenKind::Text, CacheReadMode::Explicit) => {
                let own = self.banded(&Meter::CacheRead { kind, mode });
                match (self.off_peak(|rates| rates.explicit_cache_read), own) {
                    (Some(off_peak), _) => Some(off_peak),
                    (None, Some(own)) => {
                        Some(self.off_peak(|rates| rates.cache_read).unwrap_or(own))
                    }
                    (None, None) => borrowed_implicit(),
                }
            }
            _ => self
                .banded(&Meter::CacheRead { kind, mode })
                .or_else(borrowed_implicit),
        }
    }

    fn cache_write(&self, ttl: CacheTtl) -> Option<Rate> {
        match ttl {
            CacheTtl::Default => self.default_cache_write(),
            CacheTtl::OneHour | CacheTtl::TwelveHours => self
                .banded(&Meter::CacheWrite {
                    kind: TokenKind::Text,
                    ttl,
                })
                .or_else(|| {
                    self.default_cache_write()
                        .map(|rate| rate.borrowed_from(&CACHE_WRITE))
                }),
        }
    }

    /// Off-peak first, then the tier's reasoning price. A tier that prices output and not reasoning bills reasoning at
    /// its output rate, otherwise the standard reasoning price, then the output rate. Reasoning prices are never banded
    /// outside a whole-call tier table
    fn reasoning(&self) -> Option<Rate> {
        let tier_base = When {
            tier: self.tier,
            band: Band::Base,
        };
        let tier_prices_output_only =
            || self.tier != ServiceTier::Standard && self.at(&TEXT_OUTPUT, &[tier_base]).is_some();
        self.off_peak(|rates| rates.reasoning)
            .or_else(|| match self.whole_call() {
                true => self.banded(&REASONING),
                false => self.at(&REASONING, &[tier_base]).or_else(|| {
                    (!tier_prices_output_only())
                        .then(|| self.at(&REASONING, &[When::default()]))
                        .flatten()
                }),
            })
            .or_else(|| self.borrowed_output())
    }

    fn borrowed_input(&self) -> Option<Rate> {
        self.text_input
            .clone()
            .map(|rate| rate.borrowed_from(&TEXT_INPUT))
    }

    fn borrowed_output(&self) -> Option<Rate> {
        self.text_output
            .clone()
            .map(|rate| rate.borrowed_from(&TEXT_OUTPUT))
    }

    fn implicit_cache_read(&self) -> Option<Rate> {
        self.off_peak(|rates| rates.cache_read)
            .or_else(|| self.banded(&CACHE_READ))
            .or_else(|| match self.prices.billing().cache_read_fallback {
                CacheReadFallback::InputRate => self.borrowed_input(),
                CacheReadFallback::HalfInputRate => self.borrowed_input()?.halved(),
            })
    }

    fn default_cache_write(&self) -> Option<Rate> {
        self.off_peak(|rates| rates.cache_write)
            .or_else(|| self.banded(&CACHE_WRITE))
            .or_else(|| self.borrowed_input())
    }

    /// main reads a missing or zero output price as unknown and bills image models at their image token rate. A
    /// whole-call tier with no output price bills the row's own output price
    fn output_or_image_rate(&self) -> Option<Rate> {
        let own = self.banded(&TEXT_OUTPUT).or_else(|| {
            self.whole_call()
                .then(|| self.at(&TEXT_OUTPUT, &[When::default()]))?
        });
        match own {
            Some(rate) if rate.price.get() != 0.0 => Some(rate),
            own => self
                .at(&IMAGE_OUTPUT, &[When::default()])
                .map(|rate| rate.borrowed_from(&IMAGE_OUTPUT))
                .or(own),
        }
    }

    fn off_peak(&self, rate: fn(&OffPeakRates) -> Option<UnitPrice>) -> Option<Rate> {
        self.off_peak.as_ref().and_then(rate).map(Rate::own)
    }

    fn whole_call(&self) -> bool {
        self.whole_call
    }

    /// The meter's own rate in this call's band, tried at `(tier, band)`, `(standard, band)`, `(tier, base)` and
    /// `(standard, base)`. A whole-call tier table prices every meter at the band's own rate only
    fn banded(&self, meter: &Meter) -> Option<Rate> {
        let standard = ServiceTier::Standard;
        let whens: &[When] = match self.whole_call() {
            true => &[When {
                tier: standard,
                band: self.band,
            }],
            false => &[
                When {
                    tier: self.tier,
                    band: self.band,
                },
                When {
                    tier: standard,
                    band: self.band,
                },
                When {
                    tier: self.tier,
                    band: Band::Base,
                },
                When::default(),
            ],
        };
        self.at(meter, whens)
    }

    /// The first price for `meter` at `whens` on this call's channel. A per-call overlay that prices the meter at all
    /// replaces the row's prices for it (decision 12). Decisions bills the online prices it does not set, and a batch
    /// call with no batch price bills half the online price, main's `_batch_or_half`
    fn at(&self, meter: &Meter, whens: &[When]) -> Option<Rate> {
        let source = match self.overlay {
            Some(overlay) if overlay.iter().any(|(price, _)| &price.meter == meter) => overlay,
            _ => self.prices,
        };
        let on = |channel: Channel| {
            whens.iter().find_map(|when| {
                source.get(&Price {
                    meter: meter.clone(),
                    channel,
                    when: *when,
                })
            })
        };
        match self.channel {
            Channel::Online => on(Channel::Online).map(Rate::own),
            Channel::Decisions => on(Channel::Decisions)
                .or_else(|| on(Channel::Online))
                .map(Rate::own),
            Channel::Batch => on(Channel::Batch).map(Rate::own).or_else(|| {
                Rate::own(on(Channel::Online)?)
                    .borrowed_from(meter)
                    .halved()
            }),
        }
    }
}

/// Meters with usage that no arm prices yet. Characters and seconds land in 09, the per-unit meters in 10
fn without_an_arm(usage: &Usage) -> impl Iterator<Item = Meter> + '_ {
    let counted = [
        (usage.characters.prompt, Meter::Input(InputUnit::Character)),
        (
            usage.characters.completion,
            Meter::Output(OutputUnit::Character),
        ),
        (usage.images.input, Meter::Input(InputUnit::Image)),
        (usage.images.input_pixels, Meter::Input(InputUnit::Pixel)),
        (
            usage.images.output.len() as u64,
            Meter::Output(OutputUnit::Image(None)),
        ),
        (
            usage.search.queries,
            Meter::Input(InputUnit::Query { over_results: None }),
        ),
        (
            usage.search.maps_grounding,
            Meter::Feature(Feature::GoogleMapsGroundingQuery),
        ),
        (
            usage.search.browser_opens,
            Meter::Feature(Feature::BrowserOpen),
        ),
        (
            usage.search.additional_results,
            Meter::Feature(Feature::AdditionalResult),
        ),
        (usage.text_messages, Meter::Input(InputUnit::TextMessage)),
        (
            usage.tokens.citation,
            Meter::Feature(Feature::CitationToken),
        ),
    ];
    let web_search = usage
        .search
        .web_search
        .filter(|search| search.requests > 0)
        .map(|search| Meter::WebSearch(search.context));
    let durations = usage
        .durations
        .iter()
        .filter(|(_, duration)| !duration.is_zero())
        .map(|(meter, _)| match meter {
            DurationMeter::WallClock => Meter::WallClockSecond,
            DurationMeter::InputAudio => Meter::Input(InputUnit::AudioSecond),
            DurationMeter::LiveVoice => Meter::Input(InputUnit::Second),
            DurationMeter::OutputVideo { resolution } => {
                Meter::Output(OutputUnit::Second(resolution.clone()))
            }
        });
    counted
        .into_iter()
        .filter_map(|(count, meter)| (count > 0).then_some(meter))
        .chain(web_search)
        .chain(durations)
}

fn uplifts(adjustments: &Adjustments, call: &CallContext) -> Vec<Multiplier> {
    let residency = call
        .residency
        .and_then(|residency| adjustments.get(&Uplift::Residency(residency)));
    let endpoint = call
        .non_global_endpoint
        .then(|| adjustments.get(&Uplift::NonGlobalEndpoint))
        .flatten();
    residency.into_iter().chain(endpoint).collect()
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::num::NonZeroU64;

    use rstest::rstest;

    use super::*;
    use crate::price::{Billing, Residency};

    fn price(
        meter: &Meter,
        channel: Channel,
        tier: ServiceTier,
        band: Band,
        rate: f64,
    ) -> (Price, UnitPrice) {
        (
            Price {
                meter: meter.clone(),
                channel,
                when: When { tier, band },
            },
            UnitPrice::new(rate).unwrap(),
        )
    }

    fn standard(meter: &Meter, rate: f64) -> (Price, UnitPrice) {
        price(
            meter,
            Channel::Online,
            ServiceTier::Standard,
            Band::Base,
            rate,
        )
    }

    fn row(rates: impl IntoIterator<Item = (Price, UnitPrice)>) -> Prices {
        Prices::new(rates.into_iter().collect(), None, Billing::default())
    }

    fn call() -> CallContext {
        CallContext {
            tier: ServiceTier::Standard,
            channel: Channel::Online,
            residency: None,
            non_global_endpoint: false,
            speed: None,
            inference_geo: None,
            at: jiff::Timestamp::UNIX_EPOCH,
            overlay: None,
        }
    }

    fn usage(input: u64, cache_read: u64, output: u64) -> Usage {
        let mut usage = Usage {
            prompt_tokens: input + cache_read,
            ..Usage::default()
        };
        usage.tokens.input = BTreeMap::from([(TokenKind::Text, input)]);
        usage.tokens.cache_read =
            BTreeMap::from([((TokenKind::Text, CacheReadMode::Implicit), cache_read)]);
        usage.tokens.output = BTreeMap::from([(TokenKind::Text, output)]);
        usage
    }

    fn priced(outcome: Outcome) -> Priced {
        match outcome {
            Outcome::Priced(priced) => priced,
            Outcome::Unpriced { missing } => panic!("unpriced, missing {missing:?}"),
        }
    }

    fn total(prices: &Prices, usage: &Usage, call: &CallContext) -> f64 {
        priced(calculate(prices, &Adjustments::default(), usage, call))
            .total
            .get()
    }

    fn close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() <= expected.abs() * 1e-12,
            "{actual} != {expected}"
        );
    }

    #[rstest]
    fn a_cache_read_with_no_price_borrows_the_input_rate_as_a_visible_line() {
        let prices = row([standard(&TEXT_INPUT, 1e-6), standard(&TEXT_OUTPUT, 2e-6)]);
        let outcome = priced(calculate(
            &prices,
            &Adjustments::default(),
            &usage(200, 800, 10),
            &call(),
        ));
        let cache_line = outcome
            .lines
            .iter()
            .find(|line| line.meter == CACHE_READ)
            .unwrap();
        assert_eq!(cache_line.fallback_from, Some(TEXT_INPUT));
        close(outcome.total.get(), 1000.0 * 1e-6 + 10.0 * 2e-6);
    }

    #[rstest]
    fn a_meter_with_usage_and_no_price_is_missing_and_the_rest_is_still_priced() {
        let prices = row([standard(&TEXT_OUTPUT, 2e-6)]);
        let outcome = priced(calculate(
            &prices,
            &Adjustments::default(),
            &usage(100, 0, 10),
            &call(),
        ));
        assert_eq!(outcome.missing, [TEXT_INPUT]);
        close(outcome.total.get(), 10.0 * 2e-6);
    }

    #[rstest]
    fn a_meter_no_arm_prices_yet_is_missing() {
        let prices = row([standard(&TEXT_INPUT, 1e-6)]);
        let usage = Usage {
            characters: crate::usage::Characters {
                prompt: 40,
                completion: 0,
            },
            ..usage(100, 0, 0)
        };
        let outcome = priced(calculate(&prices, &Adjustments::default(), &usage, &call()));
        assert_eq!(outcome.missing, [Meter::Input(InputUnit::Character)]);
    }

    #[rstest]
    fn a_call_with_no_priced_meter_is_unpriced() {
        let outcome = calculate(
            &row([]),
            &Adjustments::default(),
            &usage(100, 0, 10),
            &call(),
        );
        assert_eq!(
            outcome,
            Outcome::Unpriced {
                missing: vec![TEXT_INPUT, TEXT_OUTPUT]
            }
        );
    }

    #[rstest]
    fn the_overlay_replaces_the_meters_it_prices_and_keeps_the_rest() {
        let prices = row([standard(&TEXT_INPUT, 1e-6), standard(&TEXT_OUTPUT, 2e-6)]);
        let call = CallContext {
            overlay: Some(row([standard(&TEXT_INPUT, 5e-6)])),
            ..call()
        };
        close(
            total(&prices, &usage(100, 0, 10), &call),
            100.0 * 5e-6 + 10.0 * 2e-6,
        );
    }

    #[rstest]
    #[case::requested_region(Residency::Eu, true)]
    #[case::other_region(Residency::Us, false)]
    fn a_residency_uplift_multiplies_every_line(
        #[case] priced_region: Residency,
        #[case] applies: bool,
    ) {
        let prices = row([standard(&TEXT_INPUT, 1e-6), standard(&TEXT_OUTPUT, 2e-6)]);
        let adjustments = Adjustments::new([(
            Uplift::Residency(priced_region),
            Multiplier::new(1.1).unwrap(),
        )]);
        let call = CallContext {
            residency: Some(Residency::Eu),
            ..call()
        };
        let outcome = priced(calculate(&prices, &adjustments, &usage(100, 0, 10), &call));
        let base = 100.0 * 1e-6 + 10.0 * 2e-6;
        close(outcome.total.get(), if applies { base * 1.1 } else { base });
    }

    #[rstest]
    #[case::missing(None)]
    #[case::zero(Some(0.0))]
    fn an_image_model_with_no_output_price_bills_output_at_its_image_token_rate(
        #[case] output: Option<f64>,
    ) {
        let prices = row(std::iter::once(standard(&IMAGE_OUTPUT, 4e-5))
            .chain(output.map(|rate| standard(&TEXT_OUTPUT, rate))));
        let outcome = priced(calculate(
            &prices,
            &Adjustments::default(),
            &usage(0, 0, 10),
            &call(),
        ));
        assert_eq!(outcome.lines[0].fallback_from, Some(IMAGE_OUTPUT));
        close(outcome.total.get(), 10.0 * 4e-5);
    }

    #[rstest]
    #[case::below_the_band(1_000, 1e-6)]
    #[case::above_the_band(300_000, 2e-6)]
    fn the_prompt_band_prices_every_token_meter(#[case] prompt: u64, #[case] input_rate: f64) {
        let over = Band::over_prompt_tokens(200_000);
        let prices = row([
            standard(&TEXT_INPUT, 1e-6),
            price(
                &TEXT_INPUT,
                Channel::Online,
                ServiceTier::Standard,
                over,
                2e-6,
            ),
        ]);
        let usage = Usage {
            prompt_tokens: prompt,
            ..usage(100, 0, 0)
        };
        close(total(&prices, &usage, &call()), 100.0 * input_rate);
    }

    #[rstest]
    #[case::batch_price(Some(3e-7), 3e-7, None)]
    #[case::half_the_online_price(None, 5e-7, Some(TEXT_INPUT))]
    fn a_batch_call_bills_its_batch_price_or_half_the_online_one(
        #[case] batch: Option<f64>,
        #[case] rate: f64,
        #[case] fallback: Option<Meter>,
    ) {
        let batch_price = batch.map(|rate| {
            price(
                &TEXT_INPUT,
                Channel::Batch,
                ServiceTier::Standard,
                Band::Base,
                rate,
            )
        });
        let prices = row(std::iter::once(standard(&TEXT_INPUT, 1e-6)).chain(batch_price));
        let call = CallContext {
            channel: Channel::Batch,
            ..call()
        };
        let outcome = priced(calculate(
            &prices,
            &Adjustments::default(),
            &usage(100, 0, 0),
            &call,
        ));
        assert_eq!(outcome.lines[0].fallback_from, fallback);
        close(outcome.total.get(), 100.0 * rate);
    }

    #[rstest]
    #[case::tier_reasoning_price(Some(9e-6), true, 9e-6)]
    #[case::tier_output_without_reasoning(None, true, 7e-6)]
    #[case::standard_reasoning_price(None, false, 3e-6)]
    fn reasoning_follows_mains_tier_rules(
        #[case] tier_reasoning: Option<f64>,
        #[case] tier_output: bool,
        #[case] expected: f64,
    ) {
        let priority = |meter: &Meter, rate| {
            price(
                meter,
                Channel::Online,
                ServiceTier::Priority,
                Band::Base,
                rate,
            )
        };
        let prices = row([standard(&TEXT_OUTPUT, 2e-6), standard(&REASONING, 3e-6)]
            .into_iter()
            .chain(tier_reasoning.map(|rate| priority(&REASONING, rate)))
            .chain(tier_output.then(|| priority(&TEXT_OUTPUT, 7e-6))));
        let mut usage = usage(0, 0, 0);
        usage.tokens.reasoning = 10;
        let call = CallContext {
            tier: ServiceTier::Priority,
            ..call()
        };
        close(total(&prices, &usage, &call), 10.0 * expected);
    }

    #[rstest]
    fn a_long_cache_write_with_no_price_borrows_the_default_write_rate() {
        let prices = row([standard(&TEXT_INPUT, 1e-6), standard(&CACHE_WRITE, 1.25e-6)]);
        let mut usage = usage(0, 0, 0);
        usage.tokens.cache_write = BTreeMap::from([((TokenKind::Text, CacheTtl::OneHour), 100)]);
        let outcome = priced(calculate(&prices, &Adjustments::default(), &usage, &call()));
        assert_eq!(outcome.lines[0].fallback_from, Some(CACHE_WRITE));
        close(outcome.total.get(), 100.0 * 1.25e-6);
    }

    #[rstest]
    fn a_half_input_cache_read_row_bills_cache_reads_at_half_the_input_rate() {
        let prices = Prices::new(
            [standard(&TEXT_INPUT, 1e-6)].into_iter().collect(),
            None,
            Billing {
                cache_read_fallback: CacheReadFallback::HalfInputRate,
                ..Billing::default()
            },
        );
        close(total(&prices, &usage(0, 100, 0), &call()), 100.0 * 0.5e-6);
    }

    #[rstest]
    #[case::low_tier(1_000, 1e-6)]
    #[case::high_tier(40_000, 4e-6)]
    fn a_whole_call_tier_bills_cache_reads_at_its_own_input_rate(
        #[case] prompt: u64,
        #[case] rate: f64,
    ) {
        let over = Band::OverPromptTokens(NonZeroU64::new(32_000).unwrap());
        let prices = Prices::new(
            [
                standard(&TEXT_INPUT, 1e-6),
                standard(&CACHE_READ, 0.1e-6),
                price(
                    &TEXT_INPUT,
                    Channel::Online,
                    ServiceTier::Standard,
                    over,
                    4e-6,
                ),
            ]
            .into_iter()
            .collect(),
            None,
            Billing {
                tier_scope: TierScope::WholeCall,
                ..Billing::default()
            },
        );
        let usage = Usage {
            prompt_tokens: prompt,
            ..usage(0, 100, 0)
        };
        let expected = if prompt > 32_000 { rate } else { 0.1e-6 };
        close(total(&prices, &usage, &call()), 100.0 * expected);
    }
}
