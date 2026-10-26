use std::collections::BTreeSet;

use crate::calculate::{Usage, When, resolve};
use crate::money::UnitPrice;
use crate::price::{Band, Channel, Meter, OutputUnit, Price, Prices, ServiceTier, TokenKind};

#[derive(Clone, Copy, Debug, Default)]
pub enum SelectionPolicy<'a> {
    #[default]
    Generic,
    SdkTokens(&'a TokenPriceSources),
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TokenPriceSources {
    pub flat: BTreeSet<Price>,
    pub tiers: Vec<TokenTier>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TokenTier {
    pub range: [u64; 2],
    pub meters: BTreeSet<Meter>,
}

impl SelectionPolicy<'_> {
    pub(crate) fn resolve(
        self,
        prices: &Prices,
        meter: &Meter,
        band: Band,
        usage: &Usage,
        when: &When,
    ) -> Option<(Meter, UnitPrice)> {
        let Self::SdkTokens(sources) = self else {
            return resolve(prices, meter, band, usage, when);
        };
        if when.channel != Channel::Online {
            return resolve(prices, meter, band, usage, when);
        }
        let output = Meter::Output(OutputUnit::Token(TokenKind::Text));
        match meter {
            Meter::Output(OutputUnit::ReasoningToken) => {
                let off_peak = prices
                    .off_peak()
                    .filter(|off_peak| off_peak.schedule().contains(when.at))
                    .and_then(|off_peak| off_peak.rates().reasoning);
                if let Some(rate) = off_peak {
                    return Some((meter.clone(), rate));
                }
                if let Some(tier) = sources.priced_tier(usage.prompt_tokens) {
                    if let Some(rate) = tier.rate(prices, meter) {
                        return Some((meter.clone(), rate));
                    }
                    if let Some(rate) = tier.rate(prices, &output) {
                        return Some((output, rate));
                    }
                }
                if let Some(rate) = sources.flat_rate(prices, meter, when.tier) {
                    return Some((meter.clone(), rate));
                }
                if when.tier != ServiceTier::Standard
                    && sources.flat_rate(prices, &output, when.tier).is_some()
                {
                    return resolve(prices, &output, band, usage, when);
                }
                sources
                    .flat_rate(prices, meter, ServiceTier::Standard)
                    .map(|rate| (meter.clone(), rate))
                    .or_else(|| resolve(prices, &output, band, usage, when))
            }
            Meter::Output(OutputUnit::Token(kind)) if *kind != TokenKind::Text => sources
                .flat_rate(prices, meter, ServiceTier::Standard)
                .map(|rate| (meter.clone(), rate))
                .or_else(|| resolve(prices, &output, band, usage, when)),
            _ => resolve(prices, meter, band, usage, when),
        }
    }
}

impl TokenPriceSources {
    fn flat_rate(&self, prices: &Prices, meter: &Meter, tier: ServiceTier) -> Option<UnitPrice> {
        let key = Price::online(
            meter.clone(),
            crate::price::When {
                tier,
                band: Band::Base,
            },
        );
        self.flat.contains(&key).then(|| prices.get(&key)).flatten()
    }

    fn priced_tier(&self, prompt_tokens: u64) -> Option<&TokenTier> {
        if prompt_tokens == 0 {
            return None;
        }
        self.tiers
            .iter()
            .filter(|tier| tier.range[0] < prompt_tokens && prompt_tokens <= tier.range[1])
            .min_by_key(|tier| tier.range[0])
            .or_else(|| self.tiers.iter().max_by_key(|tier| tier.range[0]))
            .filter(|tier| tier.meters.contains(&Meter::TEXT_INPUT))
    }
}

impl TokenTier {
    fn rate(&self, prices: &Prices, meter: &Meter) -> Option<UnitPrice> {
        self.meters
            .contains(meter)
            .then(|| {
                prices.get(&Price::online(
                    meter.clone(),
                    crate::price::When {
                        band: Band::over_prompt_tokens(self.range[0]),
                        ..Default::default()
                    },
                ))
            })
            .flatten()
    }
}

#[cfg(test)]
mod tests {
    use rstest::{fixture, rstest};
    use time::OffsetDateTime;

    use super::*;
    use crate::price::{InputUnit, OffPeak, OffPeakRates, When as PriceWhen};
    use crate::schedule::{Schedule, Window};
    use crate::{Breakdown, Line, Priced, calculate, calculate_with_policy};

    const OUTPUT: Meter = Meter::Output(OutputUnit::Token(TokenKind::Text));
    const REASONING: Meter = Meter::Output(OutputUnit::ReasoningToken);

    #[fixture]
    fn when() -> When {
        When {
            tier: ServiceTier::Priority,
            channel: Channel::Online,
            at: OffsetDateTime::UNIX_EPOCH,
        }
    }

    fn rate(meter: Meter, tier: ServiceTier, band: Band, value: f64) -> (Price, UnitPrice) {
        (
            Price::online(meter, PriceWhen { tier, band }),
            UnitPrice::new(value).unwrap(),
        )
    }

    fn sources(prices: &Prices) -> TokenPriceSources {
        TokenPriceSources {
            flat: prices.iter().map(|(key, _)| key.clone()).collect(),
            tiers: vec![],
        }
    }

    fn overlay(reasoning: Option<f64>, output: Option<f64>) -> Option<OffPeak> {
        OffPeak::new(
            Schedule {
                windows: vec![Window {
                    hours: vec!["00:00-01:00".parse().unwrap()],
                    weekdays: None,
                }],
                weekday_timezone: None,
            },
            OffPeakRates {
                reasoning: reasoning.map(|value| UnitPrice::new(value).unwrap()),
                output: output.map(|value| UnitPrice::new(value).unwrap()),
                ..Default::default()
            },
        )
    }

    fn usage(meter: Meter, prompt_tokens: u64, inclusive_threshold: bool) -> Usage {
        Usage {
            quantities: [(meter, 5)].into(),
            prompt_tokens,
            inclusive_threshold,
            ..Default::default()
        }
    }

    fn expected(meter: Meter, original: Meter, value: f64) -> Priced {
        let unit_price = UnitPrice::new(value).unwrap();
        Priced::Known(Breakdown {
            total: unit_price * 5,
            lines: vec![Line {
                fallback_from: (meter != original).then_some(original),
                meter,
                quantity: 5,
                unit_price,
                amount: unit_price * 5,
            }],
        })
    }

    #[rstest]
    #[case::explicit_tier_output(None, Some(4.0), Some(3.0), OUTPUT, 4.0)]
    #[case::explicit_tier_reasoning(Some(7.0), Some(4.0), Some(3.0), REASONING, 7.0)]
    #[case::free_tier_output(None, Some(0.0), Some(3.0), OUTPUT, 0.0)]
    #[case::free_tier_reasoning(Some(0.0), Some(4.0), Some(3.0), REASONING, 0.0)]
    #[case::standard_reasoning(None, None, Some(3.0), REASONING, 3.0)]
    #[case::free_standard_reasoning(None, None, Some(0.0), REASONING, 0.0)]
    #[case::text_fallback(None, None, None, OUTPUT, 2.0)]
    fn reasoning_uses_declared_service_rates(
        when: When,
        #[values(
            ServiceTier::Priority,
            ServiceTier::Flex,
            ServiceTier::Balanced,
            ServiceTier::Ultrafast
        )]
        tier: ServiceTier,
        #[case] tier_reasoning: Option<f64>,
        #[case] tier_output: Option<f64>,
        #[case] standard_reasoning: Option<f64>,
        #[case] meter: Meter,
        #[case] value: f64,
    ) {
        let prices = Prices::new(
            [
                (OUTPUT, ServiceTier::Standard, Some(2.0)),
                (OUTPUT, tier, tier_output),
                (REASONING, tier, tier_reasoning),
                (REASONING, ServiceTier::Standard, standard_reasoning),
            ]
            .into_iter()
            .filter_map(|(meter, tier, value)| Some(rate(meter, tier, Band::Base, value?)))
            .collect(),
            None,
        );
        let result = calculate_with_policy(
            &prices,
            &usage(REASONING, 10, false),
            &When { tier, ..when },
            SelectionPolicy::SdkTokens(&sources(&prices)),
        );
        assert_eq!(result, expected(meter, REASONING, value));
    }

    #[rstest]
    #[case::below(99, false, None, None, 4.0, OUTPUT)]
    #[case::at_exclusive(100, false, None, None, 4.0, OUTPUT)]
    #[case::at_inclusive(100, true, None, None, 8.0, OUTPUT)]
    #[case::above(101, false, None, None, 8.0, OUTPUT)]
    #[case::offpeak_output(101, false, None, Some(6.0), 6.0, OUTPUT)]
    #[case::offpeak_reasoning(101, false, Some(0.0), Some(6.0), 0.0, REASONING)]
    fn reasoning_output_fallback_uses_selected_threshold_and_offpeak(
        when: When,
        #[case] prompt: u64,
        #[case] inclusive: bool,
        #[case] offpeak_reasoning: Option<f64>,
        #[case] offpeak_output: Option<f64>,
        #[case] value: f64,
        #[case] meter: Meter,
    ) {
        let prices = Prices::new(
            [
                rate(OUTPUT, ServiceTier::Standard, Band::Base, 2.0),
                rate(REASONING, ServiceTier::Standard, Band::Base, 3.0),
                rate(OUTPUT, ServiceTier::Priority, Band::Base, 4.0),
                rate(
                    OUTPUT,
                    ServiceTier::Priority,
                    Band::over_prompt_tokens(100),
                    8.0,
                ),
            ]
            .into(),
            overlay(offpeak_reasoning, offpeak_output),
        );
        let result = calculate_with_policy(
            &prices,
            &usage(REASONING, prompt, inclusive),
            &when,
            SelectionPolicy::SdkTokens(&sources(&prices)),
        );
        assert_eq!(result, expected(meter, REASONING, value));
        assert_eq!(
            prices.rate(
                &REASONING,
                ServiceTier::Priority,
                Band::over_prompt_tokens(100)
            ),
            UnitPrice::new(3.0)
        );
    }

    #[rstest]
    #[case::zero_prompt(0, (true, true), (None, None), REASONING, 5.0)]
    #[case::first(10, (true, true), (None, None), OUTPUT, 3.0)]
    #[case::boundary(100, (true, true), (None, None), OUTPUT, 3.0)]
    #[case::second(101, (true, true), (None, None), OUTPUT, 6.0)]
    #[case::above_last(1001, (true, true), (None, None), OUTPUT, 6.0)]
    #[case::no_input(101, (false, true), (None, None), REASONING, 5.0)]
    #[case::no_tier_output(101, (true, false), (None, None), REASONING, 5.0)]
    #[case::tier_reasoning(101, (true, true), (Some(9.0), None), REASONING, 9.0)]
    #[case::free_tier_reasoning(101, (true, true), (Some(0.0), None), REASONING, 0.0)]
    #[case::offpeak_wins(101, (true, true), (Some(9.0), Some(0.0)), REASONING, 0.0)]
    fn reasoning_uses_eligible_tier_before_global_rates(
        when: When,
        #[case] prompt: u64,
        #[case] declared: (bool, bool),
        #[case] reasoning: (Option<f64>, Option<f64>),
        #[case] meter: Meter,
        #[case] value: f64,
        #[values(false, true)] inclusive: bool,
    ) {
        let (input_declared, output_declared) = declared;
        let (tier_reasoning, offpeak_reasoning) = reasoning;
        let prices = Prices::new(
            [
                rate(REASONING, ServiceTier::Standard, Band::Base, 5.0),
                rate(OUTPUT, ServiceTier::Standard, Band::Base, 3.0),
                rate(
                    OUTPUT,
                    ServiceTier::Standard,
                    Band::over_prompt_tokens(100),
                    6.0,
                ),
            ]
            .into_iter()
            .chain(tier_reasoning.map(|value| {
                rate(
                    REASONING,
                    ServiceTier::Standard,
                    Band::over_prompt_tokens(100),
                    value,
                )
            }))
            .collect(),
            overlay(offpeak_reasoning, Some(1.0)),
        );
        let declared = [
            input_declared.then_some(Meter::TEXT_INPUT),
            output_declared.then_some(OUTPUT),
            tier_reasoning.map(|_| REASONING),
        ]
        .into_iter()
        .flatten()
        .collect();
        let sources = TokenPriceSources {
            flat: [Price::online(REASONING, PriceWhen::default())].into(),
            tiers: vec![
                TokenTier {
                    range: [100, 1000],
                    meters: declared,
                },
                TokenTier {
                    range: [0, 100],
                    meters: [Meter::TEXT_INPUT, OUTPUT].into(),
                },
            ],
        };
        let result = calculate_with_policy(
            &prices,
            &usage(REASONING, prompt, inclusive),
            &when,
            SelectionPolicy::SdkTokens(&sources),
        );
        assert_eq!(result, expected(meter, REASONING, value));
    }

    #[rstest]
    #[case::standard(Some(3.0), true, false, 3.0)]
    #[case::free(Some(0.0), true, false, 0.0)]
    #[case::priority_only(None, true, true, 8.0)]
    #[case::no_modality(None, false, true, 8.0)]
    fn output_modalities_use_only_flat_standard_rates(
        when: When,
        #[values(TokenKind::Audio, TokenKind::Image, TokenKind::Video)] kind: TokenKind,
        #[case] standard: Option<f64>,
        #[case] priority: bool,
        #[case] fallback: bool,
        #[case] value: f64,
    ) {
        let meter = Meter::Output(OutputUnit::Token(kind));
        let prices = Prices::new(
            [
                rate(OUTPUT, ServiceTier::Priority, Band::Base, 4.0),
                rate(
                    OUTPUT,
                    ServiceTier::Priority,
                    Band::over_prompt_tokens(100),
                    8.0,
                ),
            ]
            .into_iter()
            .chain(
                standard.map(|value| rate(meter.clone(), ServiceTier::Standard, Band::Base, value)),
            )
            .chain(priority.then(|| rate(meter.clone(), ServiceTier::Priority, Band::Base, 7.0)))
            .collect(),
            None,
        );
        let result = calculate_with_policy(
            &prices,
            &usage(meter.clone(), 101, false),
            &when,
            SelectionPolicy::SdkTokens(&sources(&prices)),
        );
        assert_eq!(
            result,
            expected(if fallback { OUTPUT } else { meter.clone() }, meter, value)
        );
    }

    #[rstest]
    #[case::input(Meter::TEXT_INPUT, Channel::Online)]
    #[case::input_audio(Meter::Input(InputUnit::Token(TokenKind::Audio)), Channel::Online)]
    #[case::output_text(OUTPUT, Channel::Online)]
    #[case::image(Meter::Output(OutputUnit::Image(None)), Channel::Online)]
    #[case::batch_reasoning(REASONING, Channel::Batch)]
    #[case::batch_audio(Meter::Output(OutputUnit::Token(TokenKind::Audio)), Channel::Batch)]
    #[case::decisions(REASONING, Channel::Decisions)]
    fn unrelated_meters_and_channels_use_the_generic_engine(
        when: When,
        #[case] meter: Meter,
        #[case] channel: Channel,
    ) {
        let prices = Prices::new(
            [
                rate(meter.clone(), ServiceTier::Standard, Band::Base, 3.0),
                rate(meter.clone(), ServiceTier::Priority, Band::Base, 7.0),
            ]
            .into(),
            None,
        );
        let usage = usage(meter, 101, false);
        let when = When { channel, ..when };
        assert_eq!(
            calculate_with_policy(
                &prices,
                &usage,
                &when,
                SelectionPolicy::SdkTokens(&sources(&prices))
            ),
            calculate(&prices, &usage, &when)
        );
    }

    #[rstest]
    fn unpriced_preserves_missing_meters_and_partial_amounts(when: When) {
        let prices = Prices::new(
            [rate(
                Meter::TEXT_INPUT,
                ServiceTier::Standard,
                Band::Base,
                2.0,
            )]
            .into(),
            None,
        );
        let usage = Usage {
            quantities: [(Meter::TEXT_INPUT, 5), (REASONING, 5), (OUTPUT, 0)].into(),
            ..Default::default()
        };
        assert_eq!(
            calculate_with_policy(
                &prices,
                &usage,
                &when,
                SelectionPolicy::SdkTokens(&sources(&prices))
            ),
            Priced::Unpriced {
                missing: vec![REASONING],
                partial: match expected(Meter::TEXT_INPUT, Meter::TEXT_INPUT, 2.0) {
                    Priced::Known(breakdown) => breakdown,
                    _ => unreachable!(),
                },
            }
        );
    }
}
