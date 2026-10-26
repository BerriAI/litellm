use std::collections::BTreeSet;
use std::num::NonZeroU64;

use crate::money::UnitPrice;
use crate::price::{
    Band, Channel, Feature, InputUnit, Meter, OutputUnit, Price, Prices, TokenKind,
    When as PriceWhen,
};
use crate::{PriceEvaluationError, SelectionPolicy, TokenPriceSources, Usage, When};

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum OtherSelection {
    #[default]
    Additive,
    PromptUnits,
    FirstUnit(Vec<Meter>),
    SearchTiers,
    Ocr,
    DirectionalCharacters {
        threshold: NonZeroU64,
        comparison_scale: NonZeroU64,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::EnumString, strum::VariantArray)]
pub enum DefaultEligibility {
    #[strum(serialize = "absent")]
    Absent,
    #[strum(serialize = "absent_or_null")]
    AbsentOrNull,
}

#[derive(
    Clone, Copy, Debug, PartialEq, Eq, strum::EnumString, strum::VariantArray, strum::IntoStaticStr,
)]
pub enum ReportTo {
    #[strum(serialize = "input")]
    Input,
    #[strum(serialize = "output")]
    Output,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct OtherPriceSource {
    pub id: String,
    pub declared: BTreeSet<Meter>,
    pub search_tiers: Vec<SearchTier>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SearchTier {
    pub range: [u64; 2],
    pub meter: Meter,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct OtherContext {
    pub selection: OtherSelection,
    pub sources: Vec<OtherPriceSource>,
    pub defaults: Vec<Meter>,
    pub report_to: Option<ReportTo>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedCharge {
    pub meter: Meter,
    pub price: UnitPrice,
    pub source_id: Option<String>,
    pub caller_default: bool,
    pub quantity: Option<crate::Quantity>,
}

impl From<(Meter, UnitPrice)> for ResolvedCharge {
    fn from((meter, price): (Meter, UnitPrice)) -> Self {
        Self {
            meter,
            price,
            source_id: None,
            caller_default: false,
            quantity: None,
        }
    }
}

impl OtherContext {
    pub fn insert_default(
        &mut self,
        prices: &mut Prices,
        meter: Meter,
        price: UnitPrice,
        per_units: NonZeroU64,
        eligibility: DefaultEligibility,
    ) {
        let known = self.flat(prices, &meter, Channel::Online).is_some();
        let declared = self
            .sources
            .iter()
            .any(|source| source.declared.contains(&meter));
        if known || (eligibility == DefaultEligibility::Absent && declared) {
            return;
        }
        let normalized = UnitPrice::new(price.get() / per_units.get() as f64)
            .expect("finite nonnegative price divided by positive units");
        prices.insert_default(self.defaults.len(), meter.clone(), normalized);
        self.defaults.push(meter);
    }

    pub fn validate_usage(&self, usage: &Usage) -> Result<(), PriceEvaluationError> {
        if let OtherSelection::DirectionalCharacters {
            threshold,
            comparison_scale,
        } = self.selection
        {
            for meter in [
                Meter::Input(InputUnit::Character),
                Meter::Output(OutputUnit::Character),
            ] {
                if let Some(quantity) = usage.quantities.get(&meter) {
                    quantity.exceeds_scaled(threshold.get(), comparison_scale.get())?;
                }
            }
            token_quantity(usage, false)?;
            token_quantity(usage, true)?;
        }
        Ok(())
    }

    fn row_rate(
        &self,
        prices: &Prices,
        row: usize,
        meter: &Meter,
        channel: Channel,
        band: Band,
    ) -> Option<ResolvedCharge> {
        let price = prices.source_flat(
            row,
            &Price {
                meter: meter.clone(),
                channel,
                when: PriceWhen {
                    band,
                    ..PriceWhen::default()
                },
            },
        )?;
        Some(ResolvedCharge {
            meter: meter.clone(),
            price,
            source_id: Some(self.sources[row].id.clone()),
            caller_default: false,
            quantity: None,
        })
    }

    fn default(&self, prices: &Prices, meter: &Meter) -> Option<ResolvedCharge> {
        self.defaults
            .iter()
            .enumerate()
            .find_map(|(index, candidate)| {
                (candidate == meter)
                    .then(|| prices.default_rate(index, meter))
                    .flatten()
                    .map(|price| ResolvedCharge {
                        meter: meter.clone(),
                        price,
                        source_id: None,
                        caller_default: true,
                        quantity: None,
                    })
            })
    }

    fn flat(&self, prices: &Prices, meter: &Meter, channel: Channel) -> Option<ResolvedCharge> {
        (0..self.sources.len())
            .find_map(|row| self.row_rate(prices, row, meter, channel, Band::Base))
            .or_else(|| self.default(prices, meter))
    }

    fn first(&self, prices: &Prices, meters: &[Meter], usage: &Usage) -> Option<ResolvedCharge> {
        (0..self.sources.len())
            .find_map(|row| {
                meters
                    .iter()
                    .filter(|meter| used(usage, meter))
                    .find_map(|meter| {
                        self.row_rate(prices, row, meter, Channel::Online, Band::Base)
                    })
            })
            .or_else(|| {
                meters
                    .iter()
                    .filter(|meter| used(usage, meter))
                    .find_map(|meter| self.default(prices, meter))
            })
    }

    fn ocr_family(
        &self,
        prices: &Prices,
        meter: &Meter,
        channel: Channel,
    ) -> Option<ResolvedCharge> {
        (0..self.sources.len())
            .find_map(|row| {
                self.row_rate(prices, row, meter, channel, Band::Base)
                    .or_else(|| {
                        (channel == Channel::Batch)
                            .then(|| self.row_rate(prices, row, meter, Channel::Online, Band::Base))
                            .flatten()
                    })
            })
            .or_else(|| self.default(prices, meter))
    }

    fn character_rate(
        &self,
        prices: &Prices,
        meter: &Meter,
        usage: &Usage,
    ) -> Option<ResolvedCharge> {
        let OtherSelection::DirectionalCharacters {
            threshold,
            comparison_scale,
        } = self.selection
        else {
            return None;
        };
        let quantity = usage.quantities.get(meter)?;
        let over = quantity
            .exceeds_scaled(threshold.get(), comparison_scale.get())
            .ok()?;
        let band = if over {
            Band::over_prompt_tokens(threshold.get())
        } else {
            Band::Base
        };
        (0..self.sources.len())
            .find_map(|row| self.row_rate(prices, row, meter, Channel::Online, band))
    }

    fn directional_token_threshold(&self, prices: &Prices) -> Option<NonZeroU64> {
        let OtherSelection::DirectionalCharacters { threshold, .. } = self.selection else {
            return None;
        };
        [
            Meter::TEXT_INPUT,
            Meter::Output(OutputUnit::Token(TokenKind::Text)),
        ]
        .into_iter()
        .any(|meter| {
            prices
                .get(&Price::online(
                    meter,
                    PriceWhen {
                        band: Band::OverPromptTokens(threshold),
                        ..Default::default()
                    },
                ))
                .is_some()
        })
        .then_some(threshold)
    }

    pub(crate) fn charges(
        &self,
        prices: &Prices,
        meter: &Meter,
        usage: &Usage,
        when: &When,
    ) -> bool {
        match &self.selection {
            OtherSelection::FirstUnit(meters) if meters.contains(meter) => self
                .first(prices, meters, usage)
                .is_none_or(|rate| rate.meter == *meter),
            OtherSelection::PromptUnits => token_alternative(meter).is_none_or(|alternative| {
                !used(usage, &alternative)
                    || self.flat(prices, &alternative, Channel::Online).is_none()
            }),
            OtherSelection::Ocr => {
                let credit = Meter::Feature(Feature::OcrCredit);
                let credits = used(usage, &credit)
                    && self.ocr_family(prices, &credit, when.channel).is_some();
                match meter {
                    Meter::Feature(Feature::OcrPage | Feature::AnnotationPage) => !credits,
                    Meter::Feature(Feature::OcrCredit) => {
                        credits || !used(usage, &Meter::Feature(Feature::OcrPage))
                    }
                    _ => true,
                }
            }
            OtherSelection::DirectionalCharacters { .. } => {
                if let Some(character) = direction_character(meter) {
                    if self.character_rate(prices, &character, usage).is_some() {
                        return false;
                    }
                    if self.directional_token_threshold(prices).is_some() {
                        return usage
                            .quantities
                            .iter()
                            .find(|(candidate, quantity)| {
                                !quantity.is_zero()
                                    && direction_character(candidate) == Some(character.clone())
                            })
                            .is_some_and(|(candidate, _)| candidate == meter);
                    }
                    return true;
                }
                if matches!(
                    meter,
                    Meter::Input(InputUnit::Character) | Meter::Output(OutputUnit::Character)
                ) {
                    return self.character_rate(prices, meter, usage).is_some()
                        || !usage
                            .quantities
                            .keys()
                            .any(|token| direction_character(token).as_ref() == Some(meter));
                }
                true
            }
            _ => true,
        }
    }

    pub(crate) fn resolve(
        &self,
        prices: &Prices,
        meter: &Meter,
        band: Band,
        usage: &Usage,
        when: &When,
        tokens: &TokenPriceSources,
    ) -> Option<ResolvedCharge> {
        if let OtherSelection::FirstUnit(meters) = &self.selection
            && meters.contains(meter)
        {
            return self
                .first(prices, meters, usage)
                .filter(|charge| !meter.is_token() || charge.meter.is_token());
        }
        if meter.is_token() {
            if let Some(threshold) = self.directional_token_threshold(prices) {
                let output = matches!(meter, Meter::Output(_));
                let quantity = token_quantity(usage, output).ok()?;
                let source = if output {
                    Meter::Output(OutputUnit::Token(TokenKind::Text))
                } else {
                    Meter::TEXT_INPUT
                };
                return crate::selection::directional_rate(prices, source, quantity, threshold)
                    .map(|rate| ResolvedCharge {
                        source_id: self.sources.first().map(|source| source.id.clone()),
                        quantity: Some(quantity),
                        ..ResolvedCharge::from(rate)
                    });
            }
            return SelectionPolicy::SdkTokens(tokens)
                .resolve(prices, meter, band, usage, when)
                .map(|rate| {
                    let mut charge = ResolvedCharge::from(rate);
                    charge.source_id = self.sources.first().map(|source| source.id.clone());
                    charge
                })
                .or_else(|| self.flat(prices, meter, when.channel));
        }
        match &self.selection {
            OtherSelection::SearchTiers
                if matches!(meter, Meter::Input(InputUnit::Query { .. })) =>
            {
                self.sources
                    .iter()
                    .enumerate()
                    .find_map(|(row, source)| {
                        let tier = source
                            .search_tiers
                            .iter()
                            .find(|tier| {
                                (tier.range[0]..=tier.range[1]).contains(&usage.search_results)
                            })
                            .or_else(|| source.search_tiers.last());
                        match tier {
                            Some(tier) => {
                                self.row_rate(prices, row, &tier.meter, Channel::Online, Band::Base)
                            }
                            None => self.row_rate(prices, row, meter, Channel::Online, Band::Base),
                        }
                    })
                    .or_else(|| self.default(prices, meter))
            }
            OtherSelection::Ocr if matches!(meter, Meter::Feature(Feature::AnnotationPage)) => {
                self.ocr_family(prices, meter, when.channel).or_else(|| {
                    self.ocr_family(prices, &Meter::Feature(Feature::OcrPage), when.channel)
                })
            }
            OtherSelection::Ocr
                if matches!(meter, Meter::Feature(Feature::OcrPage | Feature::OcrCredit)) =>
            {
                self.ocr_family(prices, meter, when.channel)
            }
            OtherSelection::DirectionalCharacters { .. }
                if matches!(
                    meter,
                    Meter::Input(InputUnit::Character) | Meter::Output(OutputUnit::Character)
                ) =>
            {
                self.character_rate(prices, meter, usage)
            }
            _ => SelectionPolicy::SdkTokens(tokens)
                .resolve(prices, meter, band, usage, when)
                .map(|rate| ResolvedCharge {
                    source_id: self.sources.first().map(|source| source.id.clone()),
                    ..ResolvedCharge::from(rate)
                })
                .or_else(|| self.flat(prices, meter, when.channel)),
        }
    }
}

fn token_quantity(usage: &Usage, output: bool) -> Result<crate::Quantity, PriceEvaluationError> {
    if output && let Some(completion) = usage.completion_tokens {
        return Ok(completion.into());
    }
    if !output && usage.prompt_tokens > 0 {
        return Ok(usage.prompt_tokens.into());
    }
    let text = if output {
        Meter::Output(OutputUnit::Token(TokenKind::Text))
    } else {
        Meter::TEXT_INPUT
    };
    if usage.quantities.iter().any(|(meter, quantity)| {
        meter.is_token()
            && matches!(meter, Meter::Output(_)) == output
            && *meter != text
            && !quantity.is_zero()
    }) {
        return Err(PriceEvaluationError::InvalidQuantity);
    }
    match usage.quantities.get(&text).copied().unwrap_or(0.into()) {
        crate::Quantity::Count(count) => Ok(count.into()),
        crate::Quantity::Measured(_) => Err(PriceEvaluationError::InvalidQuantity),
    }
}

fn used(usage: &Usage, meter: &Meter) -> bool {
    usage
        .quantities
        .get(meter)
        .is_some_and(|quantity| !quantity.is_zero())
}

fn token_alternative(meter: &Meter) -> Option<Meter> {
    match meter {
        Meter::Input(InputUnit::Token(TokenKind::Text)) => Some(Meter::Input(InputUnit::Character)),
        Meter::Input(InputUnit::Token(TokenKind::Audio)) => {
            Some(Meter::Input(InputUnit::AudioSecond))
        }
        Meter::Input(InputUnit::Token(TokenKind::Image)) => Some(Meter::Input(InputUnit::Image)),
        Meter::Input(InputUnit::Token(TokenKind::Video)) => {
            Some(Meter::Input(InputUnit::VideoSecond { over_seconds: None }))
        }
        _ => None,
    }
}

fn direction_character(meter: &Meter) -> Option<Meter> {
    match meter {
        Meter::Input(InputUnit::Token(_)) | Meter::CacheRead { .. } | Meter::CacheWrite { .. } => {
            Some(Meter::Input(InputUnit::Character))
        }
        Meter::Output(OutputUnit::Token(_) | OutputUnit::ReasoningToken) => {
            Some(Meter::Output(OutputUnit::Character))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    const INPUT: Meter = Meter::TEXT_INPUT;
    use rstest::{fixture, rstest};
    use time::OffsetDateTime;

    use super::*;
    use crate::money::ScaleFactor;
    use crate::price::{CacheReadMode, ServiceTier};
    use crate::{Measurement, Priced, Quantity, QuantityUnit, calculate_with_rates};

    const INPUT_CHAR: Meter = Meter::Input(InputUnit::Character);
    const OUTPUT_CHAR: Meter = Meter::Output(OutputUnit::Character);
    const OUTPUT: Meter = Meter::Output(OutputUnit::Token(TokenKind::Text));
    const PAGE: Meter = Meter::Feature(Feature::OcrPage);
    const ANNOTATION: Meter = Meter::Feature(Feature::AnnotationPage);
    const FILE: Meter = Meter::Feature(Feature::FileSearchCall);

    #[fixture]
    fn when() -> When {
        When {
            tier: ServiceTier::Standard,
            channel: Channel::Online,
            at: OffsetDateTime::UNIX_EPOCH,
        }
    }

    #[fixture]
    fn context() -> OtherContext {
        OtherContext {
            sources: vec![OtherPriceSource {
                id: "primary".to_owned(),
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    fn prices(rates: impl IntoIterator<Item = (Meter, Channel, f64)>) -> Prices {
        Prices::new(
            rates
                .into_iter()
                .map(|(meter, channel, rate)| {
                    (
                        Price {
                            meter,
                            channel,
                            when: Default::default(),
                        },
                        UnitPrice::new(rate).unwrap(),
                    )
                })
                .collect(),
            None,
        )
    }

    fn evaluated(
        prices: &Prices,
        context: &OtherContext,
        usage: &Usage,
        when: &When,
        factor: f64,
    ) -> crate::PriceEvaluation {
        calculate_with_rates(
            prices,
            usage,
            when,
            SelectionPolicy::SdkOther {
                tokens: &TokenPriceSources::default(),
                context,
            },
            &[],
            &[ScaleFactor::new(factor).unwrap()],
        )
        .unwrap()
    }

    #[rstest]
    #[case::at_threshold(32000, 4.0)]
    #[case::above_threshold(32001, 6.0)]
    fn directional_character_output_bills_actual_characters(
        mut context: OtherContext,
        when: When,
        #[case] characters: u64,
        #[case] rate: f64,
        #[values(1.0, 1.1)] factor: f64,
    ) {
        context.selection = OtherSelection::DirectionalCharacters {
            threshold: NonZeroU64::new(128000).unwrap(),
            comparison_scale: NonZeroU64::new(4).unwrap(),
        };
        let prices = Prices::new(
            [
                (
                    Price::online(INPUT_CHAR, Default::default()),
                    UnitPrice::new(3.0).unwrap(),
                ),
                (
                    Price::online(OUTPUT_CHAR, Default::default()),
                    UnitPrice::new(4.0).unwrap(),
                ),
                (
                    Price::online(
                        OUTPUT_CHAR,
                        PriceWhen {
                            band: Band::over_prompt_tokens(128000),
                            ..Default::default()
                        },
                    ),
                    UnitPrice::new(6.0).unwrap(),
                ),
                (
                    Price::online(OUTPUT, Default::default()),
                    UnitPrice::new(2.0).unwrap(),
                ),
            ]
            .into(),
            None,
        );
        let usage = Usage {
            quantities: [
                (INPUT_CHAR, Quantity::Count(10)),
                (OUTPUT_CHAR, characters.into()),
                (OUTPUT, 2.into()),
            ]
            .into(),
            ..Default::default()
        };
        let Priced::Known(result) = evaluated(&prices, &context, &usage, &when, factor).pricing
        else {
            panic!("priced characters")
        };
        assert_eq!(result.lines.len(), 2);
        assert_eq!(result.lines[1].quantity, Quantity::Count(characters));
        assert_eq!(result.output().get(), characters as f64 * rate * factor);
        assert_eq!(result.input().get(), 30.0 * factor);
    }

    #[rstest]
    fn directional_token_fallback_uses_aggregate_counts_once(
        mut context: OtherContext,
        mut when: When,
    ) {
        context.selection = OtherSelection::DirectionalCharacters {
            threshold: NonZeroU64::new(128000).unwrap(),
            comparison_scale: NonZeroU64::new(4).unwrap(),
        };
        when.tier = ServiceTier::Priority;
        let prices = Prices::new(
            [
                (
                    Price::online(Meter::TEXT_INPUT, Default::default()),
                    UnitPrice::new(1.0).unwrap(),
                ),
                (
                    Price::online(OUTPUT, Default::default()),
                    UnitPrice::new(2.0).unwrap(),
                ),
                (
                    Price::online(
                        OUTPUT,
                        PriceWhen {
                            band: Band::over_prompt_tokens(128000),
                            ..Default::default()
                        },
                    ),
                    UnitPrice::new(4.0).unwrap(),
                ),
                (
                    Price::online(
                        Meter::TEXT_INPUT,
                        PriceWhen {
                            tier: ServiceTier::Priority,
                            ..Default::default()
                        },
                    ),
                    UnitPrice::new(10.0).unwrap(),
                ),
            ]
            .into(),
            None,
        );
        let read = Meter::CacheRead {
            kind: TokenKind::Text,
            mode: CacheReadMode::Implicit,
        };
        let usage = Usage {
            quantities: [
                (Meter::TEXT_INPUT, 20.into()),
                (read, 80.into()),
                (OUTPUT, 1.into()),
                (Meter::Output(OutputUnit::ReasoningToken), 200000.into()),
                (OUTPUT_CHAR, 40000.into()),
            ]
            .into(),
            prompt_tokens: 100,
            completion_tokens: Some(128001),
            ..Default::default()
        };
        let Priced::Known(result) = evaluated(&prices, &context, &usage, &when, 1.0).pricing else {
            panic!("priced aggregate fallback")
        };
        assert_eq!(result.lines.len(), 2);
        assert_eq!(result.input().get(), 100.0);
        assert_eq!(result.output().get(), 512004.0);
        assert_eq!(result.lines[1].quantity, Quantity::Count(128001));
    }

    #[rstest]
    #[case::known(Some(3.0), 12.0, true)]
    #[case::free(Some(0.0), 0.0, true)]
    #[case::unknown(None, 10.0, false)]
    fn prompt_units_suppress_only_known_alternatives(
        mut context: OtherContext,
        when: When,
        #[case] rate: Option<f64>,
        #[case] input: f64,
        #[case] known: bool,
    ) {
        context.selection = OtherSelection::PromptUnits;
        let prices = prices(
            [
                (Meter::TEXT_INPUT, Channel::Online, 1.0),
                (OUTPUT, Channel::Online, 2.0),
            ]
            .into_iter()
            .chain(rate.map(|rate| (INPUT_CHAR, Channel::Online, rate))),
        );
        let usage = Usage {
            quantities: [
                (Meter::TEXT_INPUT, 10.into()),
                (INPUT_CHAR, 4.into()),
                (OUTPUT, 2.into()),
            ]
            .into(),
            ..Default::default()
        };
        let result = evaluated(&prices, &context, &usage, &when, 1.0).pricing;
        assert_eq!(matches!(result, Priced::Known(_)), known);
        let result = match result {
            Priced::Known(result)
            | Priced::Unpriced {
                partial: result, ..
            } => result,
        };
        assert_eq!(result.input().get(), input);
        assert_eq!(result.output().get(), 4.0);
    }

    #[rstest]
    #[case::source_before_unit(8.0, 26.0)]
    #[case::zero_wins(0.0, 10.0)]
    fn first_unit_keeps_row_precedence_and_genuine_tokens(
        mut context: OtherContext,
        when: When,
        #[case] rate: f64,
        #[case] total: f64,
    ) {
        let image = Meter::Input(InputUnit::Image);
        let output_image = Meter::Output(OutputUnit::Image(None));
        context.selection = OtherSelection::FirstUnit(vec![image.clone(), output_image.clone()]);
        context.sources.push(OtherPriceSource {
            id: "published".to_owned(),
            ..Default::default()
        });
        let mut prices = prices([
            (output_image.clone(), Channel::Online, rate),
            (Meter::TEXT_INPUT, Channel::Online, 1.0),
        ]);
        prices.append_source(1, self::prices([(image.clone(), Channel::Online, 2.0)]));
        let usage = Usage {
            quantities: [
                (image, 1.into()),
                (output_image, 2.into()),
                (Meter::TEXT_INPUT, 10.into()),
            ]
            .into(),
            ..Default::default()
        };
        let Priced::Known(result) = evaluated(&prices, &context, &usage, &when, 1.0).pricing else {
            panic!("priced alternatives")
        };
        assert_eq!(result.total.get(), total);
        assert_eq!(result.lines.len(), 2);
        assert_eq!(result.lines[1].source_id.as_deref(), Some("primary"));
    }

    #[rstest]
    #[case::annotation_family(Some(10.0), 16.0, "published")]
    #[case::annotation_zero(Some(0.0), 6.0, "published")]
    #[case::page_fallback(None, 9.0, "primary")]
    fn ocr_resolves_annotation_family_before_page_fallback(
        mut context: OtherContext,
        mut when: When,
        #[case] annotation: Option<f64>,
        #[case] total: f64,
        #[case] source: &str,
    ) {
        context.selection = OtherSelection::Ocr;
        context.sources.push(OtherPriceSource {
            id: "published".to_owned(),
            ..Default::default()
        });
        when.channel = Channel::Batch;
        let mut prices = prices([(PAGE, Channel::Batch, 3.0), (PAGE, Channel::Online, 2.0)]);
        prices.append_source(
            1,
            self::prices(annotation.map(|rate| (ANNOTATION, Channel::Online, rate))),
        );
        let usage = Usage {
            quantities: [(PAGE, 2.into()), (ANNOTATION, 1.into())].into(),
            ..Default::default()
        };
        let Priced::Known(result) = evaluated(&prices, &context, &usage, &when, 1.0).pricing else {
            panic!("priced OCR")
        };
        assert_eq!(result.total.get(), total);
        assert_eq!(result.lines[0].source_id.as_deref(), Some(source));
    }

    #[rstest]
    #[case::absent(DefaultEligibility::Absent, false, true)]
    #[case::null_stays_unknown(DefaultEligibility::Absent, true, false)]
    #[case::null_eligible(DefaultEligibility::AbsentOrNull, true, true)]
    fn defaults_normalize_occurrences_without_erasing_unknown(
        mut context: OtherContext,
        when: When,
        #[case] eligibility: DefaultEligibility,
        #[case] declared: bool,
        #[case] known: bool,
        #[values(0.0, 1.0)] factor: f64,
    ) {
        if declared {
            context.sources[0].declared.insert(FILE);
        }
        let mut prices = Prices::default();
        context.insert_default(
            &mut prices,
            FILE,
            UnitPrice::new(2.5).unwrap(),
            NonZeroU64::new(1000).unwrap(),
            eligibility,
        );
        let usage = Usage {
            quantities: [(FILE, 3.into())].into(),
            ..Default::default()
        };
        let result = evaluated(&prices, &context, &usage, &when, factor).pricing;
        assert_eq!(matches!(result, Priced::Known(_)), known);
        let result = match result {
            Priced::Known(result)
            | Priced::Unpriced {
                partial: result, ..
            } => result,
        };
        assert_eq!(
            result.total.get(),
            if known { 0.0075 * factor } else { 0.0 }
        );
        assert_eq!(
            result.lines.first().map(|line| line.caller_default),
            known.then_some(true)
        );
    }

    #[rstest]
    #[case::normalized_seconds(1500.0, QuantityUnit::Milliseconds)]
    #[case::native_seconds(1.5, QuantityUnit::Meter)]
    fn measured_time_uses_native_units(
        context: OtherContext,
        when: When,
        #[case] value: f64,
        #[case] unit: QuantityUnit,
    ) {
        let meter = Meter::WallClockSecond;
        let prices = prices([(meter.clone(), Channel::Online, 0.0014)]);
        let usage = Usage {
            quantities: [(
                meter.clone(),
                Quantity::measured(Measurement::new(value).unwrap(), unit, &meter).unwrap(),
            )]
            .into(),
            ..Default::default()
        };
        let Priced::Known(result) = evaluated(&prices, &context, &usage, &when, 1.0).pricing else {
            panic!("priced time")
        };
        assert_eq!(result.total.get(), 0.0021);
        assert_eq!(
            result.lines[0].quantity,
            Quantity::Measured(Measurement::new(1.5).unwrap())
        );
    }

    #[rstest]
    #[case::below_first(0, 2.0)]
    #[case::first_start(1, 1.0)]
    #[case::first_end(25, 1.0)]
    #[case::second_start(26, 2.0)]
    #[case::last_end(100, 2.0)]
    #[case::above_last(101, 2.0)]
    fn search_tiers_use_inclusive_ranges_and_last_outside(
        mut context: OtherContext,
        when: When,
        #[case] results: u64,
        #[case] expected: f64,
    ) {
        let query = Meter::Input(InputUnit::Query { over_results: None });
        let above = Meter::Input(InputUnit::Query {
            over_results: NonZeroU64::new(25),
        });
        context.selection = OtherSelection::SearchTiers;
        context.sources[0].search_tiers = vec![
            SearchTier {
                range: [1, 25],
                meter: query.clone(),
            },
            SearchTier {
                range: [26, 100],
                meter: above.clone(),
            },
        ];
        let prices = prices([
            (query.clone(), Channel::Online, 1.0),
            (above, Channel::Online, 2.0),
        ]);
        let usage = Usage {
            quantities: [(query, 3.into())].into(),
            search_results: results,
            ..Default::default()
        };
        let Priced::Known(result) = evaluated(&prices, &context, &usage, &when, 1.0).pricing else {
            panic!("priced search")
        };
        assert_eq!(result.total.get(), expected * 3.0);
    }

    #[rstest]
    fn chosen_second_units_do_not_become_projected_token_prices(
        mut context: OtherContext,
        when: When,
    ) {
        let seconds = Meter::Input(InputUnit::Second);
        context.selection = OtherSelection::FirstUnit(vec![seconds.clone(), INPUT]);
        let prices = prices([
            (seconds.clone(), Channel::Online, 2.0),
            (INPUT, Channel::Online, 1.0),
        ]);
        let usage = Usage {
            quantities: [(seconds, 3.into()), (INPUT, 10.into())].into(),
            ..Default::default()
        };
        let result = calculate_with_rates(
            &prices,
            &usage,
            &when,
            SelectionPolicy::SdkOther {
                tokens: &TokenPriceSources::default(),
                context: &context,
            },
            &[INPUT],
            &[],
        )
        .unwrap();
        assert!(result.rates[0].resolved.is_none());
        let Priced::Known(pricing) = result.pricing else {
            panic!("priced seconds")
        };
        assert_eq!(pricing.total.get(), 6.0);
        assert_eq!(pricing.lines.len(), 1);
    }

    #[rstest]
    fn additive_token_fallbacks_share_actual_and_projection_source(
        mut context: OtherContext,
        when: When,
    ) {
        let mut primary = Prices::default();
        primary.append_source(1, prices([(INPUT, Channel::Online, 2.0)]));
        context.sources.push(OtherPriceSource {
            id: "published".to_owned(),
            ..Default::default()
        });
        let usage = Usage {
            quantities: [(INPUT, 10.into())].into(),
            ..Default::default()
        };
        let result = calculate_with_rates(
            &primary,
            &usage,
            &when,
            SelectionPolicy::SdkOther {
                tokens: &TokenPriceSources::default(),
                context: &context,
            },
            &[INPUT],
            &[ScaleFactor::new(1.1).unwrap()],
        )
        .unwrap();
        assert_eq!(result.rates[0].source_id.as_deref(), Some("published"));
        assert_eq!(result.rates[0].resolved.as_ref().unwrap().1.get(), 2.2);
        let Priced::Known(pricing) = result.pricing else {
            panic!("priced tokens")
        };
        assert_eq!(pricing.total.get(), 22.0);
        assert_eq!(pricing.lines[0].source_id, result.rates[0].source_id);
    }

    #[rstest]
    #[case::missing_aggregate(None, None)]
    #[case::provided_aggregate(Some(10), Some(10))]
    #[case::zero_aggregate(Some(0), Some(0))]
    fn directional_counts_never_infer_overlapping_completion_details(
        #[case] completion_tokens: Option<u64>,
        #[case] expected: Option<u64>,
    ) {
        let usage = Usage {
            quantities: [
                (OUTPUT, 10.into()),
                (Meter::Output(OutputUnit::ReasoningToken), 5.into()),
            ]
            .into(),
            completion_tokens,
            ..Default::default()
        };
        assert_eq!(token_quantity(&usage, true).ok(), expected.map(Into::into));
    }
}
