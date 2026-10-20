use std::collections::BTreeMap;

use serde::de::Error as _;

use crate::error::Problem;
use crate::pricing::{
    Multiplier, OffPeakPricing, PricingFields, SearchContextCostPerQuery, TieredRate, Usd,
    UtcWindow, Weekday,
};
use crate::rate::{Charge, Rate, Threshold, Tier};

/// What a price counts.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Meter {
    Charge(Charge),
    WebSearch(SearchContextSize),
    Guardrail(Box<str>),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SearchContextSize {
    Low,
    Medium,
    High,
}

/// The request-level quantity a band is measured on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Basis {
    PromptTokens,
    Seconds,
    Results,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Floor {
    Over(u64),
    AtLeast(u64),
}

/// A price that applies to the whole request once its quantity reaches `floor`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Band {
    pub basis: Basis,
    pub floor: Floor,
}

impl From<Threshold> for Band {
    fn from(threshold: Threshold) -> Self {
        match threshold {
            Threshold::KiloTokens(count) => Self {
                basis: Basis::PromptTokens,
                floor: Floor::Over(u64::from(count) * 1000),
            },
            Threshold::Seconds(count) => Self {
                basis: Basis::Seconds,
                floor: Floor::Over(u64::from(count)),
            },
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Clock {
    Peak,
    OffPeak,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Region {
    RegionalEndpoint,
    Eu,
    Us,
}

/// The meter a price counts and the circumstances it applies under. `tier: None` is the
/// standard tier and `band: None` is the lowest band.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PriceKey {
    pub meter: Meter,
    pub tier: Option<Tier>,
    pub band: Option<Band>,
    pub clock: Clock,
}

impl PriceKey {
    fn from_rate(rate: &Rate, clock: Clock) -> Self {
        Self {
            meter: Meter::Charge(rate.charge.clone()),
            tier: rate.tier,
            band: rate.above.map(Band::from),
            clock,
        }
    }
}

/// UTC hours, limited to `weekdays` when it is set.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OffPeakHours {
    pub hours: Vec<UtcWindow>,
    pub weekdays: Option<Vec<Weekday>>,
}

/// Off-peak prices apply inside any of `hours`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OffPeakSchedule {
    pub hours: Vec<OffPeakHours>,
    pub weekday_timezone: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Prices {
    rates: BTreeMap<PriceKey, Usd>,
    off_peak: Option<OffPeakSchedule>,
    uplifts: BTreeMap<Region, Multiplier>,
}

impl Prices {
    pub fn get(&self, key: &PriceKey) -> Option<Usd> {
        self.rates.get(key).copied()
    }
    pub fn iter(&self) -> impl Iterator<Item = (&PriceKey, Usd)> {
        self.rates.iter().map(|(key, usd)| (key, *usd))
    }
    pub fn off_peak(&self) -> Option<&OffPeakSchedule> {
        self.off_peak.as_ref()
    }
    pub fn uplift(&self, region: Region) -> Option<Multiplier> {
        self.uplifts.get(&region).copied()
    }

    /// Every price an entry writes, from flat rate keys and the nested pricing fields, under one
    /// key space so two spellings of the same price are caught.
    pub(crate) fn lower(
        model: &str,
        flat: Vec<(Rate, Usd)>,
        fields: &PricingFields,
    ) -> Result<Self, Vec<Problem>> {
        let off_peak = fields
            .off_peak_pricing
            .as_ref()
            .map(|pricing| off_peak(model, pricing))
            .transpose();
        let lowered = flat
            .into_iter()
            .map(|(rate, usd)| {
                Ok(Priced {
                    source: rate.to_string(),
                    key: PriceKey::from_rate(&rate, Clock::Peak),
                    usd,
                })
            })
            .chain(
                fields
                    .tiered_pricing
                    .iter()
                    .flatten()
                    .enumerate()
                    .flat_map(|(index, tier)| tiered(model, index, tier)),
            )
            .chain(
                fields
                    .off_peak_pricing
                    .iter()
                    .flat_map(|pricing| off_peak_rates(pricing).map(Ok)),
            )
            .chain(
                fields
                    .search_context_cost_per_query
                    .iter()
                    .flat_map(search_context)
                    .map(Ok),
            )
            .chain(
                fields
                    .guardrail_cost_per_unit
                    .iter()
                    .flatten()
                    .map(|(unit, usd)| {
                        Ok(Priced {
                            key: PriceKey {
                                meter: Meter::Guardrail(unit.as_str().into()),
                                tier: None,
                                band: None,
                                clock: Clock::Peak,
                            },
                            usd: *usd,
                            source: format!("guardrail_cost_per_unit.{unit}"),
                        })
                    }),
            );

        let mut problems = Vec::new();
        let mut rates: BTreeMap<PriceKey, (Usd, String)> = BTreeMap::new();
        for priced in lowered {
            match priced {
                Err(problem) => problems.push(problem),
                Ok(Priced { key, usd, source }) => match rates.get(&key) {
                    Some((_, first)) => problems.push(Problem::DuplicatePrice {
                        model: model.to_owned(),
                        first: first.clone(),
                        second: source,
                    }),
                    None => {
                        rates.insert(key, (usd, source));
                    }
                },
            }
        }
        let off_peak = off_peak
            .map_err(|problem| problems.push(problem))
            .ok()
            .flatten();
        if !problems.is_empty() {
            return Err(problems);
        }
        Ok(Self {
            rates: rates
                .into_iter()
                .map(|(key, (usd, _))| (key, usd))
                .collect(),
            off_peak,
            uplifts: [
                (
                    Region::RegionalEndpoint,
                    fields.regional_endpoint_uplift_multiplier,
                ),
                (Region::Eu, fields.regional_processing_uplift_multiplier_eu),
                (Region::Us, fields.regional_processing_uplift_multiplier_us),
            ]
            .into_iter()
            .filter_map(|(region, factor)| Some((region, factor?)))
            .collect(),
        })
    }
}

struct Priced {
    key: PriceKey,
    usd: Usd,
    source: String,
}

fn invalid(model: &str, key: String, message: &str) -> Problem {
    Problem::InvalidField {
        model: model.to_owned(),
        key,
        error: serde_json::Error::custom(message),
    }
}

fn tiered(model: &str, index: usize, tier: &TieredRate) -> Vec<Result<Priced, Problem>> {
    let path = format!("tiered_pricing[{index}]");
    let band = match tier_band(tier) {
        Ok(band) => band,
        Err(message) => return vec![Err(invalid(model, path, message))],
    };
    tier.rates
        .iter()
        .map(|(rate, usd)| {
            let source = format!("{path}.{rate}");
            if rate.above.is_some() {
                return Err(invalid(
                    model,
                    source,
                    "a tier rate takes its band from the tier range, not an _above_ suffix",
                ));
            }
            Ok(Priced {
                key: PriceKey {
                    band: Some(band),
                    ..PriceKey::from_rate(rate, Clock::Peak)
                },
                usd: *usd,
                source,
            })
        })
        .collect()
}

fn tier_band(tier: &TieredRate) -> Result<Band, &'static str> {
    let (basis, [start, end]) = match (tier.range, tier.max_results_range) {
        (Some(range), None) => (Basis::PromptTokens, range),
        (None, Some(range)) => (Basis::Results, range),
        _ => return Err("a tier needs exactly one of range and max_results_range"),
    };
    let start = whole(start)?;
    if whole(end)? < start {
        return Err("a tier range must not end below its start");
    }
    Ok(Band {
        basis,
        floor: match basis {
            Basis::Results => Floor::AtLeast(start),
            Basis::PromptTokens | Basis::Seconds => Floor::Over(start),
        },
    })
}

fn whole(bound: f64) -> Result<u64, &'static str> {
    (bound.is_finite() && bound >= 0.0 && bound.fract() == 0.0)
        .then_some(bound as u64)
        .ok_or("a tier bound must be a non-negative whole number")
}

fn off_peak(model: &str, pricing: &OffPeakPricing) -> Result<OffPeakSchedule, Problem> {
    let problem = |key: &str, message| invalid(model, format!("off_peak_pricing.{key}"), message);
    let windows = pricing.windows.as_deref().unwrap_or_default();
    if pricing.windows.as_ref().is_some_and(Vec::is_empty) {
        return Err(problem("windows", "must list at least one window"));
    }
    if pricing.hours_utc.is_none() && windows.is_empty() {
        return Err(problem(
            "hours_utc",
            "off-peak pricing needs hours_utc or windows",
        ));
    }
    if windows
        .iter()
        .any(|window| window.weekdays.as_ref().is_some_and(Vec::is_empty))
    {
        return Err(problem(
            "windows.weekdays",
            "must list at least one weekday",
        ));
    }
    let flat = pricing.hours_utc.iter().map(|hours| OffPeakHours {
        hours: hours.windows().to_vec(),
        weekdays: None,
    });
    let scheduled = windows.iter().map(|window| OffPeakHours {
        hours: window.hours_utc.windows().to_vec(),
        weekdays: window.weekdays.clone(),
    });
    let hours: Vec<_> = flat.chain(scheduled).collect();
    if hours.iter().any(|hours| hours.hours.is_empty()) {
        return Err(problem("hours_utc", "must list at least one window"));
    }
    Ok(OffPeakSchedule {
        hours,
        weekday_timezone: pricing.weekday_timezone.clone(),
    })
}

fn off_peak_rates(pricing: &OffPeakPricing) -> impl Iterator<Item = Priced> + '_ {
    pricing.rates.iter().map(|(rate, usd)| Priced {
        key: PriceKey::from_rate(rate, Clock::OffPeak),
        usd: *usd,
        source: format!("off_peak_pricing.{rate}"),
    })
}

fn search_context(costs: &SearchContextCostPerQuery) -> impl Iterator<Item = Priced> {
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
    .filter_map(|(size, name, usd)| {
        Some(Priced {
            key: PriceKey {
                meter: Meter::WebSearch(size),
                tier: None,
                band: None,
                clock: Clock::Peak,
            },
            usd: usd?,
            source: format!("search_context_cost_per_query.search_context_size_{name}"),
        })
    })
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::{Value, json};

    use super::*;
    use crate::rate::{SimpleUnit, TokenKind, Unit};

    fn lower(entry: Value) -> Result<Prices, Vec<String>> {
        let Value::Object(fields) = entry else {
            panic!("entry must be an object")
        };
        let flat = fields
            .iter()
            .filter_map(|(key, value)| Some((key.parse().ok()?, Usd::deserialize(value).unwrap())))
            .collect();
        let pricing: PricingFields = serde_json::from_value(Value::Object(fields)).unwrap();
        Prices::lower("m", flat, &pricing)
            .map_err(|problems| problems.iter().map(ToString::to_string).collect())
    }

    use serde::Deserialize;

    fn input_token() -> Meter {
        Meter::Charge(Charge::Input(Unit::Token(TokenKind::Text)))
    }

    fn key(meter: Meter, tier: Option<Tier>, band: Option<Band>, clock: Clock) -> PriceKey {
        PriceKey {
            meter,
            tier,
            band,
            clock,
        }
    }

    fn prompt_over(tokens: u64) -> Option<Band> {
        Some(Band {
            basis: Basis::PromptTokens,
            floor: Floor::Over(tokens),
        })
    }

    #[rstest]
    #[case::flat_key(
        json!({"input_cost_per_token": 1e-6}),
        key(input_token(), None, None, Clock::Peak)
    )]
    #[case::above_suffix_and_tier(
        json!({"input_cost_per_token_above_272k_tokens_priority": 1e-6}),
        key(input_token(), Some(Tier::Priority), prompt_over(272_000), Clock::Peak)
    )]
    #[case::seconds_interval(
        json!({"input_cost_per_video_per_second_above_8s_interval": 1e-6}),
        key(
            Meter::Charge(Charge::Input(Unit::Simple(SimpleUnit::VideoSecond))),
            None,
            Some(Band { basis: Basis::Seconds, floor: Floor::Over(8) }),
            Clock::Peak
        )
    )]
    #[case::tier_range(
        json!({"tiered_pricing": [{"range": [256000.0, 1000000.0], "input_cost_per_token": 1e-6}]}),
        key(input_token(), None, prompt_over(256_000), Clock::Peak)
    )]
    #[case::result_count_range(
        json!({"tiered_pricing": [{"max_results_range": [26, 100], "input_cost_per_query": 1e-6}]}),
        key(
            Meter::Charge(Charge::Input(Unit::Simple(SimpleUnit::Query))),
            None,
            Some(Band { basis: Basis::Results, floor: Floor::AtLeast(26) }),
            Clock::Peak
        )
    )]
    #[case::off_peak(
        json!({"off_peak_pricing": {"hours_utc": "16:30-00:30", "input_cost_per_token": 1e-6}}),
        key(input_token(), None, None, Clock::OffPeak)
    )]
    #[case::search_context(
        json!({"search_context_cost_per_query": {"search_context_size_low": 1e-6}}),
        key(Meter::WebSearch(SearchContextSize::Low), None, None, Clock::Peak)
    )]
    #[case::guardrail(
        json!({"guardrail_cost_per_unit": {"contentPolicyUnits": 1e-6}}),
        key(Meter::Guardrail("contentPolicyUnits".into()), None, None, Clock::Peak)
    )]
    fn every_encoding_lowers_to_one_price_key(#[case] entry: Value, #[case] expected: PriceKey) {
        let prices = lower(entry).unwrap();
        assert_eq!(
            prices.iter().collect::<Vec<_>>(),
            vec![(&expected, Usd::new(1e-6).unwrap())]
        );
    }

    #[rstest]
    #[case::suffix_and_tier_range(
        json!({
            "input_cost_per_token_above_256k_tokens": 1e-6,
            "tiered_pricing": [{"range": [256000, 1000000], "input_cost_per_token": 2e-6}]
        }),
        "input_cost_per_token_above_256k_tokens and tiered_pricing[0].input_cost_per_token"
    )]
    #[case::two_tiers_same_start(
        json!({"tiered_pricing": [
            {"range": [0, 10], "input_cost_per_token": 1e-6},
            {"range": [0, 20], "input_cost_per_token": 2e-6}
        ]}),
        "tiered_pricing[0].input_cost_per_token and tiered_pricing[1].input_cost_per_token"
    )]
    #[case::tier_without_range(
        json!({"tiered_pricing": [{"input_cost_per_token": 1e-6}]}),
        "m.tiered_pricing[0]: a tier needs exactly one of range and max_results_range"
    )]
    #[case::tier_with_suffix(
        json!({"tiered_pricing": [{"range": [0, 10], "input_cost_per_token_above_8k_tokens": 1e-6}]}),
        "m.tiered_pricing[0].input_cost_per_token_above_8k_tokens"
    )]
    #[case::fractional_bound(
        json!({"tiered_pricing": [{"range": [0.5, 10], "input_cost_per_token": 1e-6}]}),
        "non-negative whole number"
    )]
    #[case::reversed_range(
        json!({"tiered_pricing": [{"range": [10, 5], "input_cost_per_token": 1e-6}]}),
        "must not end below its start"
    )]
    #[case::off_peak_without_hours(
        json!({"off_peak_pricing": {"input_cost_per_token": 1e-6}}),
        "m.off_peak_pricing.hours_utc: off-peak pricing needs hours_utc or windows"
    )]
    #[case::off_peak_empty_windows(
        json!({"off_peak_pricing": {"windows": []}}),
        "m.off_peak_pricing.windows"
    )]
    #[case::off_peak_empty_weekdays(
        json!({"off_peak_pricing": {"windows": [{"hours_utc": "00:00-01:00", "weekdays": []}]}}),
        "m.off_peak_pricing.windows.weekdays"
    )]
    fn conflicting_or_malformed_prices_name_where_they_came_from(
        #[case] entry: Value,
        #[case] expected: &str,
    ) {
        let problems = lower(entry).unwrap_err();
        assert!(
            problems.iter().any(|problem| problem.contains(expected)),
            "{problems:?}"
        );
    }

    #[rstest]
    fn off_peak_hours_and_windows_form_one_schedule() {
        let prices = lower(json!({"off_peak_pricing": {
            "hours_utc": ["00:00-01:00"],
            "windows": [{"hours_utc": "10:00-12:00", "weekdays": [6, "sun"]}],
            "weekday_timezone": "Asia/Shanghai"
        }}))
        .unwrap();
        assert_eq!(
            prices.off_peak(),
            Some(&OffPeakSchedule {
                hours: vec![
                    OffPeakHours {
                        hours: vec!["00:00-01:00".parse().unwrap()],
                        weekdays: None,
                    },
                    OffPeakHours {
                        hours: vec!["10:00-12:00".parse().unwrap()],
                        weekdays: Some(vec![Weekday::Saturday, Weekday::Sunday]),
                    },
                ],
                weekday_timezone: Some("Asia/Shanghai".into()),
            })
        );
    }

    #[rstest]
    fn uplifts_are_kept_per_region_not_as_prices() {
        let prices = lower(json!({
            "regional_endpoint_uplift_multiplier": 1.1,
            "regional_processing_uplift_multiplier_eu": 1.2
        }))
        .unwrap();
        assert_eq!(prices.iter().count(), 0);
        assert_eq!(
            prices.uplift(Region::RegionalEndpoint).map(Multiplier::get),
            Some(1.1)
        );
        assert_eq!(prices.uplift(Region::Eu).map(Multiplier::get), Some(1.2));
        assert_eq!(prices.uplift(Region::Us), None);
    }
}
