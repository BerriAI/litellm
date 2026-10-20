use std::collections::BTreeMap;
use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Deserializer, Serialize, de};

use crate::error::{InvalidUtcWindow, InvalidWeekday};
use crate::rate::Rate;

/// USD for one unit of a meter. Numeric strings are accepted because proxy configs write prices
/// as YAML strings.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct Usd(f64);

impl Usd {
    pub fn new(amount: f64) -> Option<Self> {
        (amount.is_finite() && amount >= 0.0).then_some(Self(amount))
    }
    pub fn get(self) -> f64 {
        self.0
    }
}

impl<'de> Deserialize<'de> for Usd {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let expected = Number("a non-negative USD amount");
        let amount = deserializer.deserialize_any(expected)?;
        Self::new(amount)
            .ok_or_else(|| de::Error::invalid_value(de::Unexpected::Float(amount), &expected))
    }
}

/// A factor applied to every token price, never a discount.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct Multiplier(f64);

impl Multiplier {
    pub fn new(factor: f64) -> Option<Self> {
        (factor.is_finite() && factor >= 1.0).then_some(Self(factor))
    }
    pub fn get(self) -> f64 {
        self.0
    }
}

impl<'de> Deserialize<'de> for Multiplier {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let expected = Number("a multiplier of at least 1");
        let factor = deserializer.deserialize_any(expected)?;
        Self::new(factor)
            .ok_or_else(|| de::Error::invalid_value(de::Unexpected::Float(factor), &expected))
    }
}

#[derive(Clone, Copy)]
struct Number(&'static str);

impl de::Visitor<'_> for Number {
    type Value = f64;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.0)
    }

    fn visit_i64<E: de::Error>(self, value: i64) -> Result<f64, E> {
        Ok(value as f64)
    }

    fn visit_u64<E: de::Error>(self, value: u64) -> Result<f64, E> {
        Ok(value as f64)
    }

    fn visit_f64<E: de::Error>(self, value: f64) -> Result<f64, E> {
        Ok(value)
    }

    fn visit_str<E: de::Error>(self, value: &str) -> Result<f64, E> {
        value
            .trim()
            .parse()
            .map_err(|_| E::invalid_value(de::Unexpected::Str(value), &self))
    }
}

/// Whether web search is billed per query or per prompt.
#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Copy, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum WebSearchBillingUnit {
    PerQuery,
    PerPrompt,
}

/// A UTC "HH:MM-HH:MM" window, in minutes after midnight. It wraps past midnight when `end` is
/// not after `start`.
#[derive(
    Clone,
    Copy,
    Debug,
    PartialEq,
    Eq,
    Hash,
    serde_with::DeserializeFromStr,
    serde_with::SerializeDisplay,
)]
pub struct UtcWindow {
    pub start: u16,
    pub end: u16,
}

impl FromStr for UtcWindow {
    type Err = InvalidUtcWindow;

    fn from_str(window: &str) -> Result<Self, Self::Err> {
        let (start, end) = window.split_once('-').ok_or(InvalidUtcWindow)?;
        Ok(Self {
            start: minute_of_day(start).ok_or(InvalidUtcWindow)?,
            end: minute_of_day(end).ok_or(InvalidUtcWindow)?,
        })
    }
}

fn minute_of_day(clock: &str) -> Option<u16> {
    let (hour, minute) = clock.split_once(':')?;
    if hour.len() != 2 || minute.len() != 2 {
        return None;
    }
    let hour: u16 = hour.parse().ok().filter(|hour| *hour < 24)?;
    let minute: u16 = minute.parse().ok().filter(|minute| *minute < 60)?;
    Some(hour * 60 + minute)
}

impl fmt::Display for UtcWindow {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:02}:{:02}-{:02}:{:02}",
            self.start / 60,
            self.start % 60,
            self.end / 60,
            self.end % 60
        )
    }
}

/// One UTC window or a list of them.
#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Eq, Hash)]
#[serde(untagged)]
pub enum UtcHours {
    Single(UtcWindow),
    Multiple(Vec<UtcWindow>),
}

impl UtcHours {
    pub fn windows(&self) -> &[UtcWindow] {
        match self {
            Self::Single(window) => std::slice::from_ref(window),
            Self::Multiple(windows) => windows,
        }
    }
}

/// Written as an ISO-8601 weekday number (1 = Monday .. 7 = Sunday) or an English day name.
#[derive(
    Clone,
    Copy,
    Debug,
    PartialEq,
    Eq,
    Hash,
    Serialize,
    Deserialize,
    strum::EnumString,
    strum::VariantArray,
)]
#[serde(try_from = "WrittenWeekday", into = "u8")]
#[strum(ascii_case_insensitive)]
pub enum Weekday {
    #[strum(serialize = "mon", serialize = "monday")]
    Monday,
    #[strum(serialize = "tue", serialize = "tues", serialize = "tuesday")]
    Tuesday,
    #[strum(serialize = "wed", serialize = "wednesday")]
    Wednesday,
    #[strum(
        serialize = "thu",
        serialize = "thur",
        serialize = "thurs",
        serialize = "thursday"
    )]
    Thursday,
    #[strum(serialize = "fri", serialize = "friday")]
    Friday,
    #[strum(serialize = "sat", serialize = "saturday")]
    Saturday,
    #[strum(serialize = "sun", serialize = "sunday")]
    Sunday,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum WrittenWeekday {
    Number(u8),
    Name(String),
}

impl TryFrom<WrittenWeekday> for Weekday {
    type Error = InvalidWeekday;

    fn try_from(written: WrittenWeekday) -> Result<Self, Self::Error> {
        use strum::VariantArray;
        match written {
            WrittenWeekday::Number(number) => Self::VARIANTS
                .get(usize::from(number).wrapping_sub(1))
                .copied()
                .ok_or(InvalidWeekday),
            WrittenWeekday::Name(name) => name.parse().map_err(|_| InvalidWeekday),
        }
    }
}

impl From<Weekday> for u8 {
    fn from(day: Weekday) -> Self {
        day as u8 + 1
    }
}

/// One off-peak window entry inside `windows`.
#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Eq, Hash)]
#[serde(deny_unknown_fields)]
pub struct OffPeakWindow {
    pub hours_utc: UtcHours,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub weekdays: Option<Vec<Weekday>>,
}

/// Rates that replace the same-named base fields inside the stated UTC windows.
#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Default)]
pub struct OffPeakPricing {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hours_utc: Option<UtcHours>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub windows: Option<Vec<OffPeakWindow>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub weekday_timezone: Option<String>,
    #[serde(flatten)]
    #[cfg_attr(feature = "schema", schemars(with = "BTreeMap<String, Usd>"))]
    pub rates: BTreeMap<Rate, Usd>,
}

/// USD cost per web search query, keyed by search context size.
#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Default)]
#[serde(deny_unknown_fields)]
pub struct SearchContextCostPerQuery {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub search_context_size_low: Option<Usd>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub search_context_size_medium: Option<Usd>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub search_context_size_high: Option<Usd>,
}

/// One tier of a context-length or result-count tiered rate.
#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Default)]
pub struct TieredRate {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub range: Option<[f64; 2]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_results_range: Option<[f64; 2]>,
    #[serde(flatten)]
    #[cfg_attr(feature = "schema", schemars(with = "BTreeMap<String, Usd>"))]
    pub rates: BTreeMap<Rate, Usd>,
}

/// The entry fields that price something but are not a flat rate key.
#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Default)]
pub struct PricingFields {
    /// USD cost per billable guardrail unit, keyed by the provider's usage counter name (e.g. Bedrock's contentPolicyUnits).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub guardrail_cost_per_unit: Option<BTreeMap<String, Usd>>,
    /// Rates that replace the same-named base fields while the request falls inside the stated UTC windows.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub off_peak_pricing: Option<OffPeakPricing>,
    /// Multiplier applied to all token costs when served from a non-global Vertex AI endpoint (e.g. 1.10 = +10%).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub regional_endpoint_uplift_multiplier: Option<Multiplier>,
    /// Multiplier applied to all token costs for EU data residency (e.g. 1.10 = +10%).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub regional_processing_uplift_multiplier_eu: Option<Multiplier>,
    /// Multiplier applied to all token costs for US data residency (e.g. 1.10 = +10%).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub regional_processing_uplift_multiplier_us: Option<Multiplier>,
    /// USD cost per web search query, keyed by search context size.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub search_context_cost_per_query: Option<SearchContextCostPerQuery>,
    /// Context-length or result-count tiered rates; each tier's costs apply within its range.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tiered_pricing: Option<Vec<TieredRate>>,
}

#[cfg(feature = "schema")]
mod schema {
    use std::borrow::Cow;

    use schemars::{JsonSchema, Schema, SchemaGenerator, json_schema};

    use super::{Multiplier, Usd, UtcWindow, Weekday};

    macro_rules! inline_schema {
        ($type:ty, $name:literal, $schema:tt) => {
            impl JsonSchema for $type {
                fn inline_schema() -> bool {
                    true
                }
                fn schema_name() -> Cow<'static, str> {
                    $name.into()
                }
                fn json_schema(_: &mut SchemaGenerator) -> Schema {
                    json_schema!($schema)
                }
            }
        };
    }

    inline_schema!(Usd, "Usd", {"type": "number", "minimum": 0});
    inline_schema!(Multiplier, "Multiplier", {"type": "number", "minimum": 1});
    inline_schema!(UtcWindow, "UtcWindow", {
        "type": "string",
        "pattern": r"^([01]\d|2[0-3]):[0-5]\d-([01]\d|2[0-3]):[0-5]\d$"
    });
    inline_schema!(Weekday, "Weekday", {
        "anyOf": [
            {"type": "integer", "minimum": 1, "maximum": 7},
            {
                "type": "string",
                "pattern": r"(?i)^(mon|monday|tue|tues|tuesday|wed|wednesday|thu|thur|thurs|thursday|fri|friday|sat|saturday|sun|sunday)$"
            }
        ]
    });
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::{Value, json};

    use super::*;

    #[rstest]
    #[case::number(json!(1.5e-6), Some(1.5e-6))]
    #[case::zero(json!(0), Some(0.0))]
    #[case::yaml_string(json!(" 0.000002 "), Some(2e-6))]
    #[case::negative(json!(-1), None)]
    #[case::not_a_number(json!("abc"), None)]
    #[case::boolean(json!(true), None)]
    fn usd_accepts_non_negative_amounts(#[case] written: Value, #[case] expected: Option<f64>) {
        assert_eq!(Usd::deserialize(&written).ok().map(Usd::get), expected);
    }

    #[rstest]
    #[case::uplift(json!(1.1), Some(1.1))]
    #[case::neutral(json!(1), Some(1.0))]
    #[case::discount(json!(0.5), None)]
    fn multiplier_never_discounts(#[case] written: Value, #[case] expected: Option<f64>) {
        assert_eq!(
            Multiplier::deserialize(&written).ok().map(Multiplier::get),
            expected
        );
    }

    #[rstest]
    #[case::same_day("10:00-18:30", Some((600, 1110)))]
    #[case::wraps_midnight("22:00-02:00", Some((1320, 120)))]
    #[case::hour_out_of_range("25:00-01:00", None)]
    #[case::minute_out_of_range("10:60-11:00", None)]
    #[case::one_digit_hour("9:00-10:00", None)]
    #[case::no_separator("10:00", None)]
    fn utc_window_parses_and_displays_the_same_text(
        #[case] text: &str,
        #[case] expected: Option<(u16, u16)>,
    ) {
        let parsed = text.parse::<UtcWindow>().ok();
        assert_eq!(parsed.map(|window| (window.start, window.end)), expected);
        if let Some(window) = parsed {
            assert_eq!(window.to_string(), text);
        }
    }

    #[rstest]
    #[case::iso_number(json!(1), Some(Weekday::Monday))]
    #[case::last_number(json!(7), Some(Weekday::Sunday))]
    #[case::short_name(json!("Thurs"), Some(Weekday::Thursday))]
    #[case::full_name(json!("SATURDAY"), Some(Weekday::Saturday))]
    #[case::zero(json!(0), None)]
    #[case::eight(json!(8), None)]
    #[case::unknown_name(json!("someday"), None)]
    fn weekday_reads_numbers_and_names(#[case] written: Value, #[case] expected: Option<Weekday>) {
        assert_eq!(Weekday::deserialize(&written).ok(), expected);
    }
}
