use std::str::FromStr;

use crate::error::{InvalidUtcWindow, InvalidWeekday};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Schedule {
    pub windows: Vec<Window>,
    pub weekday_timezone: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Window {
    pub hours: Vec<UtcWindow>,
    pub weekdays: Option<Vec<Weekday>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct UtcWindow {
    pub start_minute: u16,
    pub end_minute: u16,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Weekday {
    Monday,
    Tuesday,
    Wednesday,
    Thursday,
    Friday,
    Saturday,
    Sunday,
}

impl FromStr for UtcWindow {
    type Err = InvalidUtcWindow;

    fn from_str(window: &str) -> Result<Self, Self::Err> {
        let (start, end) = window.split_once('-').ok_or(InvalidUtcWindow)?;
        Ok(Self {
            start_minute: minute_of_day(start).ok_or(InvalidUtcWindow)?,
            end_minute: minute_of_day(end).ok_or(InvalidUtcWindow)?,
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

impl Weekday {
    pub fn from_iso(number: u8) -> Option<Self> {
        match number {
            1 => Some(Self::Monday),
            2 => Some(Self::Tuesday),
            3 => Some(Self::Wednesday),
            4 => Some(Self::Thursday),
            5 => Some(Self::Friday),
            6 => Some(Self::Saturday),
            7 => Some(Self::Sunday),
            _ => None,
        }
    }
}

impl FromStr for Weekday {
    type Err = InvalidWeekday;

    fn from_str(name: &str) -> Result<Self, Self::Err> {
        match name.trim().to_ascii_lowercase().as_str() {
            "mon" | "monday" => Ok(Self::Monday),
            "tue" | "tues" | "tuesday" => Ok(Self::Tuesday),
            "wed" | "wednesday" => Ok(Self::Wednesday),
            "thu" | "thur" | "thurs" | "thursday" => Ok(Self::Thursday),
            "fri" | "friday" => Ok(Self::Friday),
            "sat" | "saturday" => Ok(Self::Saturday),
            "sun" | "sunday" => Ok(Self::Sunday),
            _ => Err(InvalidWeekday),
        }
    }
}

#[cfg(feature = "serde")]
mod serde_impls {
    use std::fmt;

    use serde::de::{self, Deserialize, Deserializer, Visitor};

    use super::{UtcWindow, Weekday};

    impl<'de> Deserialize<'de> for UtcWindow {
        fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            let window = String::deserialize(deserializer)?;
            window.parse().map_err(de::Error::custom)
        }
    }

    struct WrittenWeekday;

    impl Visitor<'_> for WrittenWeekday {
        type Value = Weekday;

        fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str("an ISO weekday number 1-7 or an English day name")
        }

        fn visit_u64<E: de::Error>(self, number: u64) -> Result<Weekday, E> {
            u8::try_from(number)
                .ok()
                .and_then(Weekday::from_iso)
                .ok_or_else(|| E::invalid_value(de::Unexpected::Unsigned(number), &self))
        }

        fn visit_i64<E: de::Error>(self, number: i64) -> Result<Weekday, E> {
            u64::try_from(number)
                .map_err(|_| E::invalid_value(de::Unexpected::Signed(number), &self))
                .and_then(|number| self.visit_u64(number))
        }

        fn visit_str<E: de::Error>(self, name: &str) -> Result<Weekday, E> {
            name.parse()
                .map_err(|_| E::invalid_value(de::Unexpected::Str(name), &self))
        }
    }

    impl<'de> Deserialize<'de> for Weekday {
        fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            deserializer.deserialize_any(WrittenWeekday)
        }
    }
}

#[cfg(feature = "schema")]
mod schema {
    use std::borrow::Cow;

    use schemars::{JsonSchema, Schema, SchemaGenerator, json_schema};

    use super::{UtcWindow, Weekday};

    impl JsonSchema for UtcWindow {
        fn inline_schema() -> bool {
            true
        }
        fn schema_name() -> Cow<'static, str> {
            "UtcWindow".into()
        }
        fn json_schema(_: &mut SchemaGenerator) -> Schema {
            json_schema!({
                "type": "string",
                "pattern": r"^([01]\d|2[0-3]):[0-5]\d-([01]\d|2[0-3]):[0-5]\d$"
            })
        }
    }

    impl JsonSchema for Weekday {
        fn inline_schema() -> bool {
            true
        }
        fn schema_name() -> Cow<'static, str> {
            "Weekday".into()
        }
        fn json_schema(_: &mut SchemaGenerator) -> Schema {
            json_schema!({
                "anyOf": [
                    {"type": "integer", "minimum": 1, "maximum": 7},
                    {
                        "type": "string",
                        "pattern": r"(?i)^\s*(mon|monday|tue|tues|tuesday|wed|wednesday|thu|thur|thurs|thursday|fri|friday|sat|saturday|sun|sunday)\s*$"
                    }
                ]
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::same_day("10:00-18:30", Some((600, 1110)))]
    #[case::wraps_midnight("22:00-02:00", Some((1320, 120)))]
    #[case::whole_day("00:00-00:00", Some((0, 0)))]
    #[case::hour_out_of_range("25:00-01:00", None)]
    #[case::minute_out_of_range("10:60-11:00", None)]
    #[case::one_digit_hour("9:00-10:00", None)]
    #[case::no_separator("10:00", None)]
    fn utc_window_reads_hh_mm_ranges(#[case] text: &str, #[case] expected: Option<(u16, u16)>) {
        assert_eq!(
            text.parse::<UtcWindow>()
                .ok()
                .map(|window| (window.start_minute, window.end_minute)),
            expected
        );
    }

    #[rstest]
    #[case::short("thurs", Some(Weekday::Thursday))]
    #[case::upper("SATURDAY", Some(Weekday::Saturday))]
    #[case::padded(" mon ", Some(Weekday::Monday))]
    #[case::unknown("someday", None)]
    #[case::empty("", None)]
    fn weekday_names_match_python_after_trimming(
        #[case] name: &str,
        #[case] expected: Option<Weekday>,
    ) {
        assert_eq!(name.parse().ok(), expected);
    }

    #[rstest]
    #[case::monday(1, Some(Weekday::Monday))]
    #[case::sunday(7, Some(Weekday::Sunday))]
    #[case::zero(0, None)]
    #[case::eight(8, None)]
    fn weekday_numbers_are_iso(#[case] number: u8, #[case] expected: Option<Weekday>) {
        assert_eq!(Weekday::from_iso(number), expected);
    }

    #[cfg(feature = "serde")]
    #[rstest]
    #[case::number(serde_json::json!(6), Some(Weekday::Saturday))]
    #[case::name(serde_json::json!("Sun"), Some(Weekday::Sunday))]
    #[case::negative(serde_json::json!(-1), None)]
    #[case::boolean(serde_json::json!(true), None)]
    fn weekday_reads_numbers_and_names(
        #[case] written: serde_json::Value,
        #[case] expected: Option<Weekday>,
    ) {
        use serde::Deserialize;

        assert_eq!(Weekday::deserialize(&written).ok(), expected);
    }
}
