use std::str::FromStr;

use jiff::Timestamp;
use jiff::tz::TimeZone;

use crate::Error;

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

impl Schedule {
    /// Whether `at` falls in a window. A window with weekdays counts only on those days, read on
    /// the `weekday_timezone` calendar, which is UTC when the name is missing or unknown
    pub fn is_open(&self, at: Timestamp) -> bool {
        let utc = at.to_zoned(TimeZone::UTC);
        let minute = utc.hour() as u16 * 60 + utc.minute() as u16;
        let calendar = self
            .weekday_timezone
            .as_deref()
            .and_then(|name| TimeZone::get(name.trim()).ok())
            .unwrap_or(TimeZone::UTC);
        let weekday = Weekday::from(at.to_zoned(calendar).weekday());
        self.windows.iter().any(|window| {
            window
                .weekdays
                .as_ref()
                .is_none_or(|days| days.contains(&weekday))
                && window.hours.iter().any(|hours| hours.contains(minute))
        })
    }
}

impl UtcWindow {
    /// The start is in the window and the end is not. A window that ends at or before its start
    /// wraps past midnight, so equal ends cover the whole day
    fn contains(self, minute: u16) -> bool {
        if self.start_minute < self.end_minute {
            self.start_minute <= minute && minute < self.end_minute
        } else {
            minute >= self.start_minute || minute < self.end_minute
        }
    }
}

impl From<jiff::civil::Weekday> for Weekday {
    fn from(day: jiff::civil::Weekday) -> Self {
        match day {
            jiff::civil::Weekday::Monday => Self::Monday,
            jiff::civil::Weekday::Tuesday => Self::Tuesday,
            jiff::civil::Weekday::Wednesday => Self::Wednesday,
            jiff::civil::Weekday::Thursday => Self::Thursday,
            jiff::civil::Weekday::Friday => Self::Friday,
            jiff::civil::Weekday::Saturday => Self::Saturday,
            jiff::civil::Weekday::Sunday => Self::Sunday,
        }
    }
}

impl FromStr for UtcWindow {
    type Err = Error;

    fn from_str(window: &str) -> Result<Self, Self::Err> {
        let (start, end) = window.split_once('-').ok_or(Error::InvalidUtcWindow)?;
        Ok(Self {
            start_minute: minute_of_day(start).ok_or(Error::InvalidUtcWindow)?,
            end_minute: minute_of_day(end).ok_or(Error::InvalidUtcWindow)?,
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
    type Err = Error;

    fn from_str(name: &str) -> Result<Self, Self::Err> {
        match name.trim().to_ascii_lowercase().as_str() {
            "mon" | "monday" => Ok(Self::Monday),
            "tue" | "tues" | "tuesday" => Ok(Self::Tuesday),
            "wed" | "wednesday" => Ok(Self::Wednesday),
            "thu" | "thur" | "thurs" | "thursday" => Ok(Self::Thursday),
            "fri" | "friday" => Ok(Self::Friday),
            "sat" | "saturday" => Ok(Self::Saturday),
            "sun" | "sunday" => Ok(Self::Sunday),
            _ => Err(Error::InvalidWeekday),
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

    fn schedule(windows: &[(&[&str], Option<&[u8]>)], weekday_timezone: Option<&str>) -> Schedule {
        Schedule {
            windows: windows
                .iter()
                .map(|(hours, weekdays)| Window {
                    hours: hours.iter().map(|hours| hours.parse().unwrap()).collect(),
                    weekdays: weekdays.map(|days| {
                        days.iter()
                            .map(|day| Weekday::from_iso(*day).unwrap())
                            .collect()
                    }),
                })
                .collect(),
            weekday_timezone: weekday_timezone.map(str::to_owned),
        }
    }

    fn is_open(schedule: &Schedule, at: &str) -> bool {
        schedule.is_open(at.parse().unwrap())
    }

    #[rstest]
    #[case::inside("09:00-17:00", "2026-01-01T12:00:00Z", true)]
    #[case::before_the_start("09:00-17:00", "2026-01-01T08:59:59Z", false)]
    #[case::the_start_is_in("09:00-17:00", "2026-01-01T09:00:00Z", true)]
    #[case::the_last_second_is_in("09:00-17:00", "2026-01-01T16:59:59Z", true)]
    #[case::the_end_is_out("09:00-17:00", "2026-01-01T17:00:00Z", false)]
    #[case::wrapped_evening("16:30-00:30", "2026-01-01T18:00:00Z", true)]
    #[case::wrapped_after_midnight("16:30-00:30", "2026-01-01T00:15:00Z", true)]
    #[case::wrapped_start_is_in("16:30-00:30", "2026-01-01T16:30:00Z", true)]
    #[case::wrapped_end_is_out("16:30-00:30", "2026-01-01T00:30:00Z", false)]
    #[case::wrapped_midday("16:30-00:30", "2026-01-01T12:00:00Z", false)]
    #[case::equal_ends_cover_the_morning("00:00-00:00", "2026-01-01T03:00:00Z", true)]
    #[case::equal_ends_cover_the_evening("10:00-10:00", "2026-01-01T23:59:59Z", true)]
    #[case::equal_ends_cover_their_own_minute("10:00-10:00", "2026-01-01T10:00:00Z", true)]
    #[case::an_offset_is_read_in_utc("01:00-05:00", "2026-01-01T09:00:00+08:00", true)]
    #[case::an_offset_can_land_on_the_previous_day(
        "01:00-05:00",
        "2026-01-01T06:00:00+08:00",
        false
    )]
    fn an_every_day_window_is_open_from_its_start_until_its_end(
        #[case] hours: &str,
        #[case] at: &str,
        #[case] open: bool,
    ) {
        assert_eq!(is_open(&schedule(&[(&[hours], None)], None), at), open);
    }

    #[rstest]
    #[case::first("2026-01-01T03:00:00Z", true)]
    #[case::second("2026-01-01T14:30:00Z", true)]
    #[case::between("2026-01-01T09:00:00Z", false)]
    fn any_window_of_the_day_opens_the_schedule(#[case] at: &str, #[case] open: bool) {
        let schedule = schedule(&[(&["01:00-05:00", "13:00-16:00"], None)], None);
        assert_eq!(is_open(&schedule, at), open);
    }

    #[rstest]
    #[case::monday_peak("2026-08-24T01:30:00Z", false)]
    #[case::wednesday_peak("2026-08-26T07:00:00Z", false)]
    #[case::friday_last_peak_minute("2026-08-28T09:59:00Z", false)]
    #[case::sunday_in_weekday_peak_hours("2026-08-23T01:30:00Z", true)]
    #[case::saturday_in_weekday_peak_hours("2026-08-29T02:00:00Z", true)]
    #[case::sunday_morning("2026-08-30T08:00:00Z", true)]
    #[case::wednesday_between_peaks("2026-08-26T05:00:00Z", true)]
    #[case::friday_evening("2026-08-28T16:30:00Z", true)]
    #[case::monday_first_half_hour("2026-08-24T00:30:00Z", true)]
    fn weekday_windows_and_an_all_day_weekend_form_one_schedule(
        #[case] at: &str,
        #[case] open: bool,
    ) {
        let deepseek = schedule(
            &[
                (
                    &["00:00-01:00", "04:00-06:00", "10:00-00:00"],
                    Some(&[1, 2, 3, 4, 5]),
                ),
                (&["00:00-00:00"], Some(&[6, 7])),
            ],
            None,
        );
        assert_eq!(is_open(&deepseek, at), open);
    }

    #[rstest]
    #[case::shanghai_is_already_saturday(Some("Asia/Shanghai"), 6, "2026-08-28T16:30:00Z", true)]
    #[case::shanghai_is_already_sunday(Some("Asia/Shanghai"), 6, "2026-08-29T16:30:00Z", false)]
    #[case::a_padded_name_is_trimmed(Some(" Asia/Shanghai "), 6, "2026-08-28T16:30:00Z", true)]
    #[case::utc_by_default(None, 5, "2026-08-28T16:30:00Z", true)]
    #[case::utc_by_default_next_day(None, 5, "2026-08-29T16:30:00Z", false)]
    #[case::an_unknown_zone_is_utc(Some("Not/AZone"), 5, "2026-08-28T16:30:00Z", true)]
    #[case::an_unknown_zone_is_utc_next_day(Some("Not/AZone"), 5, "2026-08-29T16:30:00Z", false)]
    #[case::a_blank_zone_is_utc(Some("  "), 5, "2026-08-28T16:30:00Z", true)]
    fn a_weekday_is_read_on_the_schedules_calendar(
        #[case] weekday_timezone: Option<&str>,
        #[case] weekday: u8,
        #[case] at: &str,
        #[case] open: bool,
    ) {
        let schedule = schedule(&[(&["16:00-17:00"], Some(&[weekday]))], weekday_timezone);
        assert_eq!(is_open(&schedule, at), open);
    }

    #[rstest]
    #[case::every_day_hours("2026-08-28T05:00:00Z", true)]
    #[case::sunday_all_day("2026-08-30T20:00:00Z", true)]
    #[case::friday_outside_the_hours("2026-08-28T20:00:00Z", false)]
    fn every_day_hours_and_weekday_windows_are_a_union(#[case] at: &str, #[case] open: bool) {
        let schedule = schedule(
            &[(&["04:00-06:00"], None), (&["00:00-00:00"], Some(&[7]))],
            None,
        );
        assert_eq!(is_open(&schedule, at), open);
    }

    #[rstest]
    fn a_schedule_without_windows_is_never_open() {
        assert!(!is_open(&schedule(&[], None), "2026-08-29T12:00:00Z"));
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
