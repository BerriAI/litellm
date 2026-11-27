//! The fields of `datetime.fromtimestamp(seconds, timezone.utc)`, `datetime.fromisoformat(text)`,
//! and the milliseconds Python takes between two `datetime`s.

use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// A UTC `datetime`'s fields, as Python's `datetime` exposes them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UtcDateTime {
    pub year: i64,
    pub month: u32,
    pub day: u32,
    pub hour: u32,
    pub minute: u32,
    pub second: u32,
    pub microsecond: u32,
}

impl UtcDateTime {
    /// `datetime.fromtimestamp(seconds, timezone.utc).replace(microsecond=microsecond)`.
    pub fn from_timestamp(seconds: i64, microsecond: u32) -> Self {
        let (year, month, day) = civil_from_days(seconds.div_euclid(86_400));
        let of_day = seconds.rem_euclid(86_400);
        Self {
            year,
            month,
            day,
            hour: (of_day / 3600) as u32,
            minute: (of_day / 60 % 60) as u32,
            second: (of_day % 60) as u32,
            microsecond,
        }
    }

    /// `datetime.fromtimestamp(at, timezone.utc)` for a Rust clock reading, its nanoseconds cut
    /// to whole microseconds.
    pub fn from_system_time(at: SystemTime) -> Self {
        let micros = match at.duration_since(UNIX_EPOCH) {
            Ok(after) => i64::try_from(after.as_micros()).unwrap_or(i64::MAX),
            Err(before) => i64::try_from(before.duration().as_micros()).map_or(i64::MIN, |m| -m),
        };
        Self::from_timestamp(
            micros.div_euclid(1_000_000),
            micros.rem_euclid(1_000_000) as u32,
        )
    }
}

/// Days since 1970-01-01 to a proleptic Gregorian `(year, month, day)`.
///
/// Howard Hinnant's `civil_from_days`, <https://howardhinnant.github.io/date_algorithms.html#civil_from_days>:
/// shifts the epoch to 0000-03-01 so the leap day ends the year, then counts 400-year eras
/// of 146097 days.
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let shifted = days + 719_468;
    let era = shifted.div_euclid(146_097);
    let day_of_era = shifted.rem_euclid(146_097);
    let year_of_era =
        (day_of_era - day_of_era / 1460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_index = (5 * day_of_year + 2) / 153;
    let day = (day_of_year - (153 * month_index + 2) / 5 + 1) as u32;
    let month = if month_index < 10 {
        month_index + 3
    } else {
        month_index - 9
    } as u32;
    let year = year_of_era + era * 400 + i64::from(month <= 2);
    (year, month, day)
}

/// Days since 1970-01-01 of a proleptic Gregorian date, the inverse of [`civil_from_days`].
fn days_from_civil(year: i64, month: u32, day: u32) -> i64 {
    let year = year - i64::from(month <= 2);
    let era = year.div_euclid(400);
    let year_of_era = year.rem_euclid(400);
    let month_index = i64::from(if month > 2 { month - 3 } else { month + 9 });
    let day_of_year = (153 * month_index + 2) / 5 + i64::from(day) - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    era * 146_097 + day_of_era - 719_468
}

fn is_leap(year: i64) -> bool {
    year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)
}

fn days_in_month(year: i64, month: u32) -> u32 {
    match month {
        2 if is_leap(year) => 29,
        2 => 28,
        4 | 6 | 9 | 11 => 30,
        _ => 31,
    }
}

/// A `datetime` as Python holds it, in microseconds since 1970-01-01: naive, or aware at its UTC
/// offset.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum IsoDateTime {
    /// A naive `datetime`, its fields read as UTC.
    Naive(i64),
    /// An aware `datetime`: its UTC instant and its offset.
    Aware { utc: i64, offset: i64 },
}

impl IsoDateTime {
    /// An aware UTC `datetime`, as a database read hands it out.
    pub fn utc(microseconds: i64) -> Self {
        Self::Aware {
            utc: microseconds,
            offset: 0,
        }
    }

    /// `datetime.fromtimestamp(seconds, timezone.utc)`, or `None` outside years 1 to 9999 where
    /// it raises.
    pub fn from_timestamp(seconds: f64) -> Option<Self> {
        Self::from_epoch(seconds, 1_000_000)
    }

    /// A UTC `datetime` `milliseconds` after 1970-01-01, or `None` outside years 1 to 9999.
    pub fn from_milliseconds(milliseconds: f64) -> Option<Self> {
        Self::from_epoch(milliseconds, 1_000)
    }

    fn from_epoch(count: f64, microseconds_each: i64) -> Option<Self> {
        const YEAR_1: i64 = -62_135_596_800_000_000;
        const YEAR_10000: i64 = 253_402_300_800_000_000;
        if !count.is_finite() {
            return None;
        }
        let whole = count.floor();
        let fraction = ((count - whole) * microseconds_each as f64).round() as i64;
        let microseconds = (whole as i64)
            .checked_mul(microseconds_each)?
            .checked_add(fraction)?;
        (YEAR_1..YEAR_10000)
            .contains(&microseconds)
            .then(|| Self::utc(microseconds))
    }

    /// The instant in microseconds since 1970-01-01, a naive `datetime` read as UTC.
    pub fn utc_microseconds(self) -> i64 {
        match self {
            Self::Naive(local) => local,
            Self::Aware { utc, .. } => utc,
        }
    }

    /// What pydantic's JSON mode writes: `isoformat()` with a fraction only when there is one,
    /// `Z` for a zero offset and none for a naive `datetime`.
    pub fn isoformat(self) -> String {
        let (local, offset) = match self {
            Self::Naive(local) => (local, None),
            Self::Aware { utc, offset } => (utc + offset, Some(offset)),
        };
        let at = UtcDateTime::from_timestamp(
            local.div_euclid(1_000_000),
            local.rem_euclid(1_000_000) as u32,
        );
        let fraction = match at.microsecond {
            0 => String::new(),
            microsecond => format!(".{microsecond:06}"),
        };
        let zone = match offset {
            None => String::new(),
            Some(0) => "Z".to_owned(),
            Some(offset) => {
                let sign = if offset < 0 { '-' } else { '+' };
                let seconds = offset.abs() / 1_000_000;
                let (hours, minutes, rest) = (seconds / 3600, seconds / 60 % 60, seconds % 60);
                match rest {
                    0 => format!("{sign}{hours:02}:{minutes:02}"),
                    rest => format!("{sign}{hours:02}:{minutes:02}:{rest:02}"),
                }
            }
        };
        format!(
            "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}{fraction}{zone}",
            at.year, at.month, at.day, at.hour, at.minute, at.second
        )
    }
}

/// `datetime.fromisoformat(text)` as CPython 3.14's C parser reads it, or `None` where it
/// raises `ValueError`. Python before 3.11 accepts only `isoformat()` output.
///
/// This follows `Modules/_datetimemodule.c` rather than the documented grammar, quirks
/// included: any one character separates date from time, one stray character may sit
/// between the time and its offset, and `T24:00` rolls over to the next midnight.
pub fn fromisoformat(text: &str) -> Option<IsoDateTime> {
    if text.chars().count() < 7 {
        return None;
    }
    let bytes = Bytes(text.as_bytes());
    let separator = bytes.separator();
    let (mut year, mut month, mut day) = bytes.date(separator)?;
    let mut time = Clock::default();
    let mut offset = None;
    if bytes.len() > separator {
        let lead = bytes.at(separator);
        let width = match lead {
            _ if lead & 0x80 == 0 => 1,
            _ if lead & 0xf0 == 0xe0 => 3,
            _ if lead & 0xf0 == 0xf0 => 4,
            _ => 2,
        };
        let start = separator + width;
        if start > bytes.len() {
            return None;
        }
        (time, offset) = bytes.time(start)?;
    }
    if time.hour == 24 && (1..=12).contains(&month) && day <= days_in_month(year, month) {
        if time.minute != 0 || time.second != 0 || time.microsecond != 0 {
            return None;
        }
        time.hour = 0;
        day += 1;
        if day > days_in_month(year, month) {
            day = 1;
            month += 1;
            if month > 12 {
                month = 1;
                year += 1;
            }
        }
    }
    if !(1..=9999).contains(&year)
        || !(1..=12).contains(&month)
        || !(1..=days_in_month(year, month)).contains(&day)
        || time.hour > 23
        || time.minute > 59
        || time.second > 59
    {
        return None;
    }
    let seconds = days_from_civil(year, month, day) * 86_400
        + i64::from(time.hour) * 3600
        + i64::from(time.minute) * 60
        + i64::from(time.second);
    let local = seconds * 1_000_000 + i64::from(time.microsecond);
    Some(match offset {
        None => IsoDateTime::Naive(local),
        Some(offset) if offset.abs() >= 86_400_000_000 => return None,
        Some(offset) => IsoDateTime::Aware {
            utc: local - offset,
            offset,
        },
    })
}

#[derive(Clone, Copy, Default)]
struct Clock {
    hour: u32,
    minute: u32,
    second: u32,
    microsecond: u32,
}

/// Where `parse_hh_mm_ss_ff` stopped: at the end of the input, or with more after it.
enum Rest {
    End,
    More,
}

/// The UTF-8 bytes C reads, with the NUL terminator it relies on past the end.
struct Bytes<'a>(&'a [u8]);

impl Bytes<'_> {
    fn len(&self) -> usize {
        self.0.len()
    }

    fn at(&self, index: usize) -> u8 {
        self.0.get(index).copied().unwrap_or(0)
    }

    fn digits(&self, at: usize, count: usize) -> Option<u32> {
        (at..at + count).try_fold(0, |value, index| {
            let digit = self.at(index).wrapping_sub(b'0');
            (digit <= 9).then_some(value * 10 + u32::from(digit))
        })
    }

    /// `_find_isoformat_datetime_separator`: where the date ends, from its shape.
    fn separator(&self) -> usize {
        let len = self.len();
        if len == 7 {
            return 7;
        }
        if self.at(4) == b'-' {
            if self.at(5) != b'W' {
                return 10;
            }
            if len > 8 && self.at(8) == b'-' {
                if len == 9 {
                    return usize::MAX;
                }
                return if len > 10 && self.at(10).is_ascii_digit() {
                    8
                } else {
                    10
                };
            }
            return 8;
        }
        if self.at(4) != b'W' {
            return 8;
        }
        let digits_end = (7..len)
            .find(|&index| !self.at(index).is_ascii_digit())
            .unwrap_or(len);
        match digits_end {
            end if end < 9 => end,
            end if end % 2 == 0 => 7,
            _ => 8,
        }
    }

    /// `parse_isoformat_date`: calendar `YYYY-MM-DD` or ISO week `YYYY-Www-D`, unvalidated.
    fn date(&self, len: usize) -> Option<(i64, u32, u32)> {
        let year = self.digits(0, 4)?;
        let dashed = self.at(4) == b'-';
        let mut at = 4 + usize::from(dashed);
        if self.at(at) == b'W' {
            let week = self.digits(at + 1, 2)?;
            at += 3;
            let weekday = if at < len {
                if dashed {
                    if self.at(at) != b'-' {
                        return None;
                    }
                    at += 1;
                }
                self.digits(at, 1)?
            } else {
                1
            };
            return iso_week_date(i64::from(year), week, weekday);
        }
        let month = self.digits(at, 2)?;
        at += 2;
        if dashed {
            if self.at(at) != b'-' {
                return None;
            }
            at += 1;
        }
        Some((i64::from(year), month, self.digits(at, 2)?))
    }

    /// `parse_isoformat_time`: the clock, and the offset in microseconds when there is one.
    fn time(&self, start: usize) -> Option<(Clock, Option<i64>)> {
        let end = self.len();
        let zone = (start..end)
            .find(|&index| matches!(self.at(index), b'Z' | b'+' | b'-'))
            .unwrap_or(if start == end { start + 1 } else { end });
        let (clock, rest) = self.clock(start, zone)?;
        if zone >= end {
            return matches!(rest, Rest::End).then_some((clock, None));
        }
        if self.at(zone) == b'Z' {
            return (self.at(zone + 1) == 0).then_some((clock, Some(0)));
        }
        let sign = if self.at(zone) == b'-' { -1 } else { 1 };
        let (offset, rest) = self.clock(zone + 1, end)?;
        if matches!(rest, Rest::More) {
            return None;
        }
        let seconds = i64::from(offset.hour) * 3600
            + i64::from(offset.minute) * 60
            + i64::from(offset.second);
        Some((
            clock,
            Some(sign * (seconds * 1_000_000 + i64::from(offset.microsecond))),
        ))
    }

    /// `parse_hh_mm_ss_ff`: `HH[:?MM[:?SS[{.,}f+]]]` from `start` up to `end`.
    fn clock(&self, start: usize, end: usize) -> Option<(Clock, Rest)> {
        let mut fields = [0; 3];
        let mut at = start;
        let mut colons = true;
        for (index, field) in fields.iter_mut().enumerate() {
            *field = self.digits(at, 2)?;
            let next = self.at(at + 2);
            at += 3;
            if index == 0 {
                colons = next == b':';
            }
            if matches!(next, b'.' | b',') {
                if index < 2 || at >= end {
                    return None;
                }
                break;
            } else if at >= end {
                let [hour, minute, second] = fields;
                let clock = Clock {
                    hour,
                    minute,
                    second,
                    microsecond: 0,
                };
                return Some((clock, if next == 0 { Rest::End } else { Rest::More }));
            } else if colons && next == b':' {
                if index == 2 {
                    return None;
                }
            } else if !colons {
                at -= 1;
            } else {
                return None;
            }
        }
        let count = (end - at).min(6);
        let mut microsecond = self.digits(at, count)?;
        microsecond *= 10_u32.pow(u32::try_from(6 - count).ok()?);
        at += count;
        while self.at(at).is_ascii_digit() {
            at += 1;
        }
        let [hour, minute, second] = fields;
        let clock = Clock {
            hour,
            minute,
            second,
            microsecond,
        };
        Some((
            clock,
            if self.at(at) == 0 {
                Rest::End
            } else {
                Rest::More
            },
        ))
    }
}

/// `iso_to_ymd`: the calendar date of ISO year, week and weekday, as `date.fromisocalendar`.
fn iso_week_date(year: i64, week: u32, weekday: u32) -> Option<(i64, u32, u32)> {
    if !(1..=9999).contains(&year) {
        return None;
    }
    let first = days_from_civil(year, 1, 1);
    // 0 for Monday, as `date.weekday()`; 1970-01-01 was a Thursday.
    let first_weekday = (first + 3).rem_euclid(7);
    let long_year = first_weekday == 3 || (first_weekday == 2 && is_leap(year));
    if !(1..=52).contains(&week) && !(week == 53 && long_year) {
        return None;
    }
    if !(1..=7).contains(&weekday) {
        return None;
    }
    let week_one_monday = first - first_weekday + if first_weekday > 3 { 7 } else { 0 };
    Some(civil_from_days(
        week_one_monday + i64::from((week - 1) * 7 + weekday - 1),
    ))
}

/// `(end.timestamp() - start.timestamp()) * 1000` for `elapsed`, the time from `start` to `end`:
/// its milliseconds, held to the microsecond a `datetime` keeps.
pub fn milliseconds(elapsed: Duration) -> f64 {
    let micros = u64::try_from(elapsed.as_micros()).unwrap_or(u64::MAX);
    #[expect(
        clippy::cast_precision_loss,
        reason = "a call's duration in microseconds is far below 2^53"
    )]
    let micros = micros as f64;
    micros / 1000.0
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;
    use crate::{number::round, repr::float_repr};

    #[rstest]
    #[case::whole_milliseconds(Duration::from_millis(1500), "1500.0")]
    #[case::a_fraction(Duration::from_micros(1_250_500), "1250.5")]
    #[case::nanoseconds_a_datetime_does_not_keep(Duration::from_nanos(1_250_500_999), "1250.5")]
    fn a_duration_is_pythons_str_of_its_milliseconds(
        #[case] elapsed: Duration,
        #[case] expected: &str,
    ) {
        assert_eq!(float_repr(milliseconds(elapsed)), expected);
    }

    #[rstest]
    #[case::kept_to_four_places(12.345_678, "12.3457")]
    #[case::whole(3.0, "3.0")]
    #[case::none_spent(0.0, "0.0")]
    fn a_rounded_duration_keeps_four_places(#[case] ms: f64, #[case] expected: &str) {
        assert_eq!(float_repr(round(ms, 4)), expected);
    }
}
