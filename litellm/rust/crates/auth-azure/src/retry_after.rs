//! Retry-After as azure-core 1.41.0 reads it (`pipeline/policies/_utils.py`): seconds through
//! Python's `float`, otherwise an HTTP date through `email.utils.parsedate_tz`, and the wait
//! handed to `time.sleep`.

use azure_core::http::headers::{
    HeaderName, Headers, RETRY_AFTER, RETRY_AFTER_MS, X_MS_RETRY_AFTER_MS,
};
use azure_core::time::{Duration, OffsetDateTime};
use time::{Date, Month, PrimitiveDateTime, Time, UtcOffset};

const HEADERS: [(HeaderName, f64); 3] = [
    (RETRY_AFTER, 1.0),
    (RETRY_AFTER_MS, 0.001),
    (X_MS_RETRY_AFTER_MS, 0.001),
];
const MONTHS: [&str; 24] = [
    "jan",
    "feb",
    "mar",
    "apr",
    "may",
    "jun",
    "jul",
    "aug",
    "sep",
    "oct",
    "nov",
    "dec",
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
];
const DAYS: [&str; 7] = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"];
const ZONES: [(&str, i128); 14] = [
    ("UT", 0),
    ("UTC", 0),
    ("GMT", 0),
    ("Z", 0),
    ("AST", -400),
    ("ADT", -300),
    ("EST", -500),
    ("EDT", -400),
    ("CST", -600),
    ("CDT", -500),
    ("MST", -700),
    ("MDT", -600),
    ("PST", -800),
    ("PDT", -700),
];
const SLEEP_LIMIT_NANOS: f64 = 9_223_372_036_854_775_808.0;

#[derive(Debug, PartialEq)]
pub(crate) enum RetryAfter {
    Backoff,
    Wait(Duration),
    Unparseable,
    TooLong,
}

pub(crate) fn retry_after(headers: &Headers, now: OffsetDateTime) -> RetryAfter {
    let Some((value, unit)) = HEADERS.iter().find_map(|(name, unit)| {
        headers
            .get_optional_str(name)
            .filter(|value| !value.is_empty())
            .map(|value| (value, *unit))
    }) else {
        return RetryAfter::Backoff;
    };
    let Some(seconds) = python_float(value).or_else(|| http_date_delay(value, now)) else {
        return RetryAfter::Unparseable;
    };
    let seconds = if seconds > 0.0 { seconds * unit } else { 0.0 };
    if seconds == 0.0 {
        return RetryAfter::Backoff;
    }
    if seconds * 1e9 >= SLEEP_LIMIT_NANOS {
        return RetryAfter::TooLong;
    }
    RetryAfter::Wait(Duration::seconds_f64(seconds))
}

fn http_date_delay(value: &str, now: OffsetDateTime) -> Option<f64> {
    let date = parsedate_tz(value)?;
    Some((date.at()? - now).as_seconds_f64())
}

struct ParsedDate {
    year: i128,
    month: u8,
    day: i128,
    hour: i128,
    minute: i128,
    second: i128,
    offset: i128,
}

impl ParsedDate {
    /// `datetime(*parsed[:6], tzinfo=_FixedOffset(offset / 60))`, which raises for a field out of
    /// range, and an offset `datetime` refuses once the date is compared.
    fn at(&self) -> Option<OffsetDateTime> {
        if !(1..=9999).contains(&self.year) || self.offset.abs() >= 86_400 {
            return None;
        }
        let date = Date::from_calendar_date(
            i32::try_from(self.year).ok()?,
            Month::try_from(self.month).ok()?,
            u8::try_from(self.day).ok()?,
        )
        .ok()?;
        let time = Time::from_hms(
            u8::try_from(self.hour).ok()?,
            u8::try_from(self.minute).ok()?,
            u8::try_from(self.second).ok()?,
        )
        .ok()?;
        let offset = UtcOffset::from_whole_seconds(i32::try_from(self.offset).ok()?).ok()?;
        Some(PrimitiveDateTime::new(date, time).assume_offset(offset))
    }
}

/// `email._parseaddr._parsedate_tz` with `parsedate_tz`'s unknown zone read as UTC.
fn parsedate_tz(data: &str) -> Option<ParsedDate> {
    let mut data: Vec<String> = data.split_whitespace().map(str::to_string).collect();
    let first = data.first()?;
    if first.ends_with(',') || DAYS.contains(&first.to_lowercase().as_str()) {
        data.remove(0);
    } else if let Some(comma) = first.rfind(',') {
        data[0] = first[comma + 1..].to_string();
    }
    if data.len() == 3 {
        let stuff: Vec<String> = data[0].split('-').map(str::to_string).collect();
        if stuff.len() == 3 {
            data = stuff.into_iter().chain(data[1..].iter().cloned()).collect();
        }
    }
    if data.len() == 4 {
        let s = data[3].clone();
        let sign = s.find('+').or_else(|| s.find('-'));
        match sign {
            Some(i) if i > 0 => {
                data.truncate(3);
                data.push(s[..i].to_string());
                data.push(s[i..].to_string());
            }
            _ => data.push(String::new()),
        }
    }
    if data.len() < 5 {
        return None;
    }
    let [mut dd, mut mm, mut yy, mut tm, mut tz] =
        <[String; 5]>::try_from(data[..5].to_vec()).ok()?;
    if dd.is_empty() || mm.is_empty() || yy.is_empty() {
        return None;
    }
    mm = mm.to_lowercase();
    if !MONTHS.contains(&mm.as_str()) {
        (dd, mm) = (mm, dd.to_lowercase());
        if !MONTHS.contains(&mm.as_str()) {
            return None;
        }
    }
    let month = MONTHS.iter().position(|name| *name == mm)? + 1;
    let month = u8::try_from(if month > 12 { month - 12 } else { month }).ok()?;
    if dd.ends_with(',') {
        dd.pop();
    }
    if yy.find(':').is_some_and(|i| i > 0) {
        (yy, tm) = (tm, yy);
    }
    if yy.ends_with(',') {
        yy.pop();
        if yy.is_empty() {
            return None;
        }
    }
    if !yy.chars().next().is_some_and(|c| c.is_ascii_digit()) {
        (yy, tz) = (tz, yy);
    }
    if tm.ends_with(',') {
        tm.pop();
    }
    let parts: Vec<&str> = tm.split(':').collect();
    let (thh, tmm, tss) = match parts.as_slice() {
        [thh, tmm] => (*thh, *tmm, "0"),
        [thh, tmm, tss] => (*thh, *tmm, *tss),
        [single] if single.contains('.') => match single.split('.').collect::<Vec<_>>()[..] {
            [thh, tmm] => (thh, tmm, "0"),
            [thh, tmm, tss] => (thh, tmm, tss),
            _ => return None,
        },
        _ => return None,
    };
    let mut year = python_int(&yy)?;
    let day = python_int(&dd)?;
    let hour = python_int(thh)?;
    let minute = python_int(tmm)?;
    let second = python_int(tss)?;
    if year < 100 {
        year += if year > 68 { 1900 } else { 2000 };
    }
    let tz = tz.to_uppercase();
    let zone = ZONES
        .iter()
        .find(|(name, _)| *name == tz)
        .map(|(_, offset)| *offset)
        .or_else(|| python_int(&tz).filter(|offset| *offset != 0 || !tz.starts_with('-')))
        .unwrap_or(0);
    let offset = zone.signum() * ((zone.abs() / 100) * 3600 + (zone.abs() % 100) * 60);
    Some(ParsedDate {
        year,
        month,
        day,
        hour,
        minute,
        second,
        offset,
    })
}

/// Python's `int(text)` for a token without whitespace: an optional sign and decimal digits,
/// with single underscores between digits. A value too large for `i128` saturates, which every
/// caller then rejects as Python's `datetime` rejects the big integer.
fn python_int(text: &str) -> Option<i128> {
    let (negative, digits) = match text.as_bytes().first()? {
        b'-' => (true, &text[1..]),
        b'+' => (false, &text[1..]),
        _ => (false, text),
    };
    let digits = without_underscores(digits)?;
    if digits.is_empty() || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    let magnitude = digits.bytes().fold(0_i128, |value, byte| {
        value
            .saturating_mul(10)
            .saturating_add(i128::from(byte - b'0'))
    });
    Some(if negative { -magnitude } else { magnitude })
}

/// Python's `float(text)`, whose underscores may only sit between digits.
fn python_float(text: &str) -> Option<f64> {
    let value = without_underscores(text.trim())?.parse::<f64>().ok()?;
    Some(if value.is_nan() { 0.0 } else { value })
}

fn without_underscores(text: &str) -> Option<String> {
    let bytes = text.as_bytes();
    let separated = bytes.iter().enumerate().all(|(i, byte)| {
        *byte != b'_'
            || (i > 0
                && bytes[i - 1].is_ascii_digit()
                && bytes.get(i + 1).is_some_and(u8::is_ascii_digit))
    });
    separated.then(|| text.replace('_', ""))
}

#[cfg(test)]
mod tests {
    use super::*;
    use azure_core::http::headers::Headers;

    fn now() -> OffsetDateTime {
        OffsetDateTime::from_unix_timestamp(1_893_456_000).unwrap()
    }

    fn waited(name: &'static str, value: &str) -> RetryAfter {
        let mut headers = Headers::new();
        headers.insert(HeaderName::from_static(name), value.to_string());
        retry_after(&headers, now())
    }

    // Each expectation is what azure-core 1.41.0 on CPython 3.12.3 does with the value, checked
    // 2026-11-13: the seconds it sleeps, a fall back to its backoff, or the exception that ends the
    // token request.
    #[rstest::rstest]
    #[case::seconds("retry-after", "7", RetryAfter::Wait(Duration::seconds(7)))]
    #[case::fractional_seconds(
        "retry-after",
        "1.5",
        RetryAfter::Wait(Duration::milliseconds(1500))
    )]
    #[case::underscored_seconds("retry-after", "1_0", RetryAfter::Wait(Duration::seconds(10)))]
    #[case::misplaced_underscore("retry-after", "_10", RetryAfter::Unparseable)]
    #[case::milliseconds(
        "retry-after-ms",
        "2500",
        RetryAfter::Wait(Duration::milliseconds(2500))
    )]
    #[case::zero_falls_back("retry-after", "0", RetryAfter::Backoff)]
    #[case::negative_falls_back("retry-after", "-3", RetryAfter::Backoff)]
    #[case::not_a_number_falls_back("retry-after", "nan", RetryAfter::Backoff)]
    #[case::negative_infinity_falls_back("retry-after", "-inf", RetryAfter::Backoff)]
    #[case::infinity_cannot_be_slept("retry-after", "inf", RetryAfter::TooLong)]
    #[case::past_the_sleep_limit("retry-after", "9223372036.854776", RetryAfter::TooLong)]
    #[case::just_under_the_sleep_limit(
        "retry-after",
        "9223372036",
        RetryAfter::Wait(Duration::seconds(9_223_372_036))
    )]
    #[case::infinite_milliseconds("x-ms-retry-after-ms", "1e400", RetryAfter::TooLong)]
    #[case::rfc_7231_date(
        "retry-after",
        "Tue, 01 Jan 2030 00:00:30 GMT",
        RetryAfter::Wait(Duration::seconds(30))
    )]
    #[case::no_weekday(
        "retry-after",
        "01 Jan 2030 00:00:30 GMT",
        RetryAfter::Wait(Duration::seconds(30))
    )]
    #[case::two_digit_year(
        "retry-after",
        "01 Jan 30 00:00:30 GMT",
        RetryAfter::Wait(Duration::seconds(30))
    )]
    #[case::numeric_zone(
        "retry-after",
        "Tue, 01 Jan 2030 01:00:30 +0100",
        RetryAfter::Wait(Duration::seconds(30))
    )]
    #[case::named_zone(
        "retry-after",
        "31 Dec 2029 19:00:30 EST",
        RetryAfter::Wait(Duration::seconds(30))
    )]
    #[case::zone_glued_to_time(
        "retry-after",
        "01 Jan 2030 01:00:30+0100",
        RetryAfter::Wait(Duration::seconds(30))
    )]
    #[case::unknown_zone_is_utc(
        "retry-after",
        "01 Jan 2030 00:00:30 XYZ",
        RetryAfter::Wait(Duration::seconds(30))
    )]
    #[case::missing_seconds(
        "retry-after",
        "01 Jan 2030 00:01 GMT",
        RetryAfter::Wait(Duration::seconds(60))
    )]
    #[case::dotted_time(
        "retry-after",
        "01 Jan 2030 00.01 GMT",
        RetryAfter::Wait(Duration::seconds(60))
    )]
    #[case::month_before_day(
        "retry-after",
        "Jan 01 2030 00:01 GMT",
        RetryAfter::Wait(Duration::seconds(60))
    )]
    #[case::rfc_850_date(
        "retry-after",
        "Tuesday, 01-Jan-30 00:00:30 GMT",
        RetryAfter::Wait(Duration::seconds(30))
    )]
    #[case::past_date_falls_back("retry-after", "01 Jan 2029 00:00:00 GMT", RetryAfter::Backoff)]
    #[case::day_out_of_range("retry-after", "31 Feb 2030 00:00:00 GMT", RetryAfter::Unparseable)]
    #[case::offset_past_a_day("retry-after", "01 Jan 2030 00:00:00 +2400", RetryAfter::Unparseable)]
    #[case::words("retry-after", "soon", RetryAfter::Unparseable)]
    fn retry_after_is_read_as_azure_core_reads_it(
        #[case] name: &'static str,
        #[case] value: &str,
        #[case] expected: RetryAfter,
    ) {
        assert_eq!(waited(name, value), expected);
    }
}
