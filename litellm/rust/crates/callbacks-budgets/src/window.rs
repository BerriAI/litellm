use std::time::{SystemTime, UNIX_EPOCH};

use chrono::{NaiveDate, NaiveTime};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ErrorKind {
    Type,
    Value,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReservationError {
    pub kind: ErrorKind,
    pub message: String,
}

impl ReservationError {
    pub(crate) fn type_error(message: impl Into<String>) -> Self {
        Self {
            kind: ErrorKind::Type,
            message: message.into(),
        }
    }

    pub(crate) fn value_error(message: impl Into<String>) -> Self {
        Self {
            kind: ErrorKind::Value,
            message: message.into(),
        }
    }
}

impl std::fmt::Display for ReservationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Moment(i128);

impl Moment {
    pub fn now() -> Self {
        let since = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default();
        Self(i128::try_from(since.as_micros()).unwrap_or(i128::MAX))
    }

    pub const fn from_unix_micros(micros: i128) -> Self {
        Self(micros)
    }
}

struct Cursor<'a> {
    text: &'a [u8],
    at: usize,
}

impl Cursor<'_> {
    fn digits(&mut self, count: usize) -> Option<u32> {
        let slice = self.text.get(self.at..self.at + count)?;
        if !slice.iter().all(u8::is_ascii_digit) {
            return None;
        }
        self.at += count;
        Some(
            slice
                .iter()
                .fold(0, |value, digit| value * 10 + u32::from(digit - b'0')),
        )
    }

    fn eat(&mut self, byte: u8) -> bool {
        if self.text.get(self.at) == Some(&byte) {
            self.at += 1;
            return true;
        }
        false
    }

    fn peek(&self) -> Option<u8> {
        self.text.get(self.at).copied()
    }

    fn done(&self) -> bool {
        self.at == self.text.len()
    }
}

struct Clock {
    seconds: u32,
    micros: u32,
}

fn fraction(cursor: &mut Cursor<'_>) -> Option<u32> {
    if !(cursor.eat(b'.') || cursor.eat(b',')) {
        return Some(0);
    }
    let start = cursor.at;
    while cursor.peek().is_some_and(|byte| byte.is_ascii_digit()) {
        cursor.at += 1;
    }
    let digits = &cursor.text[start..cursor.at];
    if digits.is_empty() {
        return None;
    }
    let kept = &digits[..digits.len().min(6)];
    let value = kept
        .iter()
        .fold(0, |value, digit| value * 10 + u32::from(digit - b'0'));
    Some(value * 10_u32.pow(u32::try_from(6 - kept.len()).ok()?))
}

fn clock(cursor: &mut Cursor<'_>, offset: bool) -> Option<Clock> {
    let hours = cursor.digits(2)?;
    let extended = cursor.peek() == Some(b':');
    let mut minutes = 0;
    let mut seconds = 0;
    let mut micros = 0;
    let has_more = |cursor: &Cursor<'_>| {
        if extended {
            cursor.peek() == Some(b':')
        } else {
            cursor.peek().is_some_and(|byte| byte.is_ascii_digit())
        }
    };
    if has_more(cursor) {
        let _ = cursor.eat(b':');
        minutes = cursor.digits(2)?;
        if has_more(cursor) {
            let _ = cursor.eat(b':');
            seconds = cursor.digits(2)?;
            micros = fraction(cursor)?;
        }
    } else if !offset && matches!(cursor.peek(), Some(b'.' | b',')) {
        return None;
    }
    if hours > 23 || minutes > 59 || seconds > 59 {
        return None;
    }
    Some(Clock {
        seconds: hours * 3600 + minutes * 60 + seconds,
        micros,
    })
}

fn date(cursor: &mut Cursor<'_>) -> Option<NaiveDate> {
    let year = cursor.digits(4)?;
    let extended = cursor.eat(b'-');
    let month = cursor.digits(2)?;
    if extended && !cursor.eat(b'-') {
        return None;
    }
    let day = cursor.digits(2)?;
    NaiveDate::from_ymd_opt(i32::try_from(year).ok()?, month, day)
}

enum Parsed {
    Aware(Moment),
    Naive,
}

fn parse(text: &str) -> Option<Parsed> {
    let mut cursor = Cursor {
        text: text.as_bytes(),
        at: 0,
    };
    let day = date(&mut cursor)?;
    let days = i128::from(
        day.signed_duration_since(NaiveDate::from_ymd_opt(1970, 1, 1)?)
            .num_days(),
    );
    if cursor.done() {
        return Some(Parsed::Naive);
    }
    let separator = text[cursor.at..].chars().next()?;
    cursor.at += separator.len_utf8();
    let time = clock(&mut cursor, false)?;
    NaiveTime::from_num_seconds_from_midnight_opt(time.seconds, time.micros * 1000)?;
    let local = (days * 86_400 + i128::from(time.seconds)) * 1_000_000 + i128::from(time.micros);
    if cursor.done() {
        return Some(Parsed::Naive);
    }
    let sign: i128 = match cursor.peek()? {
        b'Z' => {
            cursor.at += 1;
            return cursor.done().then_some(Parsed::Aware(Moment(local)));
        }
        b'+' => 1,
        b'-' => -1,
        _ => return None,
    };
    cursor.at += 1;
    let offset = clock(&mut cursor, true)?;
    if !cursor.done() {
        return None;
    }
    let offset_micros = sign * (i128::from(offset.seconds) * 1_000_000 + i128::from(offset.micros));
    Some(Parsed::Aware(Moment(local - offset_micros)))
}

pub(crate) fn parse_reset(text: &str) -> Result<Option<Moment>, ReservationError> {
    match parse(&text.replace('Z', "+00:00")) {
        None => Ok(None),
        Some(Parsed::Aware(moment)) => Ok(Some(moment)),
        Some(Parsed::Naive) => Err(ReservationError::type_error(
            "can't compare offset-naive and offset-aware datetimes",
        )),
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    const NOON: i128 = 1_763_035_200_000_000;

    #[rstest]
    #[case::isoformat("2025-11-13T12:00:00+00:00", Some(NOON))]
    #[case::zulu("2025-11-13T12:00:00Z", Some(NOON))]
    #[case::micros("2025-11-13T12:00:00.000001+00:00", Some(NOON + 1))]
    #[case::short_fraction("2025-11-13T12:00:00.5+00:00", Some(NOON + 500_000))]
    #[case::long_fraction("2025-11-13T12:00:00.1234567+00:00", Some(NOON + 123_456))]
    #[case::offset("2025-11-13T13:30:00+01:30", Some(NOON))]
    #[case::negative_offset("2025-11-13T11:00:00-01:00", Some(NOON))]
    #[case::space_separator("2025-11-13 12:00+00:00", Some(NOON))]
    #[case::basic("20251113T120000+0000", Some(NOON))]
    #[case::not_a_date("soon", None)]
    #[case::bad_day("2025-02-30T12:00:00+00:00", None)]
    #[case::bad_hour("2025-11-13T24:00:00+00:00", None)]
    #[case::trailing("2025-11-13T12:00:00+00:00 ", None)]
    fn an_aware_reset_parses_as_python_fromisoformat_does(
        #[case] text: &str,
        #[case] expected: Option<i128>,
    ) {
        assert_eq!(
            parse_reset(text).unwrap(),
            expected.map(Moment::from_unix_micros)
        );
    }

    #[rstest]
    #[case::date_only("2025-11-13")]
    #[case::no_offset("2025-11-13T12:00:00")]
    fn a_naive_reset_raises_the_comparison_type_error(#[case] text: &str) {
        assert_eq!(
            parse_reset(text).unwrap_err(),
            ReservationError::type_error("can't compare offset-naive and offset-aware datetimes")
        );
    }
}
