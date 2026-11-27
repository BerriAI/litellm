use std::time::{SystemTime, UNIX_EPOCH};

use litellm_python_compat::datetime::{IsoDateTime, fromisoformat};

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

/// A reservation entry's `window_reset_at`, read as the proxy's `_coerce_datetime` reads it:
/// `datetime.fromisoformat(text.replace("Z", "+00:00"))`, `None` where that raises.
pub(crate) fn parse_reset(text: &str) -> Result<Option<Moment>, ReservationError> {
    match fromisoformat(&text.replace('Z', "+00:00")) {
        None => Ok(None),
        Some(IsoDateTime::Aware { utc: micros, .. }) => Ok(Some(Moment(i128::from(micros)))),
        Some(IsoDateTime::Naive(_)) => Err(ReservationError::type_error(
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
    #[case::bad_hour("2025-11-13T25:00:00+00:00", None)]
    #[case::midnight_rollover("2025-11-13T24:00:00+00:00", Some(NOON + 43_200_000_000))]
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
