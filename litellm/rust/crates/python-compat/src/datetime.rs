//! The fields of `datetime.fromtimestamp(seconds, timezone.utc)`.

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
