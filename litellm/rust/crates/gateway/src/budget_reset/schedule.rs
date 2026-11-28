//! Calendar and elapsed schedules used by the budget reset job.

use jiff::{
    SignedDuration, Span, Timestamp,
    civil::{Date, DateTime, Time},
    tz::TimeZone,
};

/// The configured timezone and wall time for calendar resets.
#[derive(Clone)]
pub(super) struct Schedule {
    pub(super) timezone: TimeZone,
    pub(super) wall_time: Time,
}

enum Period {
    Elapsed(i64),
    Days(i64),
    Midnights(i64),
    Months(i64),
}

impl Schedule {
    pub(super) fn advance_lifetime(
        &self,
        duration: &str,
        previous: Option<Timestamp>,
        now: Timestamp,
    ) -> Option<(Timestamp, Timestamp)> {
        if let Some(previous) = previous {
            return self.advance(duration, previous, now);
        }
        let local = now.to_zoned(self.timezone.clone());
        let time = match duration.trim().chars().last()? {
            's' => Time::new(local.hour(), local.minute(), 0, 0).ok()?,
            'm' => Time::new(local.hour(), 0, 0, 0).ok()?,
            _ => Time::MIN,
        };
        self.advance(duration, self.at(local.date(), time)?, now)
    }

    pub(super) fn advance(
        &self,
        duration: &str,
        previous: Timestamp,
        now: Timestamp,
    ) -> Option<(Timestamp, Timestamp)> {
        if previous > now {
            return None;
        }
        let date = now.to_zoned(self.timezone.clone()).date();
        let (end, start) = match self.period(duration)? {
            Period::Elapsed(seconds) => {
                let interval = i128::from(seconds) * 1_000_000_000;
                let elapsed = now.duration_since(previous).as_nanos();
                let remaining = i64::try_from(interval - elapsed % interval).ok()?;
                let end = now
                    .checked_add(SignedDuration::from_nanos(remaining))
                    .ok()?;
                (
                    end,
                    end.checked_sub(SignedDuration::from_secs(seconds)).ok()?,
                )
            }
            Period::Days(days) => {
                let time = self.wall_time;
                let end_date = match days {
                    1 => self.next(date, time, now, Span::new().days(1))?,
                    7 => {
                        let weekday = i64::from(date.weekday().to_monday_zero_offset());
                        let monday = date.checked_add(Span::new().days((7 - weekday) % 7)).ok()?;
                        self.next(monday, time, now, Span::new().days(7))?
                    }
                    _ => date.checked_add(Span::new().try_days(days).ok()?).ok()?,
                };
                let start_date = end_date
                    .checked_sub(Span::new().try_days(days).ok()?)
                    .ok()?;
                (self.at(end_date, time)?, self.at(start_date, time)?)
            }
            Period::Midnights(days) => {
                let end_date = date.checked_add(Span::new().try_days(days).ok()?).ok()?;
                (self.at(end_date, Time::MIN)?, self.at(date, Time::MIN)?)
            }
            Period::Months(months) => {
                let time = self.wall_time;
                let span = Span::new().try_months(months).ok()?;
                let anchor = if months == 12 {
                    date.first_of_year()
                } else {
                    date.first_of_month()
                };
                let end_date = self.next(anchor, time, now, span)?;
                (
                    self.at(end_date, time)?,
                    self.at(end_date.checked_sub(span).ok()?, time)?,
                )
            }
        };
        let start = Timestamp::from_millisecond(start.max(previous).as_millisecond()).ok()?;
        (end > now).then_some((end, start))
    }

    fn period(&self, raw: &str) -> Option<Period> {
        let duration = match raw.trim().to_ascii_lowercase().as_str() {
            "hourly" => "1h",
            "daily" => "24h",
            "weekly" => "7d",
            "monthly" => "30d",
            _ => raw,
        };
        let (count, unit) = duration.split_at(duration.find(|c: char| !c.is_ascii_digit())?);
        let count: i64 = count.parse().ok()?;
        if count == 0 {
            return None;
        }
        Some(match unit {
            "s" => Period::Elapsed(count),
            "m" => Period::Elapsed(count.checked_mul(60)?),
            "h" if count % 24 == 0 => Period::Midnights(count / 24),
            "h" => Period::Elapsed(count.checked_mul(3_600)?),
            "d" if count == 30 => Period::Months(1),
            "d" => Period::Days(count),
            "w" => Period::Days(count.checked_mul(7)?),
            "mo" if matches!(count, 1 | 12) => Period::Months(count),
            _ => return None,
        })
    }

    fn at(&self, date: Date, time: Time) -> Option<Timestamp> {
        Some(
            date.to_datetime(time)
                .to_zoned(self.timezone.clone())
                .ok()?
                .timestamp(),
        )
    }

    fn next(&self, date: Date, time: Time, now: Timestamp, period: Span) -> Option<Date> {
        if self.at(date, time)? > now {
            Some(date)
        } else {
            date.checked_add(period).ok()
        }
    }
}

pub(super) fn stored_timestamp(value: &str) -> Option<Timestamp> {
    value.parse().ok().or_else(|| {
        Some(
            value
                .parse::<DateTime>()
                .ok()?
                .to_zoned(TimeZone::UTC)
                .ok()?
                .timestamp(),
        )
    })
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::elapsed_anchor(
        "30m",
        "UTC",
        "2025-01-02T03:02:01Z",
        "2025-01-02T04:10:00Z",
        "2025-01-02T04:32:01Z",
        "2025-01-02T04:02:01Z"
    )]
    #[case::spring_forward(
        "1d",
        "America/New_York",
        "2025-03-09T07:30:00Z",
        "2025-03-09T08:00:00Z",
        "2025-03-10T06:30:00Z",
        "2025-03-09T07:30:00Z"
    )]
    #[case::monthly(
        "30d",
        "UTC",
        "2025-01-01T02:30:00Z",
        "2025-02-01T03:00:00Z",
        "2025-03-01T02:30:00Z",
        "2025-02-01T02:30:00Z"
    )]
    fn advances_due_dates(
        #[case] duration: &str,
        #[case] timezone: &str,
        #[case] previous: Timestamp,
        #[case] now: Timestamp,
        #[case] end: Timestamp,
        #[case] start: Timestamp,
    ) {
        let schedule = Schedule {
            timezone: TimeZone::get(timezone).unwrap(),
            wall_time: "02:30".parse().unwrap(),
        };
        assert_eq!(
            schedule.advance(duration, previous, now),
            Some((end, start))
        );
    }
}
