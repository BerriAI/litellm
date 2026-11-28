//! Resolves stored budget windows for admission, charging and accounting.

use jiff::{SignedDuration, Span, Timestamp, tz::TimeZone};
use litellm_gateway_auth_core::record::BudgetWindow;
use litellm_redis_store::spend_counters::SpendCounter;
use serde::Deserialize;

#[derive(Clone, Copy, Deserialize, strum::IntoStaticStr)]
pub(crate) enum Entity {
    #[serde(rename = "key")]
    #[strum(serialize = "key")]
    Key,
    #[serde(rename = "team")]
    #[strum(serialize = "team")]
    Team,
    #[serde(rename = "user")]
    #[strum(serialize = "user")]
    User,
}

impl Entity {
    pub(crate) fn counter(self, id: &str, duration: &str) -> SpendCounter {
        let window = duration.to_owned();
        match self {
            Self::Key => SpendCounter::Key {
                token: id.to_owned(),
                window: Some(window),
            },
            Self::Team => SpendCounter::Team {
                team_id: id.to_owned(),
                window: Some(window),
            },
            Self::User => SpendCounter::UserWindow {
                user_id: id.to_owned(),
                window,
            },
        }
    }
}

pub(crate) fn start(window: &BudgetWindow, now: Timestamp) -> Option<Timestamp> {
    let start = match window.window_start {
        Some(start) => start,
        None => {
            let duration = match window.budget_duration.trim().to_ascii_lowercase().as_str() {
                "hourly" => "1h",
                "daily" => "24h",
                "weekly" => "7d",
                "monthly" => "30d",
                _ => &window.budget_duration,
            };
            let end = window.reset_at.unwrap_or(now);
            let units = duration.find(|c: char| !c.is_ascii_digit())?;
            let count: i64 = duration[..units].parse().ok()?;
            if &duration[units..] == "mo" {
                let months = i32::try_from(count).ok()?;
                if months == 12 && window.reset_at.is_some() {
                    end.to_zoned(TimeZone::UTC)
                        .checked_sub(Span::new().try_months(months).ok()?)
                        .ok()?
                        .timestamp()
                } else {
                    let from = now.to_zoned(TimeZone::UTC);
                    let to = from
                        .checked_add(Span::new().try_months(months).ok()?)
                        .ok()?;
                    end.checked_sub(to.timestamp().duration_since(now)).ok()?
                }
            } else {
                let seconds = match &duration[units..] {
                    "s" => 1,
                    "m" => 60,
                    "h" => 3600,
                    "d" => 86_400,
                    "w" => 604_800,
                    _ => return None,
                };
                end.checked_sub(SignedDuration::from_secs(count.checked_mul(seconds)?))
                    .ok()?
            }
        }
    };
    Timestamp::from_millisecond(start.as_millisecond()).ok()
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::explicit(
        "1h",
        Some("2025-01-02T03:02:01.234567Z"),
        Some("2025-01-02T03:02:01.234Z")
    )]
    #[case::derived("1h", None, Some("2025-01-02T03:00:00Z"))]
    #[case::alias("daily", None, Some("2025-01-01T04:00:00Z"))]
    #[case::invalid("unknown", None, None)]
    fn resolves_window_start(
        #[case] budget_duration: &str,
        #[case] window_start: Option<&str>,
        #[case] expected: Option<&str>,
    ) {
        let now = "2025-01-02T03:30:00Z".parse().unwrap();
        let window = BudgetWindow {
            budget_duration: budget_duration.to_owned(),
            max_budget: 10.0,
            reset_at: Some("2025-01-02T04:00:00Z".parse().unwrap()),
            window_start: window_start.map(|start| start.parse().unwrap()),
        };
        assert_eq!(
            start(&window, now),
            expected.map(|start| start.parse().unwrap())
        );
    }
}
