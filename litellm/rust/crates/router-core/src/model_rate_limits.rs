//! `enforce_model_rate_limits` (`ModelRateLimitingCheck`), which the core runs itself when
//! `RouterConfig::enforce_model_rate_limits` is on: an attempt on a deployment past its declared
//! `tpm`, `rpm`, `tpd` or `rpd` is refused, on the counters in the router's cache. A host that
//! runs Python's check leaves the config off, so the rule never runs twice.

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use jiff::civil::Time;
use litellm_python_compat::{json::from_json, pydantic::lax_float};
use serde_json::Value;

use crate::{
    cache::{LimitCounter, LimitKind, RouterCache},
    error::{AttemptError, FailureKind},
    health::current_minute,
    registry::Deployment,
};

/// The `optional_pre_call_checks` entry that turns the check on.
pub const SETTING: &str = "enforce_model_rate_limits";

/// The `litellm_params` that hold a deployment's daily limits, which no other rule reads.
pub const DAILY_PARAMS: [&str; 2] = [TPD, RPD];
const TPD: &str = "tpd";
const RPD: &str = "rpd";

/// `RoutingArgs.ttl` and `RoutingArgs.daily_ttl`.
const MINUTE: Duration = Duration::from_secs(60);
const DAY: Duration = Duration::from_secs(86_400);

/// A deployment's declared limits (`_get_deployment_limits`, `_DailyLimitDeployment`).
struct Limits {
    tpm: Option<f64>,
    rpm: Option<f64>,
    tpd: Option<f64>,
    rpd: Option<f64>,
    /// `daily_reset_time`: how far past midnight UTC the daily window starts over.
    reset: Duration,
}

/// One window a deployment's usage is counted in, with the limit the deployment declares on it.
struct Window {
    kind: LimitKind,
    limit: Option<f64>,
    /// The window as its counter names it: the minute, or the day.
    name: String,
    ttl: Duration,
    /// How long until the window starts over.
    retry_after: Duration,
}

/// A deployment's four windows at one moment, each pair per-minute then per-day.
struct Windows {
    tokens: [Window; 2],
    requests: [Window; 2],
}

/// How far past midnight UTC `reset` is, and no way past it when a deployment names no time.
fn past_midnight(reset: Option<Time>) -> Duration {
    reset.map_or(Duration::ZERO, |reset| {
        reset.duration_since(Time::midnight()).unsigned_abs()
    })
}

/// A limit as Python's `litellm_params` reads it: a number, or text that spells one.
fn number(value: &Value) -> Option<f64> {
    lax_float(&from_json(value.clone()))
}

impl Limits {
    fn of(deployment: &Deployment) -> Self {
        let declared = |key: &str| {
            [
                &deployment.extra,
                &deployment.raw_params,
                &deployment.model_info.raw,
            ]
            .iter()
            .find_map(|section| section.get(key).filter(|value| !value.is_null()))
            .and_then(number)
        };
        let daily = |key: &str| deployment.raw_params.get(key).and_then(number);
        Self {
            tpm: declared("tpm"),
            rpm: declared("rpm"),
            tpd: daily(TPD),
            rpd: daily(RPD),
            reset: past_midnight(deployment.params.daily_reset_time),
        }
    }

    fn windows(&self, now: SystemTime) -> Windows {
        let minute = |kind, limit| Window {
            kind,
            limit,
            name: current_minute(now),
            ttl: MINUTE,
            retry_after: MINUTE,
        };
        let day = |kind, limit| Window {
            kind,
            limit,
            name: self.day(now),
            ttl: DAY,
            retry_after: self.until_reset(now),
        };
        Windows {
            tokens: [
                minute(LimitKind::Tpm, self.tpm),
                day(LimitKind::Tpd, self.tpd),
            ],
            requests: [
                minute(LimitKind::Rpm, self.rpm),
                day(LimitKind::Rpd, self.rpd),
            ],
        }
    }

    /// The seconds since the epoch the daily window is measured from.
    fn shifted(&self, now: SystemTime) -> f64 {
        now.duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64()
            - self.reset.as_secs_f64()
    }

    /// `daily_window_day`: the UTC day the window started on.
    fn day(&self, now: SystemTime) -> String {
        let days = (self.shifted(now) / 86_400.0).floor() as i64;
        jiff::civil::date(1970, 1, 1)
            .checked_add(jiff::Span::new().days(days))
            .map_or_else(
                |_| String::new(),
                |date| date.strftime("%Y-%m-%d").to_string(),
            )
    }

    /// `seconds_until_daily_reset`.
    fn until_reset(&self, now: SystemTime) -> Duration {
        let into_day = self.shifted(now).rem_euclid(86_400.0);
        Duration::from_secs(((86_400.0 - into_day) as u64).max(1))
    }
}

/// A count as Python prints the `int` it is.
fn whole(value: f64) -> String {
    match value.fract() == 0.0 && value.abs() < 1e15 {
        true => format!("{}", value as i64),
        false => value.to_string(),
    }
}

impl Window {
    fn counter(&self, deployment: &Deployment) -> LimitCounter {
        LimitCounter {
            deployment_id: deployment.id.clone(),
            model: deployment.params.model.clone(),
            kind: self.kind,
            window: self.name.clone(),
        }
    }

    /// The `RateLimitError` the check raises, with its message: never retried, tried again once
    /// the window starts over, and counted against no per-minute requests, since no request was
    /// sent (`_refused` reports it `uncounted`).
    fn refused(&self, limit: f64, usage: f64) -> AttemptError {
        let name: &'static str = self.kind.into();
        AttemptError {
            retry_after: Some(self.retry_after),
            num_retries: Some(0),
            uncounted_request: true,
            detail: Some(
                format!(
                    "Model rate limit exceeded. {} limit={}, current usage={}",
                    name.to_uppercase(),
                    whole(limit),
                    whole(usage),
                )
                .into(),
            ),
            ..AttemptError::new(FailureKind::RateLimit, Some(429))
        }
    }
}

/// `ModelRateLimitingCheck.async_pre_call_check`: the deployment's token windows must have room,
/// then its request windows count this request in one round trip, refunded when that goes past a
/// limit. A cache that fails lets the request through.
pub(crate) async fn admit(
    cache: &impl RouterCache,
    deployment: &Deployment,
    now: SystemTime,
) -> Option<AttemptError> {
    let Windows { tokens, requests } = Limits::of(deployment).windows(now);
    for window in &tokens {
        let Some(limit) = window.limit else {
            continue;
        };
        let used = cache.limit_usage(&window.counter(deployment), limit).await;
        if let Ok(Some(usage)) = used
            && usage >= limit
        {
            return Some(window.refused(limit, usage));
        }
    }
    let limited: Vec<(&Window, f64)> = requests
        .iter()
        .filter_map(|window| Some((window, window.limit?)))
        .collect();
    if limited.is_empty() {
        return None;
    }
    let charge = |amount: i64| -> Vec<(LimitCounter, i64, Duration)> {
        limited
            .iter()
            .map(|(window, _)| (window.counter(deployment), amount, window.ttl))
            .collect()
    };
    let counts = cache
        .add_limit_usages(&charge(1))
        .await
        .ok()
        .flatten()
        .filter(|counts| counts.len() == limited.len())?;
    let over = limited
        .iter()
        .zip(&counts)
        .find_map(|((window, limit), count)| {
            let usage = count.trunc();
            (usage > *limit).then(|| window.refused(*limit, usage))
        })?;
    if let Err(error) = cache.add_limit_usages(&charge(-1)).await {
        tracing::debug!(%error, "the model rate limit refund was not recorded");
    }
    Some(over)
}

/// `ModelRateLimitingCheck.async_log_success_event`: a success's tokens count against the
/// deployment's minute and day, whether or not it declares a limit.
pub(crate) async fn count_tokens(
    cache: &impl RouterCache,
    deployment: &Deployment,
    total_tokens: u64,
    now: SystemTime,
) {
    if total_tokens == 0 || deployment.params.model.is_empty() {
        return;
    }
    let tokens = i64::try_from(total_tokens).unwrap_or(i64::MAX);
    for window in &Limits::of(deployment).windows(now).tokens {
        if let Err(error) = cache
            .add_limit_usage(&window.counter(deployment), tokens, window.ttl)
            .await
        {
            tracing::debug!(%error, "Error in ModelRateLimitingCheck.async_log_success_event");
            return;
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    /// 2024-03-01 00:30:00 UTC.
    fn half_past_midnight() -> SystemTime {
        UNIX_EPOCH + Duration::from_secs(1_709_253_000)
    }

    #[rstest]
    #[case::midnight(None, "2024-03-01", 86_400 - 1800)]
    #[case::before_the_reset(Some(jiff::civil::time(1, 0, 0, 0)), "2024-02-29", 1800)]
    #[case::after_the_reset(Some(jiff::civil::time(0, 15, 0, 0)), "2024-03-01", 86_400 - 900)]
    fn the_daily_window_starts_over_at_the_reset_time(
        #[case] reset: Option<Time>,
        #[case] day: &str,
        #[case] until_reset: u64,
    ) {
        let limits = Limits {
            tpm: None,
            rpm: None,
            tpd: None,
            rpd: None,
            reset: past_midnight(reset),
        };
        assert_eq!(
            (
                limits.day(half_past_midnight()),
                limits.until_reset(half_past_midnight())
            ),
            (day.to_owned(), Duration::from_secs(until_reset))
        );
    }
}
