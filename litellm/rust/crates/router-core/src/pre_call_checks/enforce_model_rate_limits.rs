//! `enforce_model_rate_limits` (`ModelRateLimitingCheck`): an attempt on a deployment past its
//! declared `tpm`, `rpm`, `tpd` or `rpd` is refused, on the counters in the router's cache.

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use jiff::civil::Time;
use litellm_router_types::Spelled;
use serde::Deserialize;
use serde_json::{Map, Value};

use crate::{
    cache::{LimitCounter, LimitKind, RouterCache},
    error::{AttemptError, Error, FailureKind, RaisedBy, Refused},
    health::current_minute,
    registry::Deployment,
};

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
    resets_in: Duration,
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

/// A limit the deployment's own entry or its `model_info` declares under `key`.
fn written(section: &Map<String, Value>, key: &str) -> Option<Spelled<f64>> {
    section
        .get(key)
        .filter(|value| !value.is_null())
        .and_then(|value| Spelled::deserialize(value).ok())
}

impl Limits {
    fn of(deployment: &Deployment) -> Self {
        let params = &deployment.params;
        let declared = |key: &str, param: &Option<Spelled<f64>>| {
            written(&deployment.extra, key)
                .or_else(|| param.clone())
                .or_else(|| written(&deployment.model_info.raw, key))
                .as_ref()
                .and_then(Spelled::lax)
        };
        Self {
            tpm: declared("tpm", &params.tpm),
            rpm: declared("rpm", &params.rpm),
            tpd: params.tpd.as_ref().and_then(Spelled::lax),
            rpd: params.rpd.as_ref().and_then(Spelled::lax),
            reset: past_midnight(params.daily_reset_time),
        }
    }

    fn windows(&self, now: SystemTime) -> Windows {
        let minute = |kind, limit| Window {
            kind,
            limit,
            name: current_minute(now),
            ttl: MINUTE,
            resets_in: MINUTE,
        };
        let day = |kind, limit| Window {
            kind,
            limit,
            name: self.day(now),
            ttl: DAY,
            resets_in: self.until_reset(now),
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

    /// The `RateLimitError` the check raises for `deployment`, with its message: never retried,
    /// and tried again once the window starts over.
    fn refused(&self, deployment: &Deployment, limit: f64, usage: f64) -> Error {
        let name: &'static str = self.kind.into();
        Error::Attempt {
            failure: AttemptError {
                num_retries: Some(0),
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
            },
            raised_by: RaisedBy::PreCallCheck(Box::new(Refused {
                deployment_id: deployment.id.clone(),
                resets_in: self.resets_in,
            })),
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
) -> Result<(), Error> {
    let Windows { tokens, requests } = Limits::of(deployment).windows(now);
    for window in &tokens {
        let Some(limit) = window.limit else {
            continue;
        };
        let used = cache.limit_usage(&window.counter(deployment), limit).await;
        if let Ok(Some(usage)) = used
            && usage >= limit
        {
            return Err(window.refused(deployment, limit, usage));
        }
    }
    let limited: Vec<(&Window, f64)> = requests
        .iter()
        .filter_map(|window| Some((window, window.limit?)))
        .collect();
    if limited.is_empty() {
        return Ok(());
    }
    let charge = |amount: i64| -> Vec<(LimitCounter, i64, Duration)> {
        limited
            .iter()
            .map(|(window, _)| (window.counter(deployment), amount, window.ttl))
            .collect()
    };
    let Some(counts) = cache
        .add_limit_usages(&charge(1))
        .await
        .ok()
        .flatten()
        .filter(|counts| counts.len() == limited.len())
    else {
        return Ok(());
    };
    let Some(over) = limited
        .iter()
        .zip(&counts)
        .find_map(|((window, limit), count)| {
            let usage = count.trunc();
            (usage > *limit).then(|| window.refused(deployment, *limit, usage))
        })
    else {
        return Ok(());
    };
    if let Err(error) = cache.add_limit_usages(&charge(-1)).await {
        tracing::debug!(%error, "the model rate limit refund was not recorded");
    }
    Err(over)
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

    fn limits(entry: serde_json::Value) -> (Option<f64>, Option<f64>, Option<f64>, Option<f64>) {
        let spec = serde_json::from_value(entry).unwrap();
        let limits = Limits::of(&Deployment::from_spec(spec).unwrap());
        (limits.tpm, limits.rpm, limits.tpd, limits.rpd)
    }

    #[rstest]
    #[case::the_entrys_own_key_wins(
        serde_json::json!({"model_name": "m", "tpm": 1, "litellm_params": {"model": "a/b", "tpm": 2}, "model_info": {"tpm": 3}}),
        (Some(1.0), None, None, None)
    )]
    #[case::litellm_params_win_over_model_info(
        serde_json::json!({"model_name": "m", "litellm_params": {"model": "a/b", "rpm": 2}, "model_info": {"rpm": 3}}),
        (None, Some(2.0), None, None)
    )]
    #[case::model_info_when_nothing_else_is_set(
        serde_json::json!({"model_name": "m", "tpm": null, "litellm_params": {"model": "a/b"}, "model_info": {"tpm": 3}}),
        (Some(3.0), None, None, None)
    )]
    #[case::text_that_spells_a_number(
        serde_json::json!({"model_name": "m", "litellm_params": {"model": "a/b", "tpm": "10", "tpd": "100", "rpd": 5}}),
        (Some(10.0), None, Some(100.0), Some(5.0))
    )]
    #[case::daily_limits_are_read_from_litellm_params_alone(
        serde_json::json!({"model_name": "m", "tpd": 1, "litellm_params": {"model": "a/b"}, "model_info": {"rpd": 2}}),
        (None, None, None, None)
    )]
    fn a_deployment_declares_its_limits_where_python_looks(
        #[case] entry: serde_json::Value,
        #[case] expected: (Option<f64>, Option<f64>, Option<f64>, Option<f64>),
    ) {
        assert_eq!(limits(entry), expected);
    }

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
