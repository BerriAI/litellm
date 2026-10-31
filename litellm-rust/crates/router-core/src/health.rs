//! [`Health`]: cooldowns, allowed-fails counters and per-minute usage, kept in the shared
//! [`RouterCache`] under the Python router's keys, encodings and TTLs:
//!
//! | state | key | value | TTL |
//! |---|---|---|---|
//! | cooldown | `deployment:{id}:cooldown` | `{exception_received, status_code, timestamp, cooldown_time}` | cooldown time |
//! | allowed fails | `deployment:{id}:allowed_fails[:{suffix}]` | int counter | cooldown time |
//! | usage | `global_router:{id}:{model}:tpm:{HH-MM}` and `:rpm:` | int counter | 60s |
//! | outcomes (local only) | `{id}:successes`, `{id}:fails` | int counter | 60s |
//!
//! From `router_utils/cooldown_cache.py`, `router_utils/cooldown_handlers.py`,
//! `router_utils/router_callbacks/track_deployment_metrics.py`, `RouterCacheEnum` in
//! `types/router.py` and `Router._increment_usage` / `deployment_callback_on_failure` in
//! `router.py`.

use std::{
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use serde::{Deserialize, Serialize};
use tracing::debug;

use crate::{
    cache::{InMemoryRouterCache, RouterCache},
    error::{AttemptError, Error, FailureKind},
    registry::Deployment,
    settings::Settings,
};

/// `RoutingArgs.ttl`: per-minute usage keys expire after a minute.
pub const USAGE_TTL: Duration = Duration::from_secs(60);
/// `DEFAULT_FAILURE_THRESHOLD_PERCENT`.
pub const FAILURE_THRESHOLD_PERCENT: f64 = 0.5;
/// `DEFAULT_FAILURE_THRESHOLD_MINIMUM_REQUESTS`.
pub const FAILURE_THRESHOLD_MINIMUM_REQUESTS: u64 = 5;
/// `SINGLE_DEPLOYMENT_TRAFFIC_FAILURE_THRESHOLD`.
pub const SINGLE_DEPLOYMENT_TRAFFIC_FAILURE_THRESHOLD: u64 = 1000;
/// `exception_masker` in `CooldownCache`: the visible prefix of `exception_received`.
const EXCEPTION_VISIBLE_PREFIX: usize = 50;

pub trait Clock: Send + Sync {
    fn now(&self) -> SystemTime;
}

pub struct SystemClock;

impl Clock for SystemClock {
    fn now(&self) -> SystemTime {
        SystemTime::now()
    }
}

fn unix_seconds(time: SystemTime) -> f64 {
    time.duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs_f64())
        .unwrap_or_default()
}

/// `get_utc_datetime().strftime("%H-%M")`.
pub fn current_minute(time: SystemTime) -> String {
    let seconds = time
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs())
        .unwrap_or_default()
        % 86_400;
    format!("{:02}-{:02}", seconds / 3600, (seconds % 3600) / 60)
}

pub fn cooldown_key(id: &str) -> String {
    format!("deployment:{id}:cooldown")
}

pub fn allowed_fails_key(id: &str, suffix: Option<&str>) -> String {
    match suffix {
        Some(suffix) => format!("deployment:{id}:allowed_fails:{suffix}"),
        None => format!("deployment:{id}:allowed_fails"),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::Display)]
pub enum UsageMetric {
    #[strum(serialize = "tpm")]
    Tpm,
    #[strum(serialize = "rpm")]
    Rpm,
}

/// `RouterCacheEnum.TPM` / `RPM`: `model` is the deployment's `litellm_params.model`.
pub fn usage_key(id: &str, model: &str, metric: UsageMetric, minute: &str) -> String {
    format!("global_router:{id}:{model}:{metric}:{minute}")
}

/// `CooldownCacheValue`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CooldownValue {
    pub exception_received: String,
    pub status_code: String,
    pub timestamp: f64,
    /// Whole seconds are written as an int, as Python's default `cooldown_time` is, so the
    /// "Try again in N seconds" text reads the same.
    #[serde(serialize_with = "whole_as_int")]
    pub cooldown_time: f64,
}

fn whole_as_int<S: serde::Serializer>(value: &f64, serializer: S) -> Result<S::Ok, S::Error> {
    if value.fract() == 0.0 && value.abs() < i64::MAX as f64 {
        serializer.serialize_i64(*value as i64)
    } else {
        serializer.serialize_f64(*value)
    }
}

/// `litellm.should_retry`.
pub fn should_retry_status(status: u16) -> bool {
    matches!(status, 408 | 409 | 429) || status >= 500
}

/// `_is_cooldown_required`: a 4xx cools down only for 429, 401, 402, 404 and 408, every other
/// status does, and a connection error never does.
pub fn is_cooldown_required(failure: &AttemptError) -> bool {
    if failure.kind == FailureKind::ApiConnection {
        return false;
    }
    match failure.cooldown_status() {
        status @ 400..500 => matches!(status, 429 | 401 | 402 | 408 | 404),
        _ => true,
    }
}

/// `SensitiveDataMasker(visible_prefix=50, visible_suffix=0, mask_short_values=False)`.
fn mask_exception(text: &str) -> String {
    let length = text.chars().count();
    if length <= EXCEPTION_VISIBLE_PREFIX {
        return text.to_owned();
    }
    let mut masked: String = text.chars().take(EXCEPTION_VISIBLE_PREFIX).collect();
    masked.extend(std::iter::repeat_n('*', length - EXCEPTION_VISIBLE_PREFIX));
    masked
}

/// What the cooldown rules need to know about the deployment's group.
#[derive(Clone, Copy, Debug, Default)]
pub struct GroupShape {
    /// The deployment is the only one in its model group.
    pub single_deployment: bool,
    /// The requested group has a fallback configured.
    pub has_fallback_group: bool,
}

/// An outcome the router did not see itself (`Health.report`).
#[derive(Clone, Debug)]
pub enum Outcome {
    Success { total_tokens: u64 },
    Failure(AttemptError),
}

#[derive(Clone)]
pub struct Health {
    cache: Arc<dyn RouterCache>,
    local: InMemoryRouterCache,
    clock: Arc<dyn Clock>,
}

impl Health {
    pub fn new(cache: Arc<dyn RouterCache>, clock: Arc<dyn Clock>) -> Self {
        Self {
            cache,
            local: InMemoryRouterCache::default(),
            clock,
        }
    }

    pub fn cache(&self) -> &Arc<dyn RouterCache> {
        &self.cache
    }

    fn now(&self) -> f64 {
        unix_seconds(self.clock.now())
    }

    /// The ids among `ids` with an unexpired cooldown (`async_get_active_cooldowns`).
    pub async fn active_cooldowns(
        &self,
        ids: &[String],
    ) -> Result<Vec<(String, CooldownValue)>, Error> {
        let keys: Vec<String> = ids.iter().map(|id| cooldown_key(id)).collect();
        let values = self.cache.async_batch_get_cache(&keys).await?;
        let now = self.now();
        Ok(ids
            .iter()
            .zip(values)
            .filter_map(|(id, value)| {
                let value: CooldownValue = serde_json::from_value(value?).ok()?;
                (value.timestamp + value.cooldown_time - now > 0.0).then(|| (id.clone(), value))
            })
            .collect())
    }

    /// `add_deployment_to_cooldown`.
    pub async fn add_cooldown(
        &self,
        id: &str,
        failure: &AttemptError,
        cooldown: Duration,
    ) -> Result<(), Error> {
        let value = CooldownValue {
            exception_received: mask_exception(failure.detail.as_deref().unwrap_or_default()),
            status_code: failure.cooldown_status().to_string(),
            timestamp: self.now(),
            cooldown_time: cooldown.as_secs_f64(),
        };
        let value = serde_json::to_value(value).unwrap_or_default();
        self.cache
            .async_set_cache(&cooldown_key(id), value, Some(cooldown))
            .await?;
        Ok(())
    }

    fn local_count(&self, key: &str) -> u64 {
        self.local
            .get(key)
            .and_then(|value| value.as_f64())
            .map_or(0, |count| count as u64)
    }

    /// The per-minute (successes, fails) this worker saw.
    pub fn outcomes_this_minute(&self, id: &str) -> (u64, u64) {
        (
            self.local_count(&format!("{id}:successes")),
            self.local_count(&format!("{id}:fails")),
        )
    }

    /// Cooldown duration precedence from `deployment_callback_on_failure`: the deployment's
    /// `cooldown_time`, then the provider's retry-after, then the router default.
    pub fn cooldown_duration(
        settings: &Settings,
        deployment: &Deployment,
        failure: &AttemptError,
    ) -> Duration {
        Self::configured_cooldown(deployment)
            .or(failure.retry_after)
            .unwrap_or(settings.cooldown_time)
    }

    /// The deployment's own `cooldown_time`: `model_info` first, then `litellm_params`
    /// (`_first_present(model_info, litellm_params, key="cooldown_time")`). It is both the
    /// cooldown and the TTL of the deployment's `allowed_fails:generic` counter.
    fn configured_cooldown(deployment: &Deployment) -> Option<Duration> {
        deployment
            .model_info
            .cooldown_time
            .or_else(|| {
                deployment
                    .raw_params
                    .get("cooldown_time")
                    .and_then(|v| v.as_f64())
            })
            .filter(|seconds| *seconds >= 0.0)
            .and_then(|seconds| Duration::try_from_secs_f64(seconds).ok())
    }

    /// Record a failed attempt and cool the deployment down when the rules say so. Returns the
    /// cooldown applied, if any.
    pub async fn record_failure(
        &self,
        settings: &Settings,
        deployment: &Deployment,
        shape: GroupShape,
        failure: &AttemptError,
    ) -> Result<Option<Duration>, Error> {
        let cooldown = Self::cooldown_duration(settings, deployment, failure);
        self.local
            .increment(&format!("{}:fails", deployment.id), 1.0, Some(USAGE_TTL));
        if !Self::should_run_cooldown_logic(settings, deployment, failure, cooldown) {
            return Ok(None);
        }
        if !self
            .should_cooldown(settings, deployment, shape, failure)
            .await?
        {
            return Ok(None);
        }
        debug!(deployment_id = %deployment.id, cooldown_seconds = cooldown.as_secs_f64(), "cooling deployment down");
        self.add_cooldown(&deployment.id, failure, cooldown).await?;
        Ok(Some(cooldown))
    }

    /// `_should_run_cooldown_logic`.
    fn should_run_cooldown_logic(
        settings: &Settings,
        deployment: &Deployment,
        failure: &AttemptError,
        cooldown: Duration,
    ) -> bool {
        !matches!(
            failure.kind,
            FailureKind::Declined | FailureKind::CallerCredential
        ) && !cooldown.is_zero()
            && !settings.disable_cooldowns
            && is_cooldown_required(failure)
            && !settings
                .provider_default_deployment_ids
                .contains(&deployment.id)
    }

    /// `_should_cooldown_deployment`.
    async fn should_cooldown(
        &self,
        settings: &Settings,
        deployment: &Deployment,
        shape: GroupShape,
        failure: &AttemptError,
    ) -> Result<bool, Error> {
        let status = failure.cooldown_status();
        let policy_fails = settings.allowed_fails_for(failure.kind);
        let lone = shape.single_deployment && !shape.has_fallback_group;
        if shape.single_deployment && status == 402 && policy_fails.is_none() {
            return Ok(false);
        }
        if let Some(allowed) = deployment.model_info.allowed_fails {
            // A deployment-wide `allowed_fails` keeps the lone-deployment safety net
            // (`_should_cooldown_based_on_deployment_policy`).
            if lone {
                return Ok(false);
            }
            let ttl = Self::configured_cooldown(deployment).unwrap_or(settings.cooldown_time);
            return self
                .exceeds_allowed_fails(&deployment.id, Some("generic"), allowed, ttl)
                .await;
        }
        if settings.allowed_fails_policy.is_none() && settings.allowed_fails.is_none() {
            let (successes, fails) = self.outcomes_this_minute(&deployment.id);
            let total = successes + fails;
            let percent_fails = if total > 0 {
                fails as f64 / total as f64
            } else {
                0.0
            };
            return Ok((status == 429 && !shape.single_deployment)
                || (percent_fails == 1.0 && total >= SINGLE_DEPLOYMENT_TRAFFIC_FAILURE_THRESHOLD)
                || (percent_fails > FAILURE_THRESHOLD_PERCENT
                    && total >= FAILURE_THRESHOLD_MINIMUM_REQUESTS
                    && !shape.single_deployment)
                || !should_retry_status(status));
        }
        if lone && policy_fails.is_none() {
            return Ok(false);
        }
        let allowed = policy_fails.unwrap_or_else(|| settings.effective_allowed_fails());
        self.exceeds_allowed_fails(&deployment.id, None, allowed, settings.cooldown_time)
            .await
    }

    /// `should_cooldown_based_on_allowed_fails_policy`: count this failure, then compare.
    async fn exceeds_allowed_fails(
        &self,
        id: &str,
        suffix: Option<&str>,
        allowed: u32,
        ttl: Duration,
    ) -> Result<bool, Error> {
        let fails = self
            .cache
            .async_increment(&allowed_fails_key(id, suffix), 1.0, Some(ttl))
            .await?;
        Ok(fails > f64::from(allowed))
    }

    /// Count a success, and the tokens and request against the deployment's per-minute usage
    /// when it has limits to track.
    pub async fn record_success(
        &self,
        deployment: &Deployment,
        total_tokens: u64,
    ) -> Result<(), Error> {
        self.local.increment(
            &format!("{}:successes", deployment.id),
            1.0,
            Some(USAGE_TTL),
        );
        if !deployment.tracks_usage() {
            return Ok(());
        }
        let minute = current_minute(self.clock.now());
        let model = &deployment.params.model;
        self.cache
            .async_increment(
                &usage_key(&deployment.id, model, UsageMetric::Tpm, &minute),
                total_tokens as f64,
                Some(USAGE_TTL),
            )
            .await?;
        self.cache
            .async_increment(
                &usage_key(&deployment.id, model, UsageMetric::Rpm, &minute),
                1.0,
                Some(USAGE_TTL),
            )
            .await?;
        Ok(())
    }

    /// This minute's (tpm, rpm) for a deployment.
    pub async fn usage(&self, deployment: &Deployment) -> Result<(u64, u64), Error> {
        let minute = current_minute(self.clock.now());
        let model = &deployment.params.model;
        let keys = [UsageMetric::Tpm, UsageMetric::Rpm]
            .map(|metric| usage_key(&deployment.id, model, metric, &minute));
        let values = self.cache.async_batch_get_cache(&keys).await?;
        let count = |index: usize| {
            values
                .get(index)
                .cloned()
                .flatten()
                .and_then(|value| value.as_f64())
                .map_or(0, |count| count as u64)
        };
        Ok((count(0), count(1)))
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::rate_limit(FailureKind::RateLimit, Some(429), true)]
    #[case::auth(FailureKind::Authentication, Some(401), true)]
    #[case::payment(FailureKind::Other, Some(402), true)]
    #[case::not_found(FailureKind::NotFound, Some(404), true)]
    #[case::timeout(FailureKind::Timeout, Some(408), true)]
    #[case::bad_request(FailureKind::BadRequest, Some(400), false)]
    #[case::forbidden(FailureKind::PermissionDenied, Some(403), false)]
    #[case::server(FailureKind::InternalServer, Some(500), true)]
    #[case::no_status(FailureKind::Other, None, true)]
    #[case::connection(FailureKind::ApiConnection, Some(500), false)]
    fn cooldown_required_follows_python(
        #[case] kind: FailureKind,
        #[case] status: Option<u16>,
        #[case] expected: bool,
    ) {
        assert_eq!(
            is_cooldown_required(&AttemptError::new(kind, status)),
            expected
        );
    }

    #[rstest]
    #[case::midnight(0, "00-00")]
    #[case::afternoon(13 * 3600 + 7 * 60 + 59, "13-07")]
    fn current_minute_is_utc_hours_and_minutes(#[case] seconds: u64, #[case] expected: &str) {
        let time = UNIX_EPOCH + Duration::from_secs(86_400 * 3 + seconds);
        assert_eq!(current_minute(time), expected);
    }

    #[rstest]
    fn keys_match_python() {
        assert_eq!(
            (
                cooldown_key("abc"),
                allowed_fails_key("abc", Some("generic")),
                usage_key("abc", "openai/gpt-4o", UsageMetric::Tpm, "09-05"),
            ),
            (
                "deployment:abc:cooldown".to_owned(),
                "deployment:abc:allowed_fails:generic".to_owned(),
                "global_router:abc:openai/gpt-4o:tpm:09-05".to_owned(),
            )
        );
    }
}
