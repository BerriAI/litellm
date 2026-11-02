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
    cache::RouterCache,
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

/// `_EXCEPTION_POLICY_FIELDS`: a deployment policy's fields in the order Python matches them
/// (a content-policy violation is a bad request, so it comes first).
const DEPLOYMENT_POLICY_FIELDS: [(&str, &str); 9] = [
    (
        "ContentPolicyViolationError",
        "ContentPolicyViolationErrorAllowedFails",
    ),
    ("BadRequestError", "BadRequestErrorAllowedFails"),
    ("AuthenticationError", "AuthenticationErrorAllowedFails"),
    ("Timeout", "TimeoutErrorAllowedFails"),
    ("RateLimitError", "RateLimitErrorAllowedFails"),
    ("InternalServerError", "InternalServerErrorAllowedFails"),
    (
        "ServiceUnavailableError",
        "ServiceUnavailableErrorAllowedFails",
    ),
    ("BadGatewayError", "BadGatewayErrorAllowedFails"),
    ("NotFoundError", "NotFoundErrorAllowedFails"),
];

/// `_resolve_allowed_fails_from_policy` over a deployment's `model_info.allowed_fails_policy`:
/// the first field set for a class the failure is an instance of.
pub fn deployment_allowed_fails(
    policy: Option<&serde_json::Map<String, serde_json::Value>>,
    failure: &AttemptError,
) -> Option<f64> {
    let policy = policy?;
    DEPLOYMENT_POLICY_FIELDS
        .iter()
        .filter(|(class, _)| failure.is_a(class))
        .find_map(|(_, field)| policy.get(*field).and_then(serde_json::Value::as_f64))
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
    clock: Arc<dyn Clock>,
}

impl Health {
    pub fn new(cache: Arc<dyn RouterCache>, clock: Arc<dyn Clock>) -> Self {
        Self { cache, clock }
    }

    pub fn cache(&self) -> &Arc<dyn RouterCache> {
        &self.cache
    }

    pub(crate) fn now_time(&self) -> SystemTime {
        self.clock.now()
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

    async fn local_count(&self, key: &str) -> Result<u64, Error> {
        Ok(self
            .cache
            .async_get_local(key)
            .await?
            .and_then(|value| value.as_f64())
            .map_or(0, |count| count as u64))
    }

    /// The per-minute (successes, fails) this worker saw (`{id}:successes`, `{id}:fails`, kept
    /// `local_only`).
    pub async fn outcomes_this_minute(&self, id: &str) -> Result<(u64, u64), Error> {
        Ok((
            self.local_count(&format!("{id}:successes")).await?,
            self.local_count(&format!("{id}:fails")).await?,
        ))
    }

    /// `increment_deployment_successes_for_current_minute`: returns the key it counted under.
    pub async fn count_success(&self, id: &str) -> Result<String, Error> {
        let key = format!("{id}:successes");
        self.cache
            .async_increment_local(&key, 1.0, Some(USAGE_TTL))
            .await?;
        Ok(key)
    }

    /// `increment_deployment_failures_for_current_minute`.
    pub async fn count_failure(&self, id: &str) -> Result<(), Error> {
        self.cache
            .async_increment_local(&format!("{id}:fails"), 1.0, Some(USAGE_TTL))
            .await?;
        Ok(())
    }

    /// Cooldown duration precedence from `deployment_callback_on_failure`: the deployment's
    /// `cooldown_time`, then the provider's retry-after, then the router default.
    pub fn cooldown_duration(
        settings: &Settings,
        deployment: &Deployment,
        failure: &AttemptError,
    ) -> Duration {
        Self::cooldown_for_call(settings, deployment, None, failure)
    }

    /// `cooldown_duration` for a call whose own params carry `cooldown_time` (a request's
    /// value wins over the deployment's `litellm_params`, not over its `model_info`).
    pub fn cooldown_for_call(
        settings: &Settings,
        deployment: &Deployment,
        call_cooldown: Option<f64>,
        failure: &AttemptError,
    ) -> Duration {
        let configured = deployment
            .model_info
            .cooldown_time
            .or(call_cooldown)
            .or_else(|| {
                deployment
                    .raw_params
                    .get("cooldown_time")
                    .and_then(|v| v.as_f64())
            })
            .filter(|seconds| *seconds >= 0.0)
            .and_then(|seconds| Duration::try_from_secs_f64(seconds).ok());
        configured
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
        call_cooldown: Option<f64>,
        failure: &AttemptError,
    ) -> Result<Option<Duration>, Error> {
        self.count_failure(&deployment.id).await?;
        let cooldown = Self::cooldown_for_call(settings, deployment, call_cooldown, failure);
        self.cool_down_after(settings, deployment, shape, failure, cooldown)
            .await
    }

    /// `set_cooldown_deployments` for a failure already counted: cool the deployment down for
    /// `cooldown` when the rules say so. Returns the cooldown applied, if any.
    pub async fn cool_down_after(
        &self,
        settings: &Settings,
        deployment: &Deployment,
        shape: GroupShape,
        failure: &AttemptError,
        cooldown: Duration,
    ) -> Result<Option<Duration>, Error> {
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
            && (is_cooldown_required(failure)
                || deployment_allowed_fails(
                    deployment.model_info.allowed_fails_policy.as_ref(),
                    failure,
                )
                .is_some())
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
        let policy_fails = settings.allowed_fails_for(failure);
        let lone = shape.single_deployment && !shape.has_fallback_group;
        let deployment_policy = deployment.model_info.allowed_fails_policy.as_ref();
        let deployment_fails = deployment_allowed_fails(deployment_policy, failure);
        if shape.single_deployment
            && status == 402
            && deployment_fails.is_none()
            && policy_fails.is_none()
        {
            return Ok(false);
        }
        let deployment_allowed = deployment.model_info.allowed_fails;
        if deployment_policy.is_some() || deployment_allowed.is_some() {
            // `_should_cooldown_based_on_deployment_policy`: a deployment-wide `allowed_fails`
            // keeps the lone-deployment safety net; a policy entry naming the error does not.
            if deployment_fails.is_none()
                && lone
                && (deployment_allowed.is_some() || policy_fails.is_none())
            {
                return Ok(false);
            }
            let (allowed, suffix) = match deployment_fails {
                Some(allowed) => (
                    Some(allowed),
                    failure.class_names().first().map(|name| (*name).to_owned()),
                ),
                None => (
                    deployment_allowed.map(f64::from),
                    deployment_allowed.map(|_| "generic".to_owned()),
                ),
            };
            let allowed = allowed
                .or_else(|| policy_fails.map(f64::from))
                .unwrap_or_else(|| f64::from(settings.effective_allowed_fails()));
            let ttl =
                Self::configured_cooldown(deployment).unwrap_or_else(|| settings.fails_window());
            return self
                .exceeds_allowed_fails(&deployment.id, suffix.as_deref(), allowed, ttl)
                .await;
        }
        if settings.allowed_fails_policy.is_none() && settings.allowed_fails.is_none() {
            let (successes, fails) = self.outcomes_this_minute(&deployment.id).await?;
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
        self.exceeds_allowed_fails(
            &deployment.id,
            None,
            f64::from(allowed),
            settings.fails_window(),
        )
        .await
    }

    /// `should_cooldown_based_on_allowed_fails_policy`: count this failure, then compare.
    async fn exceeds_allowed_fails(
        &self,
        id: &str,
        suffix: Option<&str>,
        allowed: f64,
        ttl: Duration,
    ) -> Result<bool, Error> {
        let fails = self
            .cache
            .async_increment(&allowed_fails_key(id, suffix), 1.0, Some(ttl))
            .await?;
        Ok(fails > allowed)
    }

    /// Count a success, and the tokens and request against the deployment's per-minute usage
    /// when it has limits to track.
    pub async fn record_success(
        &self,
        deployment: &Deployment,
        total_tokens: u64,
    ) -> Result<(), Error> {
        self.count_success(&deployment.id).await?;
        self.add_usage(deployment, total_tokens as f64, 1.0).await
    }

    /// `_increment_deployment_usage`: tokens and requests against a deployment that tracks
    /// usage, unless there is nothing to add.
    pub async fn add_usage(
        &self,
        deployment: &Deployment,
        total_tokens: f64,
        requests: f64,
    ) -> Result<(), Error> {
        if (total_tokens <= 0.0 && requests <= 0.0) || !deployment.tracks_usage() {
            return Ok(());
        }
        let minute = current_minute(self.clock.now());
        let model = &deployment.params.model;
        self.cache
            .async_increment(
                &usage_key(&deployment.id, model, UsageMetric::Tpm, &minute),
                total_tokens,
                Some(USAGE_TTL),
            )
            .await?;
        self.cache
            .async_increment(
                &usage_key(&deployment.id, model, UsageMetric::Rpm, &minute),
                requests,
                Some(USAGE_TTL),
            )
            .await?;
        Ok(())
    }

    /// `async_deployment_callback_on_failure`: a failed request counts against the
    /// deployment's per-minute requests, whether or not it tracks usage.
    pub async fn count_failed_request(&self, deployment: &Deployment) -> Result<(), Error> {
        let minute = current_minute(self.clock.now());
        self.cache
            .async_increment(
                &usage_key(
                    &deployment.id,
                    &deployment.params.model,
                    UsageMetric::Rpm,
                    &minute,
                ),
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
    #[case::content_policy_first(&["ContentPolicyViolationError", "BadRequestError"], Some(1.0))]
    #[case::bad_request(&["BadRequestError"], Some(2.0))]
    #[case::unnamed(&["Timeout"], None)]
    fn a_deployment_policy_matches_in_python_order(
        #[case] classes: &[&str],
        #[case] expected: Option<f64>,
    ) {
        let policy = serde_json::json!({
            "BadRequestErrorAllowedFails": 2,
            "ContentPolicyViolationErrorAllowedFails": 1,
        });
        let mut failure = AttemptError::new(FailureKind::BadRequest, Some(400));
        failure.classes = classes.iter().map(|class| (*class).into()).collect();
        assert_eq!(
            deployment_allowed_fails(policy.as_object(), &failure),
            expected
        );
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
