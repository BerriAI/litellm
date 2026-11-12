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
    outcome::{OutcomeCallbacks, OutcomeWrite},
    registry::{Deployment, Registry},
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::Display, strum::EnumString)]
pub enum UsageMetric {
    #[strum(serialize = "tpm")]
    Tpm,
    #[strum(serialize = "rpm")]
    Rpm,
    /// Input tokens per minute, which the io-token rate limit check counts.
    #[strum(serialize = "itpm")]
    Itpm,
    /// Output tokens per minute.
    #[strum(serialize = "otpm")]
    Otpm,
}

/// `RouterCacheEnum.TPM` / `RPM` / `ITPM` / `OTPM`: `model` is the deployment's
/// `litellm_params.model`.
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
/// status does, and a connection error or an empty status never does.
pub fn is_cooldown_required(failure: &AttemptError) -> bool {
    if failure.kind == FailureKind::ApiConnection || failure.status_absent {
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

/// `_core_team_model_has_alternatives`: another routable deployment serves the deployment's team
/// public name.
pub(crate) fn team_model_has_alternatives(registry: &Registry, deployment: &Deployment) -> bool {
    let info = &deployment.model_info;
    let (Some(team_id), Some(public_name)) = (&info.team_id, &info.team_public_model_name) else {
        return false;
    };
    registry
        .team_named(team_id, public_name)
        .is_some_and(|siblings| {
            siblings
                .iter()
                .filter(|sibling| !sibling.model_info.blocked)
                .count()
                > 1
        })
}

/// `_core_routing_group_has_alternatives`: `model_group` names a callable routing group whose
/// members span more than one deployment.
pub(crate) fn routing_group_has_alternatives(registry: &Registry, model_group: &str) -> bool {
    let resolved = registry.alias_of(model_group).unwrap_or(model_group);
    registry.routing_group_named(resolved).is_some_and(|group| {
        group
            .models
            .iter()
            .map(|member| registry.named(member).map_or(0, |members| members.len()))
            .sum::<usize>()
            > 1
    })
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

    pub(crate) fn timestamp(&self) -> jiff::Timestamp {
        jiff::Timestamp::try_from(self.clock.now()).unwrap_or(jiff::Timestamp::UNIX_EPOCH)
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
        if failure.outside_deployment {
            return Ok(None);
        }
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
            FailureKind::Declined
                | FailureKind::CallerCredentialAuthentication
                | FailureKind::CallerCredentialRateLimit
        ) && !failure.caller_scoped
            && !cooldown.is_zero()
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
        if settings.allowed_fails_policy.is_none() && !settings.allowed_fails_set() {
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
    /// when `tracks` says it has limits to track. The count is the success callback's; the usage
    /// is `make_call`'s, which runs whatever the callbacks do.
    pub async fn record_success(
        &self,
        callbacks: OutcomeCallbacks,
        deployment: &Deployment,
        total_tokens: u64,
        tracks: bool,
    ) -> Result<(), Error> {
        callbacks
            .record(OutcomeWrite::Success, self.count_success(&deployment.id))
            .await?;
        if !tracks {
            return Ok(());
        }
        callbacks
            .record(OutcomeWrite::AttemptUsage, async {
                self.add_usage(deployment, None, total_tokens as f64, 1.0)
                    .await
                    .map(|_| ())
            })
            .await
    }

    /// `_increment_deployment_usage`, for a deployment that tracks usage: tokens and requests
    /// under `model` (metadata's `deployment`, which a wildcard route fills; else the
    /// deployment's own), unless there is nothing to add. Returns the tokens counter's key when
    /// it added to it.
    pub async fn add_usage(
        &self,
        deployment: &Deployment,
        model: Option<&str>,
        total_tokens: f64,
        requests: f64,
    ) -> Result<Option<String>, Error> {
        if total_tokens <= 0.0 && requests <= 0.0 {
            return Ok(None);
        }
        let minute = current_minute(self.clock.now());
        let model = model.unwrap_or(&deployment.params.model);
        let tpm_key = usage_key(&deployment.id, model, UsageMetric::Tpm, &minute);
        self.cache
            .async_increment_usage(
                &[
                    (tpm_key.clone(), total_tokens),
                    (
                        usage_key(&deployment.id, model, UsageMetric::Rpm, &minute),
                        requests,
                    ),
                ],
                Some(USAGE_TTL),
            )
            .await?;
        Ok(Some(tpm_key))
    }

    /// `async_deployment_callback_on_failure`: a failed request counts against the deployment's
    /// per-minute requests under `model` (see [`Self::add_usage`]), whether or not it tracks usage.
    pub async fn count_failed_request(
        &self,
        deployment: &Deployment,
        model: Option<&str>,
    ) -> Result<(), Error> {
        let minute = current_minute(self.clock.now());
        self.cache
            .async_increment(
                &usage_key(
                    &deployment.id,
                    model.unwrap_or(&deployment.params.model),
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
    use std::collections::BTreeMap;

    use rstest::rstest;

    use super::*;
    use crate::{
        cache::InMemoryRouterCache,
        config::{RouterConfig, RouterDynamicConfig},
        registry::{DeploymentSpec, RoutingGroupSpec},
    };

    #[rstest]
    #[case::rate_limit(FailureKind::RateLimit, Some(429), true)]
    #[case::auth(FailureKind::Authentication, Some(401), true)]
    #[case::payment(FailureKind::Other, Some(402), true)]
    #[case::not_found(FailureKind::NotFound, Some(404), true)]
    #[case::timeout(FailureKind::Timeout, Some(408), true)]
    #[case::bad_request(FailureKind::BadRequest, Some(400), false)]
    #[case::forbidden(FailureKind::PermissionDenied, Some(403), false)]
    #[case::server(FailureKind::InternalServer, Some(500), true)]
    #[case::unreadable_status(FailureKind::Other, None, true)]
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

    /// An exception with no `status_code` is Python's status `""`, which no rule cools down.
    #[rstest]
    fn an_empty_status_requires_no_cooldown() {
        let mut failure = AttemptError::new(FailureKind::Other, None);
        failure.status_absent = true;
        assert!(!is_cooldown_required(&failure));
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

    fn deployment(id: &str, info: serde_json::Value) -> DeploymentSpec {
        let mut model_info = serde_json::json!({"id": id});
        model_info
            .as_object_mut()
            .unwrap()
            .extend(info.as_object().unwrap().clone());
        serde_json::from_value(serde_json::json!({
            "model_name": id,
            "litellm_params": {"model": format!("openai/{id}")},
            "model_info": model_info,
        }))
        .unwrap()
    }

    fn health() -> Health {
        Health::new(
            Arc::new(InMemoryRouterCache::default()),
            Arc::new(SystemClock),
        )
    }

    fn settings(dynamic: RouterDynamicConfig) -> Settings {
        Settings::new(&RouterConfig::default(), Arc::new(dynamic))
    }

    #[rstest]
    #[case::held(false, false, 1, true)]
    #[case::caller_scoped(false, true, 1, false)]
    #[case::outside_the_deployment(true, false, 0, false)]
    #[tokio::test]
    async fn the_failure_exits_decide_what_is_counted_and_cooled(
        #[case] outside_deployment: bool,
        #[case] caller_scoped: bool,
        #[case] counted: u64,
        #[case] cooled: bool,
    ) {
        let health = health();
        let deployment = Deployment::from_spec(deployment("a", serde_json::json!({}))).unwrap();
        let mut failure = AttemptError::new(FailureKind::RateLimit, Some(429));
        failure.outside_deployment = outside_deployment;
        failure.caller_scoped = caller_scoped;

        let cooldown = health
            .record_failure(
                &settings(RouterDynamicConfig::default()),
                &deployment,
                GroupShape::default(),
                None,
                &failure,
            )
            .await
            .unwrap();

        assert_eq!(cooldown.is_some(), cooled);
        assert_eq!(health.outcomes_this_minute("a").await.unwrap().1, counted);
    }

    #[rstest]
    #[case::authentication(
        FailureKind::CallerCredentialAuthentication,
        401,
        "AuthenticationError"
    )]
    #[case::rate_limit(FailureKind::CallerCredentialRateLimit, 429, "RateLimitError")]
    #[tokio::test]
    async fn a_caller_credential_failure_counts_without_cooling(
        #[case] kind: FailureKind,
        #[case] status: u16,
        #[case] base_class: &str,
    ) {
        let health = health();
        let deployment = Deployment::from_spec(deployment("a", serde_json::json!({}))).unwrap();
        let failure = AttemptError::new(kind, Some(status));

        let cooldown = health
            .record_failure(
                &settings(RouterDynamicConfig::default()),
                &deployment,
                GroupShape::default(),
                None,
                &failure,
            )
            .await
            .unwrap();

        assert_eq!(cooldown, None);
        assert!(failure.is_a(base_class));
        assert_eq!(health.outcomes_this_minute("a").await.unwrap().1, 1);
    }

    /// `litellm.allowed_fails` written after the build: the copied 3 starts counting instead of
    /// the failure-ratio rule, which cools a 429 down at once.
    #[rstest]
    #[case::still_the_process_default(true, true)]
    #[case::the_process_default_moved(false, false)]
    #[tokio::test]
    async fn the_copied_allowed_fails_counts_once_it_differs_from_the_process_default(
        #[case] allowed_fails_is_default: bool,
        #[case] cooled: bool,
    ) {
        let settings = settings(RouterDynamicConfig {
            allowed_fails: Some(3),
            allowed_fails_is_default,
            ..RouterDynamicConfig::default()
        });
        let deployment = Deployment::from_spec(deployment("a", serde_json::json!({}))).unwrap();
        let failure = AttemptError::new(FailureKind::RateLimit, Some(429));

        let cooldown = health()
            .record_failure(
                &settings,
                &deployment,
                GroupShape::default(),
                None,
                &failure,
            )
            .await
            .unwrap();

        assert_eq!(cooldown.is_some(), cooled);
    }

    fn team(id: &str, team_id: &str, blocked: bool) -> DeploymentSpec {
        deployment(
            id,
            serde_json::json!({"team_id": team_id, "team_public_model_name": "tp", "blocked": blocked}),
        )
    }

    #[rstest]
    #[case::a_sibling(team("b", "t1", false), true)]
    #[case::a_blocked_sibling(team("b", "t1", true), false)]
    #[case::another_teams_deployment(team("b", "t2", false), false)]
    #[case::no_team(deployment("b", serde_json::json!({})), false)]
    fn a_team_deployment_has_alternatives_under_its_public_name(
        #[case] other: DeploymentSpec,
        #[case] expected: bool,
    ) {
        let registry = Registry::build(
            vec![team("a", "t1", false), other],
            BTreeMap::new(),
            Vec::new(),
        )
        .unwrap();
        let first = registry.deployment("a").unwrap();
        assert_eq!(team_model_has_alternatives(&registry, &first), expected);
    }

    #[rstest]
    #[case::spans_two_deployments(&["a", "b"], "rg", true)]
    #[case::one_deployment(&["a"], "rg", false)]
    #[case::through_its_alias(&["a", "b"], "alias", true)]
    #[case::a_member_called_directly(&["a", "b"], "a", false)]
    fn a_routing_group_has_alternatives_past_one_deployment(
        #[case] members: &[&str],
        #[case] requested: &str,
        #[case] expected: bool,
    ) {
        let group: RoutingGroupSpec = serde_json::from_value(serde_json::json!({
            "group_name": "rg", "models": members, "routing_strategy": "simple-shuffle",
        }))
        .unwrap();
        let registry = Registry::build(
            vec![
                deployment("a", serde_json::json!({})),
                deployment("b", serde_json::json!({})),
            ],
            BTreeMap::from([("alias".to_owned(), "rg".to_owned())]),
            vec![group],
        )
        .unwrap();
        assert_eq!(
            routing_group_has_alternatives(&registry, requested),
            expected
        );
    }

    /// An empty status is counted and cools nothing down, unless the deployment's own policy
    /// names the error.
    #[rstest]
    #[case::no_policy(serde_json::json!({}), false)]
    #[case::policy_names_the_error(
        serde_json::json!({"allowed_fails_policy": {"BadRequestErrorAllowedFails": 0}}),
        true
    )]
    #[tokio::test]
    async fn an_empty_status_on_the_attempt_path_needs_the_deployments_policy_to_cool(
        #[case] info: serde_json::Value,
        #[case] cooled: bool,
    ) {
        let health = health();
        let deployment = Deployment::from_spec(deployment("a", info)).unwrap();
        let mut failure = AttemptError::new(FailureKind::BadRequest, None);
        failure.status_absent = true;

        let cooldown = health
            .record_failure(
                &settings(RouterDynamicConfig::default()),
                &deployment,
                GroupShape::default(),
                None,
                &failure,
            )
            .await
            .unwrap();

        assert_eq!(cooldown.is_some(), cooled);
        assert_eq!(health.outcomes_this_minute("a").await.unwrap().1, 1);
    }

    #[rstest]
    #[case::the_deployments_own_model(None, "openai/a")]
    #[case::a_filled_wildcard(Some("openai/gpt-4o"), "openai/gpt-4o")]
    #[tokio::test]
    async fn usage_is_keyed_by_the_model_served(#[case] model: Option<&str>, #[case] keyed: &str) {
        let cache = InMemoryRouterCache::default();
        let clock = Arc::new(SystemClock);
        let health = Health::new(Arc::new(cache.clone()), clock.clone());
        let deployment = Deployment::from_spec(deployment("a", serde_json::json!({}))).unwrap();

        health
            .add_usage(&deployment, model, 15.0, 1.0)
            .await
            .unwrap();

        let minute = current_minute(clock.now());
        assert_eq!(
            (
                cache.get(&usage_key("a", keyed, UsageMetric::Tpm, &minute)),
                cache.get(&usage_key("a", keyed, UsageMetric::Rpm, &minute)),
            ),
            (Some(serde_json::json!(15)), Some(serde_json::json!(1))),
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
