//! [`Health`]: cooldowns, allowed-fails counters and per-minute usage, kept in the host's
//! [`RouterCache`] with the Python router's TTLs, and this worker's per-minute outcomes.
//!
//! From `router_utils/cooldown_cache.py`, `router_utils/cooldown_handlers.py`,
//! `router_utils/router_callbacks/track_deployment_metrics.py` and `Router._increment_usage` /
//! `deployment_callback_on_failure` in `router.py`.

use std::{
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use serde::{Deserialize, Serialize};
use tracing::debug;

use crate::{
    cache::{Outcomes, RouterCache, UsageWindow},
    error::{AttemptError, Error, FailureKind},
    outcome::{OutcomeCallbacks, OutcomeWrite},
    registry::{Deployment, Registry},
    settings::Settings,
};

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
    /// The deployment is the only one in its model group that could take the failed call.
    pub single_deployment: bool,
    /// The requested group has a fallback configured.
    pub has_fallback_group: bool,
    /// The failed batch or file call had other deployments to go to, while the deployment alone
    /// serves the group's other calls: its cooldown covers only batch and file calls
    /// ([`batch_cooldown_id`]), so those move on and the other calls keep their only deployment.
    pub batch_calls_only: bool,
}

impl GroupShape {
    /// The shape of a failed call, `batch_call` when it was a batch or file call: `alone` says
    /// whether the deployment was the only one that could take a call of the kind it is given.
    pub(crate) fn of_failure(
        batch_call: bool,
        has_fallback_group: bool,
        alone: impl Fn(bool) -> bool,
    ) -> Self {
        let alone_outside_batches = alone(false);
        let single_deployment = if batch_call {
            alone(true)
        } else {
            alone_outside_batches
        };
        Self {
            single_deployment,
            has_fallback_group,
            batch_calls_only: alone_outside_batches && !single_deployment,
        }
    }
}

/// The cache key of a cooldown covering only the batch and file calls of deployment `id`.
pub(crate) fn batch_cooldown_id(id: &str) -> String {
    format!("{id}:batch-calls")
}

/// `_core_team_model_has_alternatives`: another routable deployment serves the deployment's team
/// public name.
pub(crate) fn team_model_has_alternatives(
    registry: &Registry,
    deployment: &Deployment,
    batch_call: bool,
) -> bool {
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
                .filter(|sibling| deployment.shares_calls_with(sibling, batch_call))
                .count()
                > 1
        })
}

/// `_core_routing_group_has_alternatives`: `model_group` names a callable routing group whose
/// members span more than one deployment that could take `deployment`'s place on the failed call.
pub(crate) fn routing_group_has_alternatives(
    registry: &Registry,
    model_group: &str,
    deployment: &Deployment,
    batch_call: bool,
) -> bool {
    let resolved = registry.alias_of(model_group).unwrap_or(model_group);
    registry.routing_group_named(resolved).is_some_and(|group| {
        group
            .models
            .iter()
            .map(|member| {
                registry
                    .named(member)
                    .map_or(0, |members| deployment.peers_in(&members, batch_call))
            })
            .sum::<usize>()
            > 1
    })
}

/// How a failure cools its deployment down. [`cooldown_rule`] decides it from the settings, the
/// deployment, its group's shape and the failure alone, and [`Health`] then evaluates it against
/// the cache.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum CooldownRule {
    /// The failure never cools the deployment down.
    Never,
    /// No `allowed_fails` is written anywhere: the status and this minute's share of failed
    /// requests decide ([`CooldownRule::ratio_cools`]).
    Ratio {
        status: u16,
        single_deployment: bool,
    },
    /// An `allowed_fails` is written: the failure counts under `suffix` for `ttl`, and the
    /// deployment cools once the count passes `allowed`.
    Counter {
        allowed: f64,
        suffix: Option<String>,
        ttl: Duration,
    },
}

impl CooldownRule {
    /// `_should_cooldown_deployment` without an `allowed_fails`: a 429 with other deployments
    /// to serve the group, a status no retry clears, or a minute that mostly failed. A group's
    /// only deployment needs a busy minute in which every request failed.
    fn ratio_cools(status: u16, single_deployment: bool, outcomes: Outcomes) -> bool {
        let Outcomes { successes, fails } = outcomes;
        let total = successes + fails;
        let percent_fails = if total > 0 {
            fails as f64 / total as f64
        } else {
            0.0
        };
        (status == 429 && !single_deployment)
            || (percent_fails == 1.0 && total >= SINGLE_DEPLOYMENT_TRAFFIC_FAILURE_THRESHOLD)
            || (percent_fails > FAILURE_THRESHOLD_PERCENT
                && total >= FAILURE_THRESHOLD_MINIMUM_REQUESTS
                && !single_deployment)
            || !should_retry_status(status)
    }
}

/// `_should_run_cooldown_logic`: the failure, the deployment or the router rules a cooldown out
/// before any counting.
fn skips_cooldown_logic(
    settings: &Settings,
    deployment: &Deployment,
    failure: &AttemptError,
    cooldown: Duration,
) -> bool {
    let caller_owned = matches!(
        failure.kind,
        FailureKind::Declined
            | FailureKind::CallerCredentialAuthentication
            | FailureKind::CallerCredentialRateLimit
    ) || failure.caller_scoped;
    let policy = deployment.model_info.allowed_fails_policy.as_ref();
    let status_cools =
        is_cooldown_required(failure) || deployment_allowed_fails(policy, failure).is_some();
    let provider_default = settings
        .provider_default_deployment_ids
        .contains(&deployment.id);
    caller_owned
        || cooldown.is_zero()
        || settings.disable_cooldowns
        || !status_cools
        || provider_default
}

/// `_should_run_cooldown_logic` and the branches of `_should_cooldown_deployment`, as the rule a
/// failure of `deployment` follows when it would cool it down for `cooldown`.
pub(crate) fn cooldown_rule(
    settings: &Settings,
    deployment: &Deployment,
    shape: GroupShape,
    failure: &AttemptError,
    cooldown: Duration,
) -> CooldownRule {
    if skips_cooldown_logic(settings, deployment, failure, cooldown) {
        return CooldownRule::Never;
    }
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
        return CooldownRule::Never;
    }
    let deployment_allowed = deployment.model_info.allowed_fails;
    if deployment_policy.is_some() || deployment_allowed.is_some() {
        // `_should_cooldown_based_on_deployment_policy`: a deployment-wide `allowed_fails`
        // keeps the lone-deployment safety net; a policy entry naming the error does not.
        if deployment_fails.is_none()
            && lone
            && (deployment_allowed.is_some() || policy_fails.is_none())
        {
            return CooldownRule::Never;
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
        let ttl = configured_cooldown(deployment).unwrap_or_else(|| settings.fails_window());
        return CooldownRule::Counter {
            allowed,
            suffix,
            ttl,
        };
    }
    if settings.allowed_fails_policy.is_none() && !settings.allowed_fails_set() {
        return CooldownRule::Ratio {
            status,
            single_deployment: shape.single_deployment,
        };
    }
    if lone && policy_fails.is_none() {
        return CooldownRule::Never;
    }
    let allowed = policy_fails.unwrap_or_else(|| settings.effective_allowed_fails());
    CooldownRule::Counter {
        allowed: f64::from(allowed),
        suffix: None,
        ttl: settings.fails_window(),
    }
}

/// The deployment's own `cooldown_time`: `model_info` first, then `litellm_params`
/// (`_first_present(model_info, litellm_params, key="cooldown_time")`). It is both the
/// cooldown and the TTL of the deployment's `allowed_fails:generic` counter.
fn configured_cooldown(deployment: &Deployment) -> Option<Duration> {
    deployment
        .cooldown_time()
        .filter(|seconds| *seconds >= 0.0)
        .and_then(|seconds| Duration::try_from_secs_f64(seconds).ok())
}

/// An outcome the router did not see itself (`Health.report`).
#[derive(Clone, Debug)]
pub enum Outcome {
    Success { total_tokens: u64 },
    Failure(AttemptError),
}

#[derive(Clone)]
pub struct Health<C> {
    cache: C,
    clock: Arc<dyn Clock>,
}

impl<C: RouterCache> Health<C> {
    pub fn new(cache: C, clock: Arc<dyn Clock>) -> Self {
        Self { cache, clock }
    }

    pub fn cache(&self) -> &C {
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
        let values = self.cache.cooldowns(ids).await?;
        let now = self.now();
        Ok(ids
            .iter()
            .zip(values)
            .filter_map(|(id, value)| {
                let value = value?;
                (value.timestamp + value.cooldown_time - now > 0.0).then(|| (id.clone(), value))
            })
            .collect())
    }

    /// [`Self::active_cooldowns`] for a call, which on a batch or file call (`batch_call`) also
    /// finds the cooldowns covering only those calls, each under its deployment's own id.
    pub async fn cooling(
        &self,
        ids: &[String],
        batch_call: bool,
    ) -> Result<Vec<(String, CooldownValue)>, Error> {
        if !batch_call {
            return self.active_cooldowns(ids).await;
        }
        let batch_ids: Vec<String> = ids.iter().map(|id| batch_cooldown_id(id)).collect();
        let found = self
            .active_cooldowns(&[ids, batch_ids.as_slice()].concat())
            .await?;
        Ok(ids
            .iter()
            .zip(&batch_ids)
            .filter_map(|(id, batch_id)| {
                found
                    .iter()
                    .find(|(cooled, _)| cooled == id || cooled == batch_id)
                    .map(|(_, value)| (id.clone(), value.clone()))
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
        self.cache.put_cooldown(id, &value, cooldown).await?;
        Ok(())
    }

    /// The successes and fails this worker saw this minute.
    pub async fn outcomes_this_minute(&self, id: &str) -> Result<Outcomes, Error> {
        Ok(self.cache.outcomes(id).await?)
    }

    /// `increment_deployment_successes_for_current_minute`.
    pub async fn count_success(&self, id: &str) -> Result<(), Error> {
        Ok(self.cache.count_success(id).await?)
    }

    /// `increment_deployment_failures_for_current_minute`.
    pub async fn count_failure(&self, id: &str) -> Result<(), Error> {
        Ok(self.cache.count_failure(id).await?)
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
            .or_else(|| deployment.params_cooldown_time())
            .filter(|seconds| *seconds >= 0.0)
            .and_then(|seconds| Duration::try_from_secs_f64(seconds).ok());
        configured
            .or(failure.retry_after)
            .unwrap_or(settings.cooldown_time)
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
        let cools = match cooldown_rule(settings, deployment, shape, failure, cooldown) {
            CooldownRule::Never => false,
            CooldownRule::Ratio {
                status,
                single_deployment,
            } => {
                let outcomes = self.outcomes_this_minute(&deployment.id).await?;
                CooldownRule::ratio_cools(status, single_deployment, outcomes)
            }
            // `should_cooldown_based_on_allowed_fails_policy`: count this failure, then compare.
            CooldownRule::Counter {
                allowed,
                suffix,
                ttl,
            } => {
                let fails = self
                    .cache
                    .increment_allowed_fails(&deployment.id, suffix.as_deref(), ttl)
                    .await?;
                fails > allowed
            }
        };
        if !cools {
            return Ok(None);
        }
        debug!(deployment_id = %deployment.id, cooldown_seconds = cooldown.as_secs_f64(), batch_calls_only = shape.batch_calls_only, "cooling deployment down");
        let id = if shape.batch_calls_only {
            batch_cooldown_id(&deployment.id)
        } else {
            deployment.id.clone()
        };
        self.add_cooldown(&id, failure, cooldown).await?;
        Ok(Some(cooldown))
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
    /// deployment's own), unless there is nothing to add. Returns the window it added to.
    pub async fn add_usage(
        &self,
        deployment: &Deployment,
        model: Option<&str>,
        total_tokens: f64,
        requests: f64,
    ) -> Result<Option<UsageWindow>, Error> {
        if total_tokens <= 0.0 && requests <= 0.0 {
            return Ok(None);
        }
        let window = self.usage_window(deployment, model);
        self.cache
            .add_usage(&window, total_tokens, requests)
            .await?;
        Ok(Some(window))
    }

    /// `async_deployment_callback_on_failure`: a failed request counts against the deployment's
    /// per-minute requests under `model` (see [`Self::add_usage`]), whether or not it tracks usage.
    pub async fn count_failed_request(
        &self,
        deployment: &Deployment,
        model: Option<&str>,
    ) -> Result<(), Error> {
        let window = self.usage_window(deployment, model);
        self.cache.add_request(&window).await?;
        Ok(())
    }

    /// This minute's (tpm, rpm) for a deployment.
    pub async fn usage(&self, deployment: &Deployment) -> Result<(u64, u64), Error> {
        let counts = self
            .cache
            .usage(&self.usage_window(deployment, None))
            .await?;
        let count = |value: Option<f64>| value.map_or(0, |count| count as u64);
        Ok((count(counts.tokens), count(counts.requests)))
    }

    fn usage_window(&self, deployment: &Deployment, model: Option<&str>) -> UsageWindow {
        UsageWindow {
            deployment_id: deployment.id.clone(),
            model: model.unwrap_or(&deployment.params.model).to_owned(),
            minute: current_minute(self.clock.now()),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use rstest::rstest;

    use super::*;
    use crate::{
        cache::{InMemoryRouterCache, UsageCounts},
        config::{RouterConfig, RouterDynamicConfig},
        registry::DeploymentSpec,
    };
    use litellm_router_types::RoutingGroupSpec;

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

    fn health() -> Health<InMemoryRouterCache> {
        Health::new(InMemoryRouterCache::default(), Arc::new(SystemClock))
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
        assert_eq!(
            health.outcomes_this_minute("a").await.unwrap().fails,
            counted
        );
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
        assert_eq!(health.outcomes_this_minute("a").await.unwrap().fails, 1);
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

    const MINUTE: Duration = Duration::from_secs(60);

    fn failed(kind: FailureKind, status: u16) -> AttemptError {
        AttemptError::new(kind, Some(status))
    }

    fn server_error() -> AttemptError {
        failed(FailureKind::InternalServer, 500)
    }

    fn caller_scoped() -> AttemptError {
        let mut failure = server_error();
        failure.caller_scoped = true;
        failure
    }

    fn allowed_fails(allowed: u32) -> RouterDynamicConfig {
        RouterDynamicConfig {
            allowed_fails: Some(allowed),
            ..RouterDynamicConfig::default()
        }
    }

    fn router(config: serde_json::Value, dynamic: RouterDynamicConfig) -> Settings {
        let config: RouterConfig = serde_json::from_value(config).unwrap();
        Settings::new(&config, Arc::new(dynamic))
    }

    fn defaults() -> Settings {
        settings(RouterDynamicConfig::default())
    }

    fn rate_limits_allowed(allowed: u32) -> Settings {
        router(
            serde_json::json!({"allowed_fails_policy": {"RateLimitErrorAllowedFails": allowed}}),
            RouterDynamicConfig::default(),
        )
    }

    const ALONE: GroupShape = GroupShape {
        single_deployment: true,
        has_fallback_group: false,
        batch_calls_only: false,
    };
    const ALONE_WITH_A_FALLBACK: GroupShape = GroupShape {
        single_deployment: true,
        has_fallback_group: true,
        batch_calls_only: false,
    };
    const AMONG_OTHERS: GroupShape = GroupShape {
        single_deployment: false,
        has_fallback_group: false,
        batch_calls_only: false,
    };

    fn ratio(status: u16, single_deployment: bool) -> CooldownRule {
        CooldownRule::Ratio {
            status,
            single_deployment,
        }
    }

    fn counter(allowed: f64, suffix: Option<&str>, ttl: Duration) -> CooldownRule {
        CooldownRule::Counter {
            allowed,
            suffix: suffix.map(str::to_owned),
            ttl,
        }
    }

    /// The counter of a router with no `cooldown_time` lives for `DEFAULT_COOLDOWN_TIME`.
    fn default_window() -> Duration {
        defaults().fails_window()
    }

    #[rstest]
    // The failure ratio of the minute decides while no `allowed_fails` is written.
    #[case::a_lone_deployment_with_a_fallback_and_no_allowed_fails(
        defaults(), serde_json::json!({}), ALONE_WITH_A_FALLBACK, server_error(), ratio(500, true)
    )]
    #[case::one_of_several_with_no_allowed_fails(
        defaults(), serde_json::json!({}), AMONG_OTHERS, failed(FailureKind::RateLimit, 429), ratio(429, false)
    )]
    #[case::an_allowed_fails_that_is_only_the_process_default(
        settings(RouterDynamicConfig { allowed_fails_is_default: true, ..allowed_fails(3) }),
        serde_json::json!({}), AMONG_OTHERS, server_error(), ratio(500, false)
    )]
    #[case::an_unreadable_status_reads_as_a_500(
        defaults(), serde_json::json!({}), AMONG_OTHERS, AttemptError::new(FailureKind::Other, None), ratio(500, false)
    )]
    // A written `allowed_fails` counts instead.
    #[case::a_lone_deployment_with_a_fallback_and_allowed_fails_zero(
        settings(allowed_fails(0)), serde_json::json!({}), ALONE_WITH_A_FALLBACK, server_error(),
        counter(0.0, None, default_window())
    )]
    #[case::one_of_several_with_allowed_fails(
        settings(allowed_fails(2)), serde_json::json!({}), AMONG_OTHERS, server_error(),
        counter(2.0, None, default_window())
    )]
    #[case::the_counter_lives_for_the_routers_cooldown(
        router(serde_json::json!({"cooldown_time": 60}), allowed_fails(2)),
        serde_json::json!({}), AMONG_OTHERS, server_error(), counter(2.0, None, MINUTE)
    )]
    #[case::a_lone_deployment_without_a_fallback_is_never_counted_out(
        settings(allowed_fails(0)), serde_json::json!({}), ALONE, server_error(), CooldownRule::Never
    )]
    // A router policy that names the failure counts, lone deployment or not.
    #[case::a_router_policy_names_the_failure(
        rate_limits_allowed(4), serde_json::json!({}), ALONE, failed(FailureKind::RateLimit, 429),
        counter(4.0, None, default_window())
    )]
    #[case::a_router_policy_that_does_not_name_it_counts_the_default_three(
        rate_limits_allowed(4), serde_json::json!({}), AMONG_OTHERS, server_error(),
        counter(3.0, None, default_window())
    )]
    #[case::a_router_policy_that_does_not_name_it_spares_a_lone_deployment(
        rate_limits_allowed(4), serde_json::json!({}), ALONE, server_error(), CooldownRule::Never
    )]
    // The deployment's own `allowed_fails` and policy count under their own key.
    #[case::the_deployments_allowed_fails(
        defaults(), serde_json::json!({"allowed_fails": 2}), AMONG_OTHERS, server_error(),
        counter(2.0, Some("generic"), default_window())
    )]
    #[case::its_counter_lives_for_the_deployments_cooldown(
        defaults(), serde_json::json!({"allowed_fails": 2, "cooldown_time": 60}), AMONG_OTHERS, server_error(),
        counter(2.0, Some("generic"), MINUTE)
    )]
    #[case::the_deployments_allowed_fails_spares_it_when_alone(
        rate_limits_allowed(4), serde_json::json!({"allowed_fails": 2}), ALONE,
        failed(FailureKind::RateLimit, 429), CooldownRule::Never
    )]
    #[case::the_deployments_policy_names_the_failure_even_when_alone(
        defaults(), serde_json::json!({"allowed_fails_policy": {"RateLimitErrorAllowedFails": 1}}), ALONE,
        failed(FailureKind::RateLimit, 429), counter(1.0, Some("RateLimitError"), default_window())
    )]
    #[case::the_deployments_policy_cools_a_status_that_otherwise_would_not(
        defaults(), serde_json::json!({"allowed_fails_policy": {"BadRequestErrorAllowedFails": 0}}), AMONG_OTHERS,
        failed(FailureKind::BadRequest, 400), counter(0.0, Some("BadRequestError"), default_window())
    )]
    #[case::a_deployment_policy_that_does_not_name_it_falls_to_the_routers(
        rate_limits_allowed(4), serde_json::json!({"allowed_fails_policy": {"BadRequestErrorAllowedFails": 0}}),
        ALONE, failed(FailureKind::RateLimit, 429), counter(4.0, None, default_window())
    )]
    #[case::a_deployment_policy_that_does_not_name_it_spares_a_lone_deployment(
        defaults(), serde_json::json!({"allowed_fails_policy": {"BadRequestErrorAllowedFails": 0}}),
        ALONE, failed(FailureKind::RateLimit, 429), CooldownRule::Never
    )]
    #[case::a_deployment_policy_that_does_not_name_it_counts_the_default_three(
        defaults(), serde_json::json!({"allowed_fails_policy": {"BadRequestErrorAllowedFails": 0}}),
        AMONG_OTHERS, failed(FailureKind::RateLimit, 429), counter(3.0, None, default_window())
    )]
    // What rules a cooldown out before any rule is picked.
    #[case::a_status_that_needs_no_cooldown(
        defaults(), serde_json::json!({}), AMONG_OTHERS, failed(FailureKind::BadRequest, 400), CooldownRule::Never
    )]
    #[case::a_request_the_deployment_declined(
        defaults(), serde_json::json!({}), AMONG_OTHERS, failed(FailureKind::Declined, 500), CooldownRule::Never
    )]
    #[case::a_failure_of_the_callers_own_credential(
        defaults(), serde_json::json!({}), AMONG_OTHERS, caller_scoped(), CooldownRule::Never
    )]
    #[case::cooldowns_are_off(
        router(serde_json::json!({"disable_cooldowns": true}), RouterDynamicConfig::default()),
        serde_json::json!({}), AMONG_OTHERS, server_error(), CooldownRule::Never
    )]
    #[case::a_provider_default_deployment(
        router(serde_json::json!({"provider_default_deployment_ids": ["a"]}), RouterDynamicConfig::default()),
        serde_json::json!({}), AMONG_OTHERS, server_error(), CooldownRule::Never
    )]
    #[case::a_lone_deployment_out_of_credit(
        defaults(), serde_json::json!({}), ALONE_WITH_A_FALLBACK, failed(FailureKind::Other, 402), CooldownRule::Never
    )]
    #[case::one_of_several_out_of_credit(
        defaults(), serde_json::json!({}), AMONG_OTHERS, failed(FailureKind::Other, 402), ratio(402, false)
    )]
    fn the_cooldown_rule_follows_where_allowed_fails_is_written(
        #[case] settings: Settings,
        #[case] info: serde_json::Value,
        #[case] shape: GroupShape,
        #[case] failure: AttemptError,
        #[case] expected: CooldownRule,
    ) {
        let deployment = Deployment::from_spec(deployment("a", info)).unwrap();
        let rule = cooldown_rule(&settings, &deployment, shape, &failure, MINUTE);
        assert_eq!(rule, expected);
    }

    #[rstest]
    fn a_cooldown_of_no_time_is_never_applied() {
        let deployment = Deployment::from_spec(deployment("a", serde_json::json!({}))).unwrap();
        let rule = cooldown_rule(
            &defaults(),
            &deployment,
            AMONG_OTHERS,
            &server_error(),
            Duration::ZERO,
        );
        assert_eq!(rule, CooldownRule::Never);
    }

    #[rstest]
    #[case::a_lone_deployments_500s_below_the_traffic_floor(500, true, 0, 999, false)]
    #[case::a_lone_deployment_whose_busy_minute_all_failed(500, true, 0, 1000, true)]
    #[case::a_lone_deployment_with_one_success_in_a_busy_minute(500, true, 1, 1000, false)]
    #[case::a_lone_deployments_rate_limit(429, true, 1, 0, false)]
    #[case::a_rate_limit_among_others(429, false, 1, 0, true)]
    #[case::most_of_the_minute_failed_among_others(500, false, 2, 3, true)]
    #[case::half_of_the_minute_failed_among_others(500, false, 3, 3, false)]
    #[case::too_few_requests_to_judge(500, false, 0, 4, false)]
    #[case::a_status_no_retry_clears(401, true, 5, 1, true)]
    #[case::a_first_failure_with_nothing_counted(500, false, 0, 0, false)]
    fn the_failure_ratio_cools_by_status_then_by_the_minutes_traffic(
        #[case] status: u16,
        #[case] single_deployment: bool,
        #[case] successes: u64,
        #[case] fails: u64,
        #[case] cools: bool,
    ) {
        let outcomes = Outcomes { successes, fails };
        assert_eq!(
            CooldownRule::ratio_cools(status, single_deployment, outcomes),
            cools
        );
    }

    /// A counter counts the failure itself before it compares, so `allowed_fails: 0` cools on
    /// the first failure.
    #[rstest]
    #[case::allowed_fails_zero(0, true, true)]
    #[case::allowed_fails_one(1, false, true)]
    #[case::allowed_fails_two(2, false, false)]
    #[tokio::test]
    async fn a_counter_cools_once_the_failures_pass_what_is_allowed(
        #[case] allowed: u32,
        #[case] first_cools: bool,
        #[case] second_cools: bool,
    ) {
        let health = health();
        let settings = settings(allowed_fails(allowed));
        let deployment = Deployment::from_spec(deployment("a", serde_json::json!({}))).unwrap();
        let fail = || async {
            health
                .record_failure(
                    &settings,
                    &deployment,
                    ALONE_WITH_A_FALLBACK,
                    None,
                    &server_error(),
                )
                .await
                .unwrap()
                .is_some()
        };
        assert_eq!((fail().await, fail().await), (first_cools, second_cools));
    }

    /// A lone deployment with a fallback group and no `allowed_fails` keeps serving through 500s.
    #[rstest]
    #[tokio::test]
    async fn the_failure_ratio_spares_a_lone_deployment_that_fails_with_500s() {
        let health = health();
        let deployment = Deployment::from_spec(deployment("a", serde_json::json!({}))).unwrap();
        let fail = || async {
            health
                .record_failure(
                    &defaults(),
                    &deployment,
                    ALONE_WITH_A_FALLBACK,
                    None,
                    &server_error(),
                )
                .await
                .unwrap()
        };
        assert_eq!((fail().await, fail().await), (None, None));
        assert_eq!(health.outcomes_this_minute("a").await.unwrap().fails, 2);
    }

    /// One of several deployments cools once most of a minute of at least five requests failed,
    /// read from what this worker counted.
    #[rstest]
    #[tokio::test]
    async fn the_failure_ratio_cools_one_of_several_on_the_fifth_failure_of_a_minute() {
        let health = health();
        let deployment = Deployment::from_spec(deployment("a", serde_json::json!({}))).unwrap();
        let fail = || async {
            health
                .record_failure(
                    &defaults(),
                    &deployment,
                    AMONG_OTHERS,
                    None,
                    &server_error(),
                )
                .await
                .unwrap()
                .is_some()
        };
        let cooled = (
            fail().await,
            fail().await,
            fail().await,
            fail().await,
            fail().await,
        );
        assert_eq!(cooled, (false, false, false, false, true));
    }

    /// What a failure counted, read back as the next count under the router's key, the
    /// deployment's `generic` key and the key of the failure's class.
    #[rstest]
    #[case::the_routers_allowed_fails(
        settings(allowed_fails(5)), serde_json::json!({}), AMONG_OTHERS, [2.0, 1.0, 1.0]
    )]
    #[case::the_deployments_allowed_fails(
        defaults(), serde_json::json!({"allowed_fails": 5}), AMONG_OTHERS, [1.0, 2.0, 1.0]
    )]
    #[case::the_deployments_policy(
        defaults(), serde_json::json!({"allowed_fails_policy": {"RateLimitErrorAllowedFails": 5}}),
        AMONG_OTHERS, [1.0, 1.0, 2.0]
    )]
    #[case::the_failure_ratio_counts_nothing(defaults(), serde_json::json!({}), AMONG_OTHERS, [1.0, 1.0, 1.0])]
    #[case::a_failure_that_never_cools_counts_nothing(
        settings(allowed_fails(5)), serde_json::json!({}), ALONE, [1.0, 1.0, 1.0]
    )]
    #[tokio::test]
    async fn a_failure_counts_under_the_key_of_the_rule_it_follows(
        #[case] settings: Settings,
        #[case] info: serde_json::Value,
        #[case] shape: GroupShape,
        #[case] next_counts: [f64; 3],
    ) {
        let health = health();
        let deployment = Deployment::from_spec(deployment("a", info)).unwrap();
        health
            .record_failure(
                &settings,
                &deployment,
                shape,
                None,
                &failed(FailureKind::RateLimit, 429),
            )
            .await
            .unwrap();
        let cache = health.cache();
        let next = |suffix| async move {
            cache
                .increment_allowed_fails("a", suffix, MINUTE)
                .await
                .unwrap()
        };
        assert_eq!(
            [
                next(None).await,
                next(Some("generic")).await,
                next(Some("RateLimitError")).await
            ],
            next_counts
        );
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
        assert_eq!(
            team_model_has_alternatives(&registry, &first, false),
            expected
        );
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
        let first = registry.deployment("a").unwrap();
        assert_eq!(
            routing_group_has_alternatives(&registry, requested, &first, false),
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
        assert_eq!(health.outcomes_this_minute("a").await.unwrap().fails, 1);
    }

    #[rstest]
    #[case::the_deployments_own_model(None, "openai/a")]
    #[case::a_filled_wildcard(Some("openai/gpt-4o"), "openai/gpt-4o")]
    #[tokio::test]
    async fn usage_is_keyed_by_the_model_served(#[case] model: Option<&str>, #[case] keyed: &str) {
        let cache = InMemoryRouterCache::default();
        let clock = Arc::new(SystemClock);
        let health = Health::new(cache.clone(), clock.clone());
        let deployment = Deployment::from_spec(deployment("a", serde_json::json!({}))).unwrap();

        health
            .add_usage(&deployment, model, 15.0, 1.0)
            .await
            .unwrap();

        let window = UsageWindow {
            deployment_id: "a".to_owned(),
            model: keyed.to_owned(),
            minute: current_minute(clock.now()),
        };
        assert_eq!(
            cache.counted(&window),
            UsageCounts {
                tokens: Some(15.0),
                requests: Some(1.0),
            },
        );
    }
}
