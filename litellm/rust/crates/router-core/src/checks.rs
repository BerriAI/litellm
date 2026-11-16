//! The optional pre-call checks the core runs itself, for a host that does not select:
//! `encrypted_content_affinity` (`EncryptedContentAffinityCheck`) and
//! `responses_api_deployment_check` (`DeploymentAffinityCheck`'s Responses API continuity) pin a
//! follow-up request to the deployment that served the turn it continues, which the host reads
//! off the request's own ids into [`Affinity`]. `enforce_model_rate_limits`
//! (`ModelRateLimitingCheck`) refuses an attempt on a deployment past its declared limits, on the
//! counters in the router's cache. A host that selects runs Python's checks instead.

use std::{
    collections::BTreeMap,
    str::FromStr,
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use serde_json::Value;

use crate::{
    cache::{LimitCounter, LimitKind, RouterCache},
    error::{AttemptError, FailureKind},
    health::current_minute,
    host::RoutingContext,
    registry::{Deployment, Registry},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::EnumString, strum::IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
pub enum NativeCheck {
    EncryptedContentAffinity,
    ResponsesApiDeploymentCheck,
    EnforceModelRateLimits,
}

/// `optional_pre_call_checks` and `model_group_affinity_config`.
#[derive(Clone, Debug, Default, PartialEq, serde::Deserialize)]
#[serde(default)]
pub struct PreCallChecks {
    pub global: Vec<String>,
    /// A group's own affinity checks, which replace the global ones for that group.
    pub groups: BTreeMap<String, Vec<String>>,
}

fn lists(names: &[String], check: NativeCheck) -> bool {
    let name: &'static str = check.into();
    names.iter().any(|listed| listed == name)
}

fn all_native(names: &[String]) -> bool {
    names.iter().all(|name| NativeCheck::from_str(name).is_ok())
}

impl PreCallChecks {
    /// Every `optional_pre_call_checks` entry is a check the core runs.
    pub fn global_is_native(&self) -> bool {
        all_native(&self.global)
    }

    /// Every `model_group_affinity_config` entry is a check the core runs.
    pub fn groups_are_native(&self) -> bool {
        self.groups.values().all(|names| all_native(names))
    }

    pub fn is_empty(&self) -> bool {
        self.global.is_empty() && self.groups.is_empty()
    }

    pub fn enforces_model_rate_limits(&self) -> bool {
        lists(&self.global, NativeCheck::EnforceModelRateLimits)
    }

    /// `EncryptedContentAffinityCheck.is_enabled_for_model_group`.
    pub fn encrypted_content_affinity(&self, model_group: &str) -> bool {
        lists(&self.global, NativeCheck::EncryptedContentAffinity)
            || self
                .groups
                .get(model_group)
                .is_some_and(|names| lists(names, NativeCheck::EncryptedContentAffinity))
    }

    /// `DeploymentAffinityCheck._get_effective_flags`: a group's own list wins over the global one.
    fn responses_api(&self, model_group: &str) -> bool {
        lists(
            self.groups.get(model_group).unwrap_or(&self.global),
            NativeCheck::ResponsesApiDeploymentCheck,
        )
    }
}

/// The deployments a request's own ids name, which the host decodes from them.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Affinity {
    /// The deployment `previous_response_id` names (`get_model_id_from_response_id`).
    pub previous_response: Option<String>,
    /// The deployment the first encrypted-content marker in the request names
    /// (`_extract_model_id_from_input`).
    pub encrypted_content: Option<String>,
}

/// The upstream account a deployment's encrypted reasoning decrypts under
/// (`_deployment_encryption_boundary`): an operator-declared `model_info.encrypted_content_boundary`,
/// else its explicit `api_base` and `api_key`. A deployment that leaves either to the provider's
/// defaults has none here, so it is never a peer.
#[derive(Debug, PartialEq)]
struct Boundary<'a> {
    declared: Option<&'a str>,
    pair: Option<(&'a str, &'a str)>,
}

impl<'a> Boundary<'a> {
    fn of(deployment: &'a Deployment) -> Option<Self> {
        let declared = deployment
            .model_info
            .raw
            .get("encrypted_content_boundary")
            .and_then(serde_json::Value::as_str)
            .filter(|name| !name.is_empty());
        let params = &deployment.params;
        let pair = params
            .api_base
            .as_deref()
            .filter(|base| !base.is_empty())
            .zip(
                params
                    .api_key
                    .as_ref()
                    .map(|key| key.expose())
                    .filter(|key| !key.is_empty()),
            );
        (declared.is_some() || pair.is_some()).then_some(Self { declared, pair })
    }

    fn is_shared_with(&self, other: &Self) -> bool {
        match (self.declared, other.declared) {
            (Some(own), Some(theirs)) => own == theirs,
            _ => self.pair.is_some() && self.pair == other.pair,
        }
    }
}

/// Whether `target` decrypts reasoning `origin` minted: it is `origin`, or shares its boundary.
pub fn decrypts(registry: &Registry, origin: &str, target: &Deployment) -> bool {
    if origin == target.id {
        return true;
    }
    let Some(origin) = registry.deployment(origin) else {
        return false;
    };
    Boundary::of(&origin)
        .zip(Boundary::of(target))
        .is_some_and(|(origin, target)| origin.is_shared_with(&target))
}

/// The checks in the order Python registers them: encrypted-content affinity, then the
/// deployment affinity check that carries Responses API continuity.
pub(crate) fn filter(
    checks: &PreCallChecks,
    ctx: &RoutingContext<'_>,
    candidates: Vec<Arc<Deployment>>,
) -> Vec<Arc<Deployment>> {
    let candidates = encrypted_content_affinity(checks, ctx, candidates);
    responses_api_continuity(checks, ctx, candidates)
}

/// `EncryptedContentAffinityCheck.async_filter_deployments`, less the tag stage: tag filtering is
/// a host stage, refused on a host that does not select.
fn encrypted_content_affinity(
    checks: &PreCallChecks,
    ctx: &RoutingContext<'_>,
    candidates: Vec<Arc<Deployment>>,
) -> Vec<Arc<Deployment>> {
    let affinity = &ctx.request.affinity;
    let Some(origin) = affinity
        .encrypted_content
        .as_deref()
        .filter(|_| checks.encrypted_content_affinity(ctx.model_group))
    else {
        return candidates;
    };
    let candidates: Vec<_> = match ctx.target_order {
        Some(order) => candidates
            .into_iter()
            .filter(|deployment| deployment.order() == Some(order))
            .collect(),
        None => candidates,
    };
    let binds_origin = affinity.previous_response.as_deref() == Some(origin);
    let listed = |own: &[String], walked: &[String]| -> Vec<String> {
        if binds_origin {
            return Vec::new();
        }
        own.iter().chain(walked).cloned().collect()
    };
    let excluded = listed(&ctx.request.excluded_ids, ctx.excluded_ids);
    let skipped = listed(&ctx.request.retry_skipped_ids, ctx.skipped_ids);
    let not_excluded: Vec<_> = candidates
        .iter()
        .filter(|deployment| !excluded.contains(&deployment.id))
        .cloned()
        .collect();
    let untried: Vec<_> = not_excluded
        .iter()
        .filter(|deployment| !skipped.contains(&deployment.id))
        .cloned()
        .collect();
    let untried = if untried.is_empty() {
        not_excluded
    } else {
        untried
    };
    if let Some(pinned) = untried.iter().find(|deployment| deployment.id == origin) {
        return vec![pinned.clone()];
    }
    let peers: Vec<_> = untried
        .iter()
        .filter(|deployment| decrypts(ctx.registry, origin, deployment))
        .cloned()
        .collect();
    if !peers.is_empty() {
        return peers;
    }
    if excluded.is_empty() && skipped.is_empty() {
        candidates
    } else {
        untried
    }
}

/// `DeploymentAffinityCheck.async_filter_deployments`'s `previous_response_id` pin. Its user-key and
/// session pins need state and are refused on a host that does not select.
fn responses_api_continuity(
    checks: &PreCallChecks,
    ctx: &RoutingContext<'_>,
    candidates: Vec<Arc<Deployment>>,
) -> Vec<Arc<Deployment>> {
    if ctx.target_order.is_some() || !checks.responses_api(ctx.model_group) {
        return candidates;
    }
    let Some(previous) = ctx.request.affinity.previous_response.as_deref() else {
        return candidates;
    };
    match candidates
        .iter()
        .find(|deployment| deployment.id == previous)
    {
        Some(pinned) => vec![pinned.clone()],
        None => candidates,
    }
}

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

/// `HH:MM` in UTC, which `parse_daily_reset_time` accepts.
fn reset_offset(value: Option<&Value>) -> Option<Duration> {
    let Some(value) = value.filter(|value| !value.is_null()) else {
        return Some(Duration::ZERO);
    };
    let (hours, minutes) = value.as_str()?.split_once(':')?;
    let digits = |part: &str, below: u64| {
        (part.len() == 2 && part.bytes().all(|byte| byte.is_ascii_digit()))
            .then(|| part.parse::<u64>().ok())
            .flatten()
            .filter(|number| *number < below)
    };
    Some(Duration::from_secs(
        digits(hours, 24)? * 3600 + digits(minutes, 60)? * 60,
    ))
}

impl Limits {
    /// `None` when the deployment's daily params do not validate, which skips the check.
    fn of(deployment: &Deployment) -> Option<Self> {
        let declared = |key: &str| {
            [
                &deployment.extra,
                &deployment.raw_params,
                &deployment.model_info.raw,
            ]
            .iter()
            .find_map(|section| section.get(key).filter(|value| !value.is_null()))
            .and_then(Value::as_f64)
        };
        let daily = |key: &str| deployment.raw_params.get(key).and_then(Value::as_f64);
        Some(Self {
            tpm: declared("tpm"),
            rpm: declared("rpm"),
            tpd: daily("tpd"),
            rpd: daily("rpd"),
            reset: reset_offset(deployment.raw_params.get("daily_reset_time"))?,
        })
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

fn counter(deployment: &Deployment, kind: LimitKind, window: String) -> LimitCounter {
    LimitCounter {
        deployment_id: deployment.id.clone(),
        model: deployment.params.model.clone(),
        kind,
        window,
    }
}

fn number(value: f64) -> String {
    if value.fract() == 0.0 && value.abs() < 1e15 {
        format!("{}", value as i64)
    } else {
        value.to_string()
    }
}

/// The `RateLimitError` the check raises: never retried, and retried after `retry_after` by a
/// caller.
fn refused(
    deployment: &Deployment,
    kind: LimitKind,
    limit: f64,
    usage: f64,
    retry_after: Duration,
) -> AttemptError {
    let name: &'static str = kind.into();
    let mut error = AttemptError::new(FailureKind::RateLimit, Some(429));
    error.retry_after = Some(retry_after);
    error.num_retries = Some(0);
    error.detail = Some(
        format!(
            "Model rate limit exceeded. {} limit={}, current usage={}. id={}, model_group={}",
            name.to_uppercase(),
            number(limit),
            number(usage),
            deployment.id,
            deployment.model_name,
        )
        .into(),
    );
    error
}

/// `ModelRateLimitingCheck.async_pre_call_check`: the deployment's token windows must have room,
/// then its request windows count this request, refunded when that goes past a limit. A cache
/// that fails lets the request through.
pub(crate) async fn admit(
    cache: &impl RouterCache,
    deployment: &Deployment,
    now: SystemTime,
) -> Option<AttemptError> {
    let limits = Limits::of(deployment)?;
    let minute = current_minute(now);
    let day = limits.day(now);
    let used = |kind, window: &str, limit: f64| {
        let counter = counter(deployment, kind, window.to_owned());
        async move { cache.limit_usage(&counter, limit).await.ok().flatten() }
    };
    if let Some(limit) = limits.tpm
        && let Some(usage) = used(LimitKind::Tpm, &minute, limit).await
        && usage >= limit
    {
        return Some(refused(deployment, LimitKind::Tpm, limit, usage, MINUTE));
    }
    if let Some(limit) = limits.tpd
        && let Some(usage) = used(LimitKind::Tpd, &day, limit).await
        && usage >= limit
    {
        return Some(refused(
            deployment,
            LimitKind::Tpd,
            limit,
            usage,
            limits.until_reset(now),
        ));
    }
    let charged: Vec<(LimitCounter, i64, Duration)> = [
        limits
            .rpm
            .map(|_| (counter(deployment, LimitKind::Rpm, minute), 1, MINUTE)),
        limits
            .rpd
            .map(|_| (counter(deployment, LimitKind::Rpd, day), 1, DAY)),
    ]
    .into_iter()
    .flatten()
    .collect();
    if charged.is_empty() {
        return None;
    }
    let counts = cache
        .add_limit_usages(&charged)
        .await
        .ok()
        .flatten()
        .filter(|counts| counts.len() == charged.len())?;
    let count = |kind| {
        charged
            .iter()
            .zip(&counts)
            .find(|((counter, ..), _)| counter.kind == kind)
            .map(|(_, count)| count.trunc())
    };
    let over = [
        (LimitKind::Rpm, limits.rpm, MINUTE),
        (LimitKind::Rpd, limits.rpd, limits.until_reset(now)),
    ]
    .into_iter()
    .find_map(|(kind, limit, retry_after)| {
        let limit = limit?;
        let usage = count(kind)?;
        (usage > limit).then(|| refused(deployment, kind, limit, usage, retry_after))
    })?;
    let refund: Vec<_> = charged
        .into_iter()
        .map(|(counter, _, ttl)| (counter, -1, ttl))
        .collect();
    if let Err(error) = cache.add_limit_usages(&refund).await {
        tracing::debug!(%error, "the model rate limit refund was not recorded");
    }
    Some(over)
}

/// `ModelRateLimitingCheck.async_log_success_event`: a success's tokens count against the
/// deployment's minute and day.
pub(crate) async fn count_tokens(
    cache: &impl RouterCache,
    deployment: &Deployment,
    total_tokens: u64,
    now: SystemTime,
) {
    let Some(limits) = Limits::of(deployment) else {
        return;
    };
    if total_tokens == 0 || deployment.params.model.is_empty() {
        return;
    }
    let tokens = i64::try_from(total_tokens).unwrap_or(i64::MAX);
    for (kind, window, ttl) in [
        (LimitKind::Tpm, current_minute(now), MINUTE),
        (LimitKind::Tpd, limits.day(now), DAY),
    ] {
        if let Err(error) = cache
            .add_limit_usage(&counter(deployment, kind, window), tokens, ttl)
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
    use serde_json::json;

    use super::*;

    /// 2024-03-01 00:30:00 UTC.
    fn half_past_midnight() -> SystemTime {
        UNIX_EPOCH + Duration::from_secs(1_709_253_000)
    }

    #[rstest]
    #[case::midnight(json!(null), "2024-03-01", 86_400 - 1800)]
    #[case::before_the_reset(json!("01:00"), "2024-02-29", 1800)]
    #[case::after_the_reset(json!("00:15"), "2024-03-01", 86_400 - 900)]
    fn the_daily_window_starts_over_at_the_reset_time(
        #[case] reset: Value,
        #[case] day: &str,
        #[case] until_reset: u64,
    ) {
        let limits = Limits {
            tpm: None,
            rpm: None,
            tpd: None,
            rpd: None,
            reset: reset_offset(Some(&reset)).unwrap(),
        };
        assert_eq!(
            (
                limits.day(half_past_midnight()),
                limits.until_reset(half_past_midnight())
            ),
            (day.to_owned(), Duration::from_secs(until_reset))
        );
    }

    #[rstest]
    #[case::hour_out_of_range("24:00")]
    #[case::one_digit("1:00")]
    #[case::not_a_time("noon")]
    fn a_reset_time_that_is_not_hh_mm_skips_the_check(#[case] reset: &str) {
        assert_eq!(reset_offset(Some(&json!(reset))), None);
    }
}
