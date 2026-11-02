//! Recovering a stream that failed after its attempt returned it. The host wraps every streamed
//! response; when the stream fails before the caller received content, the wrapper asks the core
//! what may still take over and hands the failure back here. Mirrors the decisions of the Python
//! router's stream wrappers: `_stream_can_fall_back`, `_refusal_fallback_available`,
//! `_anthropic_messages_stream_can_retry`, `_stream_retry_budget`, `_astream_retry_same_group`,
//! and the `async_function_with_fallbacks_common_utils` re-entry of the mid-stream fallback.

use std::sync::Arc;

use serde::Deserialize;

use super::{Hop, Target, Tried, Walk};
use crate::{
    Router, Snapshot,
    config::{Op, RouterDynamicConfig},
    error::{AttemptError, Error, FailureKind},
    fallback::{FallbackTarget, Fallbacks},
    health::should_retry_status,
    host::RouterRequest,
    select::{Candidates, SelectInputs},
};

/// Where a streamed response came from: the attempt that returned it.
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct StreamOrigin {
    /// The name the attempt served: the requested group, or the fallback target.
    pub group: String,
    /// The `order` level an order fallback targeted.
    pub order: Option<i64>,
    /// `fallback_depth` of the attempt.
    pub depth: u32,
    pub level: u32,
    /// The deployment the attempt ran on.
    pub deployment_id: Option<String>,
    /// Same-group stream retries this request already spent (`attempted_retries_for_request`).
    pub attempted_retries: u32,
    /// The budget the first stream retry committed to (`committed_retry_budget_for_request`).
    pub committed_budget: Option<u32>,
}

/// What may still take over a stream that fails before content.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StreamPolicy {
    /// `_anthropic_messages_stream_can_retry`.
    pub can_retry: bool,
    /// `_stream_can_fall_back`.
    pub can_fall_back: bool,
    /// `_refusal_fallback_available`.
    pub refusal_fallback: bool,
    /// `_stream_retry_budget` for the failure kind asked about: the budget, and whether a retry
    /// policy governs the retry.
    pub retry_budget: Option<(u32, bool)>,
}

impl StreamOrigin {
    fn target(&self) -> Target {
        Target {
            group: self.group.clone(),
            order: self.order,
            params: None,
            excluded: Arc::from([]),
            failover_excluded: Arc::from([]),
        }
    }

    /// `fallback_lookup_groups`: the served group, then the group the request was for.
    fn lookup<'a>(&'a self, request: &'a RouterRequest) -> Vec<&'a str> {
        let mut lookup = vec![self.group.as_str()];
        if request.model != self.group {
            lookup.push(request.model.as_str());
        }
        lookup
    }

    /// `has_unattempted_fallback_target`. The first attempt carries no attempted targets, so any
    /// resolved chain counts; a fallback hop has already tried its own group and the original one.
    fn has_unattempted(&self, request: &RouterRequest, chain: Option<Vec<String>>) -> bool {
        let Some(chain) = chain else {
            return false;
        };
        if self.depth == 0 {
            return true;
        }
        chain
            .iter()
            .any(|target| *target != self.group && *target != request.model)
    }
}

/// The group names a chain targets (a dict target counts by its `model`).
fn names(chain: Option<Vec<FallbackTarget>>) -> Option<Vec<String>> {
    chain.map(|targets| {
        targets
            .iter()
            .filter_map(|target| target.model().map(str::to_owned))
            .collect()
    })
}

/// The regular fallback targets for the lookup groups, then `default_fallbacks`, which Python
/// folds into `fallbacks` as `{"*": [...]}`.
fn regular_targets(dynamic: &RouterDynamicConfig, lookup: &[&str]) -> Option<Vec<String>> {
    names(
        dynamic
            .fallbacks
            .lookup_groups(lookup, &dynamic.fallback_keys),
    )
    .or_else(|| (!dynamic.default_fallbacks.is_empty()).then(|| dynamic.default_fallbacks.clone()))
}

fn has_default_fallbacks(fallbacks: &Fallbacks, default_fallbacks: &[String]) -> bool {
    fallbacks.has_star() || !default_fallbacks.is_empty()
}

impl Router {
    /// What may still take over a stream from `origin` that fails before content, and the retry
    /// budget for a failure of `kind`.
    pub fn stream_policy(
        &self,
        request: &RouterRequest,
        origin: &StreamOrigin,
        kind: Option<FailureKind>,
    ) -> StreamPolicy {
        let snapshot = self.core.snapshot.load_full();
        StreamPolicy {
            can_retry: stream_can_retry(&snapshot, request, origin),
            can_fall_back: stream_can_fall_back(&snapshot, request, origin),
            refusal_fallback: refusal_fallback_available(&snapshot, request, origin),
            retry_budget: kind.map(|kind| stream_retry_budget(&snapshot, request, origin, kind)),
        }
    }

    /// The fallback walk for a stream from `origin` that failed with `failure` before content:
    /// `async_function_with_fallbacks_common_utils` re-entered from the stream wrapper, so the
    /// group's own retry loop does not run again.
    pub async fn fallback_stream(
        &self,
        op: Op,
        request: RouterRequest,
        origin: StreamOrigin,
        failure: AttemptError,
    ) -> Result<super::RouterResponse, Error> {
        self.live()?;
        let snapshot = self.core.snapshot.load_full();
        let request = Arc::new(request);
        let mut walk = Walk::default();
        if request.model != origin.group {
            walk.attempted
                .push(Tried::Target(FallbackTarget::Name(request.model.clone())));
        }
        self.core
            .with_fallbacks(
                &snapshot,
                &op,
                &request,
                origin.target(),
                Hop {
                    depth: origin.depth,
                    level: origin.level,
                    scope: 0,
                },
                &mut walk,
                Some(Error::Attempt(failure)),
            )
            .await
    }

    /// `_astream_retry_same_group`: re-run the attempt in the stream's own group until a retry
    /// returns a response or the budget runs out. The failed result carries the newest failure.
    pub async fn retry_stream(
        &self,
        op: Op,
        request: RouterRequest,
        origin: StreamOrigin,
        failure: AttemptError,
    ) -> Result<super::RouterResponse, Error> {
        self.live()?;
        let snapshot = self.core.snapshot.load_full();
        let request = Arc::new(request);
        let target = origin.target();
        let at = Hop {
            depth: origin.depth,
            level: origin.level,
            scope: 0,
        };
        let counting = SelectInputs {
            count_only: true,
            ..SelectInputs::default()
        };
        let counted = |c: &Candidates, n: usize| if c.kind.is_single() { 0 } else { n };
        let (healthy, all) = self
            .core
            .candidates(&snapshot, &op, &request, &target.group, &counting)
            .await
            .map(|c| (counted(&c, c.healthy.len()), counted(&c, c.all.len())))
            .unwrap_or((0, 0));
        let (budget, policy_applies) =
            stream_retry_budget(&snapshot, &request, &origin, failure.kind);
        let inputs = SelectInputs {
            target_order: target.order,
            ..SelectInputs::default()
        };
        let mut last = failure;
        for attempt in origin.attempted_retries..budget {
            if !policy_applies
                && !self
                    .core
                    .should_retry(&snapshot, &request, &last, healthy, all)
            {
                break;
            }
            let delay = self.core.time_to_sleep(
                &snapshot,
                last.retry_after,
                budget - attempt,
                budget,
                healthy,
                all,
            );
            self.core.host.sleep(delay).await;
            match self
                .core
                .attempt(
                    &snapshot,
                    &op,
                    &request,
                    &target,
                    &inputs,
                    (attempt + 1, budget),
                    (at, &[]),
                )
                .await
            {
                Ok(mut response) => {
                    response.attempted_retries = attempt + 1;
                    response.max_retries = Some(budget);
                    return Ok(response);
                }
                Err(failed) => match failed.error {
                    // `_stream_fallback_error_for_raised`: a status the pre-stream retry rule
                    // would not retry ends the retries with that failure.
                    Error::Attempt(failure) => {
                        let stop = failure.status.is_some_and(|s| !should_retry_status(s));
                        last = failure;
                        if stop {
                            break;
                        }
                    }
                    other => return Err(other),
                },
            }
        }
        Err(Error::Attempt(last))
    }
}

/// The request's `num_retries`, then the routed deployment's, then the router's
/// (`_anthropic_messages_plain_retry_budget`).
fn plain_retry_budget(snapshot: &Snapshot, request: &RouterRequest, origin: &StreamOrigin) -> u32 {
    if let Some(retries) = request.num_retries {
        return retries;
    }
    let deployment = origin
        .deployment_id
        .as_deref()
        .and_then(|id| snapshot.registry.deployment(id));
    let configured = deployment.and_then(|d| match d.raw_params.get("num_retries") {
        Some(serde_json::Value::Number(n)) => n.as_u64().and_then(|n| u32::try_from(n).ok()),
        Some(serde_json::Value::String(s)) if s.chars().all(|c| c.is_ascii_digit()) => {
            s.parse().ok()
        }
        _ => None,
    });
    configured.unwrap_or(snapshot.settings.num_retries)
}

/// The retry policy for the request's model group (`_anthropic_messages_resolved_retry_policy`),
/// unless the request opted out with `num_retries=0`.
fn retry_policy<'a>(
    snapshot: &'a Snapshot,
    request: &'a RouterRequest,
    origin: &StreamOrigin,
) -> Option<&'a crate::config::RetryPolicy> {
    if request.num_retries == Some(0) {
        return None;
    }
    let settings = &snapshot.settings;
    crate::retry::resolve_retry_policy(
        settings.retry_policy.as_ref(),
        &origin.group,
        Some(
            request
                .model_group_retry_policy
                .as_ref()
                .unwrap_or(&settings.model_group_retry_policy),
        ),
    )
}

fn stream_retry_budget(
    snapshot: &Snapshot,
    request: &RouterRequest,
    origin: &StreamOrigin,
    kind: FailureKind,
) -> (u32, bool) {
    let failure = AttemptError::new(kind, None);
    let policy_retries =
        retry_policy(snapshot, request, origin).and_then(|p| p.retries_for(&failure));
    if origin.attempted_retries > 0
        && let Some(committed) = origin.committed_budget
    {
        return (committed, policy_retries.is_some());
    }
    match policy_retries {
        Some(retries) => (retries, true),
        None => (plain_retry_budget(snapshot, request, origin), false),
    }
}

fn stream_can_retry(snapshot: &Snapshot, request: &RouterRequest, origin: &StreamOrigin) -> bool {
    let attempted = origin.attempted_retries;
    if attempted > 0
        && let Some(committed) = origin.committed_budget
    {
        return committed > attempted;
    }
    let plain = plain_retry_budget(snapshot, request, origin);
    // `_retry_policy_ceiling`: every field the policy dumps.
    let ceiling = retry_policy(snapshot, request, origin).map_or(plain, |policy| {
        let granted = [
            policy.BadRequestErrorRetries,
            policy.AuthenticationErrorRetries,
            policy.TimeoutErrorRetries,
            policy.RateLimitErrorRetries,
            policy.ContentPolicyViolationErrorRetries,
            policy.InternalServerErrorRetries,
            policy.ServiceUnavailableErrorRetries,
            policy.NotFoundErrorRetries,
            policy.DefaultRetries,
        ];
        plain.max(granted.into_iter().flatten().max().unwrap_or(0))
    });
    ceiling > attempted
}

fn stream_can_fall_back(
    snapshot: &Snapshot,
    request: &RouterRequest,
    origin: &StreamOrigin,
) -> bool {
    if request.disable_fallbacks {
        return false;
    }
    if snapshot.settings.enable_weighted_failover {
        return true;
    }
    let mut orders: Vec<i64> = snapshot
        .registry
        .group(&origin.group)
        .unwrap_or_default()
        .iter()
        .filter(|d| d.usable_by_team(request.team_id.as_deref()))
        .filter_map(|d| d.order())
        .collect();
    orders.sort_unstable();
    orders.dedup();
    if orders.len() > 1 {
        let current = origin.order.unwrap_or(orders[0]);
        if orders.iter().any(|order| *order > current) {
            return true;
        }
    }
    let dynamic = &snapshot.settings.dynamic;
    let lookup = origin.lookup(request);
    if let Some(content_policy) = &dynamic.content_policy_fallbacks
        && origin.has_unattempted(
            request,
            names(content_policy.lookup_groups(&lookup, &dynamic.fallback_keys)),
        )
    {
        return true;
    }
    if !dynamic.has_regular_fallbacks() {
        return false;
    }
    origin.has_unattempted(request, regular_targets(dynamic, &lookup))
}

fn refusal_fallback_available(
    snapshot: &Snapshot,
    request: &RouterRequest,
    origin: &StreamOrigin,
) -> bool {
    if request.disable_fallbacks {
        return false;
    }
    let dynamic = &snapshot.settings.dynamic;
    let lookup = origin.lookup(request);
    if let Some(content_policy) = &dynamic.content_policy_fallbacks {
        return origin.has_unattempted(
            request,
            names(content_policy.lookup_groups(&lookup, &dynamic.fallback_keys)),
        );
    }
    if has_default_fallbacks(&dynamic.fallbacks, &dynamic.default_fallbacks) {
        return true;
    }
    if !dynamic.has_regular_fallbacks() {
        return false;
    }
    origin.has_unattempted(request, regular_targets(dynamic, &lookup))
}
