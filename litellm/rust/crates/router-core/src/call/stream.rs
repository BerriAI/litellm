//! Recovering a stream that failed after its attempt returned it. The host wraps every streamed
//! response; when the stream fails before the caller received content, the wrapper asks the core
//! what may still take over and hands the failure back here. Mirrors the decisions of the Python
//! router's stream wrappers: `_stream_can_fall_back`, `_refusal_fallback_available`,
//! `_anthropic_messages_stream_can_retry`, `_stream_retry_budget`, `_astream_retry_same_group`,
//! and the `async_function_with_fallbacks_common_utils` re-entry of the mid-stream fallback.

use std::sync::Arc;

use serde::Deserialize;

use super::{
    Failed, FallbackChain, FallbackHop, Level, Lists, Target, Tried, counted, counting, ends_call,
    group_members,
};
use crate::{
    Call, Router, RouterStateSnapshot,
    config::Op,
    error::{AttemptError, Error, FailureKind},
    fallback::FallbackTarget,
    health::should_retry_status,
    host::{CallError, RouterEvent, RouterHost, RouterRequest, UncountedRetries},
    retry::RetryStep,
    select::{SelectInputs, skips_batch_only},
};

/// Where a streamed response came from: the level of the fallback chain its attempt served, as
/// the attempt held it. A take-over starts from that level whole, so the params a dict fallback
/// merged and the deployments a failover hop excluded ride with the stream.
#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
#[serde(from = "Document")]
pub struct StreamOrigin {
    target: Target,
    /// `fallback_depth` of the attempt.
    depth: u32,
    level: u32,
    /// The deployment the attempt ran on.
    deployment_id: Option<String>,
    /// Same-group stream retries this request already spent (`attempted_retries_for_request`).
    attempted_retries: u32,
    /// The budget the first stream retry committed to (`committed_retry_budget_for_request`).
    committed_budget: Option<u32>,
}

/// A [`StreamOrigin`] as the Python host hands it back. It names the level without the params
/// and the exclusions the level carried, so a stream that host recovers starts without them.
#[derive(Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct Document {
    /// The name the attempt served: the requested group, or the fallback target.
    group: String,
    /// The `order` level an order fallback targeted.
    order: Option<i64>,
    depth: u32,
    level: u32,
    deployment_id: Option<String>,
    attempted_retries: u32,
    committed_budget: Option<u32>,
    /// The group a pre-routing hook selected for `group`, which the fallback lookup keys on first.
    routed: Option<String>,
}

impl From<Document> for StreamOrigin {
    fn from(document: Document) -> Self {
        Self {
            target: Target {
                group: document.group,
                order: document.order,
                routed: document.routed,
                ..Target::default()
            },
            depth: document.depth,
            level: document.level,
            deployment_id: document.deployment_id,
            attempted_retries: document.attempted_retries,
            committed_budget: document.committed_budget,
        }
    }
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
    /// Whether a `context_window_fallbacks` chain still has a target for the stream's group.
    pub context_window_fallback: bool,
    /// `_stream_retry_budget` for the failure kind asked about: the budget, and whether a retry
    /// policy governs the retry.
    pub retry_budget: Option<(u32, bool)>,
}

/// The take-over of a stream that failed before content, which the core found an attempt may
/// still run for. The failed stream is closed before it runs.
pub struct TakeOver<'a> {
    router: &'a Router,
    request: RouterRequest,
    origin: StreamOrigin,
    failure: AttemptError,
}

/// Stream retries that served nothing: the failure the fallbacks start from, and the budget the
/// group committed to, which the fallback that answers next reports as `x-litellm-max-retries`
/// (`record_retry_attempt`).
struct Spent {
    failed: Failed,
    committed_budget: Option<u32>,
}

/// A take-over whose attempts served nothing. `raised` is the host's error of the attempt the
/// take-over ended on, `None` when it ends on the stream's own failure.
#[derive(Debug, PartialEq)]
pub struct Unanswered<F> {
    pub raised: Option<F>,
}

impl StreamOrigin {
    /// The origin of the response an attempt of `target` at `at` returned.
    pub(super) fn of(
        target: Target,
        at: FallbackHop,
        deployment_id: Option<String>,
        retry: u32,
    ) -> Self {
        Self {
            target,
            depth: at.depth,
            level: at.level,
            deployment_id,
            attempted_retries: retry,
            committed_budget: None,
        }
    }

    /// The name the attempt served: the requested group, or the fallback target.
    pub fn group(&self) -> &str {
        &self.target.group
    }

    fn at(&self) -> FallbackHop {
        FallbackHop {
            depth: self.depth,
            level: self.level,
            scope: 0,
        }
    }

    /// The stream's failure as its level's, on the deployment the stream ran on.
    fn failed(&self, failure: AttemptError) -> Failed {
        Failed {
            error: Box::new(failure.into()),
            deployment_id: self.deployment_id.clone(),
            routed: self.target.routed.clone(),
            served_as: None,
        }
    }

    /// `fallback_lookup_groups`: the served group, then the group the request was for.
    fn lookup<'a>(&'a self, request: &'a RouterRequest) -> Vec<&'a str> {
        let mut lookup: Vec<&str> = Vec::new();
        for group in self
            .target
            .routed
            .as_deref()
            .into_iter()
            .chain([self.target.group.as_str(), request.model.as_str()])
        {
            if !lookup.contains(&group) {
                lookup.push(group);
            }
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
            .any(|target| *target != self.target.group && *target != request.model)
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

/// The `content_policy_fallbacks` chain for the lookup groups: the request's list where it
/// passed one, else the router's (`_has_content_policy_fallback`).
fn content_policy_targets(lists: &Lists<'_>, lookup: &[&str]) -> Option<Option<Vec<String>>> {
    lists
        .content_policy
        .map(|list| names(list.lookup_groups(lookup, lists.keys)))
}

/// The router's `default_fallbacks`, which Python folds into its `fallbacks` as `{"*": [...]}`: a
/// request's own list replaces them.
fn folded_defaults<'a>(snapshot: &'a RouterStateSnapshot, request: &RouterRequest) -> &'a [String] {
    match request.fallbacks {
        Some(_) => &[],
        None => &snapshot.settings.dynamic.default_fallbacks,
    }
}

/// The regular fallback targets for the lookup groups, then the folded `default_fallbacks`.
fn regular_targets(
    snapshot: &RouterStateSnapshot,
    request: &RouterRequest,
    lists: &Lists<'_>,
    lookup: &[&str],
) -> Option<Vec<String>> {
    let defaults = folded_defaults(snapshot, request);
    names(
        lists
            .fallbacks
            .and_then(|list| list.lookup_groups(lookup, lists.keys)),
    )
    .or_else(|| (!defaults.is_empty()).then(|| defaults.to_vec()))
}

/// `_has_default_fallbacks`: the router's list, whatever the request passed.
fn has_default_fallbacks(snapshot: &RouterStateSnapshot) -> bool {
    let dynamic = &snapshot.settings.dynamic;
    dynamic.fallbacks.has_star() || !dynamic.default_fallbacks.is_empty()
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
            context_window_fallback: context_window_fallback_available(&snapshot, request, origin),
            retry_budget: kind.map(|kind| stream_retry_budget(&snapshot, request, origin, kind)),
        }
    }

    /// Whether a retry or a fallback could still answer in place of a stream from `origin`, so
    /// its host holds the stream back until it sends content.
    pub fn may_take_over(&self, request: &RouterRequest, origin: &StreamOrigin) -> bool {
        let snapshot = self.core.snapshot.load_full();
        stream_can_retry(&snapshot, request, origin)
            || stream_can_fall_back(&snapshot, request, origin)
    }

    /// The fallback chain for a stream from `origin` that failed with `failure` before content:
    /// `async_function_with_fallbacks_common_utils` re-entered from the stream wrapper, so the
    /// group's own retry loop does not run again.
    pub async fn fallback_stream<H: RouterHost>(
        &self,
        host: &H,
        op: Op,
        request: RouterRequest,
        origin: StreamOrigin,
        failure: AttemptError,
    ) -> Result<super::RouterResponse<H::Response>, Box<CallError<H::Failure>>> {
        self.live()?;
        let call = Call::new(&self.core, host);
        let snapshot = self.core.snapshot.load_full();
        let failed = origin.failed(failure);
        call.fall_back_from(&snapshot, &op, &Arc::new(request), origin, failed)
            .await
            .map_err(|error| call.ended(error))
    }

    /// `_astream_retry_same_group`: re-run the attempt in the stream's own group until a retry
    /// returns a response or the budget runs out. The failed result carries the newest failure.
    pub async fn retry_stream<H: RouterHost>(
        &self,
        host: &H,
        op: Op,
        request: RouterRequest,
        origin: StreamOrigin,
        failure: AttemptError,
    ) -> Result<super::RouterResponse<H::Response>, Box<CallError<H::Failure>>> {
        self.live()?;
        let call = Call::new(&self.core, host);
        let snapshot = self.core.snapshot.load_full();
        call.retry_in_group(&snapshot, &op, &Arc::new(request), &origin, failure)
            .await
            .map_err(|spent| call.ended(*spent.failed.error))
    }

    /// The take-over of a stream from `origin` that failed with `failure` before content
    /// (`_anthropic_recover_stream_error`). `None` when no attempt may answer in the stream's
    /// place: no retry clears the failure, or neither a retry nor a fallback is left.
    pub fn take_over_stream(
        &self,
        request: &RouterRequest,
        origin: &StreamOrigin,
        failure: &AttemptError,
    ) -> Option<TakeOver<'_>> {
        if failure
            .status
            .is_some_and(|status| !should_retry_status(status))
        {
            return None;
        }
        let policy = self.stream_policy(request, origin, Some(failure.kind));
        let retries = policy
            .retry_budget
            .is_some_and(|(budget, _)| budget > origin.attempted_retries);
        (retries || policy.can_fall_back).then(|| TakeOver {
            router: self,
            request: request.clone(),
            origin: origin.clone(),
            failure: failure.clone(),
        })
    }
}

impl<H: RouterHost> Call<'_, H> {
    /// The fallbacks of the level a stream came from, after `failed`.
    async fn fall_back_from(
        &self,
        snapshot: &RouterStateSnapshot,
        op: &Op,
        request: &Arc<RouterRequest>,
        origin: StreamOrigin,
        failed: Failed,
    ) -> Result<super::RouterResponse<H::Response>, Error> {
        let mut chain = FallbackChain::default();
        if request.model != origin.target.group {
            chain
                .attempted
                .push(Tried::Target(FallbackTarget::Name(request.model.clone())));
        }
        let level = Level {
            snapshot,
            op,
            request,
            at: origin.at(),
            target: origin.target,
            chain: &mut chain,
        };
        self.fall_back(level, failed).await
    }

    /// The retries of the level a stream came from, which failed with `failure`. What they end
    /// on is what the level's fallbacks start from: the newest failed attempt, the stream's own
    /// when no retry ran, or the error of a retry the host ended the call from.
    async fn retry_in_group(
        &self,
        snapshot: &RouterStateSnapshot,
        op: &Op,
        request: &Arc<RouterRequest>,
        origin: &StreamOrigin,
        failure: AttemptError,
    ) -> Result<super::RouterResponse<H::Response>, Spent> {
        // A stream's retries start from a chain that has tried nothing.
        let mut chain = FallbackChain::default();
        let level = Level {
            snapshot,
            op,
            request,
            target: origin.target.clone(),
            at: origin.at(),
            chain: &mut chain,
        };
        let target = &level.target;
        let (healthy, all) = self
            .candidates(snapshot, op, request, &target.group, &counting())
            .await
            .map(|c| (counted(&c, c.healthy.len()), counted(&c, c.routable())))
            .unwrap_or((0, 0));
        let mut inputs = SelectInputs {
            target_order: target.order,
            excluded_ids: target.excluded.clone(),
            requested: Some(target.group.clone()),
            fallback_hop: level.at.level > 0,
            failed_ids: origin.deployment_id.iter().cloned().collect(),
            ..SelectInputs::default()
        };
        let (budget, policy_applies) = stream_retry_budget(snapshot, request, origin, failure.kind);
        let mut newest = origin.failed(failure);
        let mut committed_budget = origin.committed_budget;
        for attempt in origin.attempted_retries.. {
            let Some(last) = newest.error.attempt() else {
                break;
            };
            if last.ends_retries() {
                break;
            }
            let step = level.retry_step(last, (budget, policy_applies), attempt, healthy, all);
            let RetryStep::Retry { backoff } = step else {
                break;
            };
            let delay = backoff.delay(|| self.rng.next_f64());
            self.host.sleep(delay).await;
            committed_budget = Some(budget);
            self.host.observe(&RouterEvent::Retry {
                model_group: target.group.clone(),
                retry: attempt + 1,
                delay,
            });
            let retried = self.attempt(&level, &inputs, (attempt + 1, budget)).await;
            let failed = match retried {
                Ok(mut response) => {
                    response.attempted_retries = attempt + 1;
                    response.max_retries = Some(budget);
                    response.origin.committed_budget = Some(budget);
                    return Ok(response);
                }
                Err(failed) => failed,
            };
            if ends_call(&failed.error) {
                return Err(Spent {
                    failed,
                    committed_budget,
                });
            }
            // No deployment could be picked: the fallbacks start from the newest attempt.
            let Error::Attempt { failure, .. } = &*failed.error else {
                break;
            };
            if let Some(id) = failure
                .failed_deployment_id
                .as_ref()
                .or(failed.deployment_id.as_ref())
                && !inputs.failed_ids.contains(id)
            {
                inputs.failed_ids.push(id.clone());
            }
            // `_stream_fallback_error_for_raised`: a status the pre-stream retry rule would not
            // retry ends the retries with that failure.
            let stop = failure.status.is_some_and(|s| !should_retry_status(s));
            newest = failed;
            if stop {
                break;
            }
        }
        Err(Spent {
            failed: newest,
            committed_budget,
        })
    }
}

impl TakeOver<'_> {
    /// A retry in the stream's group while the failure's retry budget lasts, then the level's
    /// fallbacks, which start from what the retries ended on. The response that answers says
    /// where its own stream came from, so a take-over of that stream starts from its `origin`.
    pub async fn run<H: RouterHost>(
        self,
        host: &H,
        op: Op,
    ) -> Result<super::RouterResponse<H::Response>, Unanswered<H::Failure>> {
        let Self {
            router,
            request,
            origin,
            failure,
        } = self;
        router.live().map_err(|_| Unanswered { raised: None })?;
        let call = Call::new(&router.core, host);
        let snapshot = router.core.snapshot.load_full();
        let request = Arc::new(request);
        let Spent {
            failed,
            committed_budget,
        } = match call
            .retry_in_group(&snapshot, &op, &request, &origin, failure)
            .await
        {
            Ok(answer) => return Ok(answer),
            Err(spent) => spent,
        };
        call.fall_back_from(&snapshot, &op, &request, origin, failed)
            .await
            .map(|answer| super::RouterResponse {
                max_retries: answer.max_retries.or(committed_budget),
                ..answer
            })
            .map_err(|error| Unanswered {
                raised: call.ended(error).raised.map(|raised| raised.failure),
            })
    }
}

/// The request's `num_retries`, then the routed deployment's, then the router's
/// (`_anthropic_messages_plain_retry_budget`).
fn plain_retry_budget(
    snapshot: &RouterStateSnapshot,
    request: &RouterRequest,
    origin: &StreamOrigin,
) -> u32 {
    if let Some(retries) = request.num_retries {
        return retries;
    }
    if request.uncounted_retries == Some(UncountedRetries::Negative) {
        return 0;
    }
    let deployment = origin
        .deployment_id
        .as_deref()
        .and_then(|id| snapshot.registry.deployment(id));
    deployment
        .and_then(|deployment| deployment.num_retries())
        .unwrap_or(snapshot.settings.num_retries)
}

/// The retry policy for the request's model group (`_anthropic_messages_resolved_retry_policy`),
/// unless the request opted out with `num_retries=0`. A policy that does not read as one governs
/// nothing here, with no fall through to the router's `retry_policy`.
fn retry_policy<'a>(
    snapshot: &'a RouterStateSnapshot,
    request: &'a RouterRequest,
    origin: &StreamOrigin,
) -> Option<&'a crate::config::RetryPolicy> {
    if request.num_retries == Some(0) || request.retry_policy_raises.contains(&origin.target.group)
    {
        return None;
    }
    let settings = &snapshot.settings;
    crate::retry::resolve_retry_policy(
        settings.retry_policy.as_ref(),
        &origin.target.group,
        Some(
            request
                .model_group_retry_policy
                .as_ref()
                .unwrap_or(&settings.model_group_retry_policy),
        ),
    )
}

fn stream_retry_budget(
    snapshot: &RouterStateSnapshot,
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

fn stream_can_retry(
    snapshot: &RouterStateSnapshot,
    request: &RouterRequest,
    origin: &StreamOrigin,
) -> bool {
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
    snapshot: &RouterStateSnapshot,
    request: &RouterRequest,
    origin: &StreamOrigin,
) -> bool {
    if request.disable_fallbacks {
        return false;
    }
    if snapshot.settings.enable_weighted_failover {
        return true;
    }
    // `_stream_order_levels`, the levels the fallback walk sees for the group.
    let listing_team = request.listing_team_id.as_deref();
    let members: Vec<_> = group_members(snapshot, &origin.target.group, listing_team)
        .into_iter()
        .filter(|d| d.listed_for_team(listing_team))
        .collect();
    let skips_batch_only = skips_batch_only(request, &members);
    let mut orders: Vec<i64> = members
        .iter()
        .filter(|d| !skips_batch_only || !d.serves_only_batches())
        .filter_map(|d| d.order())
        .collect();
    orders.sort_unstable();
    orders.dedup();
    if orders.len() > 1 {
        let current = origin.target.order.unwrap_or(orders[0]);
        if orders.iter().any(|order| *order > current) {
            return true;
        }
    }
    let lists = Lists::of(snapshot, request);
    let lookup = origin.lookup(request);
    if let Some(chain) = content_policy_targets(&lists, &lookup)
        && origin.has_unattempted(request, chain)
    {
        return true;
    }
    if !lists.has_regular() && folded_defaults(snapshot, request).is_empty() {
        return false;
    }
    if lists
        .fallbacks
        .is_some_and(|list| list.is_client_side(lists.keys))
    {
        return true;
    }
    origin.has_unattempted(request, regular_targets(snapshot, request, &lists, &lookup))
}

fn context_window_fallback_available(
    snapshot: &RouterStateSnapshot,
    request: &RouterRequest,
    origin: &StreamOrigin,
) -> bool {
    if request.disable_fallbacks {
        return false;
    }
    let lists = Lists::of(snapshot, request);
    let lookup = origin.lookup(request);
    lists.context_window.is_some_and(|list| {
        origin.has_unattempted(request, names(list.lookup_groups(&lookup, lists.keys)))
    })
}

fn refusal_fallback_available(
    snapshot: &RouterStateSnapshot,
    request: &RouterRequest,
    origin: &StreamOrigin,
) -> bool {
    if request.disable_fallbacks {
        return false;
    }
    let lists = Lists::of(snapshot, request);
    let lookup = origin.lookup(request);
    if let Some(chain) = content_policy_targets(&lists, &lookup) {
        return origin.has_unattempted(request, chain);
    }
    if has_default_fallbacks(snapshot) {
        return true;
    }
    origin.has_unattempted(request, regular_targets(snapshot, request, &lists, &lookup))
}
