//! The call loop: select, run the attempt, then retry in the same group or fall back to another
//! until it succeeds or the policy is exhausted. Mirrors `Router.async_function_with_fallbacks`,
//! `async_function_with_fallbacks_common_utils`, `async_function_with_retries`,
//! `should_retry_this_error` and `_time_to_sleep_before_retry` in `litellm/router.py`, and
//! `run_async_fallback` in `router_utils/fallback_event_handlers.py`.

use std::{borrow::Cow, future::Future, pin::Pin, sync::Arc};

use serde_json::{Map, Value};

use tracing::warn;

use crate::{
    Call, FailureReport, RouterStateSnapshot,
    config::Op,
    error::{AttemptError, Error, FailureKind, RaisedBy},
    fallback::{FallbackKeys, FallbackTarget, Fallbacks},
    health::{GroupShape, should_retry_status, team_model_has_alternatives},
    host::{
        Attempt, Blocked, PickError, RetryCause, RouterEvent, RouterHost, RouterRequest, Served,
        UncountedRetries,
    },
    host_index::python_str,
    lookup::Lookup,
    mock::MockFailure,
    outcome::{OutcomeCallbacks, OutcomeWrite},
    prerouting::{PreRoute, PreRouted},
    registry::{ResolveKind, ResolveScope},
    retry::{RetryContext, RetryState, RetryStep, resolve_retry_policy, retry_step},
    select::{Candidates, SelectInputs, skips_batch_only},
};

mod stream;

pub use self::stream::{StreamOrigin, StreamPolicy, TakeOver, Unanswered};

#[derive(Clone, Debug)]
pub struct RouterResponse<R> {
    pub output: Served<R>,
    pub deployment_id: String,
    /// The name the serving attempt ran under: the requested name as written, an alias included,
    /// or a fallback target.
    pub name: String,
    pub model_group: String,
    pub attempted_retries: u32,
    /// The retry budget of the loop a retry succeeded in (`x-litellm-max-retries`); `None`
    /// when the first attempt succeeded.
    pub max_retries: Option<u32>,
    pub attempted_fallbacks: u32,
    /// Where a stream this response carries came from, which the take-over of that stream
    /// starts from ([`Router::take_over_stream`]).
    pub origin: StreamOrigin,
}

/// A failed attempt with the deployment it ran on, when one was picked.
struct Failed {
    error: Box<Error>,
    deployment_id: Option<String>,
    /// The group a pre-routing hook rewrote the level's name to for the failed attempt (an auto
    /// router's tier), which its order fallbacks and fallback lookup key off
    /// (`get_pre_routing_selection`).
    routed: Option<String>,
    /// The group the host stamped that attempt as served for ([`Candidates::served_as`]).
    served_as: Option<String>,
}

impl From<Error> for Failed {
    fn from(error: Error) -> Self {
        Self {
            error: Box::new(error),
            deployment_id: None,
            routed: None,
            served_as: None,
        }
    }
}

/// Where a level of the fallback chain sits: `depth` is `run_async_fallback`'s counter, one more
/// for every target tried, `level` how many fallbacks deep the chain is, and `scope` the level's
/// own copy of the request metadata (every hop copies its parent's).
#[derive(Clone, Copy, Debug, Default)]
struct FallbackHop {
    depth: u32,
    level: u32,
    scope: u32,
}

/// What the whole chain shares: the attempts made (`attempted_targets`) and the scopes handed out.
#[derive(Debug, Default)]
struct FallbackChain {
    attempted: Vec<Tried>,
    scopes: u32,
}

/// What one level of the chain serves: a group, or the same group at a later `order`, with the
/// request params the chain's dict targets merged so far (`kwargs.update(target)`).
#[derive(Clone, Debug, Default, PartialEq)]
struct Target {
    group: String,
    order: Option<i64>,
    params: Option<Arc<Map<String, Value>>>,
    /// `_excluded_deployment_ids` a weighted failover hop put on the request: never picked.
    excluded: Arc<[String]>,
    /// `_failover_excluded_ids` in the level's metadata: the deployments weighted failover
    /// already excluded on the way here.
    failover_excluded: Arc<[String]>,
    /// The group a pre-routing hook selected for the level's failed attempt, if it rewrote it.
    routed: Option<String>,
}

/// One level of the fallback chain as every step of it sees it: the call it belongs to, the
/// target it serves, where it sits and what the chain has tried so far.
struct Level<'a> {
    snapshot: &'a RouterStateSnapshot,
    op: &'a Op,
    request: &'a Arc<RouterRequest>,
    target: Target,
    at: FallbackHop,
    chain: &'a mut FallbackChain,
}

/// A step of a level's fallback chain: a configured target, or an order level of the group.
#[derive(Clone, Debug)]
enum Next {
    Fallback(FallbackTarget),
    Order(String, i64),
    /// A weighted failover hop: the same group without the deployments already excluded.
    Failover(String, Arc<[String]>),
}

/// An attempt the chain already made (`fallback_attempt_key`).
#[derive(Clone, Debug, PartialEq)]
enum Tried {
    Target(FallbackTarget),
    Order(String, i64),
    Failover(String, Vec<String>),
}

/// The fallback lists a call reads: the request's own where it passed them, else the router's.
struct Lists<'a> {
    fallbacks: Option<&'a Fallbacks>,
    context_window: Option<&'a Fallbacks>,
    content_policy: Option<&'a Fallbacks>,
    keys: &'a FallbackKeys,
}

impl<'a> Lists<'a> {
    fn of(snapshot: &'a RouterStateSnapshot, request: &'a RouterRequest) -> Self {
        let dynamic = &snapshot.settings.dynamic;
        let pick = |own: &'a Option<Option<Fallbacks>>, router: Option<&'a Fallbacks>| match own {
            Some(own) => own.as_ref(),
            None => router,
        };
        Self {
            fallbacks: pick(&request.fallbacks, Some(&dynamic.fallbacks)),
            context_window: pick(
                &request.context_window_fallbacks,
                dynamic.context_window_fallbacks.as_ref(),
            ),
            content_policy: pick(
                &request.content_policy_fallbacks,
                dynamic.content_policy_fallbacks.as_ref(),
            ),
            keys: &dynamic.fallback_keys,
        }
    }

    fn has_regular(&self) -> bool {
        self.fallbacks.is_some_and(|list| !list.is_empty())
    }
}

/// `has_unattempted_fallback_target` over the chain `list` resolves for the level's lookup
/// groups: before the first hop nothing counts as tried, and a dict target with its own params
/// is never skipped.
fn unattempted_chain(
    list: &Fallbacks,
    lists: &Lists<'_>,
    level: &Level<'_>,
    served: (Option<&str>, Option<&str>),
) -> bool {
    let Level {
        request,
        target,
        at,
        chain,
        ..
    } = level;
    let attempted = &chain.attempted;
    let (routed, served_as) = served;
    let mut lookup: Vec<&str> = Vec::new();
    for group in routed
        .into_iter()
        .chain(served_as)
        .chain([target.group.as_str(), request.model.as_str()])
    {
        if !lookup.contains(&group) {
            lookup.push(group);
        }
    }
    list.lookup_groups(&lookup, lists.keys)
        .is_some_and(|targets| {
            at.level == 0
                || targets.iter().any(|next| match next.identity() {
                    FallbackTarget::Params(_) => true,
                    identity => !attempted.contains(&Tried::Target(identity)),
                })
        })
}

/// The deployments `_core_get_model_list(model_name=group, team_id=team_id)` lists: the name's own
/// (a team's public name among them), an alias's target's, a routing group's members, and only
/// when none of those serve it, the wildcard routes (`_get_wildcard_deployments`), the proxy-wide
/// pattern's and then the team's.
fn group_members(
    snapshot: &RouterStateSnapshot,
    group: &str,
    team_id: Option<&str>,
) -> Vec<Arc<crate::registry::Deployment>> {
    let registry = &snapshot.registry;
    let all = |name: &str, team_id: Option<&str>| {
        crate::resolve::all_deployments(registry, name, team_id).unwrap_or_default()
    };
    let members = |routing_group: &litellm_router_types::RoutingGroupSpec| {
        crate::routing_groups::members(routing_group, |member| all(member, None))
    };
    let mut rows = all(group, team_id);
    if let Some(alias_target) = registry.alias_of(group) {
        rows.extend(match registry.routing_group_named(alias_target) {
            Some(routing_group) => members(routing_group),
            None => all(alias_target, None),
        });
    }
    if let Some(routing_group) = registry.routing_group_named(group) {
        rows.extend(members(routing_group));
    }
    if rows.is_empty() {
        if let Some((_, routed)) = registry.wildcard_for(group) {
            rows.extend(routed);
        }
        if let Some(team) = team_id {
            rows.extend(Lookup::new(registry).team_wildcard(group, team));
        }
    }
    rows
}

/// What `_maybe_run_weighted_failover` catches and defers to the regular fallbacks:
/// `openai.APIError` (every litellm provider error, a failure the host reported no classes for
/// among them), `RouterRateLimitError` and `RouterRateLimitErrorBasic`. Anything else propagates.
fn defers_to_regular_fallbacks(error: &Error) -> bool {
    retried_failure(error).is_some_and(|failure| {
        failure.classes.is_empty()
            || [
                "APIError",
                "BadRequestError",
                "ServiceUnavailableError",
                "RouterRateLimitError",
                "RouterRateLimitErrorBasic",
            ]
            .iter()
            .any(|class| failure.is_a(class))
    })
}

/// `is_bad_request_without_fallback` with `fallback_on_unsupported_params` off: an unsupported
/// params error, or a provider's plain 400, 413 or 422 bad request.
fn is_bad_request_without_fallback(failure: &AttemptError) -> bool {
    if !failure.is_a("BadRequestError") {
        return false;
    }
    if failure.is_a("UnsupportedParamsError") {
        return true;
    }
    if [
        "ContextWindowExceededError",
        "ContentPolicyViolationError",
        "UnsupportedEndpointError",
    ]
    .iter()
    .any(|class| failure.is_a(class))
    {
        return false;
    }
    matches!(failure.status, Some(400 | 413 | 422)) && failure.names_provider
}

type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// `KeyError`'s classes, most specific first.
/// The failure a retry loop decides by: an attempt's, or the exception a failed pick raised
/// (Python's loop retries a pick's error like any other: no deployment available, a stale
/// routing group's `KeyError`, ...).
fn retried_failure(error: &Error) -> Option<Cow<'_, AttemptError>> {
    match error {
        Error::Attempt { failure, .. } => Some(Cow::Borrowed(failure)),
        _ => error.pick_failure().map(Cow::Owned),
    }
}

/// The host refused an attempt, or the dynamic queue did, somewhere in the walk: the request ends
/// with that error, whichever level or fallback hop raised it.
fn ends_call(error: &Error) -> bool {
    matches!(error, Error::Refused(_))
        || retried_failure(error).is_some_and(|failure| failure.queue_rejected())
}

impl<H: RouterHost> Call<'_, H> {
    pub(crate) async fn call(
        &self,
        op: Op,
        request: Arc<RouterRequest>,
    ) -> Result<RouterResponse<H::Response>, Error> {
        let snapshot = self.snapshot.load_full();
        let mut chain = FallbackChain::default();
        let target = Target {
            group: request.model.clone(),
            order: None,
            params: None,
            excluded: Arc::from([]),
            failover_excluded: Arc::from([]),
            routed: None,
        };
        let level = Level {
            snapshot: &snapshot,
            op: &op,
            request: &request,
            target,
            at: FallbackHop::default(),
            chain: &mut chain,
        };
        // A fallbacks mock stands in for the requested group's retry loop: no pick, no attempt.
        let Some(mock) = request.mock_failure.filter(|mock| mock.before_retries()) else {
            return self.with_fallbacks(level).await;
        };
        let failed = Failed::from(self.mock_failed(mock, &level, None));
        self.fall_back(level, failed).await
    }

    /// One level of `async_function_with_fallbacks`: the target's retry loop, then its
    /// fallbacks if every attempt failed.
    fn with_fallbacks<'a>(
        &'a self,
        level: Level<'a>,
    ) -> BoxFuture<'a, Result<RouterResponse<H::Response>, Error>> {
        Box::pin(async move {
            let failed = match self.with_retries(&level).await {
                Ok(response) => return Ok(response),
                Err(failed) => failed,
            };
            self.fall_back(level, failed).await
        })
    }

    /// The level after its own failure: the no-fallback checks, weighted failover, then the
    /// fallback targets. A level that fails raises its own error, whatever its fallbacks raised,
    /// except an order chain's failure on any error but a context window or content policy one,
    /// which raises what the chain's last hop raised ([`RouterEvent::OrderChainFailed`]).
    /// A stream that fails before content re-enters here (`fallback_stream`), so the group's
    /// retry loop does not run again.
    async fn fall_back(
        &self,
        mut level: Level<'_>,
        failed: Failed,
    ) -> Result<RouterResponse<H::Response>, Error> {
        let Failed {
            error,
            deployment_id: failed_id,
            routed,
            served_as,
        } = failed;
        let error = *error;
        let (snapshot, request, at) = (level.snapshot, level.request, level.at);
        self.level_failed(&level, &error, false);
        if self.raises_without_fallback(snapshot, request, &error) {
            return Err(error);
        }
        // `async_function_with_fallbacks_common_utils` lists the group's deployments
        // (`_core_get_model_list`) for its order fallbacks: a written index position past the
        // model list raises there, before any fallback.
        if let Err(raised) = crate::resolve::all_deployments(
            &snapshot.registry,
            &level.target.group,
            request.listing_team_id.as_deref(),
        ) {
            self.level_failed(&level, &raised, true);
            return Err(raised);
        }
        let (targets, order_path) =
            self.fallback_targets(&level, routed.as_deref(), served_as.as_deref(), &error);
        if !order_path {
            self.host.observe(&RouterEvent::FallingBack {
                level: at.level,
                model: level.target.group.clone(),
            });
        }
        level.target.routed = routed;
        // `_maybe_run_weighted_failover`: its own `run_async_fallback`, before the regular one.
        let failover = if order_path {
            None
        } else {
            match self
                .weighted_failover(&level, &error, failed_id.as_deref())
                .await
            {
                Ok(failover) => failover,
                Err(raised) => {
                    // `_get_routing_context` raised inside the level's fallback handling: the
                    // level raises that error, before any regular fallback.
                    self.level_failed(&level, &raised, true);
                    return Err(raised);
                }
            }
        };
        if let Some(failover) = failover {
            let excluded = match &failover {
                Next::Failover(_, excluded) => excluded.clone(),
                _ => Arc::from([]),
            };
            level.target.failover_excluded = excluded;
            match self.run_fallbacks(&mut level, vec![failover]).await {
                Ok(response) => return Ok(response),
                Err(Some(raised)) if ends_call(&raised) => return Err(raised),
                Err(raised) if defers_to_regular_fallbacks(raised.as_ref().unwrap_or(&error)) => {}
                // Outside the `try` around the regular fallbacks: the level raises it as it is.
                Err(raised) => {
                    self.host.observe(&RouterEvent::OrderChainFailed {
                        level: at.level,
                        last_hop: raised.is_some(),
                    });
                    return Err(raised.unwrap_or(error));
                }
            }
        }
        let Some(targets) = targets else {
            self.host.observe(&RouterEvent::FallbacksFailed {
                level: at.level,
                chain: false,
                last_hop: false,
            });
            return Err(error);
        };
        let last_hop_error = match self.run_fallbacks(&mut level, targets).await {
            Ok(response) => return Ok(response),
            Err(Some(raised)) if ends_call(&raised) => return Err(raised),
            Err(raised) => raised,
        };
        if !order_path {
            self.host.observe(&RouterEvent::FallbacksFailed {
                level: at.level,
                chain: true,
                last_hop: last_hop_error.is_some(),
            });
            return Err(error);
        }
        // The order path's `raise`, outside the `try` that re-raises the original error: only a
        // context window or content policy error stays the one the caller sees.
        let keeps_original = retried_failure(&error).is_some_and(|failure| {
            failure.is_a("ContextWindowExceededError")
                || failure.is_a("ContentPolicyViolationError")
        });
        let raised = last_hop_error.filter(|_| !keeps_original);
        self.host.observe(&RouterEvent::OrderChainFailed {
            level: at.level,
            last_hop: raised.is_some(),
        });
        Err(raised.unwrap_or(error))
    }

    /// `run_async_fallback` over one chain: the depth is checked once on entry, then every
    /// target tried counts one more. When the chain served nothing, the error its last hop
    /// raised (`error_from_fallbacks`), or `None` when no hop ran.
    fn run_fallbacks<'a>(
        &'a self,
        level: &'a mut Level<'_>,
        targets: Vec<Next>,
    ) -> BoxFuture<'a, Result<RouterResponse<H::Response>, Option<Error>>> {
        Box::pin(async move {
            let (snapshot, op, request, at) = (level.snapshot, level.op, level.request, level.at);
            let (target, chain) = (&level.target, &mut *level.chain);
            let max_fallbacks = request
                .max_fallbacks
                .unwrap_or(snapshot.settings.max_fallbacks);
            if at.depth >= max_fallbacks {
                return Err(None);
            }
            let failed = Tried::Target(FallbackTarget::Name(
                target
                    .routed
                    .clone()
                    .unwrap_or_else(|| target.group.clone()),
            ));
            if !chain.attempted.contains(&failed) {
                chain.attempted.push(failed);
            }
            // Before the loop, or the refusal would be one target's failure: no target of the
            // chain is tried, and the walk raises a `ValueError` the host holds.
            let dicts: Vec<Map<String, Value>> = targets
                .iter()
                .filter_map(|next| match next {
                    Next::Fallback(FallbackTarget::Params(dict)) => Some(dict.clone()),
                    _ => None,
                })
                .collect();
            if !dicts.is_empty() && self.host.refuses_fallbacks(at.level, &dicts).await {
                let refused = AttemptError::host_raised(&["ValueError"]);
                return Err(Some(refused.into()));
            }
            let mut params = target.params.clone();
            let mut hop = at.depth;
            let mut last_hop_error = None;
            // `kwargs["model"]` as a dict target's non-string `model` left it: later targets
            // without a `model` keep it.
            let mut unroutable: Option<Value> = None;
            for next in targets {
                let (tried, group, order) = match &next {
                    Next::Order(group, order) => {
                        // An order level of the group a pre-routing hook selected passes the
                        // fallback gates as any group the level did not serve does.
                        if *group != target.group {
                            match self.host.allow_fallback(group).await {
                                Ok(true) => {}
                                Ok(false) => continue,
                                Err(raised) => return Err(Some(raised.into())),
                            }
                        }
                        (
                            Tried::Order(group.clone(), *order),
                            group.clone(),
                            Some(*order),
                        )
                    }
                    Next::Failover(group, excluded) => {
                        let mut key = excluded.to_vec();
                        key.sort();
                        (Tried::Failover(group.clone(), key), group.clone(), None)
                    }
                    // The group that just failed: the pre-routing selection when there was one, so
                    // a subagent bound to another tier can fall back to the model it asked for.
                    Next::Fallback(FallbackTarget::Name(name))
                        if *name == target.routed.as_deref().unwrap_or(&target.group) =>
                    {
                        continue;
                    }
                    // A provider-scoped request never leaves its group; a dict target with no
                    // `model` names no group, so it is skipped too.
                    Next::Fallback(fallback)
                        if request.same_group_fallbacks_only
                            && fallback.model() != Some(target.group.as_str()) =>
                    {
                        continue;
                    }
                    Next::Fallback(fallback) => {
                        if let Some(model) = fallback.model()
                            && model != target.group
                        {
                            // A check that raises ends the loop outside its `try`, as the
                            // chain's own failure.
                            match self.host.allow_fallback(model).await {
                                Ok(true) => {}
                                Ok(false) => continue,
                                Err(raised) => return Err(Some(raised.into())),
                            }
                        }
                        let group = fallback.model().unwrap_or(&target.group).to_owned();
                        (Tried::Target(fallback.identity()), group, None)
                    }
                };
                if chain.attempted.contains(&tried) {
                    continue;
                }
                chain.attempted.push(tried);
                if let Next::Fallback(fallback) = &next
                    && let Some(merged) = fallback.params()
                {
                    let mut combined = params.as_deref().cloned().unwrap_or_default();
                    combined.extend(merged.iter().map(|(k, v)| (k.clone(), v.clone())));
                    params = Some(Arc::new(combined));
                }
                match &next {
                    Next::Fallback(fallback) if fallback.model().is_none() => {
                        if let Some(model) = fallback.unroutable_model() {
                            unroutable = Some(model.clone());
                        }
                    }
                    _ => unroutable = None,
                }
                let group = unroutable.as_ref().map_or(group, python_str);
                hop += 1;
                self.host.observe(&RouterEvent::RetryLogged {
                    scope: at.scope,
                    level: at.level,
                    model: target.group.clone(),
                    attempted_retries: None,
                    cause: RetryCause::Level,
                });
                chain.scopes += 1;
                self.host.observe(&RouterEvent::Fallback {
                    from: target.group.clone(),
                    to: group.clone(),
                    order,
                    depth: hop,
                    level: at.level + 1,
                    scope: chain.scopes,
                    parent: at.scope,
                    max_fallbacks,
                    unroutable: unroutable.clone(),
                });
                let excluded = match &next {
                    Next::Failover(_, excluded) => excluded.clone(),
                    _ => target.excluded.clone(),
                };
                let next = Target {
                    group: group.clone(),
                    order,
                    params: params.clone(),
                    excluded,
                    failover_excluded: target.failover_excluded.clone(),
                    routed: None,
                };
                let parent = at;
                let at = FallbackHop {
                    depth: hop,
                    level: at.level + 1,
                    scope: chain.scopes,
                };
                // A non-string `model` is a hop that fails before any attempt, on an error its
                // own level's fallback lookup raises too, so it walks no chain of its own.
                let result = match &unroutable {
                    Some(_) => Err(Error::UnknownModel {
                        model: group.clone(),
                    }),
                    None => {
                        self.with_fallbacks(Level {
                            snapshot,
                            op,
                            request,
                            target: next,
                            at,
                            chain,
                        })
                        .await
                    }
                };
                self.host.observe(&RouterEvent::FallbackDone {
                    from: target.group.clone(),
                    to: group,
                    level: parent.level,
                    scope: at.scope,
                    parent: parent.scope,
                    succeeded: result.is_ok(),
                });
                match result {
                    Ok(mut response) => {
                        response.attempted_fallbacks = hop;
                        return Ok(response);
                    }
                    Err(raised) if ends_call(&raised) => return Err(Some(raised)),
                    Err(raised) => last_hop_error = Some(raised),
                }
            }
            Err(last_hop_error)
        })
    }

    /// `_maybe_run_weighted_failover`'s hop for a failed deployment, when it applies: the
    /// simple-shuffle group re-picked without every deployment already excluded, while one
    /// that is not cooling down remains.
    async fn weighted_failover(
        &self,
        level: &Level<'_>,
        error: &Error,
        failed_id: Option<&str>,
    ) -> Result<Option<Next>, Error> {
        let Level {
            snapshot,
            request,
            target,
            ..
        } = level;
        if !snapshot.settings.enable_weighted_failover {
            return Ok(None);
        }
        if retried_failure(error).is_some_and(|failure| {
            failure.is_a("ContextWindowExceededError")
                || failure.is_a("ContentPolicyViolationError")
        }) {
            return Ok(None);
        }
        // `_get_routing_context` runs first, whatever failed: a stale routing-group member map
        // raises its `KeyError` here.
        snapshot.missing_routing_group(request, &target.group)?;
        if request.skip_weighted_failover || error.attempt().is_none() {
            return Ok(None);
        }
        let kind = snapshot
            .registry
            .resolve(&target.group, ResolveScope::default())
            .map_or(ResolveKind::Group, |resolution| resolution.kind);
        let (strategy, _) = crate::strategy::routing_context(
            &snapshot.registry,
            &snapshot.settings.routing_strategy,
            request.routing_strategy.as_deref(),
            &kind,
            &target.group,
        );
        if strategy != litellm_router_types::RoutingStrategy::SimpleShuffle {
            return Ok(None);
        }
        let Some(failed_id) = failed_id else {
            return Ok(None);
        };
        let mut excluded: Vec<String> = target.failover_excluded.to_vec();
        if !excluded.iter().any(|id| id == failed_id) {
            excluded.push(failed_id.to_owned());
        }
        let listing_team = request.listing_team_id.as_deref();
        let members: Vec<_> = group_members(snapshot, &target.group, listing_team)
            .into_iter()
            .filter(|d| d.listed_for_team(listing_team))
            .collect();
        let skip_batch_only = skips_batch_only(request, &members);
        let ids: Vec<String> = members
            .iter()
            .filter(|d| !skip_batch_only || !d.serves_only_batches())
            .map(|d| d.id.clone())
            .collect();
        let cooling = self
            .health()
            .cooling(&ids, !skip_batch_only)
            .await
            .unwrap_or_default();
        let remaining = ids
            .iter()
            .any(|id| !excluded.contains(id) && !cooling.iter().any(|(cooled, _)| cooled == id));
        Ok(remaining.then(|| Next::Failover(target.group.clone(), excluded.into())))
    }

    /// `async_function_with_fallbacks_common_utils` raises before looking for any fallback:
    /// fallbacks are off, the error is a guardrail's, a bad request without
    /// `fallback_on_unsupported_params`, or no fallback serves the error at all.
    fn raises_without_fallback(
        &self,
        snapshot: &RouterStateSnapshot,
        request: &RouterRequest,
        error: &Error,
    ) -> bool {
        if request.disable_fallbacks {
            return true;
        }
        // Python decides by the exception alone, so a pick's error falls back like an attempt's:
        // an `UnsupportedParamsError` raises, and the refusals name no provider. Only an error no
        // pick raises (no Python exception stands behind it) has no fallback.
        retried_failure(error).is_none_or(|failure| {
            failure.kind == FailureKind::GuardrailIntervention
                || failure.queue_rejected()
                || (!snapshot.settings.fallback_on_unsupported_params
                    && is_bad_request_without_fallback(&failure))
        })
    }

    /// The fallback targets for a failure, or `None` when it must be raised as is
    /// (`async_function_with_fallbacks_common_utils`).
    fn fallback_targets(
        &self,
        level: &Level<'_>,
        routed: Option<&str>,
        served_as: Option<&str>,
        error: &Error,
    ) -> (Option<Vec<Next>>, bool) {
        let Level {
            snapshot,
            request,
            target,
            ..
        } = level;
        // A pick's error decides by its exception's classes as an attempt's does: a pre-call
        // context window refusal takes the context window fallbacks, any other the generic ones.
        let failure = retried_failure(error);
        let is_a = |name: &str| failure.as_ref().is_some_and(|f| f.is_a(name));
        let lists = Lists::of(snapshot, request);
        // `fallback_lookup_groups`: the group a pre-routing hook selected, the group being served,
        // then the group the request was for.
        let mut lookup: Vec<&str> = routed.into_iter().collect();
        for group in served_as
            .into_iter()
            .chain([target.group.as_str(), request.model.as_str()])
        {
            if !lookup.contains(&group) {
                lookup.push(group);
            }
        }
        // `order_model_group`: the pre-routing selection's order levels, else the level's own.
        let order_group = routed.unwrap_or(target.group.as_str());
        let chain = |list: &Fallbacks| {
            if list.is_client_side(lists.keys) {
                Some(list.as_chain())
            } else {
                list.lookup_groups(&lookup, lists.keys)
            }
        };
        let fallbacks_of = |targets: Vec<FallbackTarget>| {
            targets.into_iter().map(Next::Fallback).collect::<Vec<_>>()
        };
        // Only an error-specific chain the group matched takes precedence; an unmatched one falls
        // through to the order tiers and the regular fallbacks.
        let error_specific_list = if is_a("ContextWindowExceededError") {
            lists.context_window
        } else if is_a("ContentPolicyViolationError") {
            lists.content_policy
        } else {
            None
        };
        let error_specific_targets =
            error_specific_list.and_then(|list| list.lookup_groups(&lookup, lists.keys));
        if error_specific_targets.is_none() {
            // A routing group's members, with a priority group's priorities as their order.
            let listing_team = request.listing_team_id.as_deref();
            let members: Vec<_> = group_members(snapshot, order_group, listing_team)
                .into_iter()
                .filter(|d| d.listed_for_team(listing_team))
                .collect();
            let batch_levels_only =
                request.batch_create && members.iter().any(|d| d.is_batch_capable());
            let skips_batch_only = skips_batch_only(request, &members);
            let mut orders: Vec<i64> = members
                .iter()
                .filter(|d| !batch_levels_only || !d.cannot_run_batches())
                .filter(|d| !skips_batch_only || !d.serves_only_batches())
                .filter_map(|d| d.order())
                .collect();
            orders.sort_unstable();
            orders.dedup();
            if orders.len() > 1 {
                let current = target
                    .order
                    .filter(|_| routed.is_none())
                    .unwrap_or(orders[0]);
                let mut targets: Vec<Next> = orders
                    .into_iter()
                    .filter(|order| *order > current)
                    .map(|order| Next::Order(order_group.to_owned(), order))
                    .collect();
                if let Some(list) = lists.fallbacks {
                    targets.extend(fallbacks_of(chain(list).unwrap_or_default()));
                }
                if !targets.is_empty() {
                    return (Some(targets), true);
                }
            }
        }
        if let Some(targets) = error_specific_targets {
            return (Some(fallbacks_of(targets)), false);
        }
        if let Some(list) = lists.fallbacks
            && list.is_client_side(lists.keys)
        {
            return (Some(fallbacks_of(list.as_chain())), false);
        }
        (
            lists
                .fallbacks
                .and_then(|list| list.lookup_groups(&lookup, lists.keys))
                .map(fallbacks_of),
            false,
        )
    }

    /// A level's first attempt, or the rate-limit mock in its place
    /// (`_handle_mock_testing_rate_limit_error`), which the retry loop handles like a failed
    /// attempt.
    async fn first_attempt(
        &self,
        level: &Level<'_>,
        inputs: &SelectInputs,
        max_retries: u32,
    ) -> Result<RouterResponse<H::Response>, Failed> {
        if level.request.mock_failure != Some(MockFailure::RateLimit) {
            return self.attempt(level, inputs, (0, max_retries)).await;
        }
        // A lone deployment's `num_retries` rides on the mock error, as on a real one.
        let lone = level
            .snapshot
            .registry
            .group(&level.target.group)
            .and_then(|group| {
                let skips_batch_only = skips_batch_only(level.request, &group);
                let mut usable = group
                    .into_iter()
                    .filter(|d| !skips_batch_only || !d.serves_only_batches());
                usable.next().filter(|_| usable.next().is_none())
            })
            .and_then(|deployment| deployment.raw_params.get("num_retries")?.as_u64())
            .and_then(|retries| u32::try_from(retries).ok());
        Err(self.mock_failed(MockFailure::RateLimit, level, lone).into())
    }

    /// `async_function_with_retries` for one level: the first attempt, then a retry while
    /// [`retry_step`] grants one. The loop sleeps, attempts and reports. The step decides.
    async fn with_retries(&self, level: &Level<'_>) -> Result<RouterResponse<H::Response>, Failed> {
        let (snapshot, op, request, at) = (level.snapshot, level.op, level.request, level.at);
        let target = &level.target;
        let mut inputs = SelectInputs {
            target_order: target.order,
            excluded_ids: target.excluded.clone(),
            requested: Some(target.group.clone()),
            fallback_hop: at.level > 0,
            ..SelectInputs::default()
        };
        let counting = counting();
        let first_budget = request.num_retries.unwrap_or(snapshot.settings.num_retries);
        self.host.observe(&RouterEvent::RetriesOpened {
            scope: at.scope,
            max_retries: first_budget,
            model_group_size: snapshot.registry.model_list_len(&target.group),
        });
        let first = self.first_attempt(level, &inputs, first_budget).await;
        let mut failed = match first {
            Ok(response) => return Ok(response),
            Err(failed) => failed,
        };
        self.pick_failed(level, &failed.error);
        // A guardrail's block and a failure no retry can clear are re-raised as they are, before
        // anything else looks at them (`async_function_with_retries`).
        if matches!(
            &*failed.error,
            Error::Attempt { failure, .. }
                if failure.kind == FailureKind::GuardrailIntervention || failure.ends_retries()
        ) {
            return Err(failed);
        }
        // The first failure looks the group's retry policy up; a lookup that raises replaces the
        // failure with its own error, before any retry (`async_function_with_retries`).
        if request.retry_policy_raises.contains(&target.group) {
            self.host.observe(&RouterEvent::RetryPolicyRaised {
                level: at.level,
                model: target.group.clone(),
                cause: RetryCause::of(&failed.error),
            });
            return Err(Failed::from(Error::from(AttemptError::host_raised(&[
                "Exception",
            ]))));
        }
        let Some(first_failure) = retried_failure(&failed.error) else {
            return Err(failed);
        };
        let (num_retries, policy_applies) = unary_retry_budget(level, &first_failure);
        drop(first_failure);
        let all = self
            .candidates(snapshot, op, request, &target.group, &counting)
            .await
            .map(|c| counted(&c, c.routable()))
            .unwrap_or(0);

        for retry in 0..=num_retries {
            if retry > 0 {
                self.log_retry(level, retry, &failed.error);
            }
            let healthy_ids = self.healthy_ids(level, &counting).await;
            let healthy = healthy_ids.len();
            // A pick's error is retried as Python's loop retries any exception: a retry policy
            // keeps going, else `should_retry_this_error` decides (it raises when nothing is
            // healthy).
            let Some(failure) = retried_failure(&failed.error) else {
                return Err(failed);
            };
            let budget = (num_retries, policy_applies);
            let step = level.retry_step(&failure, budget, retry, healthy, all);
            let backoff = match step {
                RetryStep::Raise => return Err(failed),
                RetryStep::Retry { backoff } | RetryStep::Exhausted { backoff } => backoff,
            };
            let uncounted = request.uncounted_retries.filter(|_| !policy_applies);
            if retry == 0 && uncounted.is_none() {
                self.host.observe(&RouterEvent::RetryBudget {
                    scope: at.scope,
                    max_retries: num_retries,
                });
            }
            if uncounted == Some(UncountedRetries::Raises) {
                self.host.observe(&RouterEvent::RetryCountRaised {
                    level: at.level,
                    model: target.group.clone(),
                    cause: RetryCause::of(&failed.error),
                });
                return Err(Failed::from(Error::from(AttemptError::host_raised(&[
                    "Exception",
                ]))));
            }
            if num_retries == 0 {
                return Err(failed);
            }
            if retry == 0 {
                self.log_retry(level, 0, &failed.error);
            }
            let named = failure.failed_deployment_id.as_ref();
            let failed_on = named.or(failed.deployment_id.as_ref());
            if let Some(id) = failed_on
                && !inputs.failed_ids.contains(id)
            {
                inputs.failed_ids.push(id.clone());
            }
            if let (Some(status), Some(id)) = (failure.status, failed_on)
                && !should_retry_status(status)
                && !inputs.skipped_ids.contains(id)
            {
                inputs.skipped_ids.push(id.clone());
            }
            // An endpoint a deployment does not serve is retried on the others, until every
            // one refused (`_every_deployment_refused`).
            if failure.is_a("UnsupportedEndpointError")
                && healthy_ids.iter().all(|id| inputs.skipped_ids.contains(id))
            {
                return Err(failed);
            }
            let delay = backoff.delay(|| self.rng.next_f64());
            self.host.sleep(delay).await;
            if matches!(step, RetryStep::Exhausted { .. }) {
                self.host.observe(&RouterEvent::RetriesExhausted {
                    model_group: target.group.clone(),
                    attempted: num_retries,
                    max: num_retries,
                    level: at.level,
                    picked: failed.error.attempt().is_none(),
                });
                break;
            }
            self.host.observe(&RouterEvent::Retry {
                model_group: target.group.clone(),
                retry: retry + 1,
                delay,
            });
            match self.attempt(level, &inputs, (retry + 1, num_retries)).await {
                Ok(mut response) => {
                    response.max_retries = Some(num_retries);
                    return Ok(response);
                }
                Err(next) if next.error.attempt().is_some_and(AttemptError::ends_retries) => {
                    return Err(next);
                }
                Err(next) => {
                    self.pick_failed(level, &next.error);
                    failed = next;
                }
            }
        }
        Err(failed)
    }

    /// The ids `_async_get_healthy_deployments` counts as healthy for the level's group.
    async fn healthy_ids(&self, level: &Level<'_>, counting: &SelectInputs) -> Vec<String> {
        let Level {
            snapshot,
            op,
            request,
            target,
            ..
        } = level;
        match self
            .candidates(snapshot, op, request, &target.group, counting)
            .await
        {
            Ok(c) => {
                let ids = c.healthy.iter().map(|d| d.id.clone());
                ids.take(counted(&c, c.healthy.len())).collect()
            }
            // `_async_get_healthy_deployments` swallows the lookup's error and still reads the
            // cooldowns: a Redis round trip the refused pick's failure log runs across.
            Err(_) => {
                let _ = self
                    .health()
                    .active_cooldowns(&snapshot.registry.ids())
                    .await;
                Vec::new()
            }
        }
    }

    /// `error` is a pick's: reported as it happens, whatever the level then does with it.
    fn pick_failed(&self, level: &Level<'_>, error: &Error) {
        if let Some(picked) = PickError::of(error) {
            self.host.observe(&RouterEvent::PickFailed {
                level: level.at.level,
                model: level.target.group.clone(),
                error: picked,
            });
        }
    }

    /// The level serving `target` raises a pick's `error` ([`RouterEvent::LevelFailed`]);
    /// `handling` when its fallback handling raised it.
    fn level_failed(&self, level: &Level<'_>, error: &Error, handling: bool) {
        if let Some(picked) = PickError::of(error) {
            self.host.observe(&RouterEvent::LevelFailed {
                level: level.at.level,
                model: level.target.group.clone(),
                error: picked,
                handling,
            });
        }
    }

    /// `_core_log_retry` for a failure in the level's retry loop.
    fn log_retry(&self, level: &Level<'_>, retry: u32, error: &Error) {
        let cause = match RetryCause::of(error) {
            RetryCause::Other => RetryCause::Attempt,
            cause => cause,
        };
        self.host.observe(&RouterEvent::RetryLogged {
            scope: level.at.scope,
            level: level.at.level,
            model: level.target.group.clone(),
            attempted_retries: Some(retry),
            cause,
        });
    }

    /// A `mock_testing_*` failure for the level's group, reported to the host, which raises
    /// the matching exception.
    fn mock_failed(&self, mock: MockFailure, level: &Level<'_>, num_retries: Option<u32>) -> Error {
        self.host.observe(&RouterEvent::MockFailed {
            mock,
            level: level.at.level,
            model: level.target.group.clone(),
            num_retries,
        });
        mock.failure(num_retries).into()
    }

    /// One attempt: select a deployment, run it through the host, record the outcome.
    async fn attempt(
        &self,
        level: &Level<'_>,
        inputs: &SelectInputs,
        (retry, max_retries): (u32, u32),
    ) -> Result<RouterResponse<H::Response>, Failed> {
        let (snapshot, op, request, at) = (level.snapshot, level.op, level.request, level.at);
        let target = &level.target;
        if request.unrouted {
            return self.unrouted_attempt(level, (retry, max_retries)).await;
        }
        let pick_started = self.health().now_time();
        let (deployment, candidates) = match self.picked(level, inputs, retry).await {
            Ok(picked) => picked,
            Err(failed) if request.passthrough_on_no_deployment => {
                self.pick_failed(level, &failed.error);
                return self.unrouted_attempt(level, (retry, max_retries)).await;
            }
            Err(failed) => return Err(failed),
        };
        self.host.observe(&RouterEvent::Selected {
            model_group: candidates.model_group.clone(),
            deployment_id: deployment.id.clone(),
            pick: self
                .health()
                .now_time()
                .duration_since(pick_started)
                .unwrap_or_default(),
        });
        // A name no deployment serves is routed to the first default fallback
        // (`_common_checks_available_deployment`), still stamped as the name.
        let defaulted = candidates.model_group != target.group
            && matches!(
                crate::resolve::resolve_request(
                    &snapshot.registry,
                    &target.group,
                    request.specific_deployment,
                    crate::resolve::Caller {
                        team_id: request.team_id.as_deref(),
                        proxy_admin: request.proxy_admin,
                        allowed_models: &request.allowed_models,
                        wildcard_request: candidates.wildcard_request.as_deref(),
                        refuse_cross_group_litellm_model: snapshot
                            .settings
                            .enable_cross_model_group_collision_check,
                        ..crate::resolve::Caller::default()
                    },
                    None,
                    snapshot.settings.dynamic.default_ref(),
                    self.health().timestamp(),
                ),
                Ok(None)
            );
        let attempt = Attempt {
            op: op.clone(),
            request: request.clone(),
            deployment: deployment.clone(),
            name: target.group.clone(),
            model_group: if defaulted {
                target.group.clone()
            } else {
                candidates.model_group.clone()
            },
            routed: routed_name(target, &candidates),
            target_order: target.order,
            candidate_ids: candidates.all.iter().map(|d| d.id.clone()).collect(),
            retry,
            fallback_depth: at.depth,
            fallback_level: at.level,
            timeout: snapshot.settings.timeout,
            params: target.params.clone(),
            scope: at.scope,
            max_retries,
            model_group_size: snapshot.registry.model_list_len(&target.group),
        };
        // Every attempt runs through the host; the core records its outcome itself, as far as the
        // request's `OutcomeCallbacks` reach, except a deferred stream's: its request counts now
        // (`increment_deployment_usage_for_response`), its success or failure when the host reports
        // the stream's end.
        let checks = &self.core.config.pre_call_checks;
        let refused = checks
            .admit(self.host.cache(), &deployment, self.core.clock.now())
            .await;
        let ran = match refused {
            Some(refusal) => Err(refusal),
            None => self
                .host
                .attempt(&attempt)
                .await
                .map_err(|failure| self.failed(failure, Some(&deployment.id))),
        };
        match ran {
            Ok(output) => {
                if !output.deferred {
                    checks
                        .succeeded(
                            self.host.cache(),
                            &deployment,
                            output.total_tokens,
                            self.core.clock.now(),
                        )
                        .await;
                }
                // `_should_raise_content_policy_error`: a blocked response is raised only when
                // a content-policy fallback would serve it, and counts as no failure.
                let routed = routed_name(target, &candidates);
                let served = (routed.as_deref(), candidates.served_as.as_deref());
                let raised = match output.blocked {
                    Some(Blocked::ContentFilter) => has_content_policy_fallback(level, served),
                    Some(Blocked::Refusal) => refusal_fallback_available(level, served),
                    None => false,
                };
                let tracks = output
                    .tracks_usage
                    .unwrap_or_else(|| deployment.tracks_usage());
                // A shared logging object logs the success of the response that serves the
                // request alone, so a raised blocked response counts its request and no success.
                let callbacks = match raised && request.outcome_callbacks.shared_logging {
                    true => OutcomeCallbacks {
                        success: false,
                        ..request.outcome_callbacks
                    },
                    false => request.outcome_callbacks,
                };
                let recorded = match (output.deferred, tracks) {
                    (true, true) => {
                        let tokens = output.total_tokens as f64;
                        self.health()
                            .add_usage(&deployment, None, tokens, 1.0)
                            .await
                            .map(|_| ())
                    }
                    (true, false) => Ok(()),
                    (false, _) => {
                        self.health()
                            .record_success(callbacks, &deployment, output.total_tokens, tracks)
                            .await
                    }
                };
                if let Err(error) = recorded {
                    warn!(deployment_id = %deployment.id, error = %error, "usage was not recorded");
                }
                self.host.observe(&RouterEvent::Succeeded {
                    deployment_id: deployment.id.clone(),
                });
                if let Some(blocked) = output.blocked.filter(|_| raised) {
                    self.host.observe(&RouterEvent::Blocked {
                        level: at.level,
                        model: target.group.clone(),
                        blocked,
                    });
                    let mut failure =
                        AttemptError::new(FailureKind::ContentPolicyViolation, Some(400));
                    failure.classes = crate::retry::content_policy_classes();
                    return Err(Failed {
                        error: Box::new(failure.into()),
                        deployment_id: Some(deployment.id.clone()),
                        routed: routed_name(target, &candidates),
                        served_as: candidates.served_as.clone(),
                    });
                }
                Ok(RouterResponse {
                    output,
                    deployment_id: deployment.id.clone(),
                    name: target.group.clone(),
                    model_group: candidates.model_group,
                    attempted_retries: retry,
                    max_retries: None,
                    attempted_fallbacks: at.depth,
                    origin: StreamOrigin::of(
                        Target {
                            routed,
                            ..target.clone()
                        },
                        at,
                        Some(deployment.id.clone()),
                        retry,
                    ),
                })
            }
            Err(error) => {
                let failed_deployment = match &error {
                    Error::Attempt { failure, raised_by } => {
                        match raised_by {
                            RaisedBy::PreCallCheck(_) => {
                                self.cool_down_refused(snapshot, request, &deployment, failure)
                                    .await;
                            }
                            RaisedBy::Host(_) | RaisedBy::Core
                                if !failure.recorded && !failure.queue_rejected() =>
                            {
                                self.record_failure(
                                    snapshot,
                                    request,
                                    &candidates,
                                    &deployment,
                                    failure,
                                )
                                .await;
                            }
                            RaisedBy::Host(_) | RaisedBy::Core => {}
                        }
                        Some(deployment.id.clone())
                    }
                    _ => None,
                };
                Err(Failed {
                    error: Box::new(error),
                    deployment_id: failed_deployment,
                    routed: routed_name(target, &candidates),
                    served_as: candidates.served_as.clone(),
                })
            }
        }
    }

    /// The attempt's pick: the deployment and the candidates it was picked from.
    async fn picked(
        &self,
        level: &Level<'_>,
        inputs: &SelectInputs,
        retry: u32,
    ) -> Result<(Arc<crate::registry::Deployment>, Candidates), Failed> {
        let (snapshot, op, request, at) = (level.snapshot, level.op, level.request, level.at);
        let target = &level.target;
        crate::routing_plugins::refuse_sync_pick(
            snapshot.settings.routing_plugins,
            &snapshot.settings.routing_strategy,
            request,
        )?;
        self.host.observe(&RouterEvent::PickStarted {
            model: target.group.clone(),
        });
        let routed = self
            .pre_routed(PreRoute {
                op,
                request,
                name: &target.group,
                fallback_depth: at.depth,
                retry,
            })
            .await?;
        let (routed, served_as) = match routed {
            PreRouted::Bound { model, group } => (PreRouted::Model(model), Some(group)),
            other => (other, None),
        };
        match routed {
            PreRouted::Model(name) => {
                match self.resolved(snapshot, op, request, &name, inputs).await {
                    Ok((deployment, candidates)) => Ok((
                        deployment,
                        Candidates {
                            served_as,
                            ..candidates
                        },
                    )),
                    // Whatever fails once the hook rerouted the pick, the failure names the
                    // group it was routed to and the group it is served for
                    // (`record_pre_routing_selection` runs before the group is resolved).
                    Err(error) => Err(Failed {
                        routed: (name != target.group).then_some(name),
                        served_as,
                        ..error.into()
                    }),
                }
            }
            // A custom strategy's pick is served as is, for the requested name.
            PreRouted::Bound { .. } => unreachable!("mapped to a model above"),
            PreRouted::Deployment(deployment) => {
                let candidates = Candidates {
                    name: target.group.clone(),
                    model_group: target.group.clone(),
                    kind: ResolveKind::Group,
                    all: vec![deployment.clone()],
                    healthy: vec![deployment.clone()],
                    skips_batch_only: skips_batch_only(request, std::slice::from_ref(&deployment)),
                    wildcard_request: None,
                    served_as: None,
                };
                Ok((deployment, candidates))
            }
        }
    }

    /// The pick among the deployments of `name`.
    async fn resolved(
        &self,
        snapshot: &RouterStateSnapshot,
        op: &Op,
        request: &Arc<RouterRequest>,
        name: &str,
        inputs: &SelectInputs,
    ) -> Result<(Arc<crate::registry::Deployment>, Candidates), Error> {
        snapshot.missing_routing_group(request, name)?;
        let candidates = self.candidates(snapshot, op, request, name, inputs).await?;
        if candidates.healthy.is_empty() {
            return Err(candidates.none_healthy(&snapshot.registry));
        }
        let deployment = self.pick(snapshot, op, request, &candidates).await?;
        Ok((deployment, candidates))
    }

    /// An attempt on a name that is no model group: the host serves it under that name, with no
    /// deployment picked and no outcome recorded.
    async fn unrouted_attempt(
        &self,
        level: &Level<'_>,
        (retry, max_retries): (u32, u32),
    ) -> Result<RouterResponse<H::Response>, Failed> {
        let (snapshot, op, request, at) = (level.snapshot, level.op, level.request, level.at);
        let target = &level.target;
        let deployment =
            Arc::new(crate::registry::Deployment::unrouted(&target.group).map_err(Error::from)?);
        let attempt = Attempt {
            op: op.clone(),
            request: request.clone(),
            deployment,
            name: target.group.clone(),
            model_group: target.group.clone(),
            routed: None,
            target_order: target.order,
            candidate_ids: Vec::new(),
            retry,
            fallback_depth: at.depth,
            fallback_level: at.level,
            timeout: snapshot.settings.timeout,
            params: target.params.clone(),
            scope: at.scope,
            max_retries,
            model_group_size: snapshot.registry.model_list_len(&target.group),
        };
        match self.host.attempt(&attempt).await {
            Ok(output) => Ok(RouterResponse {
                output,
                deployment_id: String::new(),
                name: target.group.clone(),
                model_group: target.group.clone(),
                attempted_retries: retry,
                max_retries: None,
                attempted_fallbacks: at.depth,
                origin: StreamOrigin::of(
                    Target {
                        routed: None,
                        ..target.clone()
                    },
                    at,
                    None,
                    retry,
                ),
            }),
            Err(failure) => Err(self.failed(failure, None).into()),
        }
    }

    /// `async_deployment_callback_on_failure`, when it runs for this failure.
    async fn count_failed_request(
        &self,
        deployment: &crate::registry::Deployment,
        failure: &AttemptError,
    ) {
        if failure.uncounted_request {
            return;
        }
        if let Err(error) = self.health().count_failed_request(deployment, None).await {
            warn!(deployment_id = %deployment.id, error = %error, "usage was not recorded");
        }
    }

    /// `async_routing_strategy_pre_call_checks`' cooldown for a deployment a pre-call check
    /// refused: `set_cooldown_deployments` alone, as the Python host reports its own refusal.
    async fn cool_down_refused(
        &self,
        snapshot: &RouterStateSnapshot,
        request: &RouterRequest,
        deployment: &crate::registry::Deployment,
        failure: &AttemptError,
    ) {
        let report = FailureReport {
            uncounted: true,
            batch_api: request.batch_api,
            ..FailureReport::default()
        };
        let cooled = self
            .core
            .reported_failure(
                snapshot,
                self.host.cache(),
                &deployment.id,
                failure,
                &report,
            )
            .await;
        match cooled {
            Ok(Some(cooldown)) => self.host.observe(&RouterEvent::CooldownSet {
                deployment_id: deployment.id.clone(),
                cooldown,
            }),
            Ok(None) => {}
            Err(error) => {
                warn!(deployment_id = %deployment.id, error = %error, "cooldown was not recorded");
            }
        }
    }

    async fn record_failure(
        &self,
        snapshot: &RouterStateSnapshot,
        request: &RouterRequest,
        candidates: &Candidates,
        deployment: &crate::registry::Deployment,
        failure: &AttemptError,
    ) {
        self.host.observe(&RouterEvent::AttemptFailed {
            deployment_id: deployment.id.clone(),
            kind: failure.kind,
            status: failure.status,
        });
        let recorded = request.outcome_callbacks.record(
            OutcomeWrite::Failure,
            self.attempt_failure(snapshot, request, candidates, deployment, failure),
        );
        if let Err(error) = recorded.await {
            warn!(deployment_id = %deployment.id, error = %error, "cooldown was not recorded");
        }
    }

    /// What the failure callbacks record for a failed attempt.
    async fn attempt_failure(
        &self,
        snapshot: &RouterStateSnapshot,
        request: &RouterRequest,
        candidates: &Candidates,
        deployment: &crate::registry::Deployment,
        failure: &AttemptError,
    ) -> Result<(), Error> {
        // The default deployment's copy is no configured deployment: Python's cooldown check finds
        // no group for its id and records nothing, unless the id names a configured deployment,
        // which then takes the failure.
        let configured;
        let deployment = if candidates.kind == ResolveKind::Default {
            match snapshot.registry.deployment(&deployment.id) {
                Some(found) => {
                    configured = found;
                    &*configured
                }
                None => {
                    self.count_failed_request(deployment, failure).await;
                    return Ok(());
                }
            }
        } else {
            deployment
        };
        let has_fallback_group = snapshot
            .settings
            .dynamic
            .fallback_targets(&[candidates.model_group.as_str()])
            .is_some();
        let batch_call = !candidates.skips_batch_only;
        let shape = GroupShape::of_failure(batch_call, has_fallback_group, |batch_call| {
            // The deployment's own group decides, also when the call named the deployment by id.
            snapshot
                .registry
                .group(&deployment.model_name)
                .map_or(deployment.peers_in(&candidates.all, batch_call) == 1, |group| {
                    deployment.peers_in(&group, batch_call) == 1
                })
                // A routing group call with alternatives cools its member down so selection
                // moves on (`routing_group_alternatives`).
                && !(candidates.kind == ResolveKind::RoutingGroup
                    && deployment.peers_in(&candidates.all, batch_call) > 1)
                // As does a team deployment with a sibling under its public name
                // (`team_model_alternatives`).
                && !team_model_has_alternatives(&snapshot.registry, deployment, batch_call)
        });
        self.count_failed_request(deployment, failure).await;
        match self
            .health()
            .record_failure(
                &snapshot.settings,
                deployment,
                shape,
                request.cooldown_time,
                failure,
            )
            .await
        {
            Ok(Some(cooldown)) => self.host.observe(&RouterEvent::CooldownSet {
                deployment_id: deployment.id.clone(),
                cooldown,
            }),
            Ok(None) => {}
            Err(error) => return Err(error),
        }
        Ok(())
    }
}

/// `_has_content_policy_fallback`: the content-policy chain resolves for the level's
/// lookup groups and holds a target the chain has not tried; with no content-policy list,
/// any router `"*"` entry.
fn has_content_policy_fallback(level: &Level<'_>, served: (Option<&str>, Option<&str>)) -> bool {
    let lists = Lists::of(level.snapshot, level.request);
    match lists.content_policy {
        Some(list) => unattempted_chain(list, &lists, level, served),
        None => level.snapshot.settings.dynamic.fallbacks.has_star(),
    }
}

/// `_refusal_fallback_available`: the content-policy list decides when there is one;
/// else a router `"*"` entry, else a generic chain with a target the chain has not tried.
fn refusal_fallback_available(level: &Level<'_>, served: (Option<&str>, Option<&str>)) -> bool {
    if level.request.disable_fallbacks {
        return false;
    }
    let lists = Lists::of(level.snapshot, level.request);
    if lists.content_policy.is_some() {
        return has_content_policy_fallback(level, served);
    }
    if level.snapshot.settings.dynamic.fallbacks.has_star() {
        return true;
    }
    lists
        .fallbacks
        .is_some_and(|list| unattempted_chain(list, &lists, level, served))
}

/// A pick that only counts the group's deployments (`_async_get_healthy_deployments`).
fn counting() -> SelectInputs {
    SelectInputs {
        count_only: true,
        ..SelectInputs::default()
    }
}

/// `_async_get_healthy_deployments` finds no group for a deployment id, so a call by id counts no
/// deployments when deciding whether to retry.
fn counted(candidates: &Candidates, n: usize) -> usize {
    if candidates.kind.is_single() { 0 } else { n }
}

/// The unary loop's retry budget for the level's first failure, and whether a retry policy
/// granted it. A request's `num_retries`, a count or not, wins over the deployment's and the
/// router's, and a retry policy that names the failure wins over all three. The stream loop
/// resolves its own budget (`stream_retry_budget`).
fn unary_retry_budget(level: &Level<'_>, failure: &AttemptError) -> (u32, bool) {
    let (settings, request) = (&level.snapshot.settings, level.request);
    let plain = if request.uncounted_retries.is_some() {
        0
    } else {
        request
            .num_retries
            .or(failure.num_retries)
            .unwrap_or(settings.num_retries)
    };
    let group_policies = request
        .model_group_retry_policy
        .as_ref()
        .unwrap_or(&settings.model_group_retry_policy);
    let policy_retries = (request.num_retries != Some(0))
        .then(|| {
            resolve_retry_policy(
                settings.retry_policy.as_ref(),
                &level.target.group,
                Some(group_policies),
            )?
            .retries_for(failure)
        })
        .flatten();
    (policy_retries.unwrap_or(plain), policy_retries.is_some())
}

impl Level<'_> {
    /// [`retry_step`] for a failure in this level's group, with the fallback lists the request
    /// reads and the router's `retry_after`.
    fn retry_step(
        &self,
        failure: &AttemptError,
        (budget, policy_applies): (u32, bool),
        attempted: u32,
        healthy: usize,
        all: usize,
    ) -> RetryStep {
        let lists = Lists::of(self.snapshot, self.request);
        retry_step(RetryState {
            failure,
            budget,
            attempted,
            policy_applies,
            context: RetryContext {
                healthy,
                all,
                has_context_window_fallbacks: lists.context_window.is_some(),
                has_content_policy_fallbacks: lists.content_policy.is_some(),
                has_regular_fallbacks: lists.has_regular(),
            },
            floor: self.snapshot.settings.retry_after,
        })
    }
}

/// The name a pre-routing hook rewrote the level's group to for this pick, if it did.
fn routed_name(target: &Target, candidates: &Candidates) -> Option<String> {
    (candidates.name != target.group).then(|| candidates.name.clone())
}
