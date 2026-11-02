//! The call loop: select, run the attempt, then retry in the same group or fall back to another
//! until it succeeds or the policy is exhausted. Mirrors `Router.async_function_with_fallbacks`,
//! `async_function_with_fallbacks_common_utils`, `async_function_with_retries`,
//! `should_retry_this_error` and `_time_to_sleep_before_retry` in `litellm/router.py`, and
//! `run_async_fallback` in `router_utils/fallback_event_handlers.py`.

use std::{borrow::Cow, future::Future, pin::Pin, sync::Arc, time::Duration};

use serde_json::{Map, Value};

use tracing::warn;

use crate::{
    Core, Snapshot,
    config::Op,
    error::{AttemptError, ConfigError, Error, FailureKind, Unsupported},
    fallback::{FallbackKeys, FallbackTarget, Fallbacks},
    health::{GroupShape, should_retry_status},
    host::{Attempt, Blocked, Output, PickError, RetryCause, RouterEvent, RouterRequest},
    mock::MockFailure,
    prerouting::{PreRoute, PreRouted},
    registry::{ResolveKind, ResolveScope},
    retry::{RetryContext, resolve_retry_policy, should_retry},
    select::{Candidates, SelectInputs},
};

mod stream;

pub use self::stream::{StreamOrigin, StreamPolicy};

/// `INITIAL_RETRY_DELAY`, `MAX_RETRY_DELAY` and `JITTER` in `litellm/constants.py`.
const INITIAL_RETRY_DELAY: f64 = 0.5;
const MAX_RETRY_DELAY: f64 = 8.0;
const JITTER: f64 = 0.75;
/// `calculate_retry_after` obeys a provider's retry-after up to a minute.
const MAX_HONOURED_RETRY_AFTER: Duration = Duration::from_secs(60);

#[derive(Clone, Debug)]
pub struct RouterResponse {
    pub output: Output,
    pub deployment_id: String,
    pub model_group: String,
    pub attempted_retries: u32,
    /// The retry budget of the loop a retry succeeded in (`x-litellm-max-retries`); `None`
    /// when the first attempt succeeded.
    pub max_retries: Option<u32>,
    pub attempted_fallbacks: u32,
}

/// A failed attempt with the deployment it ran on, when one was picked.
struct Failed {
    error: Error,
    deployment_id: Option<String>,
}

impl From<Error> for Failed {
    fn from(error: Error) -> Self {
        Self {
            error,
            deployment_id: None,
        }
    }
}

/// Where a level of the fallback walk sits: `depth` is `run_async_fallback`'s counter, one more
/// for every target tried, `level` how many fallbacks deep the walk is, and `scope` the level's
/// own copy of the request metadata (every hop copies its parent's).
#[derive(Clone, Copy, Debug, Default)]
struct Hop {
    depth: u32,
    level: u32,
    scope: u32,
}

/// What the whole walk shares: the attempts made (`attempted_targets`) and the scopes handed out.
#[derive(Debug, Default)]
struct Walk {
    attempted: Vec<Tried>,
    scopes: u32,
}

/// What one level of the walk serves: a group, or the same group at a later `order`, with the
/// request params the walk's dict targets merged so far (`kwargs.update(target)`).
#[derive(Clone, Debug)]
struct Target {
    group: String,
    order: Option<i64>,
    params: Option<Arc<Map<String, Value>>>,
    /// `_excluded_deployment_ids` a weighted failover hop put on the request: never picked.
    excluded: Arc<[String]>,
    /// `_failover_excluded_ids` in the level's metadata: the deployments weighted failover
    /// already excluded on the way here.
    failover_excluded: Arc<[String]>,
}

/// A step of a level's fallback chain: a configured target, or an order level of the group.
#[derive(Clone, Debug)]
enum Next {
    Fallback(FallbackTarget),
    Order(String, i64),
    /// A weighted failover hop: the same group without the deployments already excluded.
    Failover(String, Arc<[String]>),
}

/// An attempt the walk already made (`fallback_attempt_key`).
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
    fn of(snapshot: &'a Snapshot, request: &'a RouterRequest) -> Self {
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
    request: &RouterRequest,
    target: &Target,
    at: Hop,
    attempted: &[Tried],
) -> bool {
    let mut lookup = vec![target.group.as_str()];
    if request.model != target.group {
        lookup.push(request.model.as_str());
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

/// The deployments `_core_get_model_list(model_name=group)` lists: the group's own, or a routing
/// group's members.
fn group_members(snapshot: &Snapshot, group: &str) -> Vec<Arc<crate::registry::Deployment>> {
    snapshot
        .registry
        .group(group)
        .or_else(|| {
            snapshot
                .registry
                .resolve(group, ResolveScope::default())
                .filter(|resolution| resolution.kind == ResolveKind::RoutingGroup)
                .map(|resolution| resolution.deployments)
        })
        .unwrap_or_default()
}

/// Whether the op's attempt stamps `failed_deployment_id` on its error (`_completion`,
/// `_acompletion` and the generic API calls), which weighted failover needs.
fn stamps_failed_deployment(op: &Op) -> bool {
    match op {
        Op::Embedding
        | Op::ImageGeneration
        | Op::Transcription
        | Op::Speech
        | Op::TextCompletion => false,
        Op::Other(name) => !matches!(
            name.as_str(),
            "arerank" | "embedding" | "image_generation" | "text_completion"
        ),
        Op::Completion | Op::Responses | Op::Messages | Op::Ocr => true,
    }
}

/// `is_bad_request_without_fallback` with `fallback_on_unsupported_params` off: an unsupported
/// params error, or a provider's plain 400 bad request.
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
    failure.status == Some(400) && failure.names_provider
}

type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// `KeyError`'s classes, most specific first.
/// The failure a retry loop decides by: an attempt's, or the exception a failed pick raised
/// (Python's loop retries a pick's error like any other: no deployment available, a stale
/// routing group's `KeyError`, ...).
fn retried_failure(error: &Error) -> Option<Cow<'_, AttemptError>> {
    match error {
        Error::Attempt(failure) => Some(Cow::Borrowed(failure)),
        _ => error.pick_failure().map(Cow::Owned),
    }
}

impl Core {
    pub(crate) async fn call(
        &self,
        op: Op,
        request: Arc<RouterRequest>,
    ) -> Result<RouterResponse, Error> {
        let snapshot = self.snapshot.load_full();
        let mut walk = Walk::default();
        let target = Target {
            group: request.model.clone(),
            order: None,
            params: None,
            excluded: Arc::from([]),
            failover_excluded: Arc::from([]),
        };
        self.with_fallbacks(
            &snapshot,
            &op,
            &request,
            target,
            Hop::default(),
            &mut walk,
            None,
        )
        .await
    }

    /// One level of `async_function_with_fallbacks`: the target's retry loop, then its
    /// fallbacks. A level that fails raises its own error, whatever its fallbacks raised.
    /// `resumed` is a failure the level already had (a stream that failed after its attempt
    /// returned): its fallbacks run without the retry loop.
    #[allow(clippy::too_many_arguments)]
    fn with_fallbacks<'a>(
        &'a self,
        snapshot: &'a Snapshot,
        op: &'a Op,
        request: &'a Arc<RouterRequest>,
        target: Target,
        at: Hop,
        walk: &'a mut Walk,
        resumed: Option<Error>,
    ) -> BoxFuture<'a, Result<RouterResponse, Error>> {
        Box::pin(async move {
            let mock = request
                .mock_failure
                .filter(|mock| mock.before_retries() && at.level == 0);
            let (error, failed_id) = match (resumed, mock) {
                (Some(error), _) => (error, None),
                (None, Some(mock)) => (self.mock_failed(mock, &target, at, None), None),
                (None, None) => match self
                    .with_retries(snapshot, op, request, &target, at, &walk.attempted)
                    .await
                {
                    Ok(response) => return Ok(response),
                    Err(failed) => (failed.error, failed.deployment_id),
                },
            };
            self.level_failed(at, &target, &error, false);
            if self.raises_without_fallback(snapshot, request, &error) {
                return Err(error);
            }
            // `async_function_with_fallbacks_common_utils` lists the group's deployments
            // (`_core_get_model_list`) for its order fallbacks: a written index position past the
            // model list raises there, before any fallback.
            if let Err(raised) = crate::resolve::all_deployments(
                &snapshot.registry,
                &target.group,
                request.team_id.as_deref(),
            ) {
                self.level_failed(at, &target, &raised, true);
                return Err(raised);
            }
            let (targets, order_path) = self.fallback_targets(snapshot, request, &target, &error);
            let mut target = target;
            // `_maybe_run_weighted_failover`: its own `run_async_fallback`, before the regular one.
            let failover = if order_path {
                None
            } else {
                match self
                    .weighted_failover(snapshot, op, request, &target, &error, failed_id.as_deref())
                    .await
                {
                    Ok(failover) => failover,
                    Err(raised) => {
                        // `_get_routing_context` raised inside the level's fallback handling: the
                        // level raises that error, before any regular fallback.
                        self.level_failed(at, &target, &raised, true);
                        return Err(raised);
                    }
                }
            };
            if let Some(failover) = failover {
                let excluded = match &failover {
                    Next::Failover(_, excluded) => excluded.clone(),
                    _ => Arc::from([]),
                };
                target.failover_excluded = excluded;
                if let Ok(response) = self
                    .run_fallbacks(snapshot, op, request, &target, at, walk, vec![failover])
                    .await
                {
                    return Ok(response);
                }
            }
            let Some(targets) = targets else {
                return Err(error);
            };
            match self
                .run_fallbacks(snapshot, op, request, &target, at, walk, targets)
                .await
            {
                Ok(response) => Ok(response),
                Err(()) => Err(error),
            }
        })
    }

    /// `run_async_fallback` over one chain: the depth is checked once on entry, then every
    /// target tried counts one more. `Err(())` when the chain served nothing.
    #[allow(clippy::too_many_arguments)]
    fn run_fallbacks<'a>(
        &'a self,
        snapshot: &'a Snapshot,
        op: &'a Op,
        request: &'a Arc<RouterRequest>,
        target: &'a Target,
        at: Hop,
        walk: &'a mut Walk,
        targets: Vec<Next>,
    ) -> BoxFuture<'a, Result<RouterResponse, ()>> {
        Box::pin(async move {
            let max_fallbacks = request
                .max_fallbacks
                .unwrap_or(snapshot.settings.max_fallbacks);
            if at.depth >= max_fallbacks {
                return Err(());
            }
            let failed = Tried::Target(FallbackTarget::Name(target.group.clone()));
            if !walk.attempted.contains(&failed) {
                walk.attempted.push(failed);
            }
            let mut params = target.params.clone();
            let mut hop = at.depth;
            for next in targets {
                let (tried, group, order) = match &next {
                    Next::Order(group, order) => (
                        Tried::Order(group.clone(), *order),
                        group.clone(),
                        Some(*order),
                    ),
                    Next::Failover(group, excluded) => {
                        let mut key = excluded.to_vec();
                        key.sort();
                        (Tried::Failover(group.clone(), key), group.clone(), None)
                    }
                    Next::Fallback(FallbackTarget::Name(name)) if *name == target.group => continue,
                    Next::Fallback(fallback) => {
                        if let Some(model) = fallback.model()
                            && model != target.group
                            && !self.host.allow_fallback(request, model).await
                        {
                            continue;
                        }
                        let group = fallback.model().unwrap_or(&target.group).to_owned();
                        (Tried::Target(fallback.identity()), group, None)
                    }
                };
                if walk.attempted.contains(&tried) {
                    continue;
                }
                walk.attempted.push(tried);
                if let Next::Fallback(fallback) = &next
                    && let Some(merged) = fallback.params()
                {
                    let mut combined = params.as_deref().cloned().unwrap_or_default();
                    combined.extend(merged.iter().map(|(k, v)| (k.clone(), v.clone())));
                    params = Some(Arc::new(combined));
                }
                hop += 1;
                self.host.observe(&RouterEvent::RetryLogged {
                    scope: at.scope,
                    level: at.level,
                    model: target.group.clone(),
                    attempted_retries: None,
                    cause: RetryCause::Level,
                });
                walk.scopes += 1;
                self.host.observe(&RouterEvent::Fallback {
                    from: target.group.clone(),
                    to: group.clone(),
                    order,
                    depth: hop,
                    level: at.level + 1,
                    scope: walk.scopes,
                    parent: at.scope,
                    max_fallbacks,
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
                };
                let parent = at;
                let at = Hop {
                    depth: hop,
                    level: at.level + 1,
                    scope: walk.scopes,
                };
                let result = self
                    .with_fallbacks(snapshot, op, request, next, at, walk, None)
                    .await;
                self.host.observe(&RouterEvent::FallbackDone {
                    from: target.group.clone(),
                    to: group,
                    level: parent.level,
                    scope: at.scope,
                    parent: parent.scope,
                    succeeded: result.is_ok(),
                });
                if let Ok(mut response) = result {
                    response.attempted_fallbacks = hop;
                    return Ok(response);
                }
            }
            Err(())
        })
    }

    /// `_maybe_run_weighted_failover`'s hop for a failed deployment, when it applies: the
    /// simple-shuffle group re-picked without every deployment already excluded, while one
    /// that is not cooling down remains.
    async fn weighted_failover(
        &self,
        snapshot: &Snapshot,
        op: &Op,
        request: &RouterRequest,
        target: &Target,
        error: &Error,
        failed_id: Option<&str>,
    ) -> Result<Option<Next>, Error> {
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
        if !stamps_failed_deployment(op) || error.attempt().is_none() {
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
        if strategy != crate::config::RoutingStrategy::SimpleShuffle {
            return Ok(None);
        }
        let Some(failed_id) = failed_id else {
            return Ok(None);
        };
        let mut excluded: Vec<String> = target.failover_excluded.to_vec();
        if !excluded.iter().any(|id| id == failed_id) {
            excluded.push(failed_id.to_owned());
        }
        let ids: Vec<String> = group_members(snapshot, &target.group)
            .iter()
            .filter(|d| d.usable_by_team(request.team_id.as_deref()))
            .map(|d| d.id.clone())
            .collect();
        let cooling = self.health.active_cooldowns(&ids).await.unwrap_or_default();
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
        snapshot: &Snapshot,
        request: &RouterRequest,
        error: &Error,
    ) -> bool {
        if request.disable_fallbacks {
            return true;
        }
        match error {
            Error::Attempt(failure) => {
                failure.kind == FailureKind::GuardrailIntervention
                    || (!snapshot.settings.fallback_on_unsupported_params
                        && is_bad_request_without_fallback(failure))
            }
            Error::NoHealthyDeployments { .. }
            | Error::UnknownModel { .. }
            | Error::RoutingPluginsLeftNothing { .. }
            | Error::MissingRoutingGroup { .. }
            | Error::DefaultDeploymentRaises { .. }
            | Error::IndexOutOfRange { .. }
            | Error::ContextWindowExceeded { .. } => false,
            Error::UnsupportedParams { .. }
            | Error::RoutingPluginsNeedAsync
            | Error::InvalidRouterWeights
            | Error::NonNumericWeight { .. }
            | Error::Config(_)
            | Error::Cache(_)
            | Error::Retired
            | Error::Paused { .. }
            | Error::NotPassThrough { .. }
            | Error::NoPassThroughDeployments { .. }
            | Error::OnlyMarkers { .. }
            | Error::PtuReserved { .. }
            | Error::MultipleTeams { .. }
            | Error::Reserved { .. } => true,
        }
    }

    /// The fallback targets for a failure, or `None` when it must be raised as is
    /// (`async_function_with_fallbacks_common_utils`).
    fn fallback_targets(
        &self,
        snapshot: &Snapshot,
        request: &RouterRequest,
        target: &Target,
        error: &Error,
    ) -> (Option<Vec<Next>>, bool) {
        // A plain `ValueError` (routing plugins left nothing) takes the generic fallbacks, as a
        // missing group does; a pre-call context window refusal takes the context window ones.
        // Every other error was raised before (`raises_without_fallback`).
        let failure = match error {
            Error::Attempt(failure) => Some(failure),
            _ => None,
        };
        let context_window = matches!(error, Error::ContextWindowExceeded { .. });
        let is_a = |name: &str| {
            failure.is_some_and(|f| f.is_a(name))
                || (context_window && name == "ContextWindowExceededError")
        };
        let lists = Lists::of(snapshot, request);
        // `fallback_lookup_groups`: the group being served, then the group the request was for.
        let mut lookup = vec![target.group.as_str()];
        if request.model != target.group {
            lookup.push(request.model.as_str());
        }
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
        let skip_order = is_a("ContextWindowExceededError") || is_a("ContentPolicyViolationError");
        if !skip_order {
            // A routing group's members, with a priority group's priorities as their order.
            let mut orders: Vec<i64> = group_members(snapshot, &target.group)
                .iter()
                .filter(|d| d.usable_by_team(request.team_id.as_deref()))
                .filter_map(|d| d.order())
                .collect();
            orders.sort_unstable();
            orders.dedup();
            if orders.len() > 1 {
                let current = target.order.unwrap_or(orders[0]);
                let mut targets: Vec<Next> = orders
                    .into_iter()
                    .filter(|order| *order > current)
                    .map(|order| Next::Order(target.group.clone(), order))
                    .collect();
                if let Some(list) = lists.fallbacks {
                    targets.extend(fallbacks_of(chain(list).unwrap_or_default()));
                }
                if !targets.is_empty() {
                    return (Some(targets), true);
                }
            }
        }
        if let Some(list) = lists.fallbacks
            && list.is_client_side(lists.keys)
        {
            return (Some(fallbacks_of(list.as_chain())), false);
        }
        let specific = if is_a("ContextWindowExceededError") {
            lists.context_window
        } else if is_a("ContentPolicyViolationError") {
            lists.content_policy
        } else {
            None
        };
        if let Some(list) = specific {
            return (
                list.lookup_groups(&lookup, lists.keys).map(fallbacks_of),
                false,
            );
        }
        (
            lists
                .fallbacks
                .and_then(|list| list.lookup_groups(&lookup, lists.keys))
                .map(fallbacks_of),
            false,
        )
    }

    async fn with_retries(
        &self,
        snapshot: &Snapshot,
        op: &Op,
        request: &Arc<RouterRequest>,
        target: &Target,
        at: Hop,
        attempted: &[Tried],
    ) -> Result<RouterResponse, Failed> {
        let walk = (at, attempted);
        let mut inputs = SelectInputs {
            target_order: target.order,
            excluded_ids: target.excluded.clone(),
            ..SelectInputs::default()
        };
        let counting = SelectInputs {
            count_only: true,
            ..SelectInputs::default()
        };
        let settings = &snapshot.settings;
        let first_budget = request.num_retries.unwrap_or(settings.num_retries);
        let first = match request.mock_failure {
            Some(MockFailure::RateLimit) => {
                // `_handle_mock_testing_rate_limit_error`: a lone deployment's `num_retries`
                // rides on the mock error, as on a real one.
                let lone = snapshot
                    .registry
                    .group(&target.group)
                    .filter(|group| group.len() == 1)
                    .and_then(|group| group[0].raw_params.get("num_retries")?.as_u64())
                    .and_then(|retries| u32::try_from(retries).ok());
                Err(self
                    .mock_failed(MockFailure::RateLimit, target, at, lone)
                    .into())
            }
            _ => {
                self.attempt(
                    snapshot,
                    op,
                    request,
                    target,
                    &inputs,
                    (0, first_budget),
                    walk,
                )
                .await
            }
        };
        let mut failed = match first {
            Ok(response) => return Ok(response),
            Err(failed) => failed,
        };
        let guardrail = matches!(
            &failed.error,
            Error::Attempt(failure) if failure.kind == FailureKind::GuardrailIntervention
        );
        // The first failure looks the group's retry policy up; a lookup that raises replaces the
        // failure with its own error, before any retry (`async_function_with_retries`).
        if !guardrail && request.retry_policy_raises.contains(&target.group) {
            self.host.observe(&RouterEvent::RetryPolicyRaised {
                level: at.level,
                model: target.group.clone(),
                cause: RetryCause::of(&failed.error),
            });
            return Err(Failed {
                error: Error::Attempt(AttemptError::host_raised(&["Exception"])),
                deployment_id: None,
            });
        }
        if guardrail {
            return Err(failed);
        }
        let Some(first_failure) = retried_failure(&failed.error) else {
            return Err(failed);
        };
        let failure: &AttemptError = &first_failure;
        let mut num_retries = request
            .num_retries
            .or(failure.num_retries)
            .unwrap_or(settings.num_retries);
        let group_policies = request
            .model_group_retry_policy
            .as_ref()
            .unwrap_or(&settings.model_group_retry_policy);
        let policy_retries = (request.num_retries != Some(0))
            .then(|| {
                resolve_retry_policy(
                    settings.retry_policy.as_ref(),
                    &target.group,
                    Some(group_policies),
                )?
                .retries_for(failure)
            })
            .flatten();
        let policy_applies = policy_retries.is_some();
        num_retries = policy_retries.unwrap_or(num_retries);
        drop(first_failure);
        // `_async_get_healthy_deployments` finds no group for a deployment id, so a call by id
        // counts no deployments when deciding whether to retry.
        let counted = |c: &Candidates, n: usize| if c.kind.is_single() { 0 } else { n };
        let all = self
            .candidates(snapshot, op, request, &target.group, &counting)
            .await
            .map(|c| counted(&c, c.all.len()))
            .unwrap_or(0);

        for retry in 0..=num_retries {
            if retry > 0 {
                self.log_retry(target, at, retry, &failed.error);
            }
            let healthy_ids: Vec<String> = self
                .candidates(snapshot, op, request, &target.group, &counting)
                .await
                .map(|c| {
                    let ids = c.healthy.iter().map(|d| d.id.clone());
                    ids.take(counted(&c, c.healthy.len())).collect()
                })
                .unwrap_or_default();
            let healthy = healthy_ids.len();
            // A pick's error is retried as Python's loop retries any exception: a retry policy
            // keeps going, else `should_retry_this_error` decides (it raises when nothing is
            // healthy).
            let Some(failure) = retried_failure(&failed.error) else {
                return Err(failed);
            };
            if !policy_applies && !self.should_retry(snapshot, request, &failure, healthy, all) {
                return Err(failed);
            }
            if num_retries == 0 {
                return Err(failed);
            }
            if retry == 0 {
                self.log_retry(target, at, 0, &failed.error);
            }
            if let (Some(status), Some(id)) = (failure.status, &failed.deployment_id)
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
            let delay = self.time_to_sleep(
                snapshot,
                failure.retry_after,
                num_retries - retry,
                num_retries,
                healthy,
                all,
            );
            self.host.sleep(delay).await;
            if retry == num_retries {
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
            match self
                .attempt(
                    snapshot,
                    op,
                    request,
                    target,
                    &inputs,
                    (retry + 1, num_retries),
                    walk,
                )
                .await
            {
                Ok(mut response) => {
                    response.max_retries = Some(num_retries);
                    return Ok(response);
                }
                Err(next) => failed = next,
            }
        }
        Err(failed)
    }

    /// The level serving `target` raises a pick's `error` ([`RouterEvent::LevelFailed`]);
    /// `handling` when its fallback handling raised it.
    fn level_failed(&self, at: Hop, target: &Target, error: &Error, handling: bool) {
        if let Some(picked) = PickError::of(error) {
            self.host.observe(&RouterEvent::LevelFailed {
                level: at.level,
                model: target.group.clone(),
                error: picked,
                handling,
            });
        }
    }

    /// `_core_log_retry` for a failure in the level's retry loop.
    fn log_retry(&self, target: &Target, at: Hop, retry: u32, error: &Error) {
        let cause = match RetryCause::of(error) {
            RetryCause::Other => RetryCause::Attempt,
            cause => cause,
        };
        self.host.observe(&RouterEvent::RetryLogged {
            scope: at.scope,
            level: at.level,
            model: target.group.clone(),
            attempted_retries: Some(retry),
            cause,
        });
    }

    /// A `mock_testing_*` failure for the level's group, reported to the host, which raises
    /// the matching exception.
    fn mock_failed(
        &self,
        mock: MockFailure,
        target: &Target,
        at: Hop,
        num_retries: Option<u32>,
    ) -> Error {
        self.host.observe(&RouterEvent::MockFailed {
            mock,
            level: at.level,
            model: target.group.clone(),
            num_retries,
        });
        Error::Attempt(mock.failure(num_retries))
    }

    fn should_retry(
        &self,
        snapshot: &Snapshot,
        request: &RouterRequest,
        failure: &AttemptError,
        healthy: usize,
        all: usize,
    ) -> bool {
        let lists = Lists::of(snapshot, request);
        should_retry(
            failure,
            RetryContext {
                healthy,
                all,
                has_context_window_fallbacks: lists.context_window.is_some(),
                has_content_policy_fallbacks: lists.content_policy.is_some(),
                has_regular_fallbacks: lists.has_regular(),
            },
        )
    }

    /// `_time_to_sleep_before_retry` and `calculate_retry_after`.
    fn time_to_sleep(
        &self,
        snapshot: &Snapshot,
        retry_after: Option<Duration>,
        remaining_retries: u32,
        num_retries: u32,
        healthy: usize,
        all: usize,
    ) -> Duration {
        if all != 1 && healthy > 0 {
            return Duration::ZERO;
        }
        let jitter = JITTER * self.rng.next_f64();
        if let Some(retry_after) = retry_after
            && !retry_after.is_zero()
            && retry_after <= MAX_HONOURED_RETRY_AFTER
        {
            return retry_after + Duration::from_secs_f64(jitter);
        }
        let exponent = num_retries.saturating_sub(remaining_retries);
        let backoff = (INITIAL_RETRY_DELAY * 2f64.powi(exponent as i32))
            .max(snapshot.settings.retry_after.as_secs_f64())
            .min(MAX_RETRY_DELAY);
        Duration::from_secs_f64(backoff + jitter)
    }

    /// One attempt: select a deployment, run it through the host, record the outcome.
    #[allow(clippy::too_many_arguments)]
    async fn attempt(
        &self,
        snapshot: &Snapshot,
        op: &Op,
        request: &Arc<RouterRequest>,
        target: &Target,
        inputs: &SelectInputs,
        (retry, max_retries): (u32, u32),
        (at, attempted): (Hop, &[Tried]),
    ) -> Result<RouterResponse, Failed> {
        if request.unrouted {
            return self
                .unrouted_attempt(snapshot, op, request, target, (retry, max_retries), at)
                .await;
        }
        crate::routing_plugins::refuse_sync_pick(
            snapshot.settings.routing_plugins,
            &snapshot.settings.routing_strategy,
            request,
        )?;
        let routed = self
            .pre_routed(PreRoute {
                op,
                request,
                name: &target.group,
                fallback_depth: at.depth,
                retry,
            })
            .await?;
        let (deployment, candidates) = match routed {
            PreRouted::Model(name) => {
                snapshot.missing_routing_group(request, &name)?;
                let candidates = self
                    .candidates(snapshot, op, request, &name, inputs)
                    .await?;
                if candidates.healthy.is_empty() {
                    return Err(Error::NoHealthyDeployments {
                        model: candidates.model_group,
                    }
                    .into());
                }
                let deployment = self.pick(snapshot, op, request, &candidates).await?;
                (deployment, candidates)
            }
            // A custom strategy's pick is served as is, for the requested name.
            PreRouted::Deployment(deployment) => {
                let candidates = Candidates {
                    name: target.group.clone(),
                    model_group: target.group.clone(),
                    kind: ResolveKind::Group,
                    all: vec![deployment.clone()],
                    healthy: vec![deployment.clone()],
                };
                (deployment, candidates)
            }
        };
        self.host.observe(&RouterEvent::Selected {
            model_group: candidates.model_group.clone(),
            deployment_id: deployment.id.clone(),
        });
        if !self.host.supports(op, deployment.provider()) {
            return Err(Error::Config(ConfigError::Unsupported(Unsupported::Op {
                op: op.clone(),
                provider: deployment.provider().to_owned(),
            }))
            .into());
        }
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
                    },
                    None,
                    snapshot.settings.dynamic.default_ref(),
                    jiff::Timestamp::now(),
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
        // Every attempt runs through the host; the core records every outcome itself.
        match self.host.invoke(&attempt).await {
            Ok(output) => {
                if let Err(error) = self
                    .health
                    .record_success(&deployment, output.total_tokens)
                    .await
                {
                    warn!(deployment_id = %deployment.id, error = %error, "usage was not recorded");
                }
                self.host.observe(&RouterEvent::Succeeded {
                    deployment_id: deployment.id.clone(),
                });
                // `_should_raise_content_policy_error`: a blocked response is raised only when
                // a content-policy fallback would serve it, and counts as no failure.
                let raised =
                    match output.blocked {
                        Some(Blocked::ContentFilter) => self
                            .has_content_policy_fallback(snapshot, request, target, at, attempted),
                        Some(Blocked::Refusal) => self
                            .refusal_fallback_available(snapshot, request, target, at, attempted),
                        None => false,
                    };
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
                        error: Error::Attempt(failure),
                        deployment_id: Some(deployment.id.clone()),
                    });
                }
                Ok(RouterResponse {
                    output,
                    deployment_id: deployment.id.clone(),
                    model_group: candidates.model_group,
                    attempted_retries: retry,
                    max_retries: None,
                    attempted_fallbacks: at.depth,
                })
            }
            Err(failure) => {
                if !failure.recorded {
                    self.record_failure(snapshot, request, &candidates, &deployment, &failure)
                        .await;
                }
                Err(Failed {
                    error: Error::Attempt(failure),
                    deployment_id: Some(deployment.id.clone()),
                })
            }
        }
    }

    /// An attempt on a name that is no model group: the host serves it under that name, with no
    /// deployment picked and no outcome recorded.
    async fn unrouted_attempt(
        &self,
        snapshot: &Snapshot,
        op: &Op,
        request: &Arc<RouterRequest>,
        target: &Target,
        (retry, max_retries): (u32, u32),
        at: Hop,
    ) -> Result<RouterResponse, Failed> {
        let deployment =
            Arc::new(crate::registry::Deployment::unrouted(&target.group).map_err(Error::from)?);
        let attempt = Attempt {
            op: op.clone(),
            request: request.clone(),
            deployment,
            name: target.group.clone(),
            model_group: target.group.clone(),
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
        match self.host.invoke(&attempt).await {
            Ok(output) => Ok(RouterResponse {
                output,
                deployment_id: String::new(),
                model_group: target.group.clone(),
                attempted_retries: retry,
                max_retries: None,
                attempted_fallbacks: at.depth,
            }),
            Err(failure) => Err(Error::Attempt(failure).into()),
        }
    }

    /// `_has_content_policy_fallback`: the content-policy chain resolves for the level's
    /// lookup groups and holds a target the walk has not tried; with no content-policy list,
    /// any router `"*"` entry.
    fn has_content_policy_fallback(
        &self,
        snapshot: &Snapshot,
        request: &RouterRequest,
        target: &Target,
        at: Hop,
        attempted: &[Tried],
    ) -> bool {
        let lists = Lists::of(snapshot, request);
        match lists.content_policy {
            Some(list) => unattempted_chain(list, &lists, request, target, at, attempted),
            None => snapshot.settings.dynamic.fallbacks.has_star(),
        }
    }

    /// `_refusal_fallback_available`: the content-policy list decides when there is one;
    /// else a router `"*"` entry, else a generic chain with a target the walk has not tried.
    fn refusal_fallback_available(
        &self,
        snapshot: &Snapshot,
        request: &RouterRequest,
        target: &Target,
        at: Hop,
        attempted: &[Tried],
    ) -> bool {
        if request.disable_fallbacks {
            return false;
        }
        let lists = Lists::of(snapshot, request);
        if lists.content_policy.is_some() {
            return self.has_content_policy_fallback(snapshot, request, target, at, attempted);
        }
        if snapshot.settings.dynamic.fallbacks.has_star() {
            return true;
        }
        lists
            .fallbacks
            .is_some_and(|list| unattempted_chain(list, &lists, request, target, at, attempted))
    }

    async fn record_failure(
        &self,
        snapshot: &Snapshot,
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
                    if let Err(error) = self.health.count_failed_request(deployment).await {
                        warn!(deployment_id = %deployment.id, error = %error, "usage was not recorded");
                    }
                    return;
                }
            }
        } else {
            deployment
        };
        let shape = GroupShape {
            // The deployment's own group decides, also when the call named the deployment by id.
            single_deployment: snapshot
                .registry
                .group(&deployment.model_name)
                .map_or(candidates.all.len() == 1, |group| group.len() == 1)
                // A routing group call with alternatives cools its member down so selection
                // moves on (`routing_group_alternatives`).
                && !(candidates.kind == ResolveKind::RoutingGroup && candidates.all.len() > 1),
            has_fallback_group: snapshot
                .settings
                .dynamic
                .fallback_targets(&[candidates.model_group.as_str()])
                .is_some(),
        };
        if let Err(error) = self.health.count_failed_request(deployment).await {
            warn!(deployment_id = %deployment.id, error = %error, "usage was not recorded");
        }
        match self
            .health
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
            Err(error) => {
                warn!(deployment_id = %deployment.id, error = %error, "cooldown was not recorded");
            }
        }
    }
}
