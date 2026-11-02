//! [`Host`]: the router core's only boundary. Everything the core needs from the embedding
//! process goes through it, and every attempt runs through [`Host::invoke`]: the core never runs
//! an operation itself. `PythonHost` serves it in a Python process (attempts call `litellm.<op>`);
//! `NativeHost` serves it in the Rust server (attempts call the `inference-*` crates).

use std::{collections::BTreeMap, sync::Arc, time::Duration};

use async_trait::async_trait;
use serde_json::{Map, Value};

use crate::{
    cache::RouterCache,
    config::{Op, RetryPolicy, RoutingStrategy},
    error::{AttemptError, Error, FailureKind, Payload},
    fallback::Fallbacks,
    health::Health,
    mock::MockFailure,
    precall::PreCallFacts,
    prerouting::{PreRoute, PreRouted},
    registry::{Deployment, Registry},
    settings::Settings,
    strategy::StrategyScope,
};

/// One call into the router.
#[derive(Clone, Debug, Default)]
pub struct RouterRequest {
    /// The requested name: a model group, alias, routing group, deployment id or wildcard match.
    pub model: String,
    /// The request params, for native executors and plugins that read them.
    pub body: Value,
    /// `user_api_key_team_id`.
    pub team_id: Option<String>,
    /// The caller's `user_role` is `proxy_admin`.
    pub proxy_admin: bool,
    /// The caller's `models` plus `team_models` (`user_api_key_auth`), which narrow a caller
    /// allowed only through access groups to those groups' deployments.
    pub allowed_models: Vec<String>,
    /// For a proxy admin, the model the request names (`model`, else the metadata
    /// `model_group`), which keeps one team's deployments listed under it (`filter_team_based_models`).
    pub admin_model: Option<String>,
    pub specific_deployment: bool,
    /// A per-request `num_retries`, which wins over the router's.
    pub num_retries: Option<u32>,
    pub disable_fallbacks: bool,
    /// `_router_weights` as the request passed it: model group to deployment id to weight,
    /// validated when simple-shuffle picks ([`crate::weights::validate`]).
    pub router_weights: Option<Value>,
    /// The deployment a call on an existing resource (a video id) names: picked whenever the
    /// filters keep it (`_narrow_to_video_deployment`).
    pub pinned_deployment_id: Option<String>,
    /// A per-request `routing_strategy` override, as Python normalizes it.
    pub routing_strategy: Option<String>,
    /// Whether tag routing may apply to this request (router, request or model setting), so the
    /// host's tag stage runs.
    pub filter_tags: bool,
    /// A pass-through pick: only `use_in_pass_through` deployments may serve it.
    pub pass_through: Option<crate::select::PassThrough>,
    /// `_target_order`: the order level to pick from instead of the lowest.
    pub target_order: Option<i64>,
    /// `_excluded_deployment_ids`: never picked, even when that leaves nothing.
    pub excluded_ids: Vec<String>,
    /// `_retry_skipped_deployment_ids`: skipped unless that leaves nothing.
    pub retry_skipped_ids: Vec<String>,
    /// The request carries an image (`_request_carries_image`), so deployments that declare no
    /// vision are skipped while any other remains.
    pub has_image: bool,
    /// An async pick: the host's pre-routing step runs before each pick resolves its name.
    pub pre_routes: bool,
    /// The requested name is no model group (a search tool name): nothing is selected, every
    /// attempt is handed to the host as is, and no outcome is recorded for a deployment
    /// (`_asearch_with_fallbacks`, whose helper picks among the router's search tools).
    pub unrouted: bool,
    /// Fallback lists the request passed (`Some(None)` when it passed `None`), which replace
    /// the router's.
    pub fallbacks: Option<Option<Fallbacks>>,
    pub context_window_fallbacks: Option<Option<Fallbacks>>,
    pub content_policy_fallbacks: Option<Option<Fallbacks>>,
    /// A per-request `max_fallbacks`, which replaces the router's.
    pub max_fallbacks: Option<u32>,
    /// A per-request `cooldown_time`: the call's own params carry it, so it cools a failed
    /// deployment down for that long unless its `model_info` sets one.
    pub cooldown_time: Option<f64>,
    /// A `mock_testing_*` flag the request set.
    pub mock_failure: Option<MockFailure>,
    /// A per-request `model_group_retry_policy`, which replaces the router's.
    pub model_group_retry_policy: Option<BTreeMap<String, RetryPolicy>>,
    /// The model groups whose `model_group_retry_policy` lookup raises (the policy is no mapping
    /// of policies, or the group's entry is not one): the first failure of a hop serving one ends
    /// the hop with the host's error instead ([`RouterEvent::RetryPolicyRaised`]), no retry.
    pub retry_policy_raises: Vec<String>,
    /// `health_check_probe`: a health check's own pick, which sees no deployment when every
    /// one is marked unhealthy (the safety net keeps them all for any other request).
    pub health_check_probe: bool,
    /// What the host needs to run the call itself, such as the Python kwargs.
    pub payload: Option<Payload>,
}

/// One attempt against one deployment.
#[derive(Clone, Debug)]
pub struct Attempt {
    pub op: Op,
    pub request: Arc<RouterRequest>,
    pub deployment: Arc<Deployment>,
    /// The name this attempt resolved: the requested name, or the fallback target.
    pub name: String,
    /// The model group being served, after fallbacks.
    pub model_group: String,
    /// The `order` level an order fallback targets in the same group.
    pub target_order: Option<i64>,
    /// Every deployment `name` resolved to for this caller, the picked one included.
    pub candidate_ids: Vec<String>,
    /// 0 for the first attempt in this group, then the retry number.
    pub retry: u32,
    pub fallback_depth: u32,
    /// How many fallbacks deep the walk is: 0 for the requested name.
    pub fallback_level: u32,
    pub timeout: Option<Duration>,
    /// Request params the fallback walk's dict targets merged in (`kwargs.update(target)`).
    pub params: Option<Arc<Map<String, Value>>>,
    /// The walk level whose request metadata this attempt carries (see [`RouterEvent::Fallback`]).
    pub scope: u32,
    /// The retry budget the level's loop stamps on the request before this attempt.
    pub max_retries: u32,
    /// How many deployments `get_model_list` lists for the level's group (`model_group_size`).
    pub model_group_size: usize,
}

/// What a successful attempt returns.
#[derive(Clone, Debug)]
pub struct Output {
    pub value: Payload,
    /// Prompt plus completion tokens, counted against the deployment's per-minute usage.
    pub total_tokens: u64,
    /// A response the provider blocked, which the router may raise to fall back.
    pub blocked: Option<Blocked>,
}

/// How a provider blocked a response the call otherwise served.
#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::IntoStaticStr, strum::EnumString)]
#[strum(serialize_all = "snake_case")]
pub enum Blocked {
    /// A chat response whose first choice was content-filtered (or that has no choices).
    ContentFilter,
    /// An Anthropic /v1/messages safeguard refusal carrying `stop_details`.
    Refusal,
}

/// What a plugin or strategy sees: never the router itself.
pub struct RoutingContext<'a> {
    pub registry: &'a Registry,
    pub settings: &'a Settings,
    pub health: &'a Health,
    pub request: &'a RouterRequest,
    pub op: &'a Op,
    pub model_group: &'a str,
    /// The name the request resolved from (Python's `model`), which strategies key state by.
    pub name: &'a str,
    /// Whose selector a strategy stage reads.
    pub scope: StrategyScope,
    /// The deployments the name resolved to for this caller, before health filters
    /// (`request_deployments`), for the tag stage.
    pub pool: &'a [Arc<Deployment>],
}

/// Which selection step the host is asked to run.
#[derive(Clone, Debug, PartialEq)]
pub enum SelectStage {
    /// Pre-call checks and sub-routers registered as plugins: return the deployments that remain.
    Plugins,
    /// A strategy not yet ported: return the candidates with the pick first.
    Strategy(RoutingStrategy),
    /// Tag routing (`get_deployments_for_tag`), not yet ported: return the deployments that remain.
    Tags,
    /// `_deployment_declares_no_vision`: return the candidates that declare no vision support.
    TextOnly,
}

/// Routing events, delivered through [`Host::observe`].
#[derive(Clone, Debug, PartialEq)]
pub enum RouterEvent {
    Selected {
        model_group: String,
        deployment_id: String,
    },
    Succeeded {
        deployment_id: String,
    },
    AttemptFailed {
        deployment_id: String,
        kind: FailureKind,
        status: Option<u16>,
    },
    CooldownSet {
        deployment_id: String,
        cooldown: Duration,
    },
    Retry {
        model_group: String,
        retry: u32,
        delay: Duration,
    },
    Fallback {
        from: String,
        to: String,
        /// The order level of `to` a same-group hop walks to.
        order: Option<i64>,
        depth: u32,
        /// The fallback level `to` is tried at: 1 for the request's own fallbacks.
        level: u32,
        /// The new level's request metadata scope, copied from `parent`'s at the hop.
        scope: u32,
        parent: u32,
        /// The `max_fallbacks` the walk reads, stamped on the hop's request.
        max_fallbacks: u32,
    },
    /// A hop of the fallback walk returned: `log_success_fallback_event` or
    /// `log_failure_fallback_event` for the level at `parent`.
    FallbackDone {
        from: String,
        to: String,
        level: u32,
        scope: u32,
        parent: u32,
        succeeded: bool,
    },
    /// The router raised a `ContentPolicyViolationError` for a blocked response, since a
    /// fallback would serve it (`_should_raise_content_policy_error`,
    /// `_should_raise_anthropic_refusal_error`).
    Blocked {
        level: u32,
        model: String,
        blocked: Blocked,
    },
    /// `log_retry`: the router recorded a failure in `scope`'s request metadata
    /// (`previous_models`, `request_retry_count`) before retrying or falling back.
    RetryLogged {
        scope: u32,
        level: u32,
        /// The group the failed level served (`kwargs["model"]`).
        model: String,
        /// The retry number stamped before the failed try; `None` keeps the scope's own.
        attempted_retries: Option<u32>,
        cause: RetryCause,
    },
    /// A level of the fallback walk serving `model` ended on an error a pick raised (no
    /// deployment for the caller, none available, a stale routing group, ...): Python raises
    /// that error for the level, whatever its earlier attempts raised.
    LevelFailed {
        level: u32,
        model: String,
        error: PickError,
        /// Raised by the level's fallback handling, while the level's own error was handled
        /// (so it chains to that error), rather than by the level's retry loop.
        handling: bool,
    },
    /// The router raised a `mock_testing_*` failure for `model` instead of calling it.
    MockFailed {
        mock: MockFailure,
        level: u32,
        model: String,
        num_retries: Option<u32>,
    },
    /// The first failure of a hop serving `model` looked up a `model_group_retry_policy` that
    /// raises for it: the hop ends with the error the lookup raised (the host raises it, from the
    /// failure `cause` names), without retrying, and falls back as from any plain exception.
    RetryPolicyRaised {
        level: u32,
        model: String,
        cause: RetryCause,
    },
    /// A group's retry loop ran every retry and the last one failed: Python stamps
    /// `num_retries` and `max_retries` on that error.
    RetriesExhausted {
        model_group: String,
        attempted: u32,
        max: u32,
        /// The level the retry loop served.
        level: u32,
        /// The last failure was a pick's error, which the level then raises
        /// ([`RouterEvent::LevelFailed`]), rather than an attempt's.
        picked: bool,
    },
}

/// A pick's error, as the events that name it carry it.
#[derive(Clone, Debug)]
pub struct PickError(pub Arc<Error>);

impl PickError {
    /// The copy of `error` events carry, when a pick raised it.
    pub fn of(error: &Error) -> Option<Self> {
        error.pick_copy().map(|copy| Self(Arc::new(copy)))
    }
}

impl PartialEq for PickError {
    fn eq(&self, other: &Self) -> bool {
        format!("{:?}", self.0) == format!("{:?}", other.0)
    }
}

/// Which error a [`RouterEvent::RetryLogged`] records.
#[derive(Clone, Debug, PartialEq)]
pub enum RetryCause {
    /// The last attempt's failure.
    Attempt,
    /// The error the level ended on (`run_async_fallback`'s `original_exception`).
    Level,
    /// A pick raised this error: the group was unknown or had none available, the stale
    /// routing-group member map named a group no longer configured, ...
    Pick(PickError),
    /// Any other error no attempt raised.
    Other,
}

impl RetryCause {
    /// The cause that names `error`.
    pub fn of(error: &Error) -> Self {
        match error {
            Error::Attempt(_) => Self::Attempt,
            _ => PickError::of(error).map_or(Self::Other, Self::Pick),
        }
    }
}

#[async_trait]
pub trait Host: Send + Sync {
    /// Run one attempt against the picked deployment. The core records its outcome (cooldowns,
    /// allowed-fails counters, usage) itself; the host only runs it.
    async fn invoke(&self, attempt: &Attempt) -> Result<Output, AttemptError>;

    /// Run a selection stage the core does not implement.
    async fn select(
        &self,
        stage: &SelectStage,
        ctx: &RoutingContext<'_>,
        candidates: Vec<Arc<Deployment>>,
    ) -> Result<Vec<Arc<Deployment>>, Error>;

    /// Pre-routing before a pick resolves `ask.name`: the model group to resolve instead, a
    /// custom strategy's deployment to serve, or `None` to resolve the name itself. Asked only
    /// for picks that pre-route (`pre_routes`).
    async fn pre_route(&self, ask: &PreRoute<'_>) -> Result<Option<PreRouted>, Error> {
        let _ = ask;
        Ok(None)
    }

    /// Access and budget checks before a cross-group fallback target.
    async fn allow_fallback(&self, request: &RouterRequest, target: &str) -> bool;

    async fn sleep(&self, duration: Duration);

    fn observe(&self, event: &RouterEvent);

    /// The shared store routing state lives in.
    fn cache(&self) -> Arc<dyn RouterCache>;

    /// Whether [`Self::invoke`] can run `op` on `provider`. Checked for every served op and
    /// deployment when the router is built or changed, so an unsupported pair fails the config,
    /// never a request. `PythonHost` runs every op; `NativeHost` only those with a Rust route.
    fn supports(&self, _op: &Op, _provider: &str) -> bool {
        true
    }

    /// Whether [`Self::select`] can run strategies and plugins.
    fn can_select(&self) -> bool {
        true
    }

    /// The facts the `enable_pre_call_checks` filter decides by (token count, context windows,
    /// param support), for `candidates` and the deployments in `ctx.pool`. A host without them
    /// reports no input, so nothing is checked.
    /// Run the routing plugins for `ctx.name` when `run` (the async pick's pre-routing hook), and
    /// report the candidate models they left on the request, if any.
    async fn routing_plugin_models(
        &self,
        _ctx: &RoutingContext<'_>,
        _run: bool,
    ) -> Result<Option<Vec<String>>, Error> {
        Ok(None)
    }

    async fn pre_call_facts(
        &self,
        _ctx: &RoutingContext<'_>,
        _candidates: &[Arc<Deployment>],
    ) -> Result<PreCallFacts, Error> {
        Ok(PreCallFacts::default())
    }

    /// `async_callback_filter_deployments`: the host's routing callbacks narrow `candidates` in
    /// turn. A filter that raises fails the try with [`Error::Attempt`], which the call retries and
    /// falls back from like any attempt failure. A host without filters keeps every candidate.
    async fn callback_filters(
        &self,
        _ctx: &RoutingContext<'_>,
        candidates: Vec<Arc<Deployment>>,
    ) -> Result<Vec<Arc<Deployment>>, Error> {
        Ok(candidates)
    }
}
