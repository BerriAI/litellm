//! [`Host`]: the router core's only boundary. Everything the core needs from the embedding
//! process goes through it, and every attempt runs through [`Host::invoke`]: the core never runs
//! an operation itself. `PythonHost` serves it in a Python process (attempts call `litellm.<op>`,
//! and the deployment build asks it for secrets, validation, providers and side effects).

use std::{collections::BTreeMap, sync::Arc, time::Duration};

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Map, Value};

use crate::{
    cache::RouterCache,
    config::{Op, RetryPolicy, RoutingStrategy},
    error::{AttemptError, Error, FailureKind, Payload},
    fallback::Fallbacks,
    health::Health,
    mock::MockFailure,
    outcome::OutcomeCallbacks,
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
    /// The request params, for the plugins and host steps that read them.
    pub body: Value,
    /// `user_api_key_team_id`, from the first metadata bucket that holds one
    /// (`get_request_team_id`).
    pub team_id: Option<String>,
    /// The `metadata` bucket's `user_api_key_team_id` alone: the team a failed level lists the
    /// group's deployments for (`async_function_with_fallbacks_common_utils`,
    /// `_stream_order_levels`). `None` lists every team's.
    pub listing_team_id: Option<String>,
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
    /// A per-request `num_retries` that is no count, which replaces the router's as well.
    pub uncounted_retries: Option<UncountedRetries>,
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
    /// The endpoint `endpoint_selection` names around the call, which deployments that declare
    /// `enforce_supported_endpoints` may refuse ([`crate::eligibility`]).
    pub endpoint: Option<String>,
    /// A web search request on an async pick: `filter_web_search_deployments` drops the
    /// deployments that declare no support. The sync pick never runs that filter.
    pub web_search: bool,
    /// The handler name of a call only some providers serve (`_ROUTE_CAPABILITY_RESOLVERS`):
    /// the deployments whose provider cannot serve it are dropped unless that leaves nothing.
    pub route_handler: Option<String>,
    /// A Decisions call carrying audio or videos: the deployments that refuse them are skipped
    /// unless that leaves nothing (`_without_decisions_deployments_refusing_media`).
    pub decisions_media: bool,
    /// A batch create (`batch_create_routing`): when the caller may use a deployment declared
    /// `model_info.mode: batch`, the Azure deployments that are not are skipped
    /// (`without_deployments_that_cannot_batch`).
    pub batch_create: bool,
    /// A batch file upload: the batch create's narrowing also holds while those batch deployments
    /// cool down, are paused, or fail their health check, so the file never lands on a deployment
    /// that cannot run its batch.
    pub batch_upload: bool,
    /// An embedding pick: the host keeps the candidates whose `max_embedding_batch_size` admits
    /// the input (`deployments_admitting_embedding_batch`).
    pub embedding_batch: bool,
    /// `f"{provider}/{model}"` for a lookup that cannot ask the host
    /// (`_core_get_routable_deployments`); a pick asks [`Host::provider_request`].
    pub wildcard_request: Option<String>,
    /// The request carries an image (`_request_carries_image`), so deployments that declare no
    /// vision are skipped while any other remains.
    pub has_image: bool,
    /// An async pick: the host's pre-routing step runs before each pick resolves its name.
    pub pre_routes: bool,
    /// Which of the router's outcome callbacks run for this request.
    pub outcome_callbacks: OutcomeCallbacks,
    /// The requested name is no model group (a search tool name): nothing is selected, every
    /// attempt is handed to the host as is, and no outcome is recorded for a deployment
    /// (`_asearch_with_fallbacks`, whose helper picks among the router's search tools).
    pub unrouted: bool,
    /// `passthrough_on_no_deployment`: a pick that fails hands the attempt to the host under the
    /// requested name, inside the level's retries and fallbacks, with no outcome recorded for a
    /// deployment (`_ageneric_api_call_with_fallbacks_helper`).
    pub passthrough_on_no_deployment: bool,
    /// The endpoint's attempt stamps no `failed_deployment_id` on its error (file and batch
    /// creation, whose result belongs to the deployment that served it), so weighted failover
    /// never runs for it.
    pub skip_weighted_failover: bool,
    /// The request creates or names a resource only one provider holds (a file upload, a batch
    /// on an uploaded file): it falls back within its own model group only
    /// (`run_async_fallback`'s `same_model_group_only`).
    pub same_group_fallbacks_only: bool,
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

/// A per-request `num_retries` that is no count of retries. `async_function_with_retries` only
/// reads it once a hop's first failure was not taken over by a retry policy: a call that succeeds
/// first time never does.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UncountedRetries {
    /// A negative int: `num_retries > 0` is false, so the failure stands, and a stream's plain
    /// budget is spent (`isinstance(num_retries, int)`).
    Negative,
    /// Any other value `num_retries > 0` is false for (a negative float, a NaN): the failure
    /// stands, and a stream falls through to the deployment's budget, then the router's.
    NotPositive,
    /// `num_retries > 0` or `range(num_retries)` raises (a string, a positive float): the hop ends
    /// with that error ([`RouterEvent::RetryCountRaised`]).
    Raises,
}

/// One `os.environ/` reference the deployment build found ([`crate::build`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SecretRef {
    /// The value as written, `os.environ/<name>`.
    pub name: String,
    /// Where the value sits in the deployment.
    pub at: SecretAt,
}

/// Where a [`SecretRef`] sits in a deployment.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SecretAt {
    /// A top-level `litellm_params` value: resolved through the host's secret manager.
    Param(String),
    /// `litellm_params.dataSources[source].parameters[key]` (`endpoint` or `key`): read from the
    /// environment alone, empty when unset.
    DataSource { source: usize, key: String },
}

/// Which of the router's adds a build runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::IntoStaticStr, strum::EnumString)]
#[strum(serialize_all = "snake_case")]
pub enum BuildPath {
    /// `set_model_list` (and the constructor's list): config entries, one deployment per
    /// organization, ignored when invalid if the router ignores invalid deployments.
    Create,
    /// `add_deployment` (and upserts): one validated deployment.
    Add,
}

/// What the build asks the host about one deployment.
#[derive(Clone, Debug)]
pub struct BuildAsk<'a> {
    pub path: BuildPath,
    /// The entry's position in the list the build was given.
    pub index: usize,
    /// The entry as the build holds it: `model_name`, `litellm_params` (secrets resolved, the
    /// organization of this deployment set), `model_info` (with its `id`) and any other keys.
    pub entry: &'a Map<String, Value>,
    /// The `model_info.id` the entry declared, before the build assigned one.
    pub declared_id: Option<&'a str>,
    /// Another entry of the same list declares the same id.
    pub duplicate_id: bool,
}

/// The host's answer about one deployment.
#[derive(Clone, Debug, PartialEq)]
pub enum Built {
    Entry(Map<String, Value>),
    /// Left out without an error (`model_info.supported_environments` excludes this one, or an
    /// `add_deployment` id that already exists).
    Skip,
    /// The host failed the deployment and the router ignores invalid deployments.
    Invalid,
}

/// The host's provider answer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Provided {
    Provider(Option<String>),
    Invalid,
}

/// What the model-info merge ([`crate::model_info`]) asks the host while it answers one query.
/// Asked at query time, so a price reload or a registration the host made since the router was
/// built is what the next query reads.
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
#[serde(tag = "lookup", rename_all = "snake_case")]
pub enum ModelInfoLookup {
    /// The process cost map's row for `model` (`get_model_info(model=...)`), the host choosing
    /// the keys it tries. Answered with the row, or `None`. A lookup that raises propagates,
    /// unless `tolerate` is set: then the host answers `{"failed": True}`. `exact` also asks
    /// whether the row came from a fallback-generalization rule (`{"row": ..., "generalized":
    /// bool}`).
    ModelInfo {
        model: Value,
        tolerate: bool,
        exact: bool,
    },
    /// The raw cost-map entry under `key` (`model_cost.get(key)`): answered with
    /// `{"entry": value}`, or `None` when the key is absent. `process` reads the process global
    /// (`litellm.model_cost`) rather than the host's live settings, where the Python router does.
    CostEntry { key: String, process: bool },
    /// `get_llm_provider` for the queried deployment's `model` and validated params: answered
    /// with `[model, provider]`.
    Provider { model: String },
    /// The deployments a wildcard pattern routes `name` to: answered with `[{id, model}]`, or
    /// `None` when no pattern matches.
    PatternModels { name: String },
    /// The base model a bedrock model id names (`get_bedrock_base_model`).
    BedrockBase { model_id: String },
    /// The limits the background model-info refresh discovered for deployment `id`.
    Discovered { id: Value },
    /// The queried deployment's own `model_info`, read when the merge reaches it (a malformed one
    /// raises there).
    UserModelInfo,
    /// A log record, emitted where the Python router logs it.
    Log { level: LogLevel, message: String },
    /// Group info: deployment `index`'s validated params (`LiteLLM_Params(**litellm_params)`, which
    /// raises on bad params): answered with `{"model", "configurable_clientside_auth_params"}`.
    GroupParams { index: usize },
    /// Group info: what the host knows of deployment `index` serving `model`: answered with
    /// `{"model_info"}` (its merged model info, `None` when that fails), `"mapped"`
    /// (`deployment_is_catalog_mapped`), `"litellm_model"` and `"llm_provider"`, and, when it has
    /// no model info, `"supported_openai_params"`.
    GroupDeployment { index: usize, model: Value },
    /// Group info: deployment `index`'s capabilities under `model_info`: answered with
    /// `{"supported_endpoints"}` (empty when the team may not use it), `"supports_fast_mode"` and
    /// `"reasoning_efforts"` (`resolve_supported_reasoning_efforts`, or `None`).
    GroupCapabilities {
        index: usize,
        model_info: Value,
        litellm_model: Value,
        llm_provider: Value,
        mapped: bool,
    },
    /// Group info: the group's first deployment as the host builds it (`ModelGroupInfo(**info)`,
    /// whose validation may raise): answered with the fields the merge compares, as validated.
    GroupSeed { info: Map<String, Value> },
    /// `intersect_supported_reasoning_efforts(current, resolved)`, in the host's advertisement order.
    ReasoningIntersect { current: Value, resolved: Value },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LogLevel {
    Debug,
    Error,
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
    /// The group a pre-routing hook selected for `name`, when it differs (the tier stamp the
    /// fallback lookup keys on first).
    pub routed: Option<String>,
    /// The `order` level an order fallback targets in the same group.
    pub target_order: Option<i64>,
    /// Every deployment `name` resolved to for this caller, the picked one included.
    pub candidate_ids: Vec<String>,
    /// 0 for the first attempt in this group, then the retry number.
    pub retry: u32,
    pub fallback_depth: u32,
    /// How many fallbacks deep the chain is: 0 for the requested name.
    pub fallback_level: u32,
    pub timeout: Option<Duration>,
    /// Request params the fallback chain's dict targets merged in (`kwargs.update(target)`).
    pub params: Option<Arc<Map<String, Value>>>,
    /// The chain level whose request metadata this attempt carries (see [`RouterEvent::Fallback`]).
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
    /// A stream the host reports the outcome of when it ends ([`crate::Router::report_success`],
    /// [`crate::Router::report_failure`]), or a realtime session, whose success the host counts
    /// only when it ended cleanly: the attempt counts its request and `total_tokens` now, and no
    /// success.
    pub deferred: bool,
    /// Whether the attempt counts against the deployment's per-minute usage, when the host knows
    /// better than the deployment's own limits ([`Deployment::tracks_usage`]): a limit only the
    /// model catalog sets (`_deployment_tracks_usage`), or a realtime session, whose `None`
    /// response counts nothing (`increment_deployment_usage_for_response`).
    pub tracks_usage: Option<bool>,
}

/// How a provider blocked a response the call otherwise served.
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, strum::IntoStaticStr, strum::EnumString,
)]
#[serde(rename_all = "snake_case")]
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
    /// `_excluded_deployment_ids` a weighted failover hop put on the request, which the callback
    /// filters read (`deployment_ids_failed_in_request`).
    pub excluded_ids: &'a [String],
    /// `_retry_skipped_deployment_ids` the retry loop put on the request, as `excluded_ids`.
    pub skipped_ids: &'a [String],
    /// `_target_order` as the filters read it: the level an order fallback targets, else the one
    /// the request named.
    pub target_order: Option<i64>,
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
    /// `_deployment_serves_route`: return the candidates whose provider cannot serve the call
    /// `handler` names (a Decisions call on a provider with no Decisions config).
    RouteIncapable { handler: String },
    /// `_decisions_deployment_refuses_media`: return the candidates that refuse the audio or
    /// videos a Decisions call carries.
    MediaRefusing,
    /// `deployments_admitting_embedding_batch`: return the candidates that admit the embedding
    /// input, or what the pick keeps when none does (`deployments_when_none_admits_embedding_batch`).
    EmbeddingBatch,
    /// `_async_override_selector_pre_call_check`: a deployment named by id under a per-request
    /// strategy override still passes that strategy's pre-call check (rpm accounting); return the
    /// candidates, or raise its refusal.
    OverrideCheck(RoutingStrategy),
}

#[derive(Debug)]
pub struct CallbackFiltered {
    pub kept: Vec<Arc<Deployment>>,
    pub preferred: Option<String>,
}

/// Routing events, delivered through [`Host::observe`]. They serialize as JSON objects tagged by
/// `event` (the variant name), durations in seconds.
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
#[serde(tag = "event")]
pub enum RouterEvent {
    /// A pick for `model` (the level's group) begins: its route phase opens.
    PickStarted {
        model: String,
    },
    Selected {
        model_group: String,
        deployment_id: String,
        /// How long the pick took (`_acompletion`'s `async_get_available_deployment` service event).
        #[serde(serialize_with = "seconds")]
        pick: Duration,
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
        #[serde(serialize_with = "seconds")]
        cooldown: Duration,
    },
    Retry {
        model_group: String,
        retry: u32,
        #[serde(serialize_with = "seconds")]
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
        /// The `max_fallbacks` the chain reads, stamped on the hop's request.
        max_fallbacks: u32,
        /// The non-string `model` of the dict target this hop routes
        /// ([`crate::fallback::FallbackTarget::unroutable_model`]): the hop fails on it before any
        /// attempt, and the host raises what Python's level raises for that value.
        unroutable: Option<Value>,
    },
    /// A hop of the fallback chain returned: `log_success_fallback_event` or
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
    /// `async_function_with_retries` opened a level's retry loop, before any pick: `scope`'s
    /// request metadata tracks no retry yet, `max_retries` of them, and the group's size.
    RetriesOpened {
        scope: u32,
        max_retries: u32,
        model_group_size: usize,
    },
    /// The level's first failure settled its overrides: the budget `scope` then tracks.
    RetryBudget {
        scope: u32,
        max_retries: u32,
    },
    /// A pick of the async selection or the sync one failed on `error`: the async one's `except`
    /// logs every refusal to the request's logging object (`_async_get_available_deployment`).
    PickFailed {
        level: u32,
        model: String,
        error: PickError,
    },
    /// A level of the fallback chain serving `model` ended on an error a pick raised (no
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
    /// The first failure of a hop serving `model` read a request `num_retries` that raises
    /// ([`UncountedRetries::Raises`]): the hop ends with that error (the host raises it, from the
    /// failure `cause` names), and falls back as from any plain exception.
    RetryCountRaised {
        level: u32,
        model: String,
        cause: RetryCause,
    },
    /// The level serving `model` reached the fallback handling that appends the context window
    /// and content policy hints to its error, before any hop runs
    /// (`async_function_with_fallbacks_common_utils`). The order path never gets there.
    FallingBack {
        level: u32,
        model: String,
    },
    /// The order fallback chain of the level at `level` served nothing, so the level raises from
    /// `async_function_with_fallbacks_common_utils`' order path, with no outcome message: what the
    /// chain's last hop raised when `last_hop`, else its own error (a context window error, or a
    /// chain no hop of which ran). Weighted failover raising what it does not catch
    /// (`_maybe_run_weighted_failover`) ends the level the same way.
    OrderChainFailed {
        level: u32,
        last_hop: bool,
    },
    /// The regular fallback handling of the level at `level` served nothing, where
    /// `async_function_with_fallbacks_common_utils` reaches its `except` and logs the failure.
    /// `chain` when the level had targets to walk: the error it logs is what the last hop raised
    /// when `last_hop`, else its own. With no `chain`, the host decides: only the host holds
    /// whether a level with no match had a fallback list at all, which Python does not log.
    /// The order path and weighted failover are outside that `try` and never report this.
    FallbacksFailed {
        level: u32,
        chain: bool,
        last_hop: bool,
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

/// A pick's error serializes as its message.
impl serde::Serialize for PickError {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_str(&self.0)
    }
}

impl PartialEq for PickError {
    fn eq(&self, other: &Self) -> bool {
        format!("{:?}", self.0) == format!("{:?}", other.0)
    }
}

/// Which error a [`RouterEvent::RetryLogged`] records.
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
#[serde(tag = "cause", content = "error", rename_all = "snake_case")]
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
    /// `Err` when a check raised: the chain ends there, outside any target's `try`.
    async fn allow_fallback(
        &self,
        request: &RouterRequest,
        target: &str,
    ) -> Result<bool, AttemptError>;

    /// Whether `run_async_fallback` refuses a chain for one of its dict `targets`, a server-owned
    /// federation field or an `oidc/` api key. The chain is walked by `level`.
    async fn refuses_fallbacks(&self, _level: u32, _targets: &[Map<String, Value>]) -> bool {
        false
    }

    async fn sleep(&self, duration: Duration);

    fn observe(&self, event: &RouterEvent);

    /// The shared store routing state lives in.
    fn cache(&self) -> Arc<dyn RouterCache>;

    /// Whether [`Self::invoke`] can run `op` on `provider`. Checked for every served op and
    /// deployment when the router is built or changed, so an unsupported pair fails the config,
    /// never a request. `PythonHost` runs every op.
    fn supports(&self, _op: &Op, _provider: &str) -> bool {
        true
    }

    /// Whether [`Self::select`] can run strategies and plugins.
    fn can_select(&self) -> bool {
        true
    }

    /// Run the routing plugins for `ctx.name` when `run` (the async pick's pre-routing hook), and
    /// report the candidate models they left on the request, if any.
    async fn routing_plugin_models(
        &self,
        _ctx: &RoutingContext<'_>,
        _run: bool,
    ) -> Result<Option<Vec<String>>, Error> {
        Ok(None)
    }

    /// The facts the `enable_pre_call_checks` filter decides by (token count, context windows,
    /// param support), for `candidates` and the deployments in `ctx.pool`. A host without them
    /// reports no input, so nothing is checked.
    async fn pre_call_facts(
        &self,
        _ctx: &RoutingContext<'_>,
        _candidates: &[Arc<Deployment>],
    ) -> Result<PreCallFacts, Error> {
        Ok(PreCallFacts::default())
    }

    /// `f"{provider}/{model}"` for `ctx.name`, which `PatternMatchRouter.get_pattern` tries when
    /// no pattern matches the name itself: only the host can infer a provider. `None` from a
    /// host that infers none.
    async fn provider_request(&self, _ctx: &RoutingContext<'_>) -> Result<Option<String>, Error> {
        Ok(None)
    }

    /// `async_callback_filter_deployments`: the host's routing callbacks narrow `candidates` in
    /// turn, and may name one deployment they prefer, which the pick takes only when the filters
    /// after the callbacks keep it. A filter that raises fails the try with [`Error::Attempt`],
    /// which the call retries and falls back from like any attempt failure. A host without
    /// filters keeps every candidate and prefers none.
    async fn callback_filters(
        &self,
        _ctx: &RoutingContext<'_>,
        candidates: Vec<Arc<Deployment>>,
    ) -> Result<CallbackFiltered, Error> {
        Ok(CallbackFiltered {
            kept: candidates,
            preferred: None,
        })
    }
    /// The values of `refs`, one per ref, in order: what the deployment build substitutes for
    /// each `os.environ/` reference it found in entry `index` ([`crate::build`]). Asked once per
    /// deployment, as the build reaches it. A host without secrets leaves every reference as
    /// written.
    async fn secrets(&self, index: usize, refs: &[SecretRef]) -> Result<Vec<Value>, Error> {
        let _ = index;
        Ok(refs.iter().map(|r| Value::String(r.name.clone())).collect())
    }

    /// Check one deployment before it is added, and answer the entry as the host normalized it,
    /// a skip (the deployment is left out), or [`Built::Invalid`] (it failed and the router
    /// ignores invalid deployments). A host without checks accepts the entry as given.
    async fn validate(&self, ask: &BuildAsk<'_>) -> Result<Built, Error> {
        Ok(Built::Entry(ask.entry.clone()))
    }

    /// The provider that serves one validated deployment (`get_llm_provider`), `None` for one
    /// served without a provider (a prompt-management model), or [`Provided::Invalid`]. A host
    /// without provider resolution answers `None`, so [`Deployment::provider`] reads the model.
    async fn provider(&self, ask: &BuildAsk<'_>) -> Result<Provided, Error> {
        let _ = ask;
        Ok(Provided::Provider(None))
    }

    /// Run the host's side effects for an accepted deployment and answer the entry as the
    /// registry stores it. A host without side effects stores the entry it validated.
    async fn added(&self, ask: &BuildAsk<'_>, provider: Option<&str>) -> Result<Built, Error> {
        let _ = provider;
        Ok(Built::Entry(ask.entry.clone()))
    }

    /// Answer one model-info lookup ([`ModelInfoLookup`]). A host without a cost map answers
    /// `None` to everything, so no deployment has model info.
    async fn model_info(&self, lookup: &ModelInfoLookup) -> Result<Value, Error> {
        let _ = lookup;
        Ok(Value::Null)
    }
}

fn seconds<S: serde::Serializer>(duration: &Duration, serializer: S) -> Result<S::Ok, S::Error> {
    serializer.serialize_f64(duration.as_secs_f64())
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::{Value, json};

    use super::*;

    #[rstest]
    #[case::selected(
        RouterEvent::Selected {
            model_group: "gpt-4o".to_owned(),
            deployment_id: "azure-eastus".to_owned(),
            pick: Duration::from_millis(1500),
        },
        json!({"event": "Selected", "model_group": "gpt-4o", "deployment_id": "azure-eastus", "pick": 1.5})
    )]
    #[case::attempt_failed(
        RouterEvent::AttemptFailed {
            deployment_id: "openai".to_owned(),
            kind: FailureKind::RateLimit,
            status: Some(429),
        },
        json!({"event": "AttemptFailed", "deployment_id": "openai", "kind": "RateLimitError", "status": 429})
    )]
    #[case::cooldown_set(
        RouterEvent::CooldownSet { deployment_id: "openai".to_owned(), cooldown: Duration::from_secs(30) },
        json!({"event": "CooldownSet", "deployment_id": "openai", "cooldown": 30.0})
    )]
    #[case::retry(
        RouterEvent::Retry { model_group: "gpt-4o".to_owned(), retry: 1, delay: Duration::ZERO },
        json!({"event": "Retry", "model_group": "gpt-4o", "retry": 1, "delay": 0.0})
    )]
    #[case::blocked(
        RouterEvent::Blocked { level: 0, model: "claude".to_owned(), blocked: Blocked::ContentFilter },
        json!({"event": "Blocked", "level": 0, "model": "claude", "blocked": "content_filter"})
    )]
    #[case::retry_logged_after_an_attempt(
        RouterEvent::RetryLogged {
            scope: 0,
            level: 0,
            model: "gpt-4o".to_owned(),
            attempted_retries: None,
            cause: RetryCause::Attempt,
        },
        json!({
            "event": "RetryLogged", "scope": 0, "level": 0, "model": "gpt-4o",
            "attempted_retries": null, "cause": {"cause": "attempt"},
        })
    )]
    #[case::retry_logged_after_a_pick(
        RouterEvent::RetryLogged {
            scope: 1,
            level: 1,
            model: "backup".to_owned(),
            attempted_retries: Some(2),
            cause: RetryCause::Pick(PickError(Arc::new(Error::UnknownModel { model: "backup".to_owned() }))),
        },
        json!({
            "event": "RetryLogged", "scope": 1, "level": 1, "model": "backup", "attempted_retries": 2,
            "cause": {"cause": "pick", "error": "no deployment serves model `backup`"},
        })
    )]
    #[case::mock_failed(
        RouterEvent::MockFailed {
            mock: MockFailure::RateLimit,
            level: 0,
            model: "gpt-4o".to_owned(),
            num_retries: None,
        },
        json!({"event": "MockFailed", "mock": "rate_limit", "level": 0, "model": "gpt-4o", "num_retries": null})
    )]
    fn an_event_serializes_tagged_by_its_name(#[case] event: RouterEvent, #[case] expected: Value) {
        assert_eq!(serde_json::to_value(&event).unwrap(), expected);
    }
}
