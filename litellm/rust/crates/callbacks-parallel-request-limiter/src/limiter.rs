use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

use futures_util::future::BoxFuture;
use litellm_callbacks::{
    Abort, Activation, CallFacts, Callback, CallbackError, Field, FieldSet, Hook, HookInput,
    HookOutput, Host, KeyIdentity, KeyLimits, Manifest, Patch, Scalar, Selection, Subscription,
    Verdict, typed,
};
use litellm_redis_store::{
    RequestRedis,
    rate_limits::{Entry, Increment, Scope},
};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::{Map, Value};

use crate::{
    arithmetic::{OutputCapFacts, output_cap},
    batch::{self, BatchFacts},
    builder::{AuthMetadata, BuildError, Lookups, Request, RequestFacts},
    descriptor::{Descriptor, PROJECT_ITPM, PROJECT_OTPM, PTU_TEAM},
    engine::{Ctx, Engine, Now, Op, Policy, Settlement, SlotAcquisition, Unverifiable},
    enqueued::{EnqueuedOutcome, EnqueuedTokens, process_monotonic},
    hosted::Reservation,
    ptu::{Ceiling, Usage, settlement_tokens},
    refusal,
    settings::{TokenRateLimitType, fail_closed_setting, force_grouping_setting},
    state::{CapDecision, IoReservation, LimiterState, RateLimitedModel, Stash},
    status::{Code, Response, Status, WindowIdentity, header_map},
};

pub const NAME: &str = "parallel_request_limiter_v3";

const PER_MODEL_KEYS: [&str; 7] = [
    "model_per_key",
    "model_per_team",
    "model_per_end_user",
    "model_per_organization",
    "model_per_project",
    "model_per_project_itpm",
    "model_per_project_otpm",
];

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PolicyFacts {
    #[serde(default, deserialize_with = "fail_closed_setting")]
    pub fail_closed: bool,
    #[serde(default, deserialize_with = "force_grouping_setting")]
    pub force_grouping: bool,
}

impl From<PolicyFacts> for Policy {
    fn from(facts: PolicyFacts) -> Self {
        Self {
            fail_closed: facts.fail_closed,
            force_grouping: facts.force_grouping,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PreCallFacts {
    #[serde(default)]
    pub call_id: Option<String>,
    #[serde(default)]
    pub identity: Option<String>,
    #[serde(default)]
    pub descriptors: Vec<Descriptor>,
    #[serde(default)]
    pub requested_model: Option<String>,
    #[serde(default)]
    pub estimated_tokens: Option<i64>,
    #[serde(default)]
    pub reserved_model: Option<RateLimitedModel>,
    #[serde(default)]
    pub policy: PolicyFacts,
    #[serde(default)]
    pub ptu_ceiling: Option<Ceiling>,
    #[serde(default)]
    pub ptu_estimate: Option<Estimate>,
    #[serde(default)]
    pub io_estimate: Option<Estimate>,
    #[serde(skip)]
    pub ptu_exact: Option<(f64, f64)>,
    #[serde(default)]
    pub reuses_request_slot: bool,
    #[serde(default)]
    pub prior_response: Option<PriorResponse>,
    #[serde(default)]
    pub batch: Option<BatchFacts>,
    #[serde(default)]
    pub phase: Phase,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    #[default]
    Whole,
    FirstPass,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PriorResponse {
    pub overall_code: Code,
    pub statuses: Vec<Status>,
}

impl From<PriorResponse> for Response {
    fn from(prior: PriorResponse) -> Self {
        Self {
            overall: prior.overall_code,
            statuses: prior.statuses,
            windows: None,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Estimate {
    #[serde(deserialize_with = "crate::python::saturating_int")]
    pub input_tokens: i64,
    #[serde(deserialize_with = "crate::python::saturating_int")]
    pub output_tokens: i64,
    #[serde(default)]
    pub explicit_output_cap: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct IoUsage {
    #[serde(default)]
    pub response: Option<Usage>,
    #[serde(default)]
    pub combined: Option<Usage>,
    #[serde(default)]
    pub aggregate_total: i64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CallIdFacts {
    #[serde(default)]
    pub call_id: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct BatchResponseFacts {
    pub id: String,
    pub terminal: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PostCallSuccessFacts {
    #[serde(default)]
    pub call_id: Option<String>,
    #[serde(default)]
    pub batch: Option<BatchResponseFacts>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PostCallFailureFacts {
    #[serde(default)]
    pub tpm_actual: i64,
    #[serde(default)]
    pub partial_usage: Option<Usage>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TargetFacts {
    pub user_api_key_hash: Option<String>,
    pub user_api_key_user_id: Option<String>,
    pub user_api_key_team_id: Option<String>,
    pub user_api_key_org_id: Option<String>,
    pub user_api_key_project_id: Option<String>,
    pub end_user_id: Option<String>,
    pub agent_id: Option<String>,
    pub session_id: Option<String>,
    #[serde(default)]
    pub customer_org_id: Option<String>,
    #[serde(default)]
    pub customer_team_id: Option<String>,
    #[serde(default)]
    pub customer_member_id: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Candidate {
    pub group: String,
    pub key_owns: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PtuFacts {
    pub value: String,
    pub actual_tokens: i64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelFacts {
    pub candidates: Vec<Candidate>,
    #[serde(default)]
    pub ptu: Option<PtuFacts>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SuccessFacts {
    #[serde(default)]
    pub call_id: Option<String>,
    #[serde(default)]
    pub exempt: bool,
    #[serde(default)]
    pub total_tokens: i64,
    #[serde(default)]
    pub targets: TargetFacts,
    #[serde(default)]
    pub served_model: Option<RateLimitedModel>,
    #[serde(default)]
    pub models: BTreeMap<String, ModelFacts>,
    #[serde(default)]
    pub team_label: String,
    #[serde(default)]
    pub policy: PolicyFacts,
    #[serde(default)]
    pub usage: Option<Usage>,
    #[serde(default)]
    pub io_usage: IoUsage,
    #[serde(default)]
    pub keep_slot: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct FailureFacts {
    #[serde(default)]
    pub call_id: Option<String>,
    #[serde(default)]
    pub tpm_actual: i64,
    #[serde(default)]
    pub partial_usage: Option<Usage>,
}

#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
#[serde(default)]
pub struct PreCallInputs {
    pub phase: Phase,
    pub call_id: Option<String>,
    pub nested: bool,
    pub identity: Option<String>,
    pub policy: PolicyFacts,
    pub metadata: AuthMetadata,
    pub lookups: Lookups,
    pub batch_call: bool,
    pub batch: Option<BatchFacts>,
    pub held_ptu_ceiling: Option<Ceiling>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(default)]
pub struct PostCallSuccessInputs {
    pub call_id: Option<String>,
    pub batch: Option<BatchResponseFacts>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(default)]
pub struct LogPrepareInputs {
    pub call_id: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
#[serde(default)]
pub struct PostCallFailureInputs {
    pub combined_is_usage: bool,
    pub token_rate_limit_type: Value,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct Plan {
    pub reserves: bool,
    pub min_tpm_limit: Option<i64>,
    pub min_otpm_limit: Option<i64>,
    pub has_io: bool,
    pub ptu: bool,
}

const EMBEDDING_CALL_TYPES: [&str; 2] = ["aembedding", "embedding"];
const RESPONSES_CALL_TYPES: [&str; 2] = ["aresponses", "responses"];
const EXTRACTION_CALL_TYPES: [&str; 2] = ["extraction", "aextraction"];
const MCP_TOOL_CALL_TYPE: &str = "call_mcp_tool";
const DECISIONS_ENDPOINT: &str = "decisions";

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CapSettings {
    pub fraction: f64,
    pub default_max_tokens_estimate: i64,
    pub responses_min_output_tokens: i64,
}

impl Default for CapSettings {
    fn default() -> Self {
        Self {
            fraction: 0.25,
            default_max_tokens_estimate: 4096,
            responses_min_output_tokens: 16,
        }
    }
}

fn non_empty(text: &str) -> Option<String> {
    (!text.is_empty()).then(|| text.to_owned())
}

fn failed(hook: Hook, what: &str, error: impl std::fmt::Display) -> CallbackError {
    CallbackError::Failed(format!("the {hook:?} {what} do not decode: {error}"))
}

fn record_members<T: Serialize>(record: Option<T>) -> Result<Map<String, Value>, CallbackError> {
    match record.map(serde_json::to_value).transpose() {
        Ok(Some(Value::Object(members))) => Ok(members
            .into_iter()
            .map(|(name, member)| (name, crate::python::scalar_member(member)))
            .collect()),
        Ok(_) => Ok(Map::new()),
        Err(error) => Err(failed(Hook::PreCall, "key records", error)),
    }
}

fn scalar_value(scalar: Option<&Scalar>) -> Result<Value, CallbackError> {
    scalar
        .map(serde_json::to_value)
        .transpose()
        .map(|value| value.map_or(Value::Null, crate::python::scalar_member))
        .map_err(|error| failed(Hook::PreCall, "key records", error))
}

fn agent_policy(
    policy: Option<&litellm_callbacks::AgentPolicy>,
) -> Result<Option<crate::builder::AgentPolicy>, CallbackError> {
    let Some(policy) = policy else {
        return Ok(None);
    };
    Ok(Some(crate::builder::AgentPolicy {
        rpm_limit: scalar_value(policy.rpm_limit.as_ref())?,
        tpm_limit: scalar_value(policy.tpm_limit.as_ref())?,
        session_rpm_limit: scalar_value(policy.session_rpm_limit.as_ref())?,
        session_tpm_limit: scalar_value(policy.session_tpm_limit.as_ref())?,
    }))
}

fn raised(error_type: Option<String>, error_text: Option<String>) -> Option<(String, String)> {
    error_type.map(|error_type| (error_type, error_text.unwrap_or_default()))
}

fn request_facts(
    request: Option<litellm_callbacks::LimiterRequest>,
) -> Result<RequestFacts, CallbackError> {
    let Some(request) = request else {
        return Ok(RequestFacts::default());
    };
    Ok(RequestFacts {
        tags: request
            .tags
            .unwrap_or_default()
            .into_iter()
            .map(Value::String)
            .collect(),
        resolved_agent_id: scalar_value(request.resolved_agent_id.as_ref())?,
        resolved_agent_error: raised(
            request.resolved_agent_id_error_type,
            request.resolved_agent_id_error_text,
        ),
        session_id: request.session_id,
        session_error: raised(request.session_id_error_type, request.session_id_error_text),
    })
}

fn auth_members(
    identity: Option<KeyIdentity>,
    limits: Option<KeyLimits>,
) -> Result<Map<String, Value>, CallbackError> {
    let mut members = record_members(identity)?;
    members.extend(record_members(limits)?);
    Ok(members)
}

fn prior_response(input: &HookInput) -> Result<Option<PriorResponse>, CallbackError> {
    let Some(Value::Object(mut statuses)) = input.fields.raw(Field::RateLimitStatuses)?.cloned()
    else {
        return Ok(None);
    };
    if let Some(Value::Array(rows)) = statuses.get_mut("statuses") {
        for row in rows.iter_mut().filter_map(Value::as_object_mut) {
            row.retain(|_, member| !member.is_null());
        }
    }
    serde_json::from_value(Value::Object(statuses))
        .map(Some)
        .map_err(|error| failed(input.hook, "rate limit statuses", error))
}

fn cap_decision(input: &HookInput, call: &CallFacts) -> Result<CapDecision, CallbackError> {
    let call_type = call.kind.call_type.as_str();
    let explicit = input
        .fields
        .get::<typed::OutputCap>()?
        .and_then(|cap| cap.explicit)
        .is_some();
    let embedding = if EMBEDDING_CALL_TYPES.contains(&call_type) {
        true
    } else if call_type.is_empty() {
        input
            .fields
            .raw(Field::Input)?
            .is_some_and(|value| !value.is_null())
    } else {
        false
    };
    Ok(CapDecision {
        explicit,
        exempt: explicit
            || embedding
            || call.kind.endpoint_type.as_deref() == Some(DECISIONS_ENDPOINT)
            || EXTRACTION_CALL_TYPES.contains(&call_type),
        responses: RESPONSES_CALL_TYPES.contains(&call_type),
    })
}

fn scopes_keyed(descriptors: &[Descriptor], key: &str) -> BTreeSet<(String, String)> {
    descriptors
        .iter()
        .filter(|descriptor| descriptor.key == key)
        .map(|descriptor| (descriptor.key.clone(), descriptor.value.clone()))
        .collect()
}

fn windows_keyed(windows: &[WindowIdentity], key: &str) -> Vec<WindowIdentity> {
    windows
        .iter()
        .filter(|identity| identity.counter.scope.descriptor == key)
        .cloned()
        .collect()
}

pub struct ParallelRequestLimiter {
    manifest: Manifest,
    engine: Engine,
    enqueued: EnqueuedTokens,
    cap: CapSettings,
}

fn truthy(value: &Option<String>) -> Option<&str> {
    value.as_deref().filter(|value| !value.is_empty())
}

pub fn manifest() -> Manifest {
    Manifest {
        selection: Selection {
            activation: Activation::Required,
            ..Selection::named(NAME)
        },
        subscriptions: vec![
            Subscription {
                reads: FieldSet::of(&[
                    Field::DeprecatedKwargs,
                    Field::KeyIdentity,
                    Field::KeyLimits,
                    Field::RateLimitStatuses,
                    Field::OutputCap,
                    Field::Input,
                    Field::LimiterRequest,
                ]),
                writes: FieldSet::of(&[Field::OptionalParams]),
                may_abort: true,
                ..Subscription::on(Hook::PreCall)
            },
            Subscription {
                reads: FieldSet::of(&[Field::DeprecatedKwargs, Field::DeprecatedResponseObj]),
                awaited: true,
                ..Subscription::on(Hook::PostCallSuccess)
            },
            Subscription {
                reads: FieldSet::of(&[Field::DeprecatedKwargs, Field::Usage]),
                awaited: true,
                ..Subscription::on(Hook::PostCallFailure)
            },
            Subscription {
                reads: FieldSet::of(&[Field::DeprecatedKwargs, Field::StandardLoggingPayload]),
                writes: FieldSet::of(&[Field::StandardLoggingPayload]),
                ..Subscription::on(Hook::LogPrepare)
            },
        ],
        has_request_state: true,
    }
}

impl ParallelRequestLimiter {
    pub fn new(engine: Engine) -> Self {
        Self::with_monotonic(engine, process_monotonic())
    }

    pub fn with_monotonic(engine: Engine, monotonic: Now) -> Self {
        let enqueued = EnqueuedTokens::new(Arc::clone(engine.store()), monotonic);
        Self {
            manifest: manifest(),
            engine,
            enqueued,
            cap: CapSettings::default(),
        }
    }

    pub fn with_cap_settings(self, cap: CapSettings) -> Self {
        Self { cap, ..self }
    }

    pub fn engine(&self) -> &Engine {
        &self.engine
    }

    pub fn enqueued(&self) -> &EnqueuedTokens {
        &self.enqueued
    }

    async fn release_stashed(&self, state: &LimiterState, in_logging_callback: bool, ctx: Ctx<'_>) {
        let _guard = state.release.lock().await;
        let Some(acquisition) = state.stash().parallel_slot.clone() else {
            return;
        };
        let deferred = in_logging_callback && self.engine.defer_release(&acquisition, ctx).await;
        if !deferred {
            self.engine.release_slots(&acquisition).await;
        }
        state.stash().parallel_slot = None;
    }

    pub async fn pre_call(
        &self,
        state: &LimiterState,
        facts: PreCallFacts,
        request: Option<&RequestRedis>,
    ) -> Option<Abort> {
        let ctx = Ctx {
            policy: facts.policy.into(),
            request,
        };
        match self.admit(state, &facts, ctx).await {
            Ok(abort) => abort,
            Err(Unverifiable) => Some(refusal::unverifiable()),
        }
    }

    async fn admit(
        &self,
        state: &LimiterState,
        facts: &PreCallFacts,
        ctx: Ctx<'_>,
    ) -> Result<Option<Abort>, Unverifiable> {
        if let Some(abort) = self.first_pass(state, facts, ctx).await? {
            return Ok(Some(abort));
        }
        if facts.phase == Phase::FirstPass || !state.stash().first_pass_admitted {
            return Ok(None);
        }
        if let Some(abort) = self.reserve_tokens(state, facts, ctx).await? {
            return Ok(Some(abort));
        }
        self.reserve_io(state, facts, self.reservation_amount(facts), ctx)
            .await
    }

    fn reserves_tokens(&self, facts: &PreCallFacts) -> bool {
        self.engine.settings().tpm_reservation_enabled
            && facts.descriptors.iter().any(|descriptor| {
                !descriptor.is_project_io() && descriptor.limits().tokens_per_unit.is_some()
            })
    }

    pub(crate) fn reservation_amount(&self, facts: &PreCallFacts) -> i64 {
        if self.reserves_tokens(facts) {
            facts.estimated_tokens.unwrap_or(0).max(1)
        } else {
            0
        }
    }

    pub(crate) async fn first_pass(
        &self,
        state: &LimiterState,
        facts: &PreCallFacts,
        ctx: Ctx<'_>,
    ) -> Result<Option<Abort>, Unverifiable> {
        let descriptors = &facts.descriptors;
        let settings = self.engine.settings();
        {
            let mut stash = state.stash();
            stash.owned = true;
            stash.first_pass_admitted = false;
            stash.ptu_ceiling = facts.ptu_ceiling.clone();
            stash.rate_limit_response = facts.prior_response.clone().map(Response::from);
            if let Some(call_id) = &facts.call_id {
                stash.owner_call_id = Some(call_id.clone());
            }
            stash.identity = facts.identity.clone().or_else(|| facts.call_id.clone());
            stash.tpm_limited_tags = descriptors
                .iter()
                .filter(|descriptor| {
                    descriptor.key == "tag"
                        && descriptor
                            .rate_limit
                            .as_ref()
                            .is_some_and(|limits| limits.tokens_per_unit.is_some())
                })
                .map(|descriptor| descriptor.value.clone())
                .collect();
        }
        if let Some(batch) = &facts.batch {
            return self.admit_batch(state, batch, facts, ctx).await;
        }
        if descriptors.is_empty() {
            return Ok(None);
        }
        let reuse = facts.reuses_request_slot;
        let gauges: Vec<_> = if reuse {
            Vec::new()
        } else {
            descriptors
                .iter()
                .filter(|descriptor| descriptor.limits().max_parallel_requests.is_some())
                .map(Descriptor::scope)
                .collect()
        };
        let slot_id = (!gauges.is_empty()).then(|| uuid::Uuid::new_v4().simple().to_string());
        let first_pass: Vec<Descriptor> = if settings.tpm_reservation_enabled {
            descriptors.clone()
        } else {
            descriptors
                .iter()
                .filter(|descriptor| !descriptor.is_project_io())
                .cloned()
                .collect()
        };
        let response = self
            .engine
            .should_rate_limit(
                &first_pass,
                false,
                settings.tpm_reservation_enabled,
                slot_id.as_deref(),
                reuse,
                ctx,
            )
            .await?;
        if response.is_over() {
            return Ok(self.refuse(&response, facts));
        }
        {
            let mut stash = state.stash();
            stash.first_pass_admitted = true;
            if !reuse {
                stash.rate_limit_response = Some(response);
            }
            if let Some(slot_id) = slot_id {
                stash.parallel_slot = Some(SlotAcquisition { slot_id, gauges });
            }
        }
        Ok(None)
    }

    pub(crate) async fn reserve_tokens(
        &self,
        state: &LimiterState,
        facts: &PreCallFacts,
        ctx: Ctx<'_>,
    ) -> Result<Option<Abort>, Unverifiable> {
        let descriptors = &facts.descriptors;
        if self.reserves_tokens(facts) {
            let estimated = self.reservation_amount(facts);
            let needed = match (&facts.ptu_ceiling, facts.ptu_exact) {
                (Some(ceiling), Some((input, output))) => Some(ceiling.needed(input, output)),
                _ => None,
            };
            let ptu_estimated = match (&facts.ptu_ceiling, facts.ptu_estimate, needed) {
                (Some(_), _, Some(needed)) => crate::ptu::saturated(needed),
                (Some(ceiling), Some(estimate), None) => {
                    ceiling.estimate(estimate.input_tokens, estimate.output_tokens)
                }
                _ => estimated,
            };
            if let Some(ceiling) = &facts.ptu_ceiling
                && (ptu_estimated > ceiling.tpm_limit || needed.is_some_and(crate::ptu::beyond_i64))
            {
                self.release_unless_reused(state, facts, ctx).await;
                let needed_text = needed.map_or_else(
                    || ptu_estimated.to_string(),
                    |needed| format!("{needed:.0}"),
                );
                return Ok(Some(refusal::ptu_share_exceeded(ceiling, &needed_text)));
            }
            let tpm_response = self
                .engine
                .reserve_tpm_tokens(descriptors, estimated, &[(PTU_TEAM, ptu_estimated)], ctx)
                .await?;
            if tpm_response.is_over() {
                self.release_unless_reused(state, facts, ctx).await;
                return Ok(self.refuse(&tpm_response, facts));
            }
            let mut stash = state.stash();
            stash.discard_released_tpm_reservation();
            stash.reserved_tokens = estimated;
            stash.ptu_reserved_tokens = ptu_estimated;
            stash.ptu_windows = windows_keyed(tpm_response.windows(), PTU_TEAM);
            stash.reserved_model = facts.reserved_model.clone();
            stash.reserved_scopes = descriptors
                .iter()
                .filter(|descriptor| {
                    !descriptor.is_project_io() && descriptor.limits().tokens_per_unit.is_some()
                })
                .map(|descriptor| (descriptor.key.clone(), descriptor.value.clone()))
                .collect();
            if let Some(stored) = stash.rate_limit_response.as_mut() {
                stored.statuses.extend(tpm_response.statuses);
            }
        }
        Ok(None)
    }

    async fn admit_batch(
        &self,
        state: &LimiterState,
        batch: &BatchFacts,
        facts: &PreCallFacts,
        ctx: Ctx<'_>,
    ) -> Result<Option<Abort>, Unverifiable> {
        state.stash().batch_tpd_refunds.clear();
        if !batch.enqueued_scopes.is_empty() {
            return Ok(
                match self
                    .enqueued
                    .reserve(batch.total_tokens, batch.enqueued_scopes.clone())
                    .await
                {
                    EnqueuedOutcome::OverLimit { scope, enqueued } => {
                        Some(refusal::batch_enqueued_limited(
                            &scope,
                            enqueued,
                            batch.total_tokens,
                            facts.requested_model.as_deref(),
                        ))
                    }
                    EnqueuedOutcome::Reservation(reservation) => {
                        state.stash().batch_enqueued = Some(reservation);
                        None
                    }
                },
            );
        }
        let response = self
            .engine
            .atomic_check_and_increment(&batch.descriptors, &batch.increments, ctx)
            .await?;
        if response.is_over()
            && let Some(over) = response
                .statuses
                .iter()
                .find(|status| status.code == Code::OverLimit)
        {
            let window_start = if over.descriptor_key.ends_with(batch::TPD_SUFFIX) {
                let scope = Scope {
                    descriptor: over.descriptor_key.clone(),
                    value: over.descriptor_value.clone().unwrap_or_default(),
                };
                match self.engine.read(Entry::Window(scope), ctx).await {
                    Ok(stored) => batch::window_start(stored),
                    Err(_) => return Ok(None),
                }
            } else {
                None
            };
            let refusal = batch.refusal(
                over,
                self.engine.now(),
                self.engine.settings().window_size,
                window_start,
            );
            return Ok(Some(refusal::batch_rate_limited(
                refusal,
                facts.requested_model.as_deref(),
            )));
        }
        state.stash().batch_tpd_refunds = batch.tpd_refunds(response.windows());
        Ok(None)
    }

    pub(crate) async fn reserve_io(
        &self,
        state: &LimiterState,
        facts: &PreCallFacts,
        reservation_amount: i64,
        ctx: Ctx<'_>,
    ) -> Result<Option<Abort>, Unverifiable> {
        let descriptors: Vec<Descriptor> = facts
            .descriptors
            .iter()
            .filter(|descriptor| descriptor.is_project_io())
            .cloned()
            .collect();
        if descriptors.is_empty() {
            return Ok(None);
        }
        let estimate = facts.io_estimate.unwrap_or_default();
        let input_tokens = estimate.input_tokens.max(1);
        let output_tokens = if estimate.explicit_output_cap {
            estimate.output_tokens
        } else {
            estimate.output_tokens.max(1)
        };
        let (response, itpm_reserved, otpm_reserved) = self
            .engine
            .reserve_io_tokens(&descriptors, input_tokens, output_tokens, ctx)
            .await?;
        if response.is_over() {
            if reservation_amount > 0 {
                let (scopes, ptu_scopes, ptu_tokens, ptu_windows) = {
                    let stash = state.stash();
                    let (ptu, other): (Vec<_>, Vec<_>) = stash
                        .reserved_scopes
                        .iter()
                        .cloned()
                        .partition(|scope| scope.0 == PTU_TEAM);
                    (
                        other,
                        ptu,
                        stash.ptu_reserved_tokens,
                        stash.ptu_windows.clone(),
                    )
                };
                self.engine
                    .refund_reserved(&scopes, reservation_amount, &[], ctx)
                    .await;
                self.engine
                    .refund_reserved(&ptu_scopes, ptu_tokens, &ptu_windows, ctx)
                    .await;
                state.stash().reservation_released = true;
            }
            self.release_unless_reused(state, facts, ctx).await;
            return Ok(self.refuse(&response, facts));
        }
        let mut stash = state.stash();
        stash.discard_released_io_reservation();
        if itpm_reserved > 0 {
            stash.itpm = IoReservation {
                tokens: itpm_reserved,
                scopes: scopes_keyed(&descriptors, PROJECT_ITPM),
                windows: windows_keyed(response.windows(), PROJECT_ITPM),
            };
        }
        if otpm_reserved > 0 {
            stash.otpm = IoReservation {
                tokens: otpm_reserved,
                scopes: scopes_keyed(&descriptors, PROJECT_OTPM),
                windows: windows_keyed(response.windows(), PROJECT_OTPM),
            };
        }
        match stash.rate_limit_response.as_mut() {
            Some(stored) => stored.statuses.extend(response.statuses),
            None if !response.statuses.is_empty() => stash.rate_limit_response = Some(response),
            None => {}
        }
        Ok(None)
    }

    async fn release_unless_reused(
        &self,
        state: &LimiterState,
        facts: &PreCallFacts,
        ctx: Ctx<'_>,
    ) {
        if !facts.reuses_request_slot {
            self.release_stashed(state, false, ctx).await;
        }
    }

    fn refuse(&self, response: &crate::status::Response, facts: &PreCallFacts) -> Option<Abort> {
        refusal::rate_limited(
            response,
            &facts.descriptors,
            facts.requested_model.as_deref(),
            self.engine.now(),
            self.engine.settings().window_size,
        )
    }

    pub async fn post_call_success(&self, state: &LimiterState, facts: PostCallSuccessFacts) {
        let applies = {
            let stash = state.stash();
            stash.owned && stash.for_call(facts.call_id.as_deref())
        };
        if applies {
            self.release_stashed(
                state,
                false,
                Ctx {
                    policy: Policy::default(),
                    request: None,
                },
            )
            .await;
        }
        if let Some(batch) = facts.batch {
            let reservation = state.stash().batch_enqueued.take();
            if let Some(reservation) = reservation {
                self.enqueued.save(&batch.id, &reservation).await;
            }
            if batch.terminal
                && let Some(popped) = self.enqueued.pop(&batch.id).await
            {
                self.enqueued.refund(&popped).await;
            }
        }
    }

    pub async fn release_on_disconnect(&self, state: &LimiterState) {
        if state.stash().owned {
            self.release_stashed(
                state,
                false,
                Ctx {
                    policy: Policy::default(),
                    request: None,
                },
            )
            .await;
        }
    }

    pub async fn post_call_failure(&self, state: &LimiterState, facts: PostCallFailureFacts) {
        if !state.stash().owned {
            return;
        }
        let ctx = Ctx {
            policy: Policy::default(),
            request: None,
        };
        self.release_stashed(state, false, ctx).await;
        let enqueued = state.stash().batch_enqueued.take();
        if let Some(reservation) = enqueued {
            self.enqueued.refund(&reservation).await;
        }
        let refunds = std::mem::take(&mut state.stash().batch_tpd_refunds);
        if !refunds.is_empty() {
            self.engine.reservation_aware(&refunds).await;
        }
        let settlements = {
            let mut stash = state.stash();
            let (reserved, itpm, otpm) = Self::open_reservations(&stash);
            if reserved <= 0 && itpm <= 0 && otpm <= 0 {
                return;
            }
            let settlements = self.failure_settlements(
                &stash,
                facts.tpm_actual,
                facts.partial_usage.as_ref(),
                (reserved, itpm, otpm),
            );
            stash.mark_reservation_released();
            settlements
        };
        for settlement in settlements {
            self.engine.settle(settlement, ctx, false).await;
        }
    }

    fn open_reservations(stash: &Stash) -> (i64, i64, i64) {
        let reserved = if stash.reservation_released {
            0
        } else {
            stash.reserved_tokens
        };
        let (itpm, otpm) = if stash.io_reservation_released {
            (0, 0)
        } else {
            (stash.itpm.tokens, stash.otpm.tokens)
        };
        (reserved, itpm, otpm)
    }

    fn failure_settlements(
        &self,
        stash: &Stash,
        tpm_actual: i64,
        partial_usage: Option<&Usage>,
        (reserved, itpm, otpm): (i64, i64, i64),
    ) -> Vec<Settlement> {
        let combined = if reserved > 0 {
            let targets: Vec<(String, String)> = stash
                .reserved_scopes
                .iter()
                .filter(|scope| scope.0 != PTU_TEAM)
                .cloned()
                .collect();
            self.engine
                .reservation_ops(&targets, &stash.reserved_scopes, tpm_actual, reserved)
        } else {
            Vec::new()
        };
        let ptu = if reserved > 0 {
            self.ptu_failure_ops(stash, tpm_actual, partial_usage)
        } else {
            Vec::new()
        };
        vec![
            Settlement::Plain(combined),
            self.io_failure_settlement(
                &stash.itpm,
                partial_usage.map_or(0, Usage::billable_input_tokens),
                itpm,
            ),
            self.io_failure_settlement(
                &stash.otpm,
                partial_usage.map_or(0, |usage| usage.completion_tokens),
                otpm,
            ),
            Settlement::Project(ptu),
        ]
    }

    fn io_failure_settlement(
        &self,
        reservation: &IoReservation,
        actual: i64,
        reserved: i64,
    ) -> Settlement {
        let targets: Vec<(String, String)> = reservation.scopes.iter().cloned().collect();
        if reserved <= 0 {
            return Settlement::Project(Vec::new());
        }
        if reservation.windows.is_empty() {
            return Settlement::Plain(self.engine.reservation_ops(
                &targets,
                &reservation.scopes,
                actual,
                reserved,
            ));
        }
        Settlement::Project(self.engine.project_ops(
            &targets,
            &reservation.scopes,
            actual,
            reserved,
            &reservation.windows,
        ))
    }

    fn ptu_failure_ops(&self, stash: &Stash, tpm_actual: i64, usage: Option<&Usage>) -> Vec<Op> {
        let targets: Vec<(String, String)> = stash
            .reserved_scopes
            .iter()
            .filter(|scope| scope.0 == PTU_TEAM)
            .cloned()
            .collect();
        if targets.is_empty() {
            return Vec::new();
        }
        self.engine.project_ops(
            &targets,
            &stash.reserved_scopes,
            settlement_tokens(stash.ptu_ceiling.as_ref(), usage, tpm_actual),
            stash.ptu_reserved_tokens,
            &stash.ptu_windows,
        )
    }

    pub async fn take_parallel_slot(&self, state: &LimiterState) -> Option<SlotAcquisition> {
        let _guard = state.release.lock().await;
        state.stash().parallel_slot.take()
    }

    pub async fn release_for_settlement(
        &self,
        state: &LimiterState,
        call_id: Option<&str>,
        request: Option<&RequestRedis>,
    ) {
        {
            let stash = state.stash();
            if !stash.owned || !stash.for_call(call_id) {
                return;
            }
        }
        let ctx = Ctx {
            policy: Policy::default(),
            request,
        };
        self.release_stashed(state, true, ctx).await;
    }

    pub async fn settle_failure(
        &self,
        state: &LimiterState,
        facts: FailureFacts,
        request: Option<&RequestRedis>,
    ) {
        let ctx = Ctx {
            policy: Policy::default(),
            request,
        };
        {
            let stash = state.stash();
            if !stash.owned || !stash.for_call(facts.call_id.as_deref()) {
                return;
            }
        }
        self.release_stashed(state, true, ctx).await;
        let settlements = {
            let mut stash = state.stash();
            let open = Self::open_reservations(&stash);
            let settlements = self.failure_settlements(
                &stash,
                facts.tpm_actual,
                facts.partial_usage.as_ref(),
                open,
            );
            if open.0 > 0 || open.1 > 0 || open.2 > 0 {
                stash.mark_reservation_released();
            }
            settlements
        };
        for settlement in settlements {
            self.engine.settle(settlement, ctx, true).await;
        }
    }

    fn tpm_ops(
        &self,
        targets: &[(String, String)],
        reserved_scopes: &BTreeSet<(String, String)>,
        actual: i64,
        reserved: i64,
    ) -> Vec<Increment> {
        self.engine
            .reservation_ops(targets, reserved_scopes, actual, reserved)
    }

    pub async fn settle_success(
        &self,
        state: &LimiterState,
        facts: SuccessFacts,
        request: Option<&RequestRedis>,
    ) {
        let ctx = Ctx {
            policy: facts.policy.into(),
            request,
        };
        {
            let stash = state.stash();
            if !stash.owned || !stash.for_call(facts.call_id.as_deref()) {
                return;
            }
        }
        if !facts.keep_slot {
            self.release_stashed(state, true, ctx).await;
        }
        let (operations, io) = {
            let mut stash = state.stash();
            let operations = if facts.exempt {
                Vec::new()
            } else {
                self.success_operations(&stash, &facts)
            };
            let io = self.io_success_settlement(&stash, &facts.io_usage);
            stash.clear_reserved_tokens();
            (operations, io)
        };
        self.engine.apply_usage(&operations, ctx).await;
        match io {
            Some(Settlement::Plain(operations)) if !operations.is_empty() => {
                self.engine
                    .increment_with_ttl_preservation(&operations, ctx)
                    .await;
            }
            Some(Settlement::Project(operations)) if !operations.is_empty() => {
                self.engine.reservation_aware(&operations).await;
            }
            _ => {}
        }
    }

    pub async fn settle_turn(
        &self,
        connection: &LimiterState,
        facts: SuccessFacts,
        request: Option<&RequestRedis>,
    ) {
        let turn = LimiterState::default();
        {
            let mut stash = turn.stash();
            stash.owned = true;
            stash.tpm_limited_tags = connection.stash().tpm_limited_tags.clone();
        }
        self.settle_success(
            &turn,
            SuccessFacts {
                call_id: None,
                keep_slot: true,
                ..facts
            },
            request,
        )
        .await;
    }

    fn io_success_settlement(&self, stash: &Stash, usage: &IoUsage) -> Option<Settlement> {
        let (itpm, otpm) = (stash.itpm.tokens, stash.otpm.tokens);
        if itpm <= 0 && otpm <= 0 {
            return None;
        }
        let released = stash.io_reservation_released;
        if usage.response.is_none()
            && usage.combined.is_none()
            && usage.aggregate_total <= 0
            && !released
        {
            return None;
        }
        let (billable, completion) = match usage.response.or(usage.combined) {
            Some(resolved) => (resolved.billable_input_tokens(), resolved.completion_tokens),
            None if usage.aggregate_total > 0 => (usage.aggregate_total, usage.aggregate_total),
            None => (itpm, otpm),
        };
        let empty = BTreeSet::new();
        let reserved_scopes = |reservation: &IoReservation| {
            if released {
                empty.clone()
            } else {
                reservation.scopes.clone()
            }
        };
        let targets = |reservation: &IoReservation| -> Vec<(String, String)> {
            reservation.scopes.iter().cloned().collect()
        };
        if released || (stash.itpm.windows.is_empty() && stash.otpm.windows.is_empty()) {
            let reserved = |amount: i64| if released { 0 } else { amount };
            let mut operations = self.engine.reservation_ops(
                &targets(&stash.itpm),
                &reserved_scopes(&stash.itpm),
                billable,
                reserved(itpm),
            );
            operations.extend(self.engine.reservation_ops(
                &targets(&stash.otpm),
                &reserved_scopes(&stash.otpm),
                completion,
                reserved(otpm),
            ));
            return Some(Settlement::Plain(operations));
        }
        let mut operations = Vec::new();
        for (reservation, actual, reserved) in [
            (&stash.itpm, billable, itpm),
            (&stash.otpm, completion, otpm),
        ] {
            if reserved > 0 {
                operations.extend(self.engine.project_ops(
                    &targets(reservation),
                    &reserved_scopes(reservation),
                    actual,
                    reserved,
                    &reservation.windows,
                ));
            }
        }
        Some(Settlement::Project(operations))
    }

    fn success_operations(&self, stash: &Stash, facts: &SuccessFacts) -> Vec<Op> {
        let reserved_tokens = if stash.reservation_released {
            0
        } else {
            stash.reserved_tokens
        };
        let reserved_model = stash.reserved_model.as_ref();
        let served = facts.served_model.as_ref();
        let fell_back = match (reserved_model, served) {
            (Some(reserved), Some(served)) => {
                served.requested != reserved.requested
                    && stash.fallback_models.contains(&served.requested)
            }
            _ => false,
        };
        let reconcile = if fell_back || reserved_model.is_none() {
            served
        } else {
            reserved_model
        };
        let reconcile_facts = reconcile.and_then(|model| facts.models.get(&model.requested));
        let candidates: Vec<Candidate> = reconcile_facts
            .map(|facts| facts.candidates.clone())
            .unwrap_or_default();
        let groups: Vec<String> = candidates
            .iter()
            .map(|candidate| candidate.group.clone())
            .collect();
        let mut targets = collect_targets(
            &facts.targets,
            reconcile.map(|model| model.group.as_str()),
            &groups,
            &stash.tpm_limited_tags,
        );
        for scope in &stash.reserved_scopes {
            if (scope.0 == "agent" || scope.0 == "agent_session") && !targets.contains(scope) {
                targets.push(scope.clone());
            }
        }
        let key_owned: BTreeSet<(String, String)> = candidates
            .iter()
            .filter(|model| {
                candidates
                    .iter()
                    .filter(|candidate| candidate.group == model.group)
                    .all(|candidate| candidate.key_owns)
            })
            .map(|model| {
                (
                    "model_per_team".to_owned(),
                    format!("{}:{}", facts.team_label, model.group),
                )
            })
            .collect();
        let charged: Vec<(String, String)> = targets
            .into_iter()
            .filter(|target| !key_owned.contains(target))
            .collect();
        let abandoned: Vec<(String, String)> = if fell_back {
            stash
                .reserved_scopes
                .iter()
                .filter(|scope| {
                    PER_MODEL_KEYS.contains(&scope.0.as_str()) && !charged.contains(scope)
                })
                .cloned()
                .collect()
        } else {
            Vec::new()
        };
        let mut operations: Vec<Op> = self
            .tpm_ops(
                &charged,
                &stash.reserved_scopes,
                facts.total_tokens,
                reserved_tokens,
            )
            .into_iter()
            .chain(self.tpm_ops(&abandoned, &stash.reserved_scopes, 0, reserved_tokens))
            .map(Op::Plain)
            .collect();
        let reserved_ptu: Vec<(String, String)> = stash
            .reserved_scopes
            .iter()
            .filter(|scope| scope.0 == PTU_TEAM)
            .cloned()
            .collect();
        if !reserved_ptu.is_empty() {
            operations.extend(self.engine.project_ops(
                &reserved_ptu,
                &stash.reserved_scopes,
                settlement_tokens(
                    stash.ptu_ceiling.as_ref(),
                    facts.usage.as_ref(),
                    facts.total_tokens,
                ),
                if stash.reservation_released {
                    0
                } else {
                    stash.ptu_reserved_tokens
                },
                &stash.ptu_windows,
            ));
            return operations;
        }
        let team = truthy(&facts.targets.user_api_key_team_id);
        let ptu = match (&stash.ptu_ceiling, team, reconcile) {
            (Some(ceiling), Some(team), Some(_)) => Some((
                format!("{team}:{}", ceiling.model_group),
                settlement_tokens(Some(ceiling), facts.usage.as_ref(), facts.total_tokens),
            )),
            (None, Some(_), Some(_)) => reconcile_facts
                .and_then(|facts| facts.ptu.as_ref())
                .map(|ptu| (ptu.value.clone(), ptu.actual_tokens)),
            _ => None,
        };
        if let Some((value, actual)) = ptu {
            operations.extend(
                self.engine
                    .reservation_ops(
                        &[(PTU_TEAM.to_owned(), value)],
                        &stash.reserved_scopes,
                        actual,
                        0,
                    )
                    .into_iter()
                    .map(Op::Plain),
            );
        }
        operations
    }

    pub async fn handle_with(
        &self,
        input: &HookInput,
        state: Option<&LimiterState>,
        inputs: Value,
    ) -> HookOutput {
        self.handle_hosted_with(input, state, inputs, None, Host::detached())
            .await
    }

    pub async fn handle_hosted_with(
        &self,
        input: &HookInput,
        state: Option<&LimiterState>,
        inputs: Value,
        body: Option<Value>,
        host: Host,
    ) -> HookOutput {
        let Some(state) = state else {
            return Ok(None);
        };
        let inputs = if inputs.is_null() {
            Value::Object(Map::new())
        } else {
            inputs
        };
        match input.hook {
            Hook::PreCall => {
                let inputs: PreCallInputs = port_inputs(input.hook, inputs)?;
                let facts = match self.pre_call_facts(state, input, inputs)? {
                    Ok(Some(facts)) => facts,
                    Ok(None) => return Ok(None),
                    Err(raised) => {
                        state.stash().owned = true;
                        return Ok(Some(Verdict::Abort(raised)));
                    }
                };
                let request = state.request();
                if facts.phase == Phase::FirstPass {
                    return Ok(self
                        .pre_call(state, facts, request.as_ref())
                        .await
                        .map(Verdict::Abort));
                }
                let call = input.call.clone().unwrap_or_default();
                let limits = input.fields.get::<typed::KeyLimits>()?;
                let ctx = Ctx {
                    policy: facts.policy.into(),
                    request: request.as_ref(),
                };
                let reservation = Reservation {
                    host: &host,
                    body,
                    call_type: &call.kind.call_type,
                    model: Some(call.target.model.as_str()).filter(|model| !model.is_empty()),
                    configured: limits
                        .as_ref()
                        .and_then(|limits| limits.estimated_output_tokens.as_ref()),
                };
                Ok(
                    Box::pin(self.pre_call_hosted(state, facts, ctx, reservation))
                        .await?
                        .map(Verdict::Abort),
                )
            }
            Hook::PostCallSuccess => {
                let inputs: PostCallSuccessInputs = port_inputs(input.hook, inputs)?;
                let facts = PostCallSuccessFacts {
                    call_id: inputs.call_id,
                    batch: inputs.batch,
                };
                self.post_call_success(state, facts).await;
                Ok(None)
            }
            Hook::PostCallFailure => {
                let inputs: PostCallFailureInputs = port_inputs(input.hook, inputs)?;
                let facts = failure_facts(input, &inputs)?;
                self.post_call_failure(state, facts).await;
                Ok(None)
            }
            Hook::LogPrepare => {
                let inputs: LogPrepareInputs = port_inputs(input.hook, inputs)?;
                self.log_prepare(state, input, inputs.call_id)
            }
            _ => Ok(None),
        }
    }

    fn pre_call_facts(
        &self,
        state: &LimiterState,
        input: &HookInput,
        inputs: PreCallInputs,
    ) -> Result<Result<Option<PreCallFacts>, Abort>, CallbackError> {
        let call = input.call.clone().unwrap_or_default();
        let call_id = inputs.call_id.clone();
        let requested_model = non_empty(&call.target.model);
        let cap = cap_decision(input, &call)?;
        let reuses_request_slot = call.kind.call_type == MCP_TOOL_CALL_TYPE
            && inputs.nested
            && state.stash().parallel_slot.is_some();
        let base = PreCallFacts {
            identity: inputs.identity.clone(),
            call_id,
            prior_response: prior_response(input)?,
            policy: inputs.policy,
            batch: inputs.batch.clone(),
            phase: inputs.phase,
            ..PreCallFacts::default()
        };
        let facts = if inputs.batch_call {
            PreCallFacts {
                requested_model,
                ..base
            }
        } else {
            let limits = input.fields.get::<typed::KeyLimits>()?;
            let rpm_limit_type = limits
                .as_ref()
                .and_then(|limits| limits.rpm_limit_type.clone());
            let tpm_limit_type = limits
                .as_ref()
                .and_then(|limits| limits.tpm_limit_type.clone());
            let lookups = Lookups {
                managed_agent_policy: agent_policy(
                    limits
                        .as_ref()
                        .and_then(|limits| limits.managed_agent_policy.as_ref()),
                )?,
                invoked_agent_policy: agent_policy(
                    limits
                        .as_ref()
                        .and_then(|limits| limits.invoked_agent_policy.as_ref()),
                )?,
                ..inputs.lookups.clone()
            };
            let request = request_facts(input.fields.get::<typed::LimiterRequest>()?)?;
            let auth = auth_members(input.fields.get::<typed::KeyIdentity>()?, limits)?;
            let built = match (Request {
                auth: &auth,
                rpm_limit_type: rpm_limit_type.as_deref(),
                tpm_limit_type: tpm_limit_type.as_deref(),
                metadata: &inputs.metadata,
                lookups: &lookups,
                request: &request,
                requested_model: requested_model.as_deref(),
                window_size: self.engine.settings().window_size,
            }
            .build())
            {
                Ok(built) => built,
                Err(BuildError::Python { exception, message }) => {
                    return Ok(Err(refusal::python_error(&exception, &message)));
                }
                Err(error) => return Err(CallbackError::Failed(error.to_string())),
            };
            let descriptors: Vec<Descriptor> =
                match serde_json::from_value(Value::Array(built.descriptors)) {
                    Ok(descriptors) => descriptors,
                    Err(error) => {
                        let message = error.to_string();
                        return match python_int_error(&message) {
                            Some((exception, raised)) => {
                                Ok(Err(refusal::python_error(exception, raised)))
                            }
                            None => Err(failed(input.hook, "descriptors", message)),
                        };
                    }
                };
            PreCallFacts {
                descriptors,
                reserved_model: inputs
                    .lookups
                    .rate_limited_model(requested_model.as_deref()),
                requested_model,
                ptu_ceiling: built.ptu_ceiling.or(inputs.held_ptu_ceiling),
                reuses_request_slot,
                ..base
            }
        };
        {
            let mut stash = state.stash();
            stash.cap = cap;
            stash.pre_call = Some(facts.clone());
        }
        Ok(Ok(Some(facts)))
    }

    pub fn plan(&self, state: &LimiterState) -> Plan {
        let stash = state.stash();
        let empty = Vec::new();
        let (descriptors, ceiling) = stash.pre_call.as_ref().map_or((&empty, None), |facts| {
            (&facts.descriptors, facts.ptu_ceiling.as_ref())
        });
        let tpm_limits: Vec<i64> = descriptors
            .iter()
            .filter(|descriptor| !descriptor.is_project_io())
            .filter_map(|descriptor| {
                let limit = descriptor.limits().tokens_per_unit?;
                Some(match ceiling {
                    Some(ceiling) if descriptor.key == PTU_TEAM => ceiling.raw_output_token_limit(),
                    _ => limit,
                })
            })
            .collect();
        Plan {
            reserves: !tpm_limits.is_empty() && self.engine.settings().tpm_reservation_enabled,
            min_tpm_limit: tpm_limits.iter().min().copied(),
            min_otpm_limit: descriptors
                .iter()
                .filter(|descriptor| descriptor.key == PROJECT_OTPM)
                .filter_map(|descriptor| descriptor.limits().tokens_per_unit)
                .min(),
            has_io: descriptors.iter().any(Descriptor::is_project_io),
            ptu: ceiling.is_some(),
        }
    }

    pub fn cap_settings(&self) -> CapSettings {
        self.cap
    }

    pub fn descriptors(&self, state: &LimiterState) -> Vec<Descriptor> {
        state
            .stash()
            .pre_call
            .as_ref()
            .map(|facts| facts.descriptors.clone())
            .unwrap_or_default()
    }

    pub fn implicit_output_cap(
        &self,
        state: &LimiterState,
        min_configured_limit: Option<i64>,
        configured_output_tokens: Option<i64>,
        cap_applied: bool,
    ) -> Option<i64> {
        let cap = state.stash().cap;
        output_cap(&OutputCapFacts {
            min_configured_limit,
            responses: cap.responses,
            exempt: cap.exempt || cap_applied,
            configured_output_tokens,
            fraction: self.cap.fraction,
            default_max_tokens_estimate: self.cap.default_max_tokens_estimate,
            responses_min_output_tokens: self.cap.responses_min_output_tokens,
        })
    }

    pub async fn fallback_target_allowed(
        &self,
        state: Option<&LimiterState>,
        descriptors: Vec<Descriptor>,
        model: &str,
        policy: PolicyFacts,
        request: Option<&RequestRedis>,
    ) -> Result<bool, Box<Abort>> {
        let ctx = Ctx {
            policy: policy.into(),
            request,
        };
        let allowed = descriptors.is_empty()
            || !self
                .engine
                .should_rate_limit(&descriptors, false, false, None, false, ctx)
                .await
                .map_err(|Unverifiable| Box::new(refusal::unverifiable()))?
                .is_over();
        if allowed && let Some(state) = state {
            state.stash().fallback_models.insert(model.to_owned());
        }
        Ok(allowed)
    }

    fn log_prepare(
        &self,
        state: &LimiterState,
        input: &HookInput,
        call_id: Option<String>,
    ) -> HookOutput {
        let statuses = {
            let stash = state.stash();
            if !stash.owned || !stash.for_call(call_id.as_deref()) {
                return Ok(None);
            }
            stash
                .rate_limit_response
                .as_ref()
                .map(|response| response.statuses.clone())
                .unwrap_or_default()
        };
        if statuses.is_empty() {
            return Ok(None);
        }
        let Some(Value::Object(payload)) =
            input.fields.raw(Field::StandardLoggingPayload)?.cloned()
        else {
            return Ok(None);
        };
        let mut hidden = match payload.get("hidden_params") {
            Some(Value::Object(hidden)) => hidden.clone(),
            _ => Map::new(),
        };
        let mut headers = match hidden.get("additional_headers") {
            Some(Value::Object(headers)) => headers.clone(),
            _ => Map::new(),
        };
        headers.extend(header_map(&statuses));
        hidden.insert("additional_headers".into(), Value::Object(headers));
        Ok(Some(Verdict::Patch(Patch::default().merge(
            Field::StandardLoggingPayload,
            Map::from_iter([("hidden_params".to_owned(), Value::Object(hidden))]),
        ))))
    }
}

fn python_int_error(message: &str) -> Option<(&'static str, &str)> {
    for exception in [crate::descriptor::NOT_A_NUMBER, crate::descriptor::OVERFLOW] {
        if let Some(raised) = message
            .strip_prefix(exception)
            .and_then(|rest| rest.strip_prefix(": "))
        {
            return Some((exception, raised));
        }
    }
    if message.starts_with(crate::descriptor::INVALID_INT_LITERAL) {
        Some(("ValueError", message))
    } else if message.starts_with(crate::descriptor::INT_ARGUMENT) {
        Some(("TypeError", message))
    } else {
        None
    }
}

fn port_inputs<T: DeserializeOwned>(hook: Hook, inputs: Value) -> Result<T, CallbackError> {
    serde_json::from_value(inputs).map_err(|error| failed(hook, "port inputs", error))
}

fn token_count(count: Option<&Scalar>) -> i64 {
    match count {
        Some(Scalar::Int(count)) => {
            i64::try_from(*count).unwrap_or(if *count < 0 { i64::MIN } else { i64::MAX })
        }
        Some(Scalar::Bool(flag)) => i64::from(*flag),
        _ => 0,
    }
}

fn failure_facts(
    input: &HookInput,
    inputs: &PostCallFailureInputs,
) -> Result<PostCallFailureFacts, CallbackError> {
    let combined = input
        .fields
        .get::<typed::Usage>()?
        .and_then(|usage| usage.combined)
        .filter(|combined| {
            inputs.combined_is_usage && token_count(combined.completion_tokens.as_ref()) > 0
        });
    let Some(combined) = combined else {
        return Ok(PostCallFailureFacts::default());
    };
    let prompt_tokens = token_count(combined.prompt_tokens.as_ref());
    let completion_tokens = token_count(combined.completion_tokens.as_ref());
    let cached_tokens = token_count(combined.cached_tokens.as_ref());
    let without_cached = |tokens: i64| {
        if cached_tokens > 0 {
            (tokens - cached_tokens).max(0)
        } else {
            tokens
        }
    };
    let tpm_actual = match TokenRateLimitType::from_setting(&inputs.token_rate_limit_type) {
        TokenRateLimitType::Output => completion_tokens,
        TokenRateLimitType::Input => without_cached(prompt_tokens),
        TokenRateLimitType::Total => without_cached(token_count(combined.total_tokens.as_ref())),
    };
    Ok(PostCallFailureFacts {
        tpm_actual,
        partial_usage: Some(Usage {
            prompt_tokens,
            completion_tokens,
            cached_tokens,
        }),
    })
}

fn collect_targets(
    facts: &TargetFacts,
    model_group: Option<&str>,
    model_groups: &[String],
    tags: &BTreeSet<String>,
) -> Vec<(String, String)> {
    let key = truthy(&facts.user_api_key_hash);
    let user = truthy(&facts.user_api_key_user_id);
    let team = truthy(&facts.user_api_key_team_id);
    let organization = truthy(&facts.user_api_key_org_id);
    let project = truthy(&facts.user_api_key_project_id);
    let end_user = truthy(&facts.end_user_id);
    let agent = truthy(&facts.agent_id);
    let session = truthy(&facts.session_id);
    let mut targets: Vec<(String, String)> = Vec::new();
    let mut push = |key: &str, value: String| targets.push((key.to_owned(), value));
    if let Some(key) = key {
        push("api_key", key.to_owned());
    }
    if let Some(user) = user {
        push("user", user.to_owned());
    }
    if let Some(team) = team {
        push("team", team.to_owned());
    }
    if let (Some(team), Some(user)) = (team, user) {
        push("team_member", format!("{team}:{user}"));
    }
    if let Some(end_user) = end_user {
        push("end_user", end_user.to_owned());
    }
    if let Some(organization) = organization {
        push("organization", organization.to_owned());
    }
    let customers = [
        ("customer_org", truthy(&facts.customer_org_id)),
        ("customer_team", truthy(&facts.customer_team_id)),
        ("customer_member", truthy(&facts.customer_member_id)),
    ];
    for (scope, owner) in customers {
        if let Some(owner) = owner {
            push(scope, owner.to_owned());
        }
    }
    let owners = [
        ("model_per_key", key),
        ("model_per_team", team),
        ("model_per_end_user", end_user),
        ("model_per_organization", organization),
        ("model_per_project", project),
        ("model_per_customer_org", customers[0].1),
        ("model_per_customer_team", customers[1].1),
        ("model_per_customer_member", customers[2].1),
    ];
    if let Some(model_group) = model_group.filter(|group| !group.is_empty()) {
        let mut seen: Vec<&str> = Vec::new();
        let groups: Vec<&str> = if model_groups.is_empty() {
            vec![model_group]
        } else {
            model_groups.iter().map(String::as_str).collect()
        };
        for group in groups {
            if seen.contains(&group) {
                continue;
            }
            seen.push(group);
            for (scope, owner) in owners {
                if let Some(owner) = owner {
                    push(scope, format!("{owner}:{group}"));
                }
            }
        }
    }
    if let Some(agent) = agent {
        push("agent", agent.to_owned());
        if let Some(session) = session {
            push("agent_session", format!("{agent}:{session}"));
        }
    }
    for tag in tags {
        push("tag", tag.clone());
    }
    targets
}

impl Callback for ParallelRequestLimiter {
    fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    type State = LimiterState;

    fn request_state(&self) -> Option<LimiterState> {
        Some(LimiterState::default())
    }

    fn handle<'a>(
        &'a self,
        input: &'a HookInput,
        state: Option<&'a LimiterState>,
    ) -> BoxFuture<'a, HookOutput> {
        Box::pin(self.handle_with(input, state, Value::Null))
    }

    fn handle_hosted<'a>(
        &'a self,
        input: &'a HookInput,
        state: Option<&'a LimiterState>,
        host: Host,
    ) -> BoxFuture<'a, HookOutput> {
        Box::pin(self.handle_hosted_with(input, state, Value::Null, None, host))
    }
}
