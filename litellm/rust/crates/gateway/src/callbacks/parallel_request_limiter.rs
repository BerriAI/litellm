//! The proxy's `parallel_request_limiter_v3` for a key's own `rpm_limit`, `tpm_limit`,
//! `max_parallel_requests` and per-model limits, as a callback of the stage: the pre-call hook
//! counts the request, reserves the tokens its body asks for and takes a parallel slot, or refuses
//! the call. A complete answer gives the slot back before it goes out and carries the
//! `x-ratelimit-*` headers, and the success log charges the tokens the call used to the caller's
//! counters, for every caller, the master key included. The limit check Python's authentication
//! runs before the body asks the same limiter through [`KeyLimiter::before_body`].
//!
//! Accepted difference: a route that tells no usage, every one but `/v1/messages`, charges no
//! tokens.

use std::{
    collections::BTreeMap,
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use futures_util::future::BoxFuture;
use litellm_callbacks::{
    Abort, Activation, Callback, Field, FieldSet, Hook, HookInput, HookOutput, Manifest, Outcome,
    Patch, Scalar, Selection, Subscription, UsageRecord, Verdict, typed,
};
use litellm_callbacks_model_capacity::RATE_LIMIT_ERROR as CAPACITY_RATE_LIMITED;
use litellm_callbacks_parallel_request_limiter::{
    AuthMetadata, CapSettings, Descriptor, DescriptorRequest, Engine, EstimateRequest,
    LimiterState, Lookups, OutputCapFacts, ParallelRequestLimiter, PolicyFacts,
    PostCallFailureFacts, PostCallSuccessFacts, PreCallFacts, PythonServed, RATE_LIMITED,
    RateLimitedModel, RequestFacts, Settings as EngineSettings, SuccessFacts, TargetFacts,
    TokenRateLimitType, UsageObject, aggregate_only, charged_tokens, explicit_output_cap,
    header_map, output_cap, prior_response, tokens_for_request, write_output_cap,
};
use litellm_config::{Config, Environment};
use litellm_gateway_auth::Authenticated;
use litellm_gateway_auth_core::{self as auth_core, Belongs, Named, Team, record::MinuteLimits};
use litellm_gateway_inference::{RateLimited, call_type};
use litellm_redis_client::RedisClient;
use litellm_redis_store::{Remote, rate_limits::RateLimits};
use litellm_router_core::Router;
use parking_lot::Mutex;
use serde_json::{Map, Value, json};

use super::Registered;
use crate::host::NativeRows;

/// The name Python registers the hook under.
pub(crate) const NAME: &str = "parallel_request_limiter";
/// `RedisCache.default_ttl` when `default_redis_ttl` is unset.
pub(crate) const REDIS_DEFAULT_TTL: Duration = Duration::from_secs(60);
/// The `default_in_memory_ttl` of the proxy's `internal_usage_cache`, the limiter's memory tier.
const MEMORY_TTL: Duration = Duration::from_secs(1);
/// The api key and the user Python names a master key caller by, `LITELLM_PROXY_MASTER_KEY_ALIAS`
/// and `LITELLM_PROXY_ADMIN_NAME`.
pub(super) const MASTER_KEY_ALIAS: &str = "litellm_proxy_master_key";
pub(super) const PROXY_ADMIN_NAME: &str = "default_user_id";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Settings {
    window_size: i64,
    tpm_reservation_enabled: bool,
    token_type: TokenRateLimitType,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            window_size: 60,
            tpm_reservation_enabled: true,
            token_type: TokenRateLimitType::Total,
        }
    }
}

/// The proxy's `internal_usage_cache` counters, over the coordination Redis when there is one and
/// over this process's memory otherwise. A key's limits and a model group's capacity are counted
/// on the same ones, as they are in the proxy.
pub fn counters(redis: Option<RedisClient>) -> Arc<RateLimits> {
    let remote = redis.map(|client| Remote {
        client,
        default_ttl: REDIS_DEFAULT_TTL,
    });
    Arc::new(RateLimits::new(remote, Some(MEMORY_TTL)))
}

impl Settings {
    pub(crate) fn window_size(self) -> i64 {
        self.window_size
    }

    pub(crate) fn token_type(self) -> TokenRateLimitType {
        self.token_type
    }

    /// The limiter's engine over `counters`, which each limiter builds its own of, as the proxy
    /// does.
    pub(crate) fn engine(self, counters: Arc<RateLimits>) -> Engine {
        Engine::new(
            counters,
            EngineSettings {
                window_size: self.window_size,
                tpm_reservation_enabled: self.tpm_reservation_enabled,
            },
            Arc::new(unix_now),
        )
    }

    /// The window and the reservation as the environment sets them, and
    /// `general_settings.token_rate_limit_type`.
    pub fn read(config: &Config, environment: &Environment) -> Self {
        let defaults = Self::default();
        Self {
            window_size: environment
                .litellm_rate_limit_window_size
                .unwrap_or(defaults.window_size),
            tpm_reservation_enabled: environment
                .litellm_tpm_token_reservation_enabled
                .map_or(defaults.tpm_reservation_enabled, |flag| flag.0),
            token_type: TokenRateLimitType::from_setting(&json!(
                config.general_settings.token_rate_limit_type
            )),
        }
    }
}

/// The tokens a call asks for, input and output, as the proxy estimates them before it runs. `Err`
/// for a body only Python's estimator reads, one that carries media.
fn estimated(
    body: &Value,
    call_type: &str,
    min_tpm_limit: Option<i64>,
    cap: &CapSettings,
) -> Result<i64, PythonServed> {
    let tokens = tokens_for_request(
        EstimateRequest {
            data: body,
            call_type,
            min_configured_tpm_limit: min_tpm_limit,
            configured_output_tokens: None,
            include_media: true,
        },
        cap,
    )?;
    Ok(i64::try_from(tokens).unwrap_or(i64::MAX))
}

/// Whether only Python's estimator tells the tokens the body sent to `path` asks for.
pub(crate) fn python_estimates(body: &mut Map<String, Value>, path: &str) -> bool {
    let mut sent = Value::Object(std::mem::take(body));
    let python = estimated(
        &sent,
        call_type(path).map_or("", <&str>::from),
        None,
        &CapSettings::default(),
    )
    .is_err();
    if let Value::Object(sent) = sent.take() {
        *body = sent;
    }
    python
}

/// What a call reserves from a token limit, after the output cap the proxy writes into a body
/// that sets none, so one unbounded answer cannot overshoot the smallest limit.
fn reserved(
    body: &mut Value,
    call_type: &str,
    min_tpm_limit: i64,
    cap: &CapSettings,
) -> Result<i64, PythonServed> {
    let explicit = match &*body {
        Value::Object(request) => explicit_output_cap(request, call_type, cap)?,
        _ => None,
    };
    let implicit = output_cap(&OutputCapFacts {
        min_configured_limit: Some(min_tpm_limit),
        responses: false,
        exempt: explicit.is_some(),
        configured_output_tokens: None,
        fraction: cap.fraction,
        default_max_tokens_estimate: cap.default_max_tokens_estimate,
        responses_min_output_tokens: cap.responses_min_output_tokens,
    });
    if let Some(implicit) = implicit {
        write_output_cap(body, call_type, implicit);
    }
    estimated(body, call_type, Some(min_tpm_limit), cap)
}

fn unix_now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0.0, |elapsed| elapsed.as_secs_f64())
}

/// A limit for each model a key names one for, by the name a caller asks for or its model group.
pub type ModelLimits = Arc<BTreeMap<String, i64>>;

/// The limits a virtual key holds, which the limiter enforces after authentication. The key's
/// row carries them, read once where the native host reads the key.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct KeyLimits {
    pub rpm: Option<i64>,
    pub tpm: Option<i64>,
    pub max_parallel_requests: Option<i64>,
    pub model_rpm: ModelLimits,
    pub model_tpm: ModelLimits,
}

impl KeyLimits {
    /// Whether any limit counts tokens, which a call reserves from its body before it runs.
    pub fn counts_tokens(&self) -> bool {
        self.tpm.is_some() || !self.model_tpm.is_empty()
    }

    /// Whether any limit is told from the request body: its tokens or the model it names.
    pub fn reads_body(&self) -> bool {
        self.counts_tokens() || !self.model_rpm.is_empty()
    }
}

/// Who Python's auth object says the caller is, with the limits of each thing it belongs to.
pub(crate) struct Caller {
    pub(crate) api_key: Option<String>,
    user_id: Option<String>,
    team_id: Option<String>,
    org_id: Option<String>,
    project_id: Option<String>,
    end_user_id: Option<String>,
    pub(crate) limits: KeyLimits,
    user: MinuteLimits,
    team: MinuteLimits,
    team_member: MinuteLimits,
    end_user: MinuteLimits,
    org: MinuteLimits,
}

impl Caller {
    /// A key is named by the hash its row is stored under, as Python's `api_key` is, and the
    /// master key by Python's alias for it. The master key holds no limit. A limit is only read
    /// from a stored row: a key copy of a team holds none here.
    pub(crate) fn of(identity: &Authenticated<NativeRows>) -> Self {
        let auth_core::Caller {
            credential,
            belongs:
                Belongs {
                    holder,
                    org,
                    project,
                },
            end_user,
        } = &**identity;
        let user = holder.user();
        let team = holder.team();
        let (api_key, user_id, limits) = match credential {
            auth_core::Credential::MasterKey => (
                Some(MASTER_KEY_ALIAS.to_owned()),
                Some(PROXY_ADMIN_NAME.to_owned()),
                KeyLimits::default(),
            ),
            auth_core::Credential::Key { key } => (
                identity.hash().map(ToString::to_string),
                user.map(|user| user.id().to_owned()),
                key.limits().clone(),
            ),
        };
        Self {
            api_key,
            user_id,
            team_id: team.map(|team| team.id().to_owned()),
            org_id: org.as_ref().map(|org| org.id().to_owned()),
            project_id: project.as_ref().map(|project| project.id().to_owned()),
            end_user_id: end_user.end_user_id.clone(),
            limits,
            user: user
                .and_then(Named::stored)
                .map(|user| user.limits)
                .unwrap_or_default(),
            team: match team {
                Some(Team::Row(team)) => team.limits,
                Some(Team::Unstored(_) | Team::KeyCopy(_)) | None => MinuteLimits::default(),
            },
            team_member: holder
                .membership()
                .map(|member| member.limits)
                .unwrap_or_default(),
            end_user: end_user.limits,
            org: org
                .as_ref()
                .and_then(Named::stored)
                .map(|org| org.limits)
                .unwrap_or_default(),
        }
    }

    /// Whether any limit counts tokens, which a call reserves from its body before it runs.
    pub(crate) fn counts_tokens(&self) -> bool {
        self.limits.counts_tokens()
            || [
                self.user,
                self.team,
                self.team_member,
                self.end_user,
                self.org,
            ]
            .into_iter()
            .any(MinuteLimits::counts_tokens)
    }

    /// The members of Python's auth object the limiter's descriptors read.
    fn auth(&self) -> Map<String, Value> {
        let KeyLimits {
            rpm,
            tpm,
            max_parallel_requests,
            ..
        } = &self.limits;
        Map::from_iter(
            [
                ("api_key", json!(self.api_key)),
                ("rpm_limit", json!(rpm)),
                ("tpm_limit", json!(tpm)),
                ("max_parallel_requests", json!(max_parallel_requests)),
                ("user_id", json!(self.user_id)),
                ("user_rpm_limit", json!(self.user.rpm)),
                ("user_tpm_limit", json!(self.user.tpm)),
                ("team_id", json!(self.team_id)),
                ("team_rpm_limit", json!(self.team.rpm)),
                ("team_tpm_limit", json!(self.team.tpm)),
                ("team_member_rpm_limit", json!(self.team_member.rpm)),
                ("team_member_tpm_limit", json!(self.team_member.tpm)),
                ("end_user_id", json!(self.end_user_id)),
                ("end_user_rpm_limit", json!(self.end_user.rpm)),
                ("end_user_tpm_limit", json!(self.end_user.tpm)),
                ("org_id", json!(self.org_id)),
                ("organization_rpm_limit", json!(self.org.rpm)),
                ("organization_tpm_limit", json!(self.org.tpm)),
                ("project_id", json!(self.project_id)),
            ]
            .map(|(name, value)| (name.to_owned(), value)),
        )
    }

    fn targets(&self) -> TargetFacts {
        TargetFacts {
            user_api_key_hash: self.api_key.clone(),
            user_api_key_user_id: self.user_id.clone(),
            user_api_key_team_id: self.team_id.clone(),
            user_api_key_org_id: self.org_id.clone(),
            user_api_key_project_id: self.project_id.clone(),
            end_user_id: self.end_user_id.clone(),
            ..TargetFacts::default()
        }
    }
}

/// The tokens Python charges a served call: those `token_type` counts, less the prompt tokens read
/// from cache.
fn charged(used: &UsageRecord, token_type: TokenRateLimitType) -> i64 {
    let count = |tokens: &Option<Scalar>| match tokens {
        Some(Scalar::Int(tokens)) => Value::from(i64::try_from(*tokens).unwrap_or(i64::MAX)),
        _ => Value::from(0),
    };
    let usage = UsageObject::Chat {
        members: Map::from_iter([
            ("prompt_tokens".to_owned(), count(&used.prompt_tokens)),
            (
                "completion_tokens".to_owned(),
                count(&used.completion_tokens),
            ),
            ("total_tokens".to_owned(), count(&used.total_tokens)),
            ("cached_tokens".to_owned(), count(&used.cached_tokens)),
        ]),
    };
    match charged_tokens(Some(&usage), token_type) {
        0 => aggregate_only(Some(&usage)),
        charged => charged,
    }
}

/// The limit's ordered headers and words of an abort that is Python's `ProxyRateLimitError`.
pub(crate) fn rate_limited(abort: &Abort) -> Option<RateLimited> {
    let error = abort
        .error
        .as_ref()
        .filter(|error| error.kind == RATE_LIMITED || error.kind == CAPACITY_RATE_LIMITED)?;
    let headers = error
        .fields
        .get("headers")
        .and_then(|headers| serde_json::from_value::<Vec<(String, String)>>(headers.clone()).ok())
        .unwrap_or_else(|| abort.headers.clone().into_iter().collect());
    Some(RateLimited {
        message: abort.message.clone(),
        headers,
    })
}

/// The proxy's `parallel_request_limiter_v3` over [`counters`].
pub struct KeyLimiter {
    manifest: Manifest,
    limiter: ParallelRequestLimiter,
    token_type: TokenRateLimitType,
    /// Tells the model group `model_group_alias` gives the model a body names.
    router: Router,
}

fn manifest() -> Manifest {
    Manifest {
        selection: Selection {
            activation: Activation::Required,
            ..Selection::named(NAME)
        },
        subscriptions: vec![
            Subscription {
                reads: FieldSet::of(&[Field::Model, Field::RequestBody, Field::RateLimitStatuses]),
                writes: FieldSet::of(&[Field::OptionalParams]),
                may_abort: true,
                ..Subscription::on(Hook::PreCall)
            },
            Subscription::on(Hook::PostCallSuccess),
            Subscription {
                writes: FieldSet::of(&[Field::ResponseHeaders]),
                ..Subscription::on(Hook::ResponseHeaders)
            },
            Subscription::on(Hook::PostCallFailure),
            Subscription {
                reads: FieldSet::of(&[
                    Field::Outcome,
                    Field::Usage,
                    Field::Model,
                    Field::LoggedModelGroup,
                ]),
                ..Subscription::on(Hook::AttemptEnded)
            },
        ],
        has_request_state: true,
    }
}

/// The members of `capped` that `sent` does not hold the same, which is the output cap the
/// limiter wrote.
fn written(sent: &Map<String, Value>, capped: Map<String, Value>) -> Map<String, Value> {
    capped
        .into_iter()
        .filter(|(name, value)| sent.get(name) != Some(value))
        .collect()
}

impl KeyLimiter {
    pub fn new(settings: Settings, counters: Arc<RateLimits>, router: Router) -> Arc<Self> {
        Arc::new(Self {
            manifest: manifest(),
            limiter: ParallelRequestLimiter::new(settings.engine(counters)),
            token_type: settings.token_type,
            router,
        })
    }

    #[cfg(test)]
    pub(crate) fn window_size(&self) -> i64 {
        self.limiter.engine().settings().window_size
    }

    /// The limiter as the stage registers it, with each request's admission made from its caller.
    pub(crate) fn registered(self: &Arc<Self>) -> Registered {
        let limiter = self.clone();
        Registered::for_caller(self.clone(), move |identity| Admission {
            limiter: limiter.clone(),
            caller: Caller::of(identity),
            state: LimiterState::default(),
            phase: Mutex::new(Phase::Asked),
        })
    }

    /// `requested` with the group `model_group_alias` gives it, so a limit keyed by either is
    /// found.
    fn asked(&self, requested: String) -> RateLimitedModel {
        let registry = self.router.registry();
        let group = registry
            .alias_of(&requested)
            .filter(|group| !group.is_empty())
            .unwrap_or(&requested)
            .to_owned();
        RateLimitedModel { requested, group }
    }

    /// The limits the caller holds, as the proxy describes them: its key's, its user's, team's,
    /// membership's, end user's and organization's, and its key's `model_per_key` for the model
    /// asked for.
    fn descriptors(&self, caller: &Caller, model: Option<&RateLimitedModel>) -> Vec<Descriptor> {
        let auth = caller.auth();
        let metadata = AuthMetadata {
            key_model_rpm_limit: json!(*caller.limits.model_rpm),
            key_model_tpm_limit: json!(*caller.limits.model_tpm),
            ..AuthMetadata::default()
        };
        let lookups = Lookups {
            candidates: model
                .map(|model| model.requested.clone())
                .into_iter()
                .collect(),
            groups: model
                .map(|model| (model.requested.clone(), Some(model.group.clone())))
                .into_iter()
                .collect(),
            ..Lookups::default()
        };
        DescriptorRequest {
            auth: &auth,
            rpm_limit_type: None,
            tpm_limit_type: None,
            metadata: &metadata,
            lookups: &lookups,
            request: &RequestFacts::default(),
            requested_model: model.map(|model| model.requested.as_str()),
            window_size: self.limiter.engine().settings().window_size,
        }
        .build()
        .map(|built| built.descriptors)
        .unwrap_or_default()
        .into_iter()
        .filter_map(|descriptor| serde_json::from_value(descriptor).ok())
        .collect()
    }

    /// What the pre-call hook is told of a call: the caller's limits and, when one counts tokens,
    /// what `body` asks for, after the output cap the proxy writes into a body that sets none. A
    /// body nobody here estimates reserves the least.
    fn pre_call_facts(
        &self,
        caller: &Caller,
        call_type: &str,
        model: Option<RateLimitedModel>,
        body: Option<&mut Value>,
    ) -> PreCallFacts {
        let descriptors = self.descriptors(caller, model.as_ref());
        let min_tpm_limit = descriptors
            .iter()
            .filter_map(|descriptor| descriptor.limits().tokens_per_unit)
            .min();
        let reserves = self.limiter.engine().settings().tpm_reservation_enabled;
        let estimated_tokens = match (min_tpm_limit, body) {
            (Some(min_tpm_limit), Some(body)) if reserves => {
                reserved(body, call_type, min_tpm_limit, &self.limiter.cap_settings()).ok()
            }
            _ => None,
        };
        PreCallFacts {
            descriptors,
            requested_model: model.as_ref().map(|model| model.requested.clone()),
            estimated_tokens,
            reserved_model: model,
            ..PreCallFacts::default()
        }
    }

    /// `raise_if_over_limit_before_body`: the refusal of a caller already over a limit that needs
    /// no body. Counters are only read, so the pre-call hook still counts an admitted call.
    pub(crate) async fn before_body(&self, caller: &Caller) -> Option<Abort> {
        self.limiter
            .before_body(
                &self.descriptors(caller, None),
                PolicyFacts::default(),
                None,
            )
            .await
    }

    async fn pre_call(&self, input: &HookInput, admission: &Admission) -> HookOutput {
        let model = input
            .fields
            .get::<typed::Model>()?
            .filter(|model| !model.is_empty())
            .map(|model| self.asked(model));
        let call_type = input
            .call
            .as_ref()
            .map_or("", |call| call.kind.call_type.as_str());
        let sent = match input.fields.raw(Field::RequestBody)? {
            Some(Value::Object(sent)) if admission.caller.counts_tokens() => Some(sent),
            _ => None,
        };
        let mut capped = sent.cloned().map(Value::Object);
        let facts = PreCallFacts {
            prior_response: prior_response(input)?,
            ..self.pre_call_facts(&admission.caller, call_type, model, capped.as_mut())
        };
        if let Some(abort) = self.limiter.pre_call(&admission.state, facts, None).await {
            return Ok(Some(Verdict::Abort(abort)));
        }
        *admission.phase.lock() = Phase::Admitted;
        let cap = match (sent, capped) {
            (Some(sent), Some(Value::Object(capped))) => written(sent, capped),
            _ => Map::new(),
        };
        Ok((!cap.is_empty())
            .then(|| Verdict::Patch(Patch::default().merge(Field::OptionalParams, cap))))
    }

    /// `async_log_success_event`: gives back what the call still holds and charges the tokens it
    /// used to the caller's counters, which a TPM limit reads. A call whose route told no usage
    /// only gives back what it holds.
    async fn log_success(&self, input: &HookInput, admission: &Admission) -> HookOutput {
        let used = input
            .fields
            .get::<typed::Usage>()?
            .and_then(|usage| usage.response);
        let Some(used) = used else {
            self.limiter.release_on_disconnect(&admission.state).await;
            return Ok(None);
        };
        let group = input
            .fields
            .get::<typed::LoggedModelGroup>()?
            .and_then(|group| match group.value {
                Some(Scalar::Text(group)) => Some(group),
                _ => None,
            });
        let served_model = input
            .fields
            .get::<typed::Model>()?
            .zip(group)
            .map(|(requested, group)| RateLimitedModel { requested, group });
        let facts = SuccessFacts {
            total_tokens: charged(&used, self.token_type),
            targets: admission.caller.targets(),
            served_model,
            ..SuccessFacts::default()
        };
        self.limiter
            .settle_success(&admission.state, facts, None)
            .await;
        Ok(None)
    }
}

#[cfg(feature = "test-support")]
impl KeyLimiter {
    /// What this process's memory holds of a counter, for a test to read what was counted.
    pub fn counted(&self, kind: &str, descriptor: &str, value: &str) -> Option<f64> {
        use litellm_redis_store::{
            Number,
            rate_limits::{Entry, Local, Scope},
        };
        let entry = Entry::new(
            kind,
            Scope {
                descriptor: descriptor.into(),
                value: value.into(),
            },
        )
        .ok()?;
        match self.limiter.engine().store().get_local(&entry).ok()? {
            Some(Local::Number(number)) => Some(Number::as_f64(number)),
            _ => None,
        }
    }
}

/// How far one request got with the limiter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    /// Not admitted, so the limiter holds nothing for it.
    Asked,
    Admitted,
    /// A complete answer gave its slot back, so its limits are known to name in its headers.
    Succeeded,
    /// Nothing is left to give back.
    Settled,
}

/// One request's admission: who it is charged to and what the limiter holds for it. A call that
/// is dropped while it still holds something, a stream the client left, gives it back off the
/// request.
pub struct Admission {
    limiter: Arc<KeyLimiter>,
    caller: Caller,
    state: LimiterState,
    phase: Mutex<Phase>,
}

impl Drop for Admission {
    fn drop(&mut self) {
        let holds = matches!(*self.phase.lock(), Phase::Admitted | Phase::Succeeded);
        let Ok(runtime) = tokio::runtime::Handle::try_current() else {
            return;
        };
        if holds {
            let limiter = self.limiter.clone();
            let state = std::mem::take(&mut self.state);
            runtime.spawn(async move { limiter.limiter.release_on_disconnect(&state).await });
        }
    }
}

impl Callback for KeyLimiter {
    type State = Admission;

    fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    fn handle<'a>(
        &'a self,
        input: &'a HookInput,
        state: Option<&'a Admission>,
    ) -> BoxFuture<'a, HookOutput> {
        Box::pin(async move {
            let Some(admission) = state else {
                return Ok(None);
            };
            let phase = *admission.phase.lock();
            match (input.hook, phase) {
                (Hook::PreCall, Phase::Asked) => self.pre_call(input, admission).await,
                (Hook::PostCallSuccess, Phase::Admitted) => {
                    self.limiter
                        .post_call_success(&admission.state, PostCallSuccessFacts::default())
                        .await;
                    *admission.phase.lock() = Phase::Succeeded;
                    Ok(None)
                }
                (Hook::ResponseHeaders, Phase::Succeeded) => {
                    let statuses = admission
                        .state
                        .stash()
                        .rate_limit_response
                        .as_ref()
                        .map(|response| header_map(&response.statuses))
                        .unwrap_or_default();
                    Ok((!statuses.is_empty()).then(|| {
                        Verdict::Patch(Patch::default().merge(Field::ResponseHeaders, statuses))
                    }))
                }
                (Hook::PostCallFailure, Phase::Admitted | Phase::Succeeded) => {
                    self.limiter
                        .post_call_failure(&admission.state, PostCallFailureFacts::default())
                        .await;
                    *admission.phase.lock() = Phase::Settled;
                    Ok(None)
                }
                (Hook::AttemptEnded, Phase::Admitted | Phase::Succeeded)
                    if input.fields.get::<typed::Outcome>()? == Some(Outcome::Success) =>
                {
                    *admission.phase.lock() = Phase::Settled;
                    self.log_success(input, admission).await
                }
                _ => Ok(None),
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use litellm_gateway_auth_core::{
        Holder, TeamCopy,
        record::{EndUserRecord, MembershipRecord, OrgRecord, TeamRecord, UserRecord},
    };
    use rstest::rstest;

    use super::*;
    use crate::test_support::{key, master};

    fn limiter() -> Arc<KeyLimiter> {
        KeyLimiter::new(
            Settings::default(),
            counters(None),
            crate::boot::unhosted(&Config::default()),
        )
    }

    /// Each descriptor's kind, the value it counts under and its request and token limits.
    fn described(
        identity: &Authenticated<NativeRows>,
    ) -> Vec<(String, String, Option<i64>, Option<i64>)> {
        limiter()
            .descriptors(&Caller::of(identity), None)
            .into_iter()
            .map(|descriptor| {
                let limits = descriptor.limits();
                (
                    descriptor.key,
                    descriptor.value,
                    limits.requests_per_unit,
                    limits.tokens_per_unit,
                )
            })
            .collect()
    }

    /// A user in a team in an organization, each with its own limits, called for an end user
    /// with its own.
    fn member() -> Authenticated<NativeRows> {
        key(
            "stored",
            None,
            Belongs {
                holder: Holder::Member {
                    user: Named::Stored(UserRecord {
                        user_id: "u1".to_owned(),
                        limits: MinuteLimits {
                            rpm: Some(1),
                            tpm: Some(10),
                        },
                        ..UserRecord::default()
                    }),
                    team: Team::Row(TeamRecord {
                        team_id: "t1".to_owned(),
                        limits: MinuteLimits {
                            rpm: Some(2),
                            tpm: Some(20),
                        },
                        ..TeamRecord::default()
                    }),
                    membership: Some(MembershipRecord {
                        user_id: "u1".to_owned(),
                        team_id: "t1".to_owned(),
                        limits: MinuteLimits {
                            rpm: Some(3),
                            tpm: Some(30),
                        },
                        ..MembershipRecord::default()
                    }),
                },
                org: Some(Named::Stored(OrgRecord {
                    organization_id: "o1".to_owned(),
                    limits: MinuteLimits {
                        rpm: Some(5),
                        tpm: Some(50),
                    },
                    ..OrgRecord::default()
                })),
                project: Some(Named::Unstored("p1".to_owned())),
            },
            EndUserRecord {
                end_user_id: Some("e1".to_owned()),
                found: true,
                limits: MinuteLimits {
                    rpm: Some(4),
                    tpm: Some(40),
                },
                ..EndUserRecord::default()
            },
        )
    }

    /// The same names with no row stored behind any of them.
    fn unstored() -> Authenticated<NativeRows> {
        key(
            "stored",
            None,
            Belongs {
                holder: Holder::Member {
                    user: Named::Unstored("u1".to_owned()),
                    team: Team::KeyCopy(TeamCopy {
                        team_id: "t1".to_owned(),
                        object_permission: None,
                    }),
                    membership: None,
                },
                org: Some(Named::Unstored("o1".to_owned())),
                project: Some(Named::Unstored("p1".to_owned())),
            },
            EndUserRecord::default(),
        )
    }

    fn described_as(
        kind: &str,
        value: &str,
        rpm: i64,
        tpm: i64,
    ) -> (String, String, Option<i64>, Option<i64>) {
        (kind.to_owned(), value.to_owned(), Some(rpm), Some(tpm))
    }

    #[rstest]
    fn each_thing_a_caller_belongs_to_is_limited_under_its_own_name() {
        assert_eq!(
            described(&member()),
            [
                described_as("user", "u1", 1, 10),
                described_as("team", "t1", 2, 20),
                described_as("team_member", "t1:u1", 3, 30),
                described_as("end_user", "e1", 4, 40),
                described_as("organization", "o1", 5, 50),
            ]
        );
    }

    #[rstest]
    fn the_master_key_is_limited_under_the_end_user_its_request_names() {
        let master = Authenticated::from(auth_core::Caller::master_key(EndUserRecord {
            end_user_id: Some("e1".to_owned()),
            found: true,
            limits: MinuteLimits {
                rpm: Some(4),
                tpm: Some(40),
            },
            ..EndUserRecord::default()
        }));

        assert_eq!(described(&master), [described_as("end_user", "e1", 4, 40)]);
    }

    #[rstest]
    #[case::named_with_no_row(unstored())]
    #[case::the_master_key(master())]
    fn a_caller_with_no_stored_row_holds_no_limit_beyond_its_keys(
        #[case] identity: Authenticated<NativeRows>,
    ) {
        assert_eq!(described(&identity), []);
    }

    #[rstest]
    #[case::stored(member(), Some("e1"))]
    #[case::unstored(unstored(), None)]
    fn a_settled_call_is_charged_to_every_name_the_caller_has(
        #[case] identity: Authenticated<NativeRows>,
        #[case] end_user_id: Option<&str>,
    ) {
        assert_eq!(
            Caller::of(&identity).targets(),
            TargetFacts {
                user_api_key_hash: Some("stored".to_owned()),
                user_api_key_user_id: Some("u1".to_owned()),
                user_api_key_team_id: Some("t1".to_owned()),
                user_api_key_org_id: Some("o1".to_owned()),
                user_api_key_project_id: Some("p1".to_owned()),
                end_user_id: end_user_id.map(str::to_owned),
                ..TargetFacts::default()
            }
        );
    }

    #[rstest]
    fn the_master_key_is_charged_as_the_proxy_admin() {
        assert_eq!(
            Caller::of(&master()).targets(),
            TargetFacts {
                user_api_key_hash: Some(MASTER_KEY_ALIAS.to_owned()),
                user_api_key_user_id: Some(PROXY_ADMIN_NAME.to_owned()),
                ..TargetFacts::default()
            }
        );
    }

    #[rstest]
    #[case::a_team_token_limit(TeamRecord { limits: MinuteLimits { tpm: Some(1), rpm: None }, ..TeamRecord::default() }, true)]
    #[case::a_team_request_limit(TeamRecord { limits: MinuteLimits { rpm: Some(1), tpm: None }, ..TeamRecord::default() }, false)]
    fn a_token_limit_anywhere_the_caller_belongs_counts_tokens(
        #[case] team: TeamRecord,
        #[case] counts_tokens: bool,
    ) {
        let identity = key(
            "stored",
            None,
            Belongs {
                holder: Holder::Team(Team::Row(team)),
                org: None,
                project: None,
            },
            EndUserRecord::default(),
        );

        assert_eq!(Caller::of(&identity).counts_tokens(), counts_tokens);
    }

    #[rstest]
    fn the_window_and_the_reservation_come_from_the_environment() {
        let environment = Environment {
            litellm_rate_limit_window_size: Some(30),
            litellm_tpm_token_reservation_enabled: Some(litellm_config::Flag(false)),
            ..Environment::default()
        };

        assert_eq!(
            (
                Settings::read(&Config::default(), &environment),
                Settings::read(&Config::default(), &Environment::default())
            ),
            (
                Settings {
                    window_size: 30,
                    tpm_reservation_enabled: false,
                    token_type: TokenRateLimitType::Total,
                },
                Settings::default()
            )
        );
    }
}
