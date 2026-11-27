//! The proxy's `dynamic_rate_limiter_v3` callback as a callback of the stage: a model group's
//! `tpm` and `rpm` are shared out by priority, and once the group is saturated each priority is
//! held to the share `litellm_settings.priority_reservation` reserves it.
//!
//! Every decision is the engine's in `litellm-callbacks-model-capacity`, which the proxy runs too.
//! This file only tells it who is calling and for which model, and settles the tokens the call
//! used. It counts on the key limiter's counters, so the proxy and the gateway see one another's
//! calls on a shared Redis. It counts every caller of a model group that sets `tpm` or `rpm`, the
//! master key included, and the statuses it publishes are the answer the key limiter, registered
//! after it, starts its own from.
//!
//! Accepted differences: a route that tells no usage, every one but `/v1/messages`, settles no
//! tokens. The share and throttle metrics are Prometheus's, which is a callback of its own. A
//! per-model limit that borrows idle capacity is never held here, since `admit` leaves a config
//! that lists one to Python. A refusal carries its message once: the proxy's error body repeats it
//! under `provider_specific_fields`, which is only how its `HTTPException` is converted.

use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

use futures_util::future::BoxFuture;
use litellm_callbacks::{
    Abort, Activation, Callback, Field, FieldSet, Hook, HookInput, HookOutput, Manifest, Outcome,
    Patch, Scalar, Selection, Subscription, UsageRecord, Verdict, typed,
};
use litellm_callbacks_model_capacity::{
    CapacityError, Charge, ClassShareEntity, Engine, Fact, Facts, GroupInfo, KeyFacts, LoggedGroup,
    NAME, PreCall, Priority, Request as CapacityRequest, SaturationFallback,
    Settings as EngineSettings, Settlement, SharedStash, StoreLimiter, TokenRateLimitType,
    UsageTokens,
};
use litellm_callbacks_parallel_request_limiter::{
    PolicyFacts, TokenRateLimitType as LimiterTokenType,
};
use litellm_config::{Config, PriorityReservation};
use litellm_gateway_auth::Authenticated;
use litellm_gateway_auth_core::Credential;
use litellm_gateway_inference::group_limits;
use litellm_python_compat::json::loaded_str;
use litellm_redis_store::{RequestRedis, active_teams::ActiveTeams, rate_limits::RateLimits};
use litellm_router_core::Router;
use serde_json::{Map, Value, json};

use super::{Registered, parallel_request_limiter};
use crate::host::{NativeCapacityHost, NativeRows};

/// The settlement flushes its own post-call batch, so this deadline never sends it.
const POST_CALL_DEADLINE: Duration = Duration::from_secs(1);

/// What the config says of the dynamic rate limiter, as the proxy's `CapacityHost.settings` hands
/// it to the engine.
#[derive(Clone, Debug, PartialEq)]
pub struct Settings {
    engine: EngineSettings,
    limits: parallel_request_limiter::Settings,
}

fn reserved(reservation: &PriorityReservation) -> Value {
    match reservation {
        PriorityReservation::Share(share) => json!(share),
        PriorityReservation::Sized { unit, value } => {
            json!({"type": <&'static str>::from(*unit), "value": value})
        }
    }
}

fn class_share_entity(entity: litellm_config::ClassShareEntity) -> ClassShareEntity {
    match entity {
        litellm_config::ClassShareEntity::Team => ClassShareEntity::Team,
        litellm_config::ClassShareEntity::Key => ClassShareEntity::Key,
        litellm_config::ClassShareEntity::CustomerOrganization => {
            ClassShareEntity::CustomerOrganization
        }
        litellm_config::ClassShareEntity::CustomerMember => ClassShareEntity::CustomerMember,
    }
}

impl Settings {
    /// `None` unless `litellm_settings.callbacks` names `dynamic_rate_limiter_v3`. The window and
    /// the tokens a limit counts are the key limiter's. `licensed` is whether `LITELLM_LICENSE` is
    /// set at all, which is all the proxy asks before it honors a reserved priority.
    pub fn read(
        config: &Config,
        limits: parallel_request_limiter::Settings,
        licensed: bool,
    ) -> Option<Self> {
        let litellm = &config.litellm_settings;
        let named = litellm
            .callbacks
            .iter()
            .flatten()
            .any(|callback| callback.as_str() == Some(NAME));
        let written = litellm
            .priority_reservation_settings
            .clone()
            .unwrap_or_default();
        let engine = EngineSettings {
            priority_reservation: litellm.priority_reservation.as_ref().map(|reservations| {
                reservations
                    .iter()
                    .map(|(priority, reservation)| (json!(priority), reserved(reservation)))
                    .collect()
            }),
            default_priority: written.default_priority,
            saturation_threshold: written.saturation_threshold,
            class_split: written
                .priority_class_share_config
                .split_equally_among_active
                .iter()
                .copied()
                .map(class_share_entity)
                .collect(),
            saturation_check_cache_ttl: json!(written.saturation_check_cache_ttl),
            window_size: limits.window_size(),
            rate_limit_type: match limits.token_type() {
                LimiterTokenType::Input => TokenRateLimitType::Input,
                LimiterTokenType::Output => TokenRateLimitType::Output,
                LimiterTokenType::Total => TokenRateLimitType::Total,
            },
            licensed,
            policy: PolicyFacts::default(),
        };
        named.then_some(Self { engine, limits })
    }
}

/// The proxy's `dynamic_rate_limiter_v3` over the limiter's counters.
pub struct ModelCapacity {
    manifest: Manifest,
    engine: Engine,
    settings: EngineSettings,
    /// Tells what a model group's deployments serve a minute.
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
                reads: FieldSet::of(&[Field::Model]),
                writes: FieldSet::of(&[Field::RateLimitStatuses]),
                may_abort: true,
                ..Subscription::on(Hook::PreCall)
            },
            Subscription::on(Hook::PostCallSuccess),
            Subscription {
                writes: FieldSet::of(&[Field::ResponseHeaders]),
                ..Subscription::on(Hook::ResponseHeaders)
            },
            Subscription {
                reads: FieldSet::of(&[Field::Outcome, Field::Usage, Field::LoggedModelGroup]),
                ..Subscription::on(Hook::AttemptEnded)
            },
        ],
        has_request_state: true,
    }
}

/// One request as the engine counts it: who it counts under, and what the hooks it ran left for
/// the ones after them.
pub struct Counted {
    key: KeyFacts,
    priority: Option<Arc<str>>,
    stash: SharedStash,
    /// The pre-call hook counted the call against a model group, so its tokens settle there.
    admitted: AtomicBool,
    /// The answer is complete, so it names the priority it was counted under.
    complete: AtomicBool,
}

impl Counted {
    /// The master key names no priority class. A key names the one its row holds.
    fn of(identity: &Authenticated<NativeRows>) -> Self {
        let priority = match &identity.credential {
            Credential::MasterKey => None,
            Credential::Key { key } => key.priority().cloned(),
        };
        Self {
            key: KeyFacts {
                api_key: parallel_request_limiter::Caller::of(identity).api_key,
                priority: priority.as_deref().map(|priority| Priority {
                    value: json!(priority),
                    hash_error: None,
                }),
                ..KeyFacts::default()
            },
            priority,
            stash: SharedStash::default(),
            admitted: AtomicBool::new(false),
            complete: AtomicBool::new(false),
        }
    }
}

fn tokens(count: &Option<Scalar>) -> i64 {
    match count {
        Some(Scalar::Int(count)) => i64::try_from(*count).unwrap_or(i64::MAX),
        _ => 0,
    }
}

/// The proxy's 503 when `fail_closed_rate_limit_enforcement` could not verify a counter. The
/// gateway only serves a config that leaves it off, so nothing else is ever raised here.
fn unverifiable(error: &CapacityError) -> Abort {
    Abort::new(503, error.to_string())
}

impl ModelCapacity {
    /// `counters` are the ones the key limits count on, and `active` is the registry of callers a
    /// priority class's share is split among.
    pub fn new(
        settings: Settings,
        counters: Arc<RateLimits>,
        active: ActiveTeams,
        router: Router,
    ) -> Arc<Self> {
        Arc::new(Self {
            manifest: manifest(),
            engine: Engine {
                limiter: Arc::new(StoreLimiter::new(settings.limits.engine(counters), None)),
                host: Arc::new(NativeCapacityHost::new(active)),
            },
            settings: settings.engine,
            router,
        })
    }

    pub(crate) fn registered(self: &Arc<Self>) -> Registered {
        Registered::for_caller(self.clone(), Counted::of)
    }

    /// What `model`'s group serves a minute. `None` for a name that is no model group, which the
    /// proxy's hook leaves alone.
    fn group(&self, model: &str) -> Option<GroupInfo> {
        let limits = group_limits(&self.router, model)?;
        Some(GroupInfo {
            tpm: limits.tpm,
            rpm: limits.rpm,
            input_cost_per_token: None,
            output_cost_per_token: None,
        })
    }

    async fn pre_call(&self, input: &HookInput, counted: &Counted) -> HookOutput {
        let Some(model) = input.fields.get::<typed::Model>()? else {
            return Ok(None);
        };
        let Some(group) = self.group(&model) else {
            return Ok(None);
        };
        let facts = Facts {
            borrowing: Some(Fact::Ok(Vec::new())),
            ..Facts::default()
        };
        let request = CapacityRequest {
            key: &counted.key,
            call_type: input.call.as_ref().map(|call| call.kind.call_type.as_str()),
            settings: &self.settings,
            stash: &counted.stash,
            fallback: SaturationFallback::default(),
            redis: None,
            facts: &facts,
        };
        match self.engine.pre_call(&request, &model, &group).await {
            PreCall::Allow(published) => {
                counted.admitted.store(true, Ordering::Release);
                Ok(published.map(|published| {
                    Verdict::Patch(Patch::default().set(Field::RateLimitStatuses, json!(published)))
                }))
            }
            PreCall::Refuse(refusal) => Ok(Some(Verdict::Abort(refusal.into_abort()))),
            PreCall::Raise(error) => Ok(Some(Verdict::Abort(unverifiable(&error)))),
        }
    }

    /// `x-litellm-priority` and `x-litellm-rate-limiter-version`, which the proxy's post-call hook
    /// adds to every complete answer, whatever model it was for.
    fn priority_headers(counted: &Counted) -> Map<String, Value> {
        Engine::post_call_headers(&counted.key)
            .into_iter()
            .map(|(name, value)| (name, Value::from(loaded_str(&value))))
            .collect()
    }

    /// The success log's half: the tokens the call used are added to the group's saturation and
    /// to its priority's counter, on the post-call batch the proxy's settlement rides.
    async fn settle(&self, counted: &Counted, used: &UsageRecord, model_group: String) {
        let used = UsageTokens {
            prompt_tokens: tokens(&used.prompt_tokens),
            completion_tokens: tokens(&used.completion_tokens),
            total_tokens: tokens(&used.total_tokens),
            cached_tokens: tokens(&used.cached_tokens),
        };
        let group = LoggedGroup {
            truthy: !model_group.is_empty(),
            text: model_group,
            keyed: true,
            hash_error: None,
        };
        let settlement = Settlement {
            model_group: Some(&group),
            team_id: None,
            team_alias: None,
            key_hash: counted.key.api_key.as_deref(),
            key_alias: None,
            customer_org_id: None,
            customer_org_alias: None,
        };
        let request = RequestRedis::new(tokio::runtime::Handle::current(), POST_CALL_DEADLINE);
        let settled = self
            .engine
            .settle_success(
                &self.settings,
                Some(&counted.stash),
                &settlement,
                counted.priority.as_deref(),
                Charge::Reported(UsageTokens::settled(
                    Some(&used),
                    self.settings.rate_limit_type,
                )),
                Some(&request),
            )
            .await;
        request.flush_post().await;
        if let Err(error) = settled {
            tracing::error!(
                "Error in dynamic rate limiter success event: {}",
                error.python_message()
            );
        }
    }

    /// A call the route told no usage of settles nothing: no caller the gateway serves holds a
    /// reservation to give back, since a limit that borrows belongs to a config Python serves.
    async fn log(&self, input: &HookInput, counted: &Counted) -> HookOutput {
        let succeeded = input.fields.get::<typed::Outcome>()? == Some(Outcome::Success);
        if !succeeded || !counted.admitted.load(Ordering::Acquire) {
            return Ok(None);
        }
        let used = input
            .fields
            .get::<typed::Usage>()?
            .and_then(|usage| usage.response);
        let group = input
            .fields
            .get::<typed::LoggedModelGroup>()?
            .and_then(|group| match group.value {
                Some(Scalar::Text(group)) => Some(group),
                _ => None,
            });
        if let Some((used, group)) = used.zip(group) {
            self.settle(counted, &used, group).await;
        }
        Ok(None)
    }
}

impl Callback for ModelCapacity {
    type State = Counted;

    fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    fn handle<'a>(
        &'a self,
        input: &'a HookInput,
        state: Option<&'a Counted>,
    ) -> BoxFuture<'a, HookOutput> {
        Box::pin(async move {
            let Some(counted) = state else {
                return Ok(None);
            };
            match input.hook {
                Hook::PreCall => self.pre_call(input, counted).await,
                Hook::PostCallSuccess => {
                    counted.complete.store(true, Ordering::Release);
                    Ok(None)
                }
                Hook::ResponseHeaders if counted.complete.load(Ordering::Acquire) => {
                    Ok(Some(Verdict::Patch(Patch::default().merge(
                        Field::ResponseHeaders,
                        Self::priority_headers(counted),
                    ))))
                }
                Hook::AttemptEnded => self.log(input, counted).await,
                _ => Ok(None),
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    fn the_callback_is_built_only_when_the_config_names_it() {
        let unnamed =
            Config::from_yaml("litellm_settings:\n  priority_reservation: {prod: 0.9}\n").unwrap();

        assert_eq!(
            Settings::read(
                &unnamed,
                parallel_request_limiter::Settings::default(),
                true
            ),
            None
        );
    }
}
