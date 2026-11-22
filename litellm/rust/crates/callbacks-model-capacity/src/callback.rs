use std::sync::Arc;

use futures_util::future::BoxFuture;
use litellm_callbacks::{
    Activation, Callback, CallbackError, Field, FieldSet, Hook, HookInput, HookOutput,
    LimiterRequest, LoggedKeyIdentity, Manifest, MetadataKind, Outcome, Patch, Scalar, Selection,
    Subscription, Verdict, typed,
};
use litellm_redis_store::RequestRedis;
use serde::{Deserialize, de::DeserializeOwned};
use serde_json::{Value, json};

use crate::{
    capacity::{Engine, Frame, PreCall, Request, SaturationFallback, Settlement},
    error::CapacityError,
    host::{CapacityHost, CounterStore, Facts},
    model::{KeyFacts, KeyGaps, LoggedGroup, UsageTokens},
    refusal::{reraise, unverifiable},
    state::SharedStash,
};

pub const NAME: &str = "dynamic_rate_limiter_v3";

#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct AttemptGaps {
    settlement_error: Option<String>,
    customer_org_id: Option<String>,
    customer_org_alias: Option<String>,
}

const FAILURE_HOOK_ERROR: &str = "[Non-Blocking] Error in async_post_call_failure_hook callback";

fn saturation_fallback(request: Option<LimiterRequest>) -> SaturationFallback {
    let request = request.unwrap_or_default();
    SaturationFallback {
        enabled: request.fallback_on_saturation == Some(true),
        invalid: request.fallback_on_saturation_invalid,
        armed: request.fallback_walk_armed == Some(true),
    }
}

#[derive(Debug, Default, PartialEq, Eq)]
struct LoggedTeam {
    team_id: Option<String>,
    team_alias: Option<String>,
    key_hash: Option<String>,
    key_alias: Option<String>,
    priority: Option<String>,
}

impl LoggedTeam {
    fn settlement<'a>(
        &'a self,
        group: Option<&'a LoggedGroup>,
        gaps: &'a AttemptGaps,
    ) -> Settlement<'a> {
        Settlement {
            model_group: group,
            team_id: self.team_id.as_deref(),
            team_alias: self.team_alias.as_deref(),
            key_hash: self.key_hash.as_deref(),
            key_alias: self.key_alias.as_deref(),
            customer_org_id: gaps.customer_org_id.as_deref(),
            customer_org_alias: gaps.customer_org_alias.as_deref(),
        }
    }
}

fn logged_text(
    identity: &LoggedKeyIdentity,
    name: &str,
    value: Option<&Scalar>,
) -> Result<Option<String>, ()> {
    if identity
        .rejected
        .iter()
        .flatten()
        .any(|rejected| rejected == name)
    {
        return Err(());
    }
    match value {
        None => Ok(None),
        Some(Scalar::Text(text)) => Ok(Some(text.clone())),
        Some(_) => Err(()),
    }
}

fn logged_team(identity: Option<&LoggedKeyIdentity>) -> Option<LoggedTeam> {
    let Some(identity) = identity else {
        return Some(LoggedTeam::default());
    };
    if identity.metadata_kind == Some(MetadataKind::Other) {
        return None;
    }
    if identity
        .rejected
        .iter()
        .flatten()
        .any(|rejected| rejected == "auth_metadata")
    {
        return None;
    }
    let team_id = logged_text(identity, "team_id", identity.team_id.as_ref()).ok()?;
    let team_alias = logged_text(identity, "team_alias", identity.team_alias.as_ref()).ok()?;
    let key_hash = logged_text(identity, "key_hash", identity.key_hash.as_ref()).ok()?;
    let key_alias = logged_text(identity, "key_alias", identity.key_alias.as_ref()).ok()?;
    let priority = match &identity.auth_metadata_priority {
        Some(Scalar::Text(priority)) => Some(priority.clone()),
        _ => None,
    };
    Some(LoggedTeam {
        team_id,
        team_alias,
        key_hash,
        key_alias,
        priority,
    })
}

#[derive(Clone, Copy)]
struct Host<'a> {
    redis: Option<&'a RequestRedis>,
    facts: &'a Facts,
}

pub struct ModelCapacity {
    manifest: Manifest,
    engine: Engine,
}

fn failed(error: CapacityError) -> CallbackError {
    CallbackError::Failed(error.to_string())
}

fn raise(error: CapacityError) -> HookOutput {
    match error {
        CapacityError::Raised(raised) => {
            Ok(Some(Verdict::Abort(reraise(raised.token, &raised.message))))
        }
        CapacityError::Unverifiable => Ok(Some(Verdict::Abort(unverifiable()))),
        error => Err(failed(error)),
    }
}

fn projected<T: DeserializeOwned + Default>(input: &HookInput) -> Result<T, CallbackError> {
    match input.fields.raw(Field::DeprecatedKwargs)? {
        None | Some(Value::Null) => Ok(T::default()),
        Some(value) => {
            T::deserialize(value).map_err(|error| CallbackError::Failed(error.to_string()))
        }
    }
}

fn key_facts(input: &HookInput) -> Result<KeyFacts, CallbackError> {
    let identity = input.fields.get::<typed::KeyIdentity>()?;
    let gaps: KeyGaps = projected(input)?;
    Ok(KeyFacts::new(identity.as_ref(), gaps))
}

impl ModelCapacity {
    pub fn new(limiter: Arc<dyn CounterStore>, host: Arc<dyn CapacityHost>) -> Self {
        Self {
            manifest: Manifest {
                selection: Selection {
                    activation: Activation::Required,
                    ..Selection::named(NAME)
                },
                subscriptions: vec![
                    Subscription {
                        reads: FieldSet::of(&[
                            Field::Model,
                            Field::KeyIdentity,
                            Field::LimiterRequest,
                            Field::DeprecatedKwargs,
                        ]),
                        writes: FieldSet::of(&[Field::OptionalParams, Field::RateLimitStatuses]),
                        may_abort: true,
                        ..Subscription::on(Hook::PreCall)
                    },
                    Subscription {
                        reads: FieldSet::of(&[
                            Field::KeyIdentity,
                            Field::DeprecatedKwargs,
                            Field::DeprecatedResponseObj,
                        ]),
                        writes: FieldSet::of(&[Field::Response]),
                        ..Subscription::on(Hook::PostCallSuccess)
                    },
                    Subscription {
                        awaited: true,
                        ..Subscription::on(Hook::PostCallFailure)
                    },
                    Subscription {
                        reads: FieldSet::of(&[
                            Field::LoggedKeyIdentity,
                            Field::LoggedModelGroup,
                            Field::Usage,
                            Field::Outcome,
                            Field::DeprecatedKwargs,
                        ]),
                        awaited: true,
                        ..Subscription::on(Hook::AttemptEnded)
                    },
                ],
                has_request_state: false,
            },
            engine: Engine { limiter, host },
        }
    }

    async fn pre_call(
        &self,
        input: &HookInput,
        stash: Option<&SharedStash>,
        redis: Option<&RequestRedis>,
        facts: &Facts,
    ) -> HookOutput {
        let Some(Value::String(model)) = input.fields.raw(Field::Model)?.cloned() else {
            return Ok(None);
        };
        let key = key_facts(input)?;
        let fallback = saturation_fallback(input.fields.get::<typed::LimiterRequest>()?);
        let stash = stash.cloned().unwrap_or_default();
        let group = match facts.group() {
            Ok(Some(group)) => group,
            Ok(None) => return Ok(None),
            Err(error) => return raise(error),
        };
        let settings = match facts.settings() {
            Ok(settings) => settings,
            Err(error) => return raise(error),
        };
        let call_type = input.call.as_ref().map(|call| call.kind.call_type.clone());
        let request = Request {
            key: &key,
            call_type: call_type.as_deref(),
            settings: &settings,
            stash: &stash,
            fallback,
            redis,
            facts,
        };
        match self.engine.pre_call(&request, &model, &group).await {
            PreCall::Allow(None) => Ok(None),
            PreCall::Allow(Some(published)) => Ok(Some(Verdict::Patch(
                Patch::default().set(Field::RateLimitStatuses, json!(published)),
            ))),
            PreCall::Refuse(refusal) => Ok(Some(Verdict::Abort(refusal.into_abort()))),
            PreCall::Raise(error) => raise(error),
        }
    }

    async fn post_call_success(&self, input: &HookInput) -> HookOutput {
        let Ok(key) = key_facts(input) else {
            return Ok(None);
        };
        let headers = Engine::post_call_headers(&key);
        let _ = self.engine.host.post_call_success(&headers).await;
        Ok(None)
    }

    async fn post_call_failure(
        &self,
        stash: Option<&SharedStash>,
        redis: Option<&RequestRedis>,
        facts: &Facts,
    ) -> HookOutput {
        let Some(stash) = stash else {
            return Ok(None);
        };
        let released = match facts.settings() {
            Ok(settings) => self.engine.release_admission(&settings, stash, redis).await,
            Err(error) => Err(error),
        };
        if let Err(error) = released {
            let message = format!("{FAILURE_HOOK_ERROR}: {}", error.python_message());
            let _ = self.engine.host.log("error", &message);
        }
        Ok(None)
    }

    async fn attempt_ended(
        &self,
        input: &HookInput,
        stash: Option<&SharedStash>,
        redis: Option<&RequestRedis>,
        facts: &Facts,
    ) -> HookOutput {
        if input
            .call
            .as_ref()
            .is_some_and(|call| call.kind.synchronous)
        {
            return Ok(None);
        }
        let Some(outcome) = input.fields.get::<typed::Outcome>()? else {
            return Ok(None);
        };
        let gaps: AttemptGaps = projected(input)?;
        let group = LoggedGroup::read(input.fields.get::<typed::LoggedModelGroup>()?.as_ref());
        let identity = input.fields.get::<typed::LoggedKeyIdentity>()?;
        let usage = input.fields.get::<typed::Usage>()?;
        let stash = stash.cloned();
        let host = Host { redis, facts };
        match outcome {
            Outcome::Failure => {
                let used = UsageTokens::failed(usage.as_ref());
                self.failure(&gaps, group, identity.as_ref(), used, stash, host)
                    .await
            }
            Outcome::Success => {
                let used = UsageTokens::delivered(usage.as_ref());
                if let Err(error) = self
                    .success(&gaps, group, identity.as_ref(), used, stash, host)
                    .await
                {
                    let message = format!("Error in dynamic rate limiter success event: {error}");
                    let _ = self.engine.host.log("error", &message);
                }
                Ok(None)
            }
            Outcome::FallbackSuccess | Outcome::FallbackFailure => Ok(None),
        }
    }

    async fn success(
        &self,
        gaps: &AttemptGaps,
        group: Result<Option<LoggedGroup>, String>,
        identity: Option<&LoggedKeyIdentity>,
        used: Option<UsageTokens>,
        stash: Option<SharedStash>,
        host: Host<'_>,
    ) -> Result<(), String> {
        let team = match (logged_team(identity), &gaps.settlement_error) {
            (Some(team), None) => team,
            (_, Some(error)) => return Err(error.clone()),
            (None, None) => return Ok(()),
        };
        let group = group?;
        let settings = host
            .facts
            .settings()
            .map_err(|error| error.python_message())?;
        let total = UsageTokens::settled(used.as_ref(), settings.rate_limit_type);
        let settlement = team.settlement(group.as_ref(), gaps);
        self.engine
            .settle_success(
                &settings,
                stash.as_ref(),
                &settlement,
                team.priority.as_deref(),
                total,
                host.redis,
            )
            .await
            .map_err(|error| error.python_message())
    }

    async fn failure(
        &self,
        gaps: &AttemptGaps,
        group: Result<Option<LoggedGroup>, String>,
        identity: Option<&LoggedKeyIdentity>,
        used: Option<UsageTokens>,
        stash: Option<SharedStash>,
        host: Host<'_>,
    ) -> HookOutput {
        let Some(stash) = stash else {
            return Ok(None);
        };
        let Ok(group) = group else {
            return Ok(None);
        };
        let Ok(settings) = host.facts.settings() else {
            return Ok(None);
        };
        let used = UsageTokens::settled(used.as_ref(), settings.rate_limit_type);
        let Ok(holds) = self
            .engine
            .settle_failure(&settings, &stash, group.as_ref(), used, host.redis)
            .await
        else {
            return Ok(None);
        };
        let (Some(team), None) = (logged_team(identity), &gaps.settlement_error) else {
            return Ok(None);
        };
        let settlement = team.settlement(group.as_ref(), gaps);
        self.engine
            .record_team_share(
                &settings,
                Some(&stash),
                group.as_ref().map(|group| group.text.as_str()),
                &settlement,
                &holds,
            )
            .await;
        Ok(None)
    }

    pub async fn handle_with(
        &self,
        input: &HookInput,
        state: Option<&SharedStash>,
        redis: Option<&RequestRedis>,
        facts: &Facts,
    ) -> HookOutput {
        match input.hook {
            Hook::PreCall => self.pre_call(input, state, redis, facts).await,
            Hook::PostCallSuccess => self.post_call_success(input).await,
            Hook::PostCallFailure => self.post_call_failure(state, redis, facts).await,
            Hook::AttemptEnded => self.attempt_ended(input, state, redis, facts).await,
            _ => Ok(None),
        }
    }

    pub fn post_call_headers(&self, key: &KeyFacts) -> Vec<(String, Value)> {
        Engine::post_call_headers(key)
    }

    pub async fn fallback_target_allowed(
        &self,
        stash: &SharedStash,
        key: &KeyFacts,
        model: &str,
        commit: bool,
        redis: Option<&RequestRedis>,
        facts: &Facts,
    ) -> Result<bool, CapacityError> {
        let settings = facts.settings()?;
        let request = Request {
            key,
            call_type: None,
            settings: &settings,
            stash,
            fallback: SaturationFallback::default(),
            redis,
            facts,
        };
        self.engine
            .fallback_target_allowed(&request, model, commit)
            .await
    }

    pub async fn enforce_frame(
        &self,
        connection: Option<&SharedStash>,
        key: &KeyFacts,
        model: &str,
        redis: Option<&RequestRedis>,
        facts: &Facts,
    ) -> Result<Frame, CapacityError> {
        let stash = connection.cloned().unwrap_or_default();
        let settings = facts.settings()?;
        let request = Request {
            key,
            call_type: None,
            settings: &settings,
            stash: &stash,
            fallback: SaturationFallback::default(),
            redis,
            facts,
        };
        self.engine.enforce_frame(&request, model, connection).await
    }
}

impl Callback for ModelCapacity {
    type State = SharedStash;

    fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    fn handle<'a>(
        &'a self,
        input: &'a HookInput,
        state: Option<&'a SharedStash>,
    ) -> BoxFuture<'a, HookOutput> {
        Box::pin(async move {
            self.handle_with(input, state, None, &Facts::default())
                .await
        })
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    fn team(id: Option<&str>, alias: Option<&str>, priority: Option<&str>) -> Option<LoggedTeam> {
        Some(LoggedTeam {
            team_id: id.map(str::to_owned),
            team_alias: alias.map(str::to_owned),
            priority: priority.map(str::to_owned),
            ..LoggedTeam::default()
        })
    }

    #[rstest]
    #[case::stamped(json!({"metadata_kind": "mapping", "team_id": "t", "team_alias": "a", "auth_metadata_priority": "gold"}), team(Some("t"), Some("a"), Some("gold")))]
    #[case::non_text_priority_is_dropped(json!({"metadata_kind": "mapping", "auth_metadata_priority": 7}), team(None, None, None))]
    #[case::missing_metadata(json!({"metadata_kind": "missing"}), team(None, None, None))]
    #[case::int_team_id(json!({"metadata_kind": "mapping", "team_id": 5}), None)]
    #[case::stringified_team_id(json!({"metadata_kind": "mapping", "team_id": {"text": "['t']", "truthy": true, "hashable": false, "integer": false, "type_name": "list", "hash_error": "unhashable type: 'list'"}}), None)]
    #[case::stringified_priority_is_dropped(json!({"metadata_kind": "mapping", "auth_metadata_priority": {"text": "['p']", "truthy": true, "hashable": false, "integer": false, "type_name": "list", "hash_error": "unhashable type: 'list'"}}), team(None, None, None))]
    #[case::rejected_alias(json!({"metadata_kind": "mapping", "rejected": ["team_alias"]}), None)]
    #[case::non_mapping_auth_metadata(json!({"metadata_kind": "mapping", "rejected": ["auth_metadata"]}), None)]
    #[case::non_mapping_metadata(json!({"metadata_kind": "other"}), None)]
    fn logged_metadata_settles_only_what_python_validates(
        #[case] members: Value,
        #[case] expected: Option<LoggedTeam>,
    ) {
        let identity: LoggedKeyIdentity = serde_json::from_value(members).unwrap();
        assert_eq!(logged_team(Some(&identity)), expected);
    }
}
