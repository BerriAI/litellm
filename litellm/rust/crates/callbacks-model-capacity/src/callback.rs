use std::sync::Arc;

use futures_util::future::BoxFuture;
use litellm_callbacks::{
    Activation, Callback, CallbackError, Field, FieldSet, Hook, HookInput, HookOutput,
    LimiterRequest, LoggedKeyIdentity, Manifest, MetadataKind, Outcome, Scalar, Selection,
    Subscription, Verdict, typed,
};
use serde::{Deserialize, de::DeserializeOwned};
use serde_json::Value;

use crate::{
    capacity::{Engine, Frame, PreCall, Request, SaturationFallback, Settlement},
    error::CapacityError,
    model::{KeyFacts, KeyGaps, LoggedGroup, UsageTokens},
    refusal::reraise,
    seam::{CapacityHost, V3Limiter},
    state::SharedStash,
};

pub const NAME: &str = "dynamic_rate_limiter_v3";

#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct AttemptGaps {
    settlement_error: Option<String>,
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
    priority: Option<String>,
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
    let priority = match &identity.auth_metadata_priority {
        Some(Scalar::Text(priority)) => Some(priority.clone()),
        _ => None,
    };
    Some(LoggedTeam {
        team_id,
        team_alias,
        priority,
    })
}

pub struct ModelCapacity {
    manifest: Manifest,
    engine: Engine,
}

fn failed(error: CapacityError) -> CallbackError {
    CallbackError::Failed(error.to_string())
}

fn raise(error: CapacityError) -> HookOutput {
    match error.raised() {
        Some(raised) => Ok(Some(Verdict::Abort(reraise(raised.token, &raised.message)))),
        None => Err(failed(error)),
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
    pub fn new(limiter: Arc<dyn V3Limiter>, host: Arc<dyn CapacityHost>) -> Self {
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
                        writes: FieldSet::of(&[Field::OptionalParams]),
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

    async fn pre_call(&self, input: &HookInput, stash: Option<&SharedStash>) -> HookOutput {
        let Some(Value::String(model)) = input.fields.raw(Field::Model)?.cloned() else {
            return Ok(None);
        };
        let key = key_facts(input)?;
        let fallback = saturation_fallback(input.fields.get::<typed::LimiterRequest>()?);
        let stash = stash.cloned().unwrap_or_default();
        let group = match self.engine.host.group(&model).await {
            Ok(Some(group)) => group,
            Ok(None) => return Ok(None),
            Err(error) => return raise(error),
        };
        let settings = match self.engine.host.settings().await {
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
        };
        match self.engine.pre_call(&request, &model, &group).await {
            PreCall::Allow => Ok(None),
            PreCall::Refuse(refusal) => Ok(Some(Verdict::Abort(refusal.into_abort()))),
            PreCall::Raise(error) => raise(error),
        }
    }

    async fn post_call_success(&self, input: &HookInput) -> HookOutput {
        let Ok(key) = key_facts(input) else {
            return Ok(None);
        };
        let headers = Engine::post_call_headers(&key);
        let _ = self.engine.limiter.post_call_success(&headers).await;
        Ok(None)
    }

    async fn post_call_failure(&self, stash: Option<&SharedStash>) -> HookOutput {
        let Some(stash) = stash else {
            return Ok(None);
        };
        let released = match self.engine.host.settings().await {
            Ok(settings) => self.engine.release_admission(&settings, stash).await,
            Err(error) => Err(error),
        };
        if let Err(error) = released {
            let message = format!("{FAILURE_HOOK_ERROR}: {}", error.python_message());
            let _ = self.engine.host.log("error", &message).await;
        }
        Ok(None)
    }

    async fn attempt_ended(&self, input: &HookInput, stash: Option<&SharedStash>) -> HookOutput {
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
        match outcome {
            Outcome::Failure => {
                let used = UsageTokens::failed(usage.as_ref());
                self.failure(&gaps, group, identity.as_ref(), used, stash)
                    .await
            }
            Outcome::Success => {
                let used = UsageTokens::delivered(usage.as_ref());
                if let Err(error) = self
                    .success(&gaps, group, identity.as_ref(), used, stash)
                    .await
                {
                    let message = format!("Error in dynamic rate limiter success event: {error}");
                    let _ = self.engine.host.log("error", &message).await;
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
    ) -> Result<(), String> {
        let team = match (logged_team(identity), &gaps.settlement_error) {
            (Some(team), None) => team,
            (_, Some(error)) => return Err(error.clone()),
            (None, None) => return Ok(()),
        };
        let group = group?;
        let settings = self
            .engine
            .host
            .settings()
            .await
            .map_err(|error| error.python_message())?;
        let total = UsageTokens::settled(used.as_ref(), settings.rate_limit_type);
        let settlement = Settlement {
            model_group: group.as_ref(),
            team_id: team.team_id.as_deref(),
            team_alias: team.team_alias.as_deref(),
        };
        self.engine
            .settle_success(
                &settings,
                stash.as_ref(),
                &settlement,
                team.priority.as_deref(),
                total,
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
    ) -> HookOutput {
        let Some(stash) = stash else {
            return Ok(None);
        };
        let Ok(group) = group else {
            return Ok(None);
        };
        let Ok(settings) = self.engine.host.settings().await else {
            return Ok(None);
        };
        let used = UsageTokens::settled(used.as_ref(), settings.rate_limit_type);
        let Ok(hold) = self
            .engine
            .settle_failure(&settings, &stash, group.as_ref(), used)
            .await
        else {
            return Ok(None);
        };
        let (Some(team), None) = (logged_team(identity), &gaps.settlement_error) else {
            return Ok(None);
        };
        let settlement = Settlement {
            model_group: group.as_ref(),
            team_id: team.team_id.as_deref(),
            team_alias: team.team_alias.as_deref(),
        };
        self.engine
            .record_team_share(
                &settings,
                Some(&stash),
                group.as_ref().map(|group| group.text.as_str()),
                &settlement,
                hold.as_deref(),
            )
            .await;
        Ok(None)
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
    ) -> Result<bool, CapacityError> {
        let settings = self.engine.host.settings().await?;
        let request = Request {
            key,
            call_type: None,
            settings: &settings,
            stash,
            fallback: SaturationFallback::default(),
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
    ) -> Result<Frame, CapacityError> {
        let stash = connection.cloned().unwrap_or_default();
        let settings = self.engine.host.settings().await?;
        let request = Request {
            key,
            call_type: None,
            settings: &settings,
            stash: &stash,
            fallback: SaturationFallback::default(),
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
            match input.hook {
                Hook::PreCall => self.pre_call(input, state).await,
                Hook::PostCallSuccess => self.post_call_success(input).await,
                Hook::PostCallFailure => self.post_call_failure(state).await,
                Hook::AttemptEnded => self.attempt_ended(input, state).await,
                _ => Ok(None),
            }
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
