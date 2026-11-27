//! How a callback reads and writes a call's record by contract field, the Rust counterpart of
//! the proxy's `v2_proxy_fields.py`. A field the gateway cannot work out yet is left out of what
//! a callback is handed, and one it has no place for is not written.

use std::time::SystemTime;

use litellm_callbacks::{
    AttemptTiming, CallFacts, CallIds, CallKind, CallTarget, CallTiming, Cost, Deployment, Field,
    FieldSet, FieldValues, Hook, HookInput, KeyIdentity, LoggedKeyIdentity, LoggedModelGroup,
    MetadataKind, ModelInfo, Moment, Outcome, PatchOp, Scalar, SpendLogId, Usage, UsageRecord,
};
use litellm_gateway_auth::Authenticated;
use litellm_gateway_auth_core::{
    Affiliation, Caller, Credential, Named, StoredKey, Team, record::Label,
};
use litellm_gateway_inference::{
    CallType,
    record::{CallState, Charged, Ended, HeadSent, unix},
};
use litellm_host::interceptors::Cost as Settled;
use litellm_llms_types::formats::chat_completions::ChatCompletionsUsage;
use litellm_python_compat::datetime::UtcDateTime;
use serde::Serialize;
use serde_json::{Map, Value};

use super::{
    parallel_request_limiter::{MASTER_KEY_ALIAS, PROXY_ADMIN_NAME},
    spend_log_input::{self, Served},
};
use crate::host::NativeRows;

const MODEL: &str = "model";

/// A patch operation the gateway has no place to write.
#[derive(Debug, PartialEq, Eq, thiserror::Error)]
pub enum Unwritable {
    #[error("the gateway has no place to write {0:?} yet")]
    Field(Field),
    #[error("{0:?} is replaced whole, never merged into")]
    Merge(Field),
    #[error("{0:?} takes keys merged into it, never a whole value")]
    Set(Field),
    #[error("a model is a name, never {0}")]
    Model(Value),
    #[error("{0:?} is part of the request body, which the route already has")]
    Routed(Field),
    #[error(transparent)]
    HeadSent(#[from] HeadSent),
}

/// Who `identity` is charged as: a key by the hash its row is stored under, its alias and its
/// owner, the master key by Python's alias for it and the proxy admin, and a JWT by the user it
/// names, as Python's `UserAPIKeyAuth` for a JWT holds no key, each with the team, organization,
/// project and end user it names. Only a stored row gives an alias.
pub(super) fn charged(identity: &Authenticated<NativeRows>) -> Charged {
    let Caller {
        credential,
        affiliation:
            Affiliation {
                holder,
                org,
                project,
            },
        end_user,
    } = &**identity;
    let (key_hash, key_alias, user_id) = match credential {
        Credential::MasterKey => (
            Some(MASTER_KEY_ALIAS.to_owned()),
            None,
            Some(PROXY_ADMIN_NAME.to_owned()),
        ),
        Credential::Key { key, .. } => (
            identity.hash().map(ToString::to_string),
            key.alias().map(str::to_owned),
            key.record().user_id.clone(),
        ),
        Credential::Jwt { .. } => (None, None, holder.user().map(|user| user.id().to_owned())),
    };
    Charged {
        key_hash,
        key_alias,
        user_id,
        team: holder.team().map(Team::label),
        org: org.as_ref().map(Named::label),
        project: project.as_ref().map(Named::label),
        end_user_id: end_user
            .as_ref()
            .and_then(|end_user| end_user.end_user_id.clone()),
    }
}

/// The route a call came in on, as Python's `user_api_key_request_route` names it.
fn route(call_type: CallType) -> &'static str {
    match call_type {
        CallType::AnthropicMessages => "/v1/messages",
        CallType::Aocr => "/v1/ocr",
    }
}

/// `at` as the UTC `datetime` Python logs it as.
fn moment(at: SystemTime) -> Option<Moment> {
    let utc = UtcDateTime::from_system_time(at);
    Some(Moment {
        year: i32::try_from(utc.year).ok()?,
        month: u8::try_from(utc.month).ok()?,
        day: u8::try_from(utc.day).ok()?,
        hour: u8::try_from(utc.hour).ok()?,
        minute: u8::try_from(utc.minute).ok()?,
        second: u8::try_from(utc.second).ok()?,
        microsecond: utc.microsecond,
        utc_offset_seconds: Some(0),
    })
}

fn record(value: impl Serialize) -> Option<Value> {
    serde_json::to_value(value).ok()
}

fn count(tokens: u64) -> Option<Scalar> {
    Some(Scalar::Int(i128::from(tokens)))
}

/// A served call's usage as the success log's `response_obj` carries it.
fn usage(used: &ChatCompletionsUsage) -> Usage {
    Usage {
        response: Some(UsageRecord {
            prompt_tokens: count(used.prompt_tokens),
            completion_tokens: count(used.completion_tokens),
            total_tokens: count(used.total_tokens),
            cached_tokens: used
                .prompt_tokens_details
                .as_ref()
                .and_then(|details| details.cached_tokens)
                .and_then(count),
            ..UsageRecord::default()
        }),
        ..Usage::default()
    }
}

fn outcome(ended: Ended) -> Outcome {
    match ended {
        Ended::Served => Outcome::Success,
        Ended::Failed => Outcome::Failure,
    }
}

fn deployment(entries: &CallState) -> Option<&litellm_router_core::Deployment> {
    entries.served()?.deployment.as_deref()
}

/// The provider the served deployment was called through, as Python's `custom_llm_provider`
/// names it.
fn provider(entries: &CallState) -> Option<String> {
    let deployment = deployment(entries)?;
    deployment
        .resolved_provider
        .clone()
        .or_else(|| deployment.params.custom_llm_provider.clone())
        .or_else(|| {
            deployment
                .params
                .model
                .split_once('/')
                .map(|(provider, _)| provider.to_owned())
        })
}

fn id(label: Option<&Label>) -> Option<Scalar> {
    label.map(|label| Scalar::from(label.id.as_str()))
}

fn alias(label: Option<&Label>) -> Option<Scalar> {
    label
        .and_then(|label| label.alias.as_deref())
        .map(Scalar::from)
}

fn key_identity(charged: &Charged) -> KeyIdentity {
    KeyIdentity {
        key_hash: charged.key_hash.as_deref().map(Scalar::from),
        api_key: charged.key_hash.as_deref().map(Scalar::from),
        key_alias: charged.key_alias.as_deref().map(Scalar::from),
        user_id: charged.user_id.as_deref().map(Scalar::from),
        team_id: id(charged.team.as_ref()),
        team_alias: alias(charged.team.as_ref()),
        org_id: id(charged.org.as_ref()),
        org_alias: alias(charged.org.as_ref()),
        project_id: id(charged.project.as_ref()),
        project_alias: alias(charged.project.as_ref()),
        end_user_id: charged.end_user_id.as_deref().map(Scalar::from),
        ..KeyIdentity::default()
    }
}

fn logged_key_identity(entries: &CallState) -> LoggedKeyIdentity {
    let charged = &entries.charged;
    LoggedKeyIdentity {
        metadata_kind: Some(MetadataKind::Mapping),
        key_hash: charged.key_hash.as_deref().map(Scalar::from),
        key_alias: charged.key_alias.as_deref().map(Scalar::from),
        user_id: charged.user_id.as_deref().map(Scalar::from),
        team_id: id(charged.team.as_ref()),
        team_alias: alias(charged.team.as_ref()),
        org_id: id(charged.org.as_ref()),
        org_alias: alias(charged.org.as_ref()),
        project_id: id(charged.project.as_ref()),
        project_alias: alias(charged.project.as_ref()),
        end_user_id: charged.end_user_id.as_deref().map(Scalar::from),
        cost_tracking_end_user_id: charged.end_user_id.as_deref().map(Scalar::from),
        request_route: entries.call_type.map(route).map(Scalar::from),
        ..LoggedKeyIdentity::default()
    }
}

fn served_deployment(entries: &CallState) -> Option<Deployment> {
    let deployment = deployment(entries)?;
    let rate = |name: &str| match deployment.raw_params.get(name)? {
        Value::Number(number) => Some(number.as_i64().map_or_else(
            || Scalar::Float(number.as_f64().unwrap_or(0.0)),
            |whole| Scalar::Int(i128::from(whole)),
        )),
        Value::String(spelled) => Some(Scalar::Text(spelled.clone())),
        _ => None,
    };
    Some(Deployment {
        model_info: Some(ModelInfo {
            id: Some(Scalar::from(deployment.id.as_str())),
            tpm: rate("tpm"),
            rpm: rate("rpm"),
            ..ModelInfo::default()
        }),
        provider: provider(entries).map(Scalar::from),
        tpm: rate("tpm"),
        rpm: rate("rpm"),
        params_present: Some(true),
        ..Deployment::default()
    })
}

/// The id the call's spend log row is written under, as `get_spend_logs_id` picks it: the
/// provider's id for the answer, else the call id.
fn spend_log_id(entries: &CallState) -> SpendLogId {
    let answered = entries
        .served()
        .and_then(|served| served.response_id.as_deref());
    SpendLogId {
        id: answered
            .into_iter()
            .chain(entries.call_id.as_deref())
            .find(|id| !id.is_empty())
            .map(str::to_owned),
        ..SpendLogId::default()
    }
}

/// When a stream's first event arrived, else when the call ended, as Python defaults
/// `completion_start_time` to the end for a complete answer.
fn completion_start(entries: &CallState) -> Option<SystemTime> {
    entries
        .served()
        .and_then(|served| served.first_event)
        .or(entries.end())
}

/// The call's times.
fn timing(entries: &CallState) -> AttemptTiming {
    let completion_start = completion_start(entries);
    AttemptTiming {
        start: moment(entries.start()),
        end: entries.end().and_then(moment),
        api_call_start: entries.api_call_start.and_then(moment),
        completion_start: completion_start.and_then(moment),
        completion_start_present: Some(completion_start.is_some()),
        logged_start: moment(entries.start()),
        logged_end: entries.end().and_then(moment),
        ..AttemptTiming::default()
    }
}

/// What the served attempt's route settled the call's cost as. A cache hit cost nothing, and a
/// stream's price is read off its meter once the stream ended. `None` for a call the route left
/// unpriced.
fn cost(entries: &CallState) -> Option<Cost> {
    let amount = match entries.attempt.as_ref()?.cost.as_ref()? {
        Settled::Reported { amount } => Scalar::Float(amount.as_f64()?),
        Settled::Priced(priced) => Scalar::Float(priced.receipt.total),
        Settled::Metered(meter) => Scalar::Float(meter.settled()?.receipt.total),
        Settled::CacheHit => Scalar::Float(0.0),
        Settled::Deferred => return None,
    };
    Some(Cost {
        response_cost: Some(amount),
        ..Cost::default()
    })
}

/// The spend log callback's input for a served call. A failed call has none yet.
fn spend_log_input(entries: &CallState) -> Option<Value> {
    let call_type = entries.call_type?;
    let provider = provider(entries);
    spend_log_input::success(&Served {
        call_type,
        route: route(call_type),
        charged: &entries.charged,
        call_id: entries.call_id.as_deref(),
        trace_id: &entries.trace_id,
        stream: entries.stream(),
        start: entries.start(),
        end: entries.end()?,
        provider: provider.as_deref(),
        call: entries.served()?,
    })
}

fn read(entries: &CallState, field: Field) -> Option<Value> {
    match field {
        Field::Model => entries.model().map(Value::from),
        Field::RequestBody => entries.body().cloned().map(Value::Object),
        Field::RateLimitStatuses => entries.rate_limit_statuses.clone(),
        Field::ResponseHeaders => Some(Value::Object(entries.response_headers().clone())),
        Field::Outcome => record(outcome(entries.ended()?)),
        Field::Usage => record(usage(entries.served()?.usage.as_ref()?)),
        Field::KeyIdentity => record(key_identity(&entries.charged)),
        Field::LoggedKeyIdentity => record(logged_key_identity(entries)),
        Field::Deployment => record(served_deployment(entries)?),
        Field::AttemptTiming => record(timing(entries)),
        Field::SpendLogId => record(spend_log_id(entries)),
        Field::SpendLogInput => spend_log_input(entries),
        Field::Cost => record(cost(entries)?),
        Field::LoggedModelGroup => record(LoggedModelGroup {
            value: Some(Scalar::Text(entries.served()?.model_group.clone())),
            ..LoggedModelGroup::default()
        }),
        _ => None,
    }
}

fn facts(entries: &CallState) -> CallFacts {
    let deployment = deployment(entries);
    CallFacts {
        ids: CallIds {
            call_id: entries.call_id.clone().unwrap_or_default(),
            ..CallIds::default()
        },
        kind: CallKind {
            call_type: entries
                .call_type
                .map(<&str>::from)
                .unwrap_or_default()
                .to_owned(),
            stream: entries.stream(),
            ..CallKind::default()
        },
        target: CallTarget {
            model: entries.model().unwrap_or_default().to_owned(),
            model_group: entries.served().map(|served| served.model_group.clone()),
            provider: provider(entries),
            api_base: deployment.and_then(|deployment| deployment.params.api_base.clone()),
            deployment_id: deployment.map(|deployment| deployment.id.clone()),
            ..CallTarget::default()
        },
        timing: CallTiming {
            start: unix(entries.start()),
            provider_start: entries.api_call_start.map(unix),
            first_token: completion_start(entries).map(unix),
            end: entries.end().map(unix),
        },
        ..CallFacts::default()
    }
}

/// What a callback that reads `reads` is handed on `hook`.
pub fn input(entries: &CallState, hook: Hook, reads: FieldSet) -> HookInput {
    let fields = reads
        .iter()
        .filter(|field| !field.by_reference())
        .filter_map(|field| Some((field, read(entries, field)?)))
        .fold(FieldValues::new(reads), |fields, (field, value)| {
            fields.with(field, value)
        });
    HookInput {
        hook,
        fields,
        call: Some(facts(entries)),
    }
}

/// The held body for a write to `field`.
fn held(entries: &mut CallState, field: Field) -> Result<&mut Map<String, Value>, Unwritable> {
    entries.body_mut().ok_or(Unwritable::Routed(field))
}

/// Writes one operation of a callback's patch: the request's own params into the body, as the
/// proxy's hooks change `data`, and the rest where a later callback or the route reads it.
pub fn write(entries: &mut CallState, op: PatchOp) -> Result<(), Unwritable> {
    match op {
        PatchOp::Merge {
            field: Field::OptionalParams,
            keys,
        } => held(entries, Field::OptionalParams)?.extend(keys),
        PatchOp::Merge {
            field: Field::ResponseHeaders,
            keys,
        } => {
            for (name, value) in keys {
                entries.add_header(name, value)?;
            }
        }
        PatchOp::Set {
            field: Field::Model,
            value: Value::String(model),
        } => {
            held(entries, Field::Model)?.insert(MODEL.to_owned(), Value::from(model));
        }
        PatchOp::Set {
            field: Field::Model,
            value,
        } => return Err(Unwritable::Model(value)),
        PatchOp::Set {
            field: Field::RateLimitStatuses,
            value,
        } => entries.rate_limit_statuses = Some(value),
        PatchOp::Set {
            field: field @ (Field::OptionalParams | Field::ResponseHeaders),
            ..
        } => return Err(Unwritable::Set(field)),
        PatchOp::Merge {
            field: field @ (Field::Model | Field::RateLimitStatuses),
            ..
        } => return Err(Unwritable::Merge(field)),
        PatchOp::Set { field, .. } | PatchOp::Merge { field, .. } => {
            return Err(Unwritable::Field(field));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use litellm_config::LitellmRole;
    use litellm_gateway_auth_core::{
        Holder, JwtActor, JwtStamp, TeamCopy,
        record::{EndUserRecord, OrgRecord, ProjectRecord, TeamRecord, UserRecord},
    };
    use litellm_gateway_inference::{
        ServedCall,
        record::{self, CallRecord},
    };
    use rstest::rstest;
    use serde_json::json;

    use super::*;
    use crate::host::auth::UserRow;

    /// Runs `look` on a fresh record of a Messages call holding a body that names `claude`.
    fn on_a_call<R>(look: impl FnOnce(&mut CallState) -> R) -> R {
        let scoped = || {
            record::write(|entries| {
                entries.call_type = Some(CallType::AnthropicMessages);
                entries.hold(Map::from_iter([("model".to_owned(), json!("claude"))]));
                look(entries)
            })
            .unwrap()
        };
        futures_util::FutureExt::now_or_never(record::scope(CallRecord::new(), async { scoped() }))
            .unwrap()
    }

    fn priced(total: f64) -> litellm_host::interceptors::PricedCost {
        litellm_host::interceptors::PricedCost {
            receipt: litellm_cost::receipt::Receipt::of_total(total),
            service_tier: None,
            data_residency: None,
        }
    }

    fn settled_meter(total: Option<f64>) -> litellm_host::interceptors::StreamMeter {
        let meter = litellm_host::interceptors::StreamMeter::default();
        if let Some(total) = total {
            meter.settle(priced(total));
        }
        meter
    }

    #[rstest]
    #[case::priced_by_the_route(Some(Settled::Priced(Box::new(priced(0.25)))), Some(0.25))]
    #[case::a_settled_stream(Some(Settled::Metered(settled_meter(Some(0.5)))), Some(0.5))]
    #[case::a_stream_that_ended_before_its_usage(Some(Settled::Metered(settled_meter(None))), None)]
    #[case::reported_by_the_provider(
        Some(Settled::Reported { amount: serde_json::Number::from_f64(1.5).unwrap() }),
        Some(1.5)
    )]
    #[case::a_cache_hit(Some(Settled::CacheHit), Some(0.0))]
    #[case::left_to_the_host(Some(Settled::Deferred), None)]
    #[case::no_attempt_settled(None, None)]
    fn a_callback_reads_the_cost_the_served_attempts_route_settled(
        #[case] settled: Option<Settled>,
        #[case] expected: Option<f64>,
    ) {
        let read = on_a_call(|entries| {
            entries.attempt = Some(litellm_gateway_inference::record::AttemptRecord {
                cost: settled,
                ..Default::default()
            });
            read(entries, Field::Cost)
        });
        assert_eq!(
            read.as_ref()
                .and_then(|cost| cost["response_cost"].as_f64()),
            expected
        );
    }

    #[rstest]
    fn a_failed_attempts_cost_is_never_the_calls() {
        let read = on_a_call(|entries| {
            entries
                .failed_attempts
                .push(litellm_gateway_inference::record::AttemptRecord {
                    cost: Some(Settled::Priced(Box::new(priced(0.25)))),
                    ..Default::default()
                });
            read(entries, Field::Cost)
        });
        assert_eq!(read, None);
    }

    fn keys() -> Map<String, Value> {
        Map::from_iter([("max_tokens".to_owned(), json!(5))])
    }

    #[rstest]
    #[case::a_field_with_no_place(
        PatchOp::Set { field: Field::Metadata, value: json!({}) },
        Unwritable::Field(Field::Metadata)
    )]
    #[case::params_replaced_whole(
        PatchOp::Set { field: Field::OptionalParams, value: json!({}) },
        Unwritable::Set(Field::OptionalParams)
    )]
    #[case::a_model_that_is_no_name(
        PatchOp::Set { field: Field::Model, value: json!(7) },
        Unwritable::Model(json!(7))
    )]
    #[case::keys_merged_into_a_model(
        PatchOp::Merge { field: Field::Model, keys: keys() },
        Unwritable::Merge(Field::Model)
    )]
    fn a_write_the_gateway_has_no_place_for_leaves_the_call_as_it_was(
        #[case] op: PatchOp,
        #[case] refused: Unwritable,
    ) {
        assert_eq!(
            on_a_call(|entries| (write(entries, op), entries.release())),
            (
                Err(refused),
                Map::from_iter([("model".to_owned(), json!("claude"))])
            )
        );
    }

    /// What a callback that reads the model is told of the call: the field, and the model and
    /// the stream flag among the call's facts.
    fn told(entries: &CallState, hook: Hook) -> (Option<Value>, String, bool) {
        let input = input(entries, hook, FieldSet::of(&[Field::Model]));
        let call = input.call.unwrap();
        (
            input.fields.raw(Field::Model).unwrap().cloned(),
            call.target.model,
            call.kind.stream,
        )
    }

    #[rstest]
    #[case::the_model_set(
        PatchOp::Set { field: Field::Model, value: json!("claude-fallback") },
        (Some(json!("claude-fallback")), "claude-fallback".to_owned(), false)
    )]
    #[case::params_that_name_a_model_and_a_stream(
        PatchOp::Merge {
            field: Field::OptionalParams,
            keys: Map::from_iter([
                ("model".to_owned(), json!("claude-fallback")),
                ("stream".to_owned(), json!(true)),
            ]),
        },
        (Some(json!("claude-fallback")), "claude-fallback".to_owned(), true)
    )]
    fn what_a_callback_changes_of_the_body_is_what_the_next_one_and_the_route_see(
        #[case] op: PatchOp,
        #[case] seen: (Option<Value>, String, bool),
    ) {
        let (on_the_same_hook, model, after) = on_a_call(|entries| {
            write(entries, op).unwrap();
            let on_the_same_hook = told(entries, Hook::PreCall);
            let body = entries.release();
            (
                on_the_same_hook,
                body.get(MODEL).cloned(),
                told(entries, Hook::AttemptEnded),
            )
        });
        assert_eq!(
            (on_the_same_hook, model, after),
            (seen.clone(), seen.0.clone(), seen)
        );
    }

    fn served(response_id: Option<&str>) -> ServedCall {
        ServedCall {
            model_group: "claude".to_owned(),
            deployment: None,
            response_id: response_id.map(str::to_owned),
            usage: None,
            first_event: None,
        }
    }

    #[rstest]
    #[case::the_answers_own_id(Some(served(Some("msg_1"))), Some("msg_1"))]
    #[case::the_call_id_for_an_answer_that_names_none(Some(served(None)), Some("call-1"))]
    #[case::the_call_id_for_an_answer_whose_id_is_empty(Some(served(Some(""))), Some("call-1"))]
    #[case::the_call_id_for_a_failure(None, Some("call-1"))]
    fn a_spend_log_row_is_written_under_the_answers_id_or_else_the_call_id(
        #[case] served: Option<ServedCall>,
        #[case] id: Option<&str>,
    ) {
        let logged = on_a_call(|entries| {
            entries.call_id = Some("call-1".to_owned());
            match served {
                Some(served) => entries.serve(served),
                None => entries.fail(),
            }
            spend_log_id(entries).id
        });
        assert_eq!(logged.as_deref(), id);
    }

    #[rstest]
    #[case::a_served_call(Some(served(Some("msg_1"))), Some("success"))]
    #[case::a_failed_call(None, None)]
    fn only_a_served_call_is_handed_a_spend_log_input(
        #[case] served: Option<ServedCall>,
        #[case] kind: Option<&str>,
    ) {
        let input = on_a_call(|entries| {
            entries.release();
            match served {
                Some(served) => entries.serve(served),
                None => entries.fail(),
            }
            input(
                entries,
                Hook::AttemptEnded,
                FieldSet::of(&[Field::SpendLogInput]),
            )
        });
        let read = input.fields.raw(Field::SpendLogInput).unwrap();
        assert_eq!(read.and_then(|input| input["kind"].as_str()), kind);
    }

    #[rstest]
    fn a_body_the_route_already_has_is_never_written() {
        assert_eq!(
            on_a_call(|entries| {
                entries.release();
                write(
                    entries,
                    PatchOp::Merge {
                        field: Field::OptionalParams,
                        keys: keys(),
                    },
                )
            }),
            Err(Unwritable::Routed(Field::OptionalParams))
        );
    }

    #[rstest]
    fn a_patch_to_the_headers_of_a_response_whose_head_was_written_is_refused() {
        assert_eq!(
            on_a_call(|entries| {
                entries.close_head();
                (
                    write(
                        entries,
                        PatchOp::Merge {
                            field: Field::ResponseHeaders,
                            keys: Map::from_iter([("x-late".to_owned(), json!("yes"))]),
                        },
                    ),
                    entries.response_headers().is_empty(),
                )
            }),
            (Err(Unwritable::HeadSent(HeadSent)), true)
        );
    }

    fn member() -> Authenticated<NativeRows> {
        crate::test_support::key(
            "stored",
            Some("ci"),
            Affiliation {
                holder: Holder::Member {
                    user: Named::Stored(UserRow::from(UserRecord {
                        user_id: "u1".to_owned(),
                        ..UserRecord::default()
                    })),
                    team: Team::Row(TeamRecord {
                        team_id: "t1".to_owned(),
                        team_alias: Some("team one".to_owned()),
                        ..TeamRecord::default()
                    }),
                    membership: None,
                },
                org: Some(Named::Stored(OrgRecord {
                    organization_id: "o1".to_owned(),
                    organization_alias: Some("org one".to_owned()),
                    ..OrgRecord::default()
                })),
                project: Some(Named::Stored(ProjectRecord {
                    project_id: "p1".to_owned(),
                    project_alias: Some("project one".to_owned()),
                    ..ProjectRecord::default()
                })),
            },
            EndUserRecord {
                end_user_id: Some("e1".to_owned()),
                ..EndUserRecord::default()
            },
        )
    }

    fn unstored() -> Authenticated<NativeRows> {
        crate::test_support::key(
            "stored",
            None,
            Affiliation {
                holder: Holder::Team(Team::KeyCopy(TeamCopy {
                    team_id: "t1".to_owned(),
                    object_permission: None,
                })),
                org: Some(Named::Unstored("o1".to_owned())),
                project: Some(Named::Unstored("p1".to_owned())),
            },
            EndUserRecord::default(),
        )
    }

    fn jwt(holder: Holder<NativeRows>) -> Authenticated<NativeRows> {
        Authenticated::from(Caller {
            credential: Credential::Jwt {
                jwt: JwtStamp {
                    claims: Map::new(),
                    user_email: None,
                },
                actor: JwtActor::User {
                    role: LitellmRole::InternalUser,
                },
            },
            affiliation: Affiliation {
                holder,
                org: None,
                project: None,
            },
            end_user: None,
        })
    }

    fn jwt_user() -> Named<UserRow> {
        Named::Stored(UserRow::from(UserRecord {
            user_id: "u1".to_owned(),
            ..UserRecord::default()
        }))
    }

    fn jwt_team() -> Team<NativeRows> {
        Team::Row(TeamRecord {
            team_id: "t1".to_owned(),
            team_alias: Some("team one".to_owned()),
            ..TeamRecord::default()
        })
    }

    fn master_for(end_user_id: &str) -> Authenticated<NativeRows> {
        Authenticated::from(Caller::master_key(Some(EndUserRecord {
            end_user_id: Some(end_user_id.to_owned()),
            ..EndUserRecord::default()
        })))
    }

    #[rstest]
    #[case::a_member_of_a_team_in_an_org_and_a_project(
        member(),
        Charged {
            key_hash: Some("stored".to_owned()),
            key_alias: Some("ci".to_owned()),
            team: Some(Label {
                id: "t1".to_owned(),
                alias: Some("team one".to_owned()),
            }),
            org: Some(Label {
                id: "o1".to_owned(),
                alias: Some("org one".to_owned()),
            }),
            project: Some(Label {
                id: "p1".to_owned(),
                alias: Some("project one".to_owned()),
            }),
            end_user_id: Some("e1".to_owned()),
            ..Charged::default()
        }
    )]
    #[case::names_no_row_stands_behind(
        unstored(),
        Charged {
            key_hash: Some("stored".to_owned()),
            team: Some(Label {
                id: "t1".to_owned(),
                alias: None,
            }),
            org: Some(Label {
                id: "o1".to_owned(),
                alias: None,
            }),
            project: Some(Label {
                id: "p1".to_owned(),
                alias: None,
            }),
            ..Charged::default()
        }
    )]
    #[case::the_master_key_for_an_end_user(
        master_for("e1"),
        Charged {
            key_hash: Some(MASTER_KEY_ALIAS.to_owned()),
            user_id: Some(PROXY_ADMIN_NAME.to_owned()),
            end_user_id: Some("e1".to_owned()),
            ..Charged::default()
        }
    )]
    #[case::a_jwt_that_names_a_user(
        jwt(Holder::User(jwt_user())),
        Charged {
            user_id: Some("u1".to_owned()),
            ..Charged::default()
        }
    )]
    #[case::a_jwt_whose_user_belongs_to_its_team(
        jwt(Holder::Member {
            user: jwt_user(),
            team: jwt_team(),
            membership: None,
        }),
        Charged {
            user_id: Some("u1".to_owned()),
            team: Some(Label {
                id: "t1".to_owned(),
                alias: Some("team one".to_owned()),
            }),
            ..Charged::default()
        }
    )]
    fn a_call_is_charged_to_everything_its_caller_belongs_to(
        #[case] identity: Authenticated<NativeRows>,
        #[case] expected: Charged,
    ) {
        assert_eq!(charged(&identity), expected);
    }

    #[rstest]
    fn callbacks_read_the_team_org_project_and_end_user_a_call_is_charged_to() {
        let input = on_a_call(|entries| {
            entries.charged = charged(&member());
            input(
                entries,
                Hook::AttemptEnded,
                FieldSet::of(&[Field::KeyIdentity, Field::LoggedKeyIdentity]),
            )
        });
        let read = |field| input.fields.raw(field).unwrap().unwrap().clone();
        let key: KeyIdentity = serde_json::from_value(read(Field::KeyIdentity)).unwrap();
        let logged: LoggedKeyIdentity =
            serde_json::from_value(read(Field::LoggedKeyIdentity)).unwrap();
        let named = |names: [&Option<Scalar>; 7]| names.map(Clone::clone);
        let expected = ["t1", "team one", "o1", "org one", "p1", "project one", "e1"]
            .map(|name| Some(Scalar::from(name)));
        assert_eq!(
            (
                named([
                    &key.team_id,
                    &key.team_alias,
                    &key.org_id,
                    &key.org_alias,
                    &key.project_id,
                    &key.project_alias,
                    &key.end_user_id,
                ]),
                named([
                    &logged.team_id,
                    &logged.team_alias,
                    &logged.org_id,
                    &logged.org_alias,
                    &logged.project_id,
                    &logged.project_alias,
                    &logged.end_user_id,
                ]),
                logged.cost_tracking_end_user_id,
            ),
            (expected.clone(), expected, Some(Scalar::from("e1")))
        );
    }
}
