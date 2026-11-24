//! How a callback reads and writes a call's record by contract field, the Rust counterpart of
//! the proxy's `v2_proxy_fields.py`. A field the gateway cannot work out yet is left out of what
//! a callback is handed, and one it has no place for is not written.

use std::time::SystemTime;

use litellm_callbacks::{
    AttemptTiming, CallFacts, CallIds, CallKind, CallTarget, CallTiming, Deployment, Field,
    FieldSet, FieldValues, Hook, HookInput, KeyIdentity, LoggedKeyIdentity, LoggedModelGroup,
    MetadataKind, ModelInfo, Moment, Outcome, PatchOp, Scalar, SpendLogId, Usage, UsageRecord,
};
use litellm_gateway_auth::Authenticated;
use litellm_gateway_auth_core::{Caller, StoredKey};
use litellm_gateway_inference::{
    CallType,
    record::{CallState, Charged, Ended, HeadSent, unix},
};
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
/// owner, and the master key by Python's alias for it and the proxy admin. Native keys name no
/// team, organization, project or end user.
pub(super) fn charged(identity: &Authenticated<NativeRows>) -> Charged {
    match &**identity {
        Caller::MasterKey => Charged {
            key_hash: Some(MASTER_KEY_ALIAS.to_owned()),
            key_alias: None,
            user_id: Some(PROXY_ADMIN_NAME.to_owned()),
        },
        Caller::Key(key) => Charged {
            key_hash: identity.hash().map(ToString::to_string),
            key_alias: key.key.alias().map(str::to_owned),
            user_id: key.key.record().user_id.clone(),
        },
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

fn key_identity(charged: &Charged) -> KeyIdentity {
    KeyIdentity {
        key_hash: charged.key_hash.as_deref().map(Scalar::from),
        api_key: charged.key_hash.as_deref().map(Scalar::from),
        key_alias: charged.key_alias.as_deref().map(Scalar::from),
        user_id: charged.user_id.as_deref().map(Scalar::from),
        ..KeyIdentity::default()
    }
}

fn logged_key_identity(entries: &CallState) -> LoggedKeyIdentity {
    let charged = entries.charged();
    LoggedKeyIdentity {
        metadata_kind: Some(MetadataKind::Mapping),
        key_hash: charged.key_hash.as_deref().map(Scalar::from),
        key_alias: charged.key_alias.as_deref().map(Scalar::from),
        user_id: charged.user_id.as_deref().map(Scalar::from),
        request_route: entries.call_type().map(route).map(Scalar::from),
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
            .chain(entries.call_id())
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
        api_call_start: entries.api_call_start().and_then(moment),
        completion_start: completion_start.and_then(moment),
        completion_start_present: Some(completion_start.is_some()),
        logged_start: moment(entries.start()),
        logged_end: entries.end().and_then(moment),
        ..AttemptTiming::default()
    }
}

/// The spend log callback's input for a served call. A failed call has none yet.
fn spend_log_input(entries: &CallState) -> Option<Value> {
    let call_type = entries.call_type()?;
    let provider = provider(entries);
    spend_log_input::success(&Served {
        call_type,
        route: route(call_type),
        charged: entries.charged(),
        call_id: entries.call_id(),
        trace_id: entries.trace_id(),
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
        Field::RateLimitStatuses => entries.rate_limit_statuses().cloned(),
        Field::ResponseHeaders => Some(Value::Object(entries.response_headers().clone())),
        Field::Outcome => record(outcome(entries.ended()?)),
        Field::Usage => record(usage(entries.served()?.usage.as_ref()?)),
        Field::KeyIdentity => record(key_identity(entries.charged())),
        Field::LoggedKeyIdentity => record(logged_key_identity(entries)),
        Field::Deployment => record(served_deployment(entries)?),
        Field::AttemptTiming => record(timing(entries)),
        Field::SpendLogId => record(spend_log_id(entries)),
        Field::SpendLogInput => spend_log_input(entries),
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
            call_id: entries.call_id().unwrap_or_default().to_owned(),
            ..CallIds::default()
        },
        kind: CallKind {
            call_type: entries
                .call_type()
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
            provider_start: entries.api_call_start().map(unix),
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
        } => entries.set_rate_limit_statuses(value),
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
    use litellm_gateway_inference::{
        ServedCall,
        record::{self, CallRecord},
    };
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    /// Runs `look` on a fresh record of a Messages call holding a body that names `claude`.
    fn on_a_call<R>(look: impl FnOnce(&mut CallState) -> R) -> R {
        let scoped = || {
            record::write(|entries| {
                entries.serve_as(CallType::AnthropicMessages);
                entries.hold(Map::from_iter([("model".to_owned(), json!("claude"))]));
                look(entries)
            })
            .unwrap()
        };
        futures_util::FutureExt::now_or_never(record::scope(CallRecord::new(), async { scoped() }))
            .unwrap()
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
            entries.name_call("call-1".to_owned());
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
}
