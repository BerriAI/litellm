//! How a callback reads and writes a call's record by contract field, the Rust counterpart of
//! the proxy's `v2_proxy_fields.py`. A field the gateway cannot work out yet is left out of what
//! a callback is handed, and one it has no place for is not written.

use litellm_callbacks::{
    CallFacts, CallKind, CallTarget, CallTiming, Field, FieldSet, FieldValues, Hook, HookInput,
    LoggedModelGroup, Outcome, PatchOp, Scalar, Usage, UsageRecord,
};
use litellm_gateway_inference::record::{Ended, Entries};
use litellm_llms_types::formats::chat_completions::ChatCompletionsUsage;
use serde::Serialize;
use serde_json::{Map, Value};

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

fn read(entries: &Entries, field: Field) -> Option<Value> {
    match field {
        Field::Model => entries.model().map(Value::from),
        Field::RequestBody => entries.body().cloned().map(Value::Object),
        Field::RateLimitStatuses => entries.rate_limit_statuses().cloned(),
        Field::ResponseHeaders => Some(Value::Object(entries.response_headers().clone())),
        Field::Outcome => record(outcome(entries.ended()?)),
        Field::Usage => record(usage(entries.served()?.usage.as_ref()?)),
        Field::LoggedModelGroup => record(LoggedModelGroup {
            value: Some(Scalar::Text(entries.served()?.model_group.clone())),
            ..LoggedModelGroup::default()
        }),
        _ => None,
    }
}

fn facts(entries: &Entries) -> CallFacts {
    CallFacts {
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
            ..CallTarget::default()
        },
        timing: CallTiming {
            start: entries.start(),
            ..CallTiming::default()
        },
        ..CallFacts::default()
    }
}

/// What a callback that reads `reads` is handed on `hook`.
pub fn input(entries: &Entries, hook: Hook, reads: FieldSet) -> HookInput {
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
fn held(entries: &mut Entries, field: Field) -> Result<&mut Map<String, Value>, Unwritable> {
    entries.body_mut().ok_or(Unwritable::Routed(field))
}

/// Writes one operation of a callback's patch: the request's own params into the body, as the
/// proxy's hooks change `data`, and the rest where a later callback or the route reads it.
pub fn write(entries: &mut Entries, op: PatchOp) -> Result<(), Unwritable> {
    match op {
        PatchOp::Merge {
            field: Field::OptionalParams,
            keys,
        } => held(entries, Field::OptionalParams)?.extend(keys),
        PatchOp::Merge {
            field: Field::ResponseHeaders,
            keys,
        } => keys
            .into_iter()
            .for_each(|(name, value)| entries.add_header(name, value)),
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
        CallType,
        record::{self, CallRecord},
    };
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    /// Runs `look` on a fresh record of a Messages call holding a body that names `claude`.
    fn on_a_call<R>(look: impl FnOnce(&mut Entries) -> R) -> R {
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
    fn told(entries: &Entries, hook: Hook) -> (Option<Value>, String, bool) {
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
}
