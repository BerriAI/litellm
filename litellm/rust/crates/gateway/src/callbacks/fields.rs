//! What the gateway knows of one call and how a callback reads and writes it by contract field,
//! the native counterpart of the proxy's `v2_proxy_fields.py`. A field the gateway cannot work
//! out yet is left out of what a callback is handed, and one it has no place for is not written.

use std::time::{SystemTime, UNIX_EPOCH};

use litellm_callbacks::{
    CallFacts, CallIds, CallKind, CallTarget, CallTiming, Field, FieldSet, FieldValues, Hook,
    HookInput, LoggedModelGroup, Outcome, PatchOp, Scalar, Usage, UsageRecord,
};
use litellm_gateway_inference::{CallType, ServedCall};
use litellm_llms_types::formats::chat_completions::ChatCompletionsUsage;
use serde::Serialize;
use serde_json::{Map, Value};

const MODEL: &str = "model";
const STREAM: &str = "stream";

/// A patch operation the gateway has no place to write.
#[derive(Debug, PartialEq, Eq, thiserror::Error)]
pub(super) enum Unwritable {
    #[error("the gateway has no place to write {0:?} yet")]
    Field(Field),
    #[error("{0:?} is replaced whole, never merged into")]
    Merge(Field),
    #[error("{0:?} takes keys merged into it, never a whole value")]
    Set(Field),
    #[error("a model is a name, never {0}")]
    Model(Value),
}

/// One call as it stands. The route owns the body, so it is only here while the pre-call hook
/// runs, and the model and the stream flag it named stay after it is handed back.
#[derive(Debug)]
pub(super) struct Known {
    call_type: CallType,
    start: f64,
    body: Map<String, Value>,
    model: Option<String>,
    stream: bool,
    rate_limit_statuses: Option<Value>,
    response_headers: Map<String, Value>,
    served: Option<ServedCall>,
    outcome: Option<Outcome>,
}

fn unix_now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0.0, |elapsed| elapsed.as_secs_f64())
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
            cached_tokens: count(used.prompt_tokens_details.cached_tokens),
            ..UsageRecord::default()
        }),
        ..Usage::default()
    }
}

impl Known {
    pub(super) fn of(call_type: CallType) -> Self {
        Self {
            call_type,
            start: unix_now(),
            body: Map::new(),
            model: None,
            stream: false,
            rate_limit_statuses: None,
            response_headers: Map::new(),
            served: None,
            outcome: None,
        }
    }

    /// Takes the route's `body` for the pre-call hook.
    pub(super) fn hold(&mut self, body: Map<String, Value>) {
        self.body = body;
        self.remember_body();
    }

    /// The body as the callbacks left it, back to the route.
    pub(super) fn release(&mut self) -> Map<String, Value> {
        self.remember_body();
        std::mem::take(&mut self.body)
    }

    fn remember_body(&mut self) {
        self.model = self
            .body
            .get(MODEL)
            .and_then(Value::as_str)
            .map(str::to_owned);
        self.stream = self.body.get(STREAM).and_then(Value::as_bool) == Some(true);
    }

    pub(super) fn serve(&mut self, served: ServedCall) {
        self.served = Some(served);
        self.outcome = Some(Outcome::Success);
    }

    pub(super) fn fail(&mut self) {
        self.outcome = Some(Outcome::Failure);
    }

    pub(super) fn response_headers(&self) -> &Map<String, Value> {
        &self.response_headers
    }

    fn read(&self, field: Field) -> Option<Value> {
        match field {
            Field::Model => self.model.clone().map(Value::from),
            Field::RequestBody if !self.body.is_empty() => Some(Value::Object(self.body.clone())),
            Field::RateLimitStatuses => self.rate_limit_statuses.clone(),
            Field::ResponseHeaders => Some(Value::Object(self.response_headers.clone())),
            Field::Outcome => record(self.outcome?),
            Field::Usage => record(usage(self.served.as_ref()?.usage.as_ref()?)),
            Field::LoggedModelGroup => record(LoggedModelGroup {
                value: Some(Scalar::Text(self.served.as_ref()?.model_group.clone())),
                ..LoggedModelGroup::default()
            }),
            _ => None,
        }
    }

    fn facts(&self) -> CallFacts {
        CallFacts {
            ids: CallIds::default(),
            kind: CallKind {
                call_type: <&str>::from(self.call_type).to_owned(),
                stream: self.stream,
                ..CallKind::default()
            },
            target: CallTarget {
                model: self.model.clone().unwrap_or_default(),
                model_group: self
                    .served
                    .as_ref()
                    .map(|served| served.model_group.clone()),
                ..CallTarget::default()
            },
            timing: CallTiming {
                start: self.start,
                ..CallTiming::default()
            },
            cache_hit: None,
        }
    }

    /// What a callback that reads `reads` is handed on `hook`.
    pub(super) fn input(&self, hook: Hook, reads: FieldSet) -> HookInput {
        let fields = reads
            .iter()
            .filter(|field| !field.by_reference())
            .filter_map(|field| Some((field, self.read(field)?)))
            .fold(FieldValues::new(reads), |fields, (field, value)| {
                fields.with(field, value)
            });
        HookInput {
            hook,
            fields,
            call: Some(self.facts()),
        }
    }

    /// Writes one operation of a callback's patch: the request's own params into the body, as
    /// the proxy's hooks change `data`, and the rest where a later callback or the route reads it.
    pub(super) fn write(&mut self, op: PatchOp) -> Result<(), Unwritable> {
        match op {
            PatchOp::Merge {
                field: Field::OptionalParams,
                keys,
            } => self.body.extend(keys),
            PatchOp::Merge {
                field: Field::ResponseHeaders,
                keys,
            } => self.response_headers.extend(keys),
            PatchOp::Set {
                field: Field::Model,
                value: Value::String(model),
            } => {
                self.body.insert(MODEL.to_owned(), Value::from(model));
            }
            PatchOp::Set {
                field: Field::Model,
                value,
            } => return Err(Unwritable::Model(value)),
            PatchOp::Set {
                field: Field::RateLimitStatuses,
                value,
            } => self.rate_limit_statuses = Some(value),
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
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    fn known() -> Known {
        let mut known = Known::of(CallType::AnthropicMessages);
        known.hold(Map::from_iter([("model".to_owned(), json!("claude"))]));
        known
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
        let mut known = known();
        assert_eq!(
            (known.write(op), known.release()),
            (
                Err(refused),
                Map::from_iter([("model".to_owned(), json!("claude"))])
            )
        );
    }

    #[rstest]
    fn a_model_a_callback_sets_is_the_model_the_route_and_later_callbacks_see() {
        let mut known = known();
        known
            .write(PatchOp::Set {
                field: Field::Model,
                value: json!("claude-fallback"),
            })
            .unwrap();
        let body = known.release();
        let input = known.input(Hook::AttemptEnded, FieldSet::of(&[Field::Model]));
        assert_eq!(
            (
                body.get(MODEL),
                input.fields.raw(Field::Model).unwrap(),
                input.call.map(|call| call.target.model)
            ),
            (
                Some(&json!("claude-fallback")),
                Some(&json!("claude-fallback")),
                Some("claude-fallback".to_owned())
            )
        );
    }
}
