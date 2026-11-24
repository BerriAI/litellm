//! What the gateway knows of one call and how a callback reads and writes it by contract field,
//! the native counterpart of the proxy's `v2_proxy_fields.py`. A field the gateway cannot work
//! out yet is left out of what a callback is handed, and one it has no place for is not written.

use std::time::{SystemTime, UNIX_EPOCH};

use litellm_callbacks::{
    CallFacts, CallIds, CallKind, CallTarget, CallTiming, Cost, Field, FieldSet, FieldValues, Hook,
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
    #[error("{0:?} is part of the request body, which the route already has")]
    Routed(Field),
}

/// The request body of one call. The route owns it, so it is only held while the pre-call hook
/// runs, and what it named of the call is all that stays once it is handed back. The model and
/// the stream flag are read from whichever is there, so a patch to a held body is what every
/// later reader sees.
enum Request {
    Held(Map<String, Value>),
    Named { model: Option<String>, stream: bool },
}

impl Request {
    fn body(&self) -> Option<&Map<String, Value>> {
        match self {
            Self::Held(body) => Some(body),
            Self::Named { .. } => None,
        }
    }

    fn model(&self) -> Option<&str> {
        match self {
            Self::Held(body) => body.get(MODEL).and_then(Value::as_str),
            Self::Named { model, .. } => model.as_deref(),
        }
    }

    fn stream(&self) -> bool {
        match self {
            Self::Held(body) => body.get(STREAM).and_then(Value::as_bool) == Some(true),
            Self::Named { stream, .. } => *stream,
        }
    }

    fn named(&self) -> Self {
        Self::Named {
            model: self.model().map(str::to_owned),
            stream: self.stream(),
        }
    }

    /// The held body for a write to `field`.
    fn held(&mut self, field: Field) -> Result<&mut Map<String, Value>, Unwritable> {
        match self {
            Self::Held(body) => Ok(body),
            Self::Named { .. } => Err(Unwritable::Routed(field)),
        }
    }
}

/// One call as it stands.
pub(super) struct Known {
    call_type: CallType,
    start: f64,
    request: Request,
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
            request: Request::Named {
                model: None,
                stream: false,
            },
            rate_limit_statuses: None,
            response_headers: Map::new(),
            served: None,
            outcome: None,
        }
    }

    /// Takes the route's `body` for the pre-call hook.
    pub(super) fn hold(&mut self, body: Map<String, Value>) {
        self.request = Request::Held(body);
    }

    /// The body as the callbacks left it, back to the route.
    pub(super) fn release(&mut self) -> Map<String, Value> {
        let named = self.request.named();
        match std::mem::replace(&mut self.request, named) {
            Request::Held(body) => body,
            Request::Named { .. } => Map::new(),
        }
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
            Field::Model => self.request.model().map(Value::from),
            Field::RequestBody => self.request.body().cloned().map(Value::Object),
            Field::RateLimitStatuses => self.rate_limit_statuses.clone(),
            Field::ResponseHeaders => Some(Value::Object(self.response_headers.clone())),
            Field::Outcome => record(self.outcome?),
            Field::Usage => record(usage(self.served.as_ref()?.usage.as_ref()?)),
            Field::Cost => record(Cost {
                response_cost: Some(Scalar::Float(self.served.as_ref()?.cost?)),
                ..Cost::default()
            }),
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
                stream: self.request.stream(),
                ..CallKind::default()
            },
            target: CallTarget {
                model: self.request.model().unwrap_or_default().to_owned(),
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
            } => self.request.held(Field::OptionalParams)?.extend(keys),
            PatchOp::Merge {
                field: Field::ResponseHeaders,
                keys,
            } => self.response_headers.extend(keys),
            PatchOp::Set {
                field: Field::Model,
                value: Value::String(model),
            } => {
                self.request
                    .held(Field::Model)?
                    .insert(MODEL.to_owned(), Value::from(model));
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
    #[case::a_cost_the_route_settled(Some(0.25), Some(Scalar::Float(0.25)))]
    #[case::a_call_the_route_settled_no_cost_for(None, None)]
    fn the_success_log_reads_what_the_route_settled_the_call_cost(
        #[case] cost: Option<f64>,
        #[case] expected: Option<Scalar>,
    ) {
        let mut known = known();
        known.serve(ServedCall {
            model_group: "claude".to_owned(),
            usage: None,
            cost,
        });

        let input = known.input(Hook::PostCallSuccess, FieldSet::of(&[Field::Cost]));

        let read = input.fields.raw(Field::Cost).unwrap().cloned();
        let read: Option<Cost> = read.map(|cost| serde_json::from_value(cost).unwrap());
        assert_eq!(read.and_then(|cost| cost.response_cost), expected);
    }

    /// What a callback that reads the model is told of the call: the field, and the model and
    /// the stream flag among the call's facts.
    fn told(known: &Known, hook: Hook) -> (Option<Value>, String, bool) {
        let input = known.input(hook, FieldSet::of(&[Field::Model]));
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
        let mut known = known();
        known.write(op).unwrap();
        let on_the_same_hook = told(&known, Hook::PreCall);
        let body = known.release();
        assert_eq!(
            (
                on_the_same_hook,
                body.get(MODEL).cloned(),
                told(&known, Hook::AttemptEnded)
            ),
            (seen.clone(), seen.0.clone(), seen)
        );
    }

    #[rstest]
    fn a_body_the_route_already_has_is_never_written() {
        let mut known = known();
        known.release();
        assert_eq!(
            known.write(PatchOp::Merge {
                field: Field::OptionalParams,
                keys: keys()
            }),
            Err(Unwritable::Routed(Field::OptionalParams))
        );
    }
}
