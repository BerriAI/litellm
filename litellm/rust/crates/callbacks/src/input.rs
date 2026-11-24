use std::{any::Any, borrow::Cow, sync::Arc};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    field::{Field, FieldSet, FieldValue, TypedField},
    hook::Hook,
    payload::PayloadRef,
};

pub type RequestState = Arc<dyn Any + Send + Sync>;

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct CallIds {
    pub call_id: String,
    pub trace_id: Option<String>,
    pub session_id: Option<String>,
    pub parent_call_id: Option<String>,
    pub connection_id: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct CallKind {
    pub call_type: String,
    pub route: Option<String>,
    pub stream: bool,
    pub synchronous: bool,
    pub endpoint_type: Option<String>,
    pub internal_origin: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct CallTarget {
    pub model: String,
    pub model_group: Option<String>,
    pub provider: Option<String>,
    pub api_base: Option<String>,
    pub deployment_id: Option<String>,
    pub attempt: Option<u32>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct CallTiming {
    pub start: f64,
    pub provider_start: Option<f64>,
    pub first_token: Option<f64>,
    pub end: Option<f64>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct CallFacts {
    pub ids: CallIds,
    pub kind: CallKind,
    pub target: CallTarget,
    pub timing: CallTiming,
    pub cache_hit: Option<bool>,
}

#[derive(Debug, thiserror::Error)]
pub enum FieldError {
    #[error("{0:?} is not among the declared reads")]
    Undeclared(Field),
    #[error("{field:?} does not decode to its typed view: {source}")]
    Decode {
        field: Field,
        source: serde_json::Error,
    },
    #[error("{field:?} does not encode to JSON")]
    Encode {
        field: Field,
        source: serde_json::Error,
    },
    #[error("{0:?} is delivered by reference, so read it with payload()")]
    ByReference(Field),
}

/// A field's value as its producer handed it over: typed from the Rust gateway, JSON from Python
/// and from tests. JSON is made from a typed value only when a reader asks for it.
#[derive(Clone, Debug, PartialEq)]
enum Slot {
    Typed(Box<FieldValue>),
    Json(Value),
}

impl Slot {
    fn to_json(&self) -> Result<Cow<'_, Value>, FieldError> {
        match self {
            Self::Typed(value) => {
                value
                    .to_json()
                    .map(Cow::Owned)
                    .map_err(|source| FieldError::Encode {
                        field: value.field(),
                        source,
                    })
            }
            Self::Json(value) => Ok(Cow::Borrowed(value)),
        }
    }
}

/// The fields one hook run reads, each held typed or as JSON as it arrived.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FieldValues {
    declared: FieldSet,
    values: Vec<(Field, Slot)>,
    payload: Option<Arc<PayloadRef>>,
}

impl FieldValues {
    pub fn new(declared: FieldSet) -> Self {
        Self {
            declared,
            values: Vec::new(),
            payload: None,
        }
    }

    #[must_use]
    pub fn with_payload(mut self, payload: PayloadRef) -> Self {
        self.payload = Some(Arc::new(payload));
        self
    }

    pub fn set_payload(&mut self, payload: PayloadRef) {
        self.payload = Some(Arc::new(payload));
    }

    pub fn payload(&self) -> Result<Option<Arc<PayloadRef>>, FieldError> {
        self.check(Field::StandardLoggingPayloadRef)?;
        Ok(self.payload.clone())
    }

    fn check(&self, field: Field) -> Result<bool, FieldError> {
        if self.declared.contains(field) {
            return Ok(true);
        }
        if cfg!(debug_assertions) {
            Err(FieldError::Undeclared(field))
        } else {
            Ok(false)
        }
    }

    #[must_use]
    pub fn with(mut self, field: Field, value: Value) -> Self {
        self.insert(field, value);
        self
    }

    pub fn insert(&mut self, field: Field, value: Value) {
        self.place(field, Slot::Json(value));
    }

    pub fn put<F: TypedField>(&mut self, value: F::Value) {
        self.place(F::FIELD, Slot::Typed(Box::new(F::wrap(value))));
    }

    fn place(&mut self, field: Field, slot: Slot) {
        match self
            .values
            .iter_mut()
            .find(|(present, _)| *present == field)
        {
            Some((_, held)) => *held = slot,
            None => self.values.push((field, slot)),
        }
    }

    fn slot(&self, field: Field) -> Result<Option<&Slot>, FieldError> {
        if field.by_reference() {
            return Err(FieldError::ByReference(field));
        }
        if !self.check(field)? {
            return Ok(None);
        }
        Ok(self
            .values
            .iter()
            .find(|(present, _)| *present == field)
            .map(|(_, slot)| slot))
    }

    pub fn raw(&self, field: Field) -> Result<Option<Cow<'_, Value>>, FieldError> {
        self.slot(field)?.map(Slot::to_json).transpose()
    }

    pub fn get<F: TypedField>(&self) -> Result<Option<F::Value>, FieldError> {
        Ok(self.get_ref::<F>()?.map(Cow::into_owned))
    }

    /// `get` without the clone when the value arrived typed: borrowed from a typed slot, decoded
    /// from a JSON one.
    pub fn get_ref<F: TypedField>(&self) -> Result<Option<Cow<'_, F::Value>>, FieldError> {
        let Some(slot) = self.slot(F::FIELD)? else {
            return Ok(None);
        };
        if let Slot::Typed(value) = slot
            && let Some(typed) = F::view(value)
        {
            return Ok(Some(Cow::Borrowed(typed)));
        }
        F::Value::deserialize(&*slot.to_json()?)
            .map(|value| Some(Cow::Owned(value)))
            .map_err(|source| FieldError::Decode {
                field: F::FIELD,
                source,
            })
    }
}

pub struct HookInput {
    pub hook: Hook,
    pub fields: FieldValues,
    pub call: Option<CallFacts>,
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;
    use crate::field::typed;

    fn values() -> FieldValues {
        FieldValues::new(FieldSet::of(&[Field::Model, Field::Usage]))
            .with(Field::Model, json!("gpt"))
    }

    #[rstest]
    fn a_declared_read_decodes_to_its_typed_view() {
        assert_eq!(
            values().get::<typed::Model>().unwrap(),
            Some("gpt".to_owned())
        );
    }

    #[rstest]
    fn a_declared_read_the_hook_point_did_not_supply_is_none() {
        assert_eq!(values().get::<typed::Usage>().unwrap(), None);
    }

    #[rstest]
    fn an_undeclared_read_is_an_error_in_tests() {
        assert!(matches!(
            values().get::<typed::Metadata>(),
            Err(FieldError::Undeclared(Field::Metadata))
        ));
    }

    #[rstest]
    fn a_value_of_the_wrong_shape_names_its_field() {
        let wrong = FieldValues::new(FieldSet::of(&[Field::Model])).with(Field::Model, json!(3));
        assert!(matches!(
            wrong.get::<typed::Model>(),
            Err(FieldError::Decode {
                field: Field::Model,
                ..
            })
        ));
    }

    #[rstest]
    fn a_typed_record_decodes_from_the_python_wire_form() {
        let values = FieldValues::new(FieldSet::of(&[Field::KeyIdentity, Field::Outcome]))
            .with(
                Field::KeyIdentity,
                json!({"team_id": "t", "team_alias": null}),
            )
            .with(Field::Outcome, json!("fallback_failure"));
        let identity = values.get::<typed::KeyIdentity>().unwrap().unwrap();
        assert_eq!(
            identity.team_id,
            Some(crate::scalar::Scalar::Text("t".to_owned()))
        );
        assert_eq!(
            values.get::<typed::Outcome>().unwrap(),
            Some(crate::record::Outcome::FallbackFailure)
        );
    }

    fn usage() -> crate::record::Usage {
        crate::record::Usage {
            response: Some(crate::record::UsageRecord {
                prompt_tokens: Some(crate::scalar::Scalar::Int(3)),
                total_tokens: Some(crate::scalar::Scalar::Float(f64::NAN)),
                ..crate::record::UsageRecord::default()
            }),
            ..crate::record::Usage::default()
        }
    }

    fn typed_usage() -> FieldValues {
        let mut values = FieldValues::new(FieldSet::of(&[Field::Usage]));
        values.put::<typed::Usage>(usage());
        values
    }

    #[rstest]
    fn a_typed_put_is_read_back_without_a_json_round_trip() {
        let values = typed_usage();
        assert!(matches!(
            values.get_ref::<typed::Usage>().unwrap(),
            Some(Cow::Borrowed(_))
        ));
        let read = values.get::<typed::Usage>().unwrap().unwrap();
        assert_eq!(
            read.response.as_ref().unwrap().prompt_tokens,
            Some(crate::scalar::Scalar::Int(3))
        );
        assert!(matches!(
            read.response.unwrap().total_tokens,
            Some(crate::scalar::Scalar::Float(number)) if number.is_nan()
        ));
    }

    #[rstest]
    fn a_typed_put_reads_raw_as_the_json_serde_would_make() {
        assert_eq!(
            typed_usage()
                .raw(Field::Usage)
                .unwrap()
                .unwrap()
                .into_owned(),
            serde_json::to_value(usage()).unwrap()
        );
    }

    #[rstest]
    fn a_typed_put_replaces_a_json_insert_of_the_same_field() {
        let mut values = values();
        values.put::<typed::Model>("claude".to_owned());
        assert_eq!(
            values.raw(Field::Model).unwrap().unwrap().into_owned(),
            json!("claude")
        );
        assert_eq!(
            values.get::<typed::Model>().unwrap(),
            Some("claude".to_owned())
        );
    }

    #[rstest]
    fn a_json_insert_still_decodes_on_get() {
        let values = FieldValues::new(FieldSet::of(&[Field::Usage]))
            .with(Field::Usage, json!({"response": {"prompt_tokens": 3}}));
        assert!(matches!(
            values.get_ref::<typed::Usage>().unwrap(),
            Some(Cow::Owned(_))
        ));
        assert_eq!(
            values
                .get::<typed::Usage>()
                .unwrap()
                .unwrap()
                .response
                .unwrap()
                .prompt_tokens,
            Some(crate::scalar::Scalar::Int(3))
        );
    }

    #[rstest]
    fn the_payload_reference_is_read_through_its_own_accessor() {
        let values = FieldValues::new(FieldSet::of(&[Field::StandardLoggingPayloadRef]));
        assert!(matches!(
            values.raw(Field::StandardLoggingPayloadRef),
            Err(FieldError::ByReference(Field::StandardLoggingPayloadRef))
        ));
        assert!(values.payload().unwrap().is_none());
        assert!(matches!(
            FieldValues::default().payload(),
            Err(FieldError::Undeclared(Field::StandardLoggingPayloadRef))
        ));
    }

    #[rstest]
    fn call_facts_read_the_python_dataclass_shape() {
        let facts: CallFacts = serde_json::from_value(json!({
            "ids": {"call_id": "c", "trace_id": null, "session_id": null, "parent_call_id": null, "connection_id": "ws"},
            "kind": {
                "call_type": "acompletion", "route": "/chat/completions", "stream": false,
                "synchronous": true, "endpoint_type": "generic", "internal_origin": null
            },
            "target": {"model": "m", "model_group": "g", "provider": null, "api_base": null, "deployment_id": null, "attempt": 0},
            "timing": {"start": 1.5, "provider_start": null, "first_token": null, "end": null},
            "cache_hit": null,
        }))
        .unwrap();
        assert_eq!(facts.target.model_group.as_deref(), Some("g"));
        assert_eq!(facts.timing.start, 1.5);
        assert!(facts.kind.synchronous);
        assert_eq!(facts.ids.connection_id.as_deref(), Some("ws"));
    }

    #[rstest]
    fn call_facts_from_an_older_sender_default_the_new_members() {
        let facts: CallFacts = serde_json::from_value(json!({
            "ids": {"call_id": "c"},
            "kind": {"call_type": "acompletion", "stream": true},
            "target": {"model": "m"},
            "timing": {"start": 1.0},
        }))
        .unwrap();
        assert!(!facts.kind.synchronous);
        assert_eq!(facts.kind.endpoint_type, None);
    }
}
