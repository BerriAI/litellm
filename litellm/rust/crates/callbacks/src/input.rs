use std::{any::Any, sync::Arc};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    field::{Field, FieldSet, TypedField},
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
    #[error("{0:?} is delivered by reference, so read it with payload()")]
    ByReference(Field),
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct FieldValues {
    declared: FieldSet,
    values: Vec<(Field, Value)>,
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
        match self
            .values
            .iter_mut()
            .find(|(present, _)| *present == field)
        {
            Some((_, slot)) => *slot = value,
            None => self.values.push((field, value)),
        }
    }

    pub fn raw(&self, field: Field) -> Result<Option<&Value>, FieldError> {
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
            .map(|(_, value)| value))
    }

    pub fn get<F: TypedField>(&self) -> Result<Option<F::Value>, FieldError> {
        self.raw(F::FIELD)?
            .map(|value| {
                F::Value::deserialize(value).map_err(|source| FieldError::Decode {
                    field: F::FIELD,
                    source,
                })
            })
            .transpose()
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
