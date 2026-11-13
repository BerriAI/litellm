use std::{any::Any, sync::Arc};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    field::{Field, FieldSet, TypedField},
    hook::Hook,
};

pub type RequestState = Arc<dyn Any + Send + Sync>;

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CallIds {
    pub call_id: String,
    pub trace_id: Option<String>,
    pub session_id: Option<String>,
    pub parent_call_id: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CallKind {
    pub call_type: String,
    pub route: Option<String>,
    pub stream: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CallTarget {
    pub model: String,
    pub model_group: Option<String>,
    pub provider: Option<String>,
    pub api_base: Option<String>,
    pub deployment_id: Option<String>,
    pub attempt: Option<u32>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct CallTiming {
    pub start: f64,
    pub provider_start: Option<f64>,
    pub first_token: Option<f64>,
    pub end: Option<f64>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
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
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct FieldValues {
    declared: FieldSet,
    values: Vec<(Field, Value)>,
}

impl FieldValues {
    pub fn new(declared: FieldSet) -> Self {
        Self {
            declared,
            values: Vec::new(),
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
        if !self.declared.contains(field) {
            return if cfg!(debug_assertions) {
                Err(FieldError::Undeclared(field))
            } else {
                Ok(None)
            };
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
    pub state: Option<RequestState>,
}

impl HookInput {
    pub fn state<T: Any>(&self) -> Option<&T> {
        self.state.as_deref()?.downcast_ref()
    }
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
    fn call_facts_read_the_python_dataclass_shape() {
        let facts: CallFacts = serde_json::from_value(json!({
            "ids": {"call_id": "c", "trace_id": null, "session_id": null, "parent_call_id": null},
            "kind": {"call_type": "acompletion", "route": "/chat/completions", "stream": false},
            "target": {"model": "m", "model_group": "g", "provider": null, "api_base": null, "deployment_id": null, "attempt": 0},
            "timing": {"start": 1.5, "provider_start": null, "first_token": null, "end": null},
            "cache_hit": null,
        }))
        .unwrap();
        assert_eq!(facts.target.model_group.as_deref(), Some("g"));
        assert_eq!(facts.timing.start, 1.5);
    }

    #[rstest]
    fn request_state_downcasts_to_the_type_the_callback_created() {
        let input = HookInput {
            hook: Hook::PreCall,
            fields: FieldValues::default(),
            call: None,
            state: Some(Arc::new(7_u32)),
        };
        assert_eq!(input.state::<u32>(), Some(&7));
        assert_eq!(input.state::<String>(), None);
    }
}
