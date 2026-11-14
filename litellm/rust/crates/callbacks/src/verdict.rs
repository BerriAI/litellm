use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::field::Field;

// Respond and Reroute are left out: the five ports (budgets, both limiters, prometheus, s3_v2) only patch or abort.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    Patch(Patch),
    Abort(Abort),
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Patch {
    pub ops: Vec<PatchOp>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum PatchOp {
    Set {
        field: Field,
        value: Value,
    },
    Merge {
        field: Field,
        keys: Map<String, Value>,
    },
}

impl Patch {
    #[must_use]
    pub fn set(mut self, field: Field, value: Value) -> Self {
        self.ops.push(PatchOp::Set { field, value });
        self
    }

    #[must_use]
    pub fn merge(mut self, field: Field, keys: Map<String, Value>) -> Self {
        self.ops.push(PatchOp::Merge { field, keys });
        self
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Abort {
    pub status: u16,
    pub message: String,
    pub code: Option<String>,
    pub headers: BTreeMap<String, String>,
    pub error: Option<AbortError>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AbortError {
    pub kind: String,
    pub fields: Map<String, Value>,
}

impl Abort {
    pub fn new(status: u16, message: impl Into<String>) -> Self {
        Self {
            status,
            message: message.into(),
            code: None,
            headers: BTreeMap::new(),
            error: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    #[rstest]
    #[case::patch(
        Verdict::Patch(Patch::default().set(Field::Model, json!("m")).merge(Field::ResponseHeaders, Map::from_iter([("x-a".into(), json!("1"))]))),
        json!({"patch": {"ops": [
            {"op": "set", "field": "model", "value": "m"},
            {"op": "merge", "field": "response_headers", "keys": {"x-a": "1"}},
        ]}})
    )]
    #[case::typed_error(
        Verdict::Abort(Abort {
            error: Some(AbortError { kind: "budget_exceeded".into(), fields: Map::from_iter([("max_budget".into(), json!(1.0))]) }),
            ..Abort::new(400, "over budget")
        }),
        json!({"abort": {"status": 400, "message": "over budget", "code": null, "headers": {},
            "error": {"kind": "budget_exceeded", "fields": {"max_budget": 1.0}}}})
    )]
    #[case::abort(
        Verdict::Abort(Abort { code: Some("limit".into()), ..Abort::new(429, "slow down") }),
        json!({"abort": {"status": 429, "message": "slow down", "code": "limit", "headers": {}, "error": null}})
    )]
    fn a_verdict_has_the_wire_shape_python_reads(#[case] verdict: Verdict, #[case] wire: Value) {
        assert_eq!(serde_json::to_value(&verdict).unwrap(), wire);
        assert_eq!(serde_json::from_value::<Verdict>(wire).unwrap(), verdict);
    }
}
