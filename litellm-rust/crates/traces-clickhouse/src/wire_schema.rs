use std::collections::BTreeMap;

use schemars::{JsonSchema, Schema, SchemaGenerator, generate::SchemaSettings};
use serde_json::json;

use crate::query::lens;

fn numeric_wire(normalized: Schema, python_type: String) -> Schema {
    json!({
        "anyOf": [normalized, {"type": "string", "pattern": "^[0-9]+$"}],
        "x-python-normalized": {"type": python_type, "minimum": 0, "maximum": u64::MAX},
    })
    .try_into()
    .unwrap()
}

pub(crate) fn u64_number(generator: &mut SchemaGenerator) -> Schema {
    numeric_wire(u64::json_schema(generator), "int".to_owned())
}

pub(crate) fn flag_number(_: &mut SchemaGenerator) -> Schema {
    json!({
        "anyOf": [{"type": "integer", "enum": [0, 1]}, {"type": "string", "enum": ["0", "1"]}],
        "x-python-normalized": {"type": "int", "minimum": 0, "maximum": 1}
    })
    .try_into()
    .unwrap()
}

pub(crate) fn boolean_flag(_: &mut SchemaGenerator) -> Schema {
    json!({
        "anyOf": [{"type": "boolean"}, {"type": "integer", "enum": [0, 1]}, {"type": "string", "enum": ["0", "1"]}],
        "default": false,
        "x-python-normalized": {"type": "bool"}
    }).try_into().unwrap()
}

pub(crate) fn selected(generator: &mut SchemaGenerator) -> Schema {
    u64_number(generator)
}

fn received<T: JsonSchema>() -> Schema {
    SchemaSettings::draft2020_12()
        .for_deserialize()
        .with_transform(litellm_traces::schema::integer_bounds)
        .into_generator()
        .into_root_schema_for::<T>()
}

pub fn schemas() -> BTreeMap<&'static str, Schema> {
    BTreeMap::from([
        ("ReadQueryName", json!({"$schema": "https://json-schema.org/draft/2020-12/schema", "title": "ReadQueryName", "type": "string", "enum": lens::LENS_QUERIES.map(|query| query.to_string())}).try_into().unwrap()),
        ("LensAccessParams", received::<lens::LensAccessParams>()),
        ("LensSampleParams", received::<lens::LensSampleParams>()),
        ("LensContentParams", received::<lens::LensContentParams>()),
        ("LensEvidenceParams", received::<lens::LensEvidenceParams>()),
        (
            "ActivityAvailability",
            received::<lens::LensAvailabilityRow>(),
        ),
        ("ExecutionRow", received::<lens::LensSampleRow>()),
        ("PartRow", received::<lens::LensContentRow>()),
        ("CountRow", received::<lens::LensEvidenceRow>()),
        ("AgentRow", received::<lens::LensAgentsRow>()),
        ("TraceQueryHelp", crate::query::help_schema()),
    ])
}
