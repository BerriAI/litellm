use std::collections::BTreeMap;

use schemars::{JsonSchema, Schema, SchemaGenerator, generate::SchemaSettings};
use serde_json::json;

pub fn flag(_: &mut SchemaGenerator) -> Schema {
    json!({"type": "integer", "enum": [0, 1]})
        .try_into()
        .unwrap()
}

pub fn integer_bounds(schema: &mut Schema) {
    let bounds = match schema.get("format").and_then(serde_json::Value::as_str) {
        Some("uint8") => Some((json!(0), json!(u8::MAX))),
        Some("uint16") => Some((json!(0), json!(u16::MAX))),
        Some("uint32") => Some((json!(0), json!(u32::MAX))),
        Some("uint64") => Some((json!(0), json!(u64::MAX))),
        Some("uint") => Some((json!(0), json!(usize::MAX))),
        Some("int32") => Some((json!(i32::MIN), json!(i32::MAX))),
        Some("int64") => Some((json!(i64::MIN), json!(i64::MAX))),
        Some("int") => Some((json!(isize::MIN), json!(isize::MAX))),
        _ => None,
    };
    if let Some((minimum, maximum)) = bounds {
        schema.insert("minimum".to_owned(), minimum);
        schema.insert("maximum".to_owned(), maximum);
    }
    schemars::transform::transform_subschemas(&mut integer_bounds, schema);
}

fn received<T: JsonSchema>() -> Schema {
    SchemaSettings::draft2020_12()
        .for_deserialize()
        .with_transform(integer_bounds)
        .into_generator()
        .into_root_schema_for::<T>()
}

fn emitted<T: JsonSchema>() -> Schema {
    SchemaSettings::draft2020_12()
        .for_serialize()
        .with_transform(integer_bounds)
        .into_generator()
        .into_root_schema_for::<T>()
}

pub fn schemas() -> BTreeMap<&'static str, Schema> {
    BTreeMap::from([
        ("QueryScope", received::<crate::QueryScope>()),
        ("Tenant", received::<crate::Tenant>()),
        ("TracePage", emitted::<crate::TracePage>()),
        ("SpanText", emitted::<crate::store::SpanText>()),
        ("Trace", emitted::<crate::Trace>()),
        ("SpanDetail", emitted::<crate::SpanDetail>()),
        ("SpanErrorPage", emitted::<crate::SpanErrorPage>()),
        ("TraceHistogram", emitted::<crate::search::TraceHistogram>()),
        ("RunValues", emitted::<crate::search::RunValues>()),
        ("RunField", received::<crate::search::RunField>()),
        ("RunOrder", received::<crate::store::RunOrder>()),
    ])
}
