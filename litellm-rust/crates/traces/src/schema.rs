use std::collections::BTreeMap;

use schemars::{JsonSchema, Schema, SchemaGenerator, generate::SchemaSettings};
use serde_json::json;

pub fn flag(_: &mut SchemaGenerator) -> Schema {
    json!({"type": "integer", "enum": [0, 1]})
        .try_into()
        .unwrap()
}

fn integer_value(value: &serde_json::Value) -> Option<i128> {
    value
        .as_i64()
        .map(i128::from)
        .or_else(|| value.as_u64().map(i128::from))
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
        let existing_minimum = schema.get("minimum").and_then(integer_value);
        let existing_maximum = schema.get("maximum").and_then(integer_value);
        if existing_minimum.is_none_or(|bound| bound < integer_value(&minimum).unwrap()) {
            schema.insert("minimum".to_owned(), minimum);
        }
        if existing_maximum.is_none_or(|bound| bound > integer_value(&maximum).unwrap()) {
            schema.insert("maximum".to_owned(), maximum);
        }
    }
    schemars::transform::transform_subschemas(&mut integer_bounds, schema);
}

pub(crate) fn received<T: JsonSchema>() -> Schema {
    SchemaSettings::draft2020_12()
        .for_deserialize()
        .with_transform(integer_bounds)
        .into_generator()
        .into_root_schema_for::<T>()
}

pub(crate) fn emitted<T: JsonSchema>() -> Schema {
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

pub fn api_schemas() -> BTreeMap<&'static str, Schema> {
    BTreeMap::from([
        (
            "TraceNoQueryRequest",
            received::<crate::api::TraceNoQueryRequest>(),
        ),
        (
            "TraceListRequest",
            received::<crate::api::TraceListRequest>(),
        ),
        (
            "TraceHistogramRequest",
            received::<crate::api::TraceHistogramRequest>(),
        ),
        (
            "TraceValuesRequest",
            received::<crate::api::TraceValuesRequest>(),
        ),
        (
            "TraceSpanPageRequest",
            received::<crate::api::TraceSpanPageRequest>(),
        ),
        (
            "TraceErrorPageRequest",
            received::<crate::api::TraceErrorPageRequest>(),
        ),
        (
            "TraceQueryRequest",
            received::<crate::api::TraceQueryRequest>(),
        ),
        ("TraceMetadata", emitted::<crate::api::TraceMetadata>()),
        ("TraceSpansPage", emitted::<crate::api::TraceSpansPage>()),
        ("TraceProblem", emitted::<crate::api::TraceProblem>()),
        (
            "TraceSQLResponse",
            emitted::<crate::api::TraceSQLResponse>(),
        ),
    ])
}
