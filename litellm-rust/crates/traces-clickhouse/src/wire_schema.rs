use std::collections::BTreeMap;

use schemars::{JsonSchema, Schema, SchemaGenerator, generate::SchemaSettings};
use serde_json::json;

use crate::query::lens;

fn quoted_u64() -> Schema {
    let upper = u64::MAX.to_string();
    let alternatives = upper
        .char_indices()
        .filter_map(|(index, digit)| {
            let lower = if index == 0 { '1' } else { '0' };
            if digit <= lower {
                return None;
            }
            Some(format!(
                "{}[{}-{}][0-9]{{{}}}",
                &upper[..index],
                lower,
                char::from(digit as u8 - 1),
                upper.len() - index - 1
            ))
        })
        .collect::<Vec<_>>()
        .join("|");
    json!({
        "type": "string",
        "pattern": format!("^(?:0|[1-9][0-9]{{0,{}}}|{alternatives}|{upper})$", upper.len() - 2),
    })
    .try_into()
    .unwrap()
}

fn numeric_wire(normalized: Schema, python_type: String) -> Schema {
    json!({
        "anyOf": [normalized, quoted_u64()],
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

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case::zero(json!(0), true)]
    #[case::quoted_zero(json!("0"), true)]
    #[case::maximum(json!(u64::MAX), true)]
    #[case::quoted_maximum(json!(u64::MAX.to_string()), true)]
    #[case::negative(json!(-1), false)]
    #[case::overflow(json!((u128::from(u64::MAX) + 1).to_string()), false)]
    #[case::fraction(json!(1.5), false)]
    fn count_schema_enforces_the_native_range(
        #[case] value: serde_json::Value,
        #[case] valid: bool,
    ) {
        let schema = received::<lens::LensEvidenceRow>();
        assert_eq!(
            jsonschema::is_valid(schema.as_value(), &json!({"count": value})),
            valid
        );
    }
}
