use litellm_llms_types::json_schema::{JsonSchema, JsonSchemaObject};
use rstest::rstest;
use serde::{Serialize, de::DeserializeOwned};
use serde_json::{Value, json};

fn round_trip<T>(wire: Value)
where
    T: DeserializeOwned + Serialize,
{
    let parsed: T = serde_json::from_value(wire.clone()).unwrap();
    assert_eq!(serde_json::to_value(parsed).unwrap(), wire);
}

#[rstest]
#[case::boolean(json!(false))]
#[case::nested(json!({
    "type":["object","null"],
    "properties":{"nested":{"type":"array","items":{"$ref":"#/$defs/item"}},"free":true},
    "additionalProperties":{"type":"string"},
    "$defs":{"item":{"anyOf":[{"type":"string"},false]}},
    "required":["nested"],
    "enum":[{"custom":[1,null]}],
    "future":null
}))]
fn recursive_schema_round_trips(#[case] wire: Value) {
    round_trip::<JsonSchema>(wire);
}

#[rstest]
fn schema_object_round_trips_supported_keywords() {
    round_trip::<JsonSchemaObject>(json!({
        "type":"object",
        "properties":{"name":{"type":"string"}},
        "required":["name"],
        "additionalProperties":false,
        "$ref":"#/$defs/value",
        "strict":true,
        "extension":{"nested":[1,null]}
    }));
}
