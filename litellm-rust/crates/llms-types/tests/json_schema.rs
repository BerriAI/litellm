use litellm_llms_types::{
    json_schema::{JsonSchema, JsonSchemaType},
    recognized::Recognized,
};
use rstest::rstest;
use serde_json::{Value, json};

#[rstest]
fn recursive_schemas_expose_known_keywords_and_preserve_arbitrary_values() {
    let wire = json!({
        "type":["object","null"],
        "properties":{"nested":{"type":"array","items":{"$ref":"#/$defs/item"}},"free":true,"broken":17},
        "additionalProperties":{"type":"string"},
        "$defs":{"item":{"anyOf":[{"type":"string"},false]}},
        "required":["nested"],"enum":[{"custom":[1,null]}],"future":null
    });
    let parsed: JsonSchema = serde_json::from_value(wire.clone()).unwrap();
    let JsonSchema::Object(schema) = &parsed else {
        panic!("expected object schema")
    };
    assert_eq!(
        schema.schema_type,
        Some(Recognized::Known(JsonSchemaType::Names(vec![
            "object".into(),
            "null".into()
        ])))
    );
    let properties = schema.properties.as_ref().unwrap().known().unwrap();
    let JsonSchema::Object(nested) = properties["nested"].known().unwrap() else {
        panic!("expected nested schema")
    };
    let JsonSchema::Object(items) = nested.items.as_ref().unwrap().known().unwrap().as_ref() else {
        panic!("expected items schema")
    };
    assert_eq!(
        items.reference,
        Some(Recognized::Known("#/$defs/item".into()))
    );
    assert_eq!(
        properties["free"],
        Recognized::Known(JsonSchema::Boolean(true))
    );
    assert_eq!(properties["broken"], Recognized::Unrecognized(json!(17)));
    assert!(
        matches!(schema.additional_properties.as_ref().unwrap().known().unwrap().as_ref(), JsonSchema::Object(value) if value.schema_type == Some(Recognized::Known(JsonSchemaType::Name("string".into()))))
    );
    assert_eq!(serde_json::to_value(parsed).unwrap(), wire);
}

#[rstest]
#[case::boolean(json!(false))]
#[case::unknown_keywords(json!({"vendor":{"format":null}}))]
#[case::malformed_keywords(json!({"properties":[],"items":null,"anyOf":17,"required":false}))]
fn schemas_keep_boolean_forms_and_permissive_keyword_values(#[case] wire: Value) {
    let parsed: JsonSchema = serde_json::from_value(wire.clone()).unwrap();
    assert_eq!(serde_json::to_value(parsed).unwrap(), wire);
}
