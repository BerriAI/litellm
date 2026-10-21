use litellm_llms_types::json_schema::{
    JsonSchema, JsonSchemaItems, JsonSchemaObject, JsonSchemaType,
};
use rstest::rstest;
use serde::{Serialize, de::DeserializeOwned};
use serde_json::{Value, json};

fn round_trip<T>(wire: Value) -> T
where
    T: DeserializeOwned + Serialize,
{
    let parsed: T = serde_json::from_value(wire.clone()).unwrap();
    assert_eq!(serde_json::to_value(&parsed).unwrap(), wire);
    parsed
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
    let schema = round_trip::<JsonSchema>(wire);
    match schema {
        JsonSchema::Boolean(allowed) => assert!(!allowed),
        JsonSchema::Object(schema) => {
            assert_eq!(
                schema.schema_type,
                Some(JsonSchemaType::Names(vec!["object".into(), "null".into()]))
            );
            let properties = schema.properties.as_ref().unwrap();
            let JsonSchema::Object(nested) = &properties["nested"] else {
                panic!("expected nested schema")
            };
            let Some(JsonSchemaItems::Schema(items)) = &nested.items else {
                panic!("expected single items schema")
            };
            let JsonSchema::Object(items) = items.as_ref() else {
                panic!("expected items schema")
            };
            assert_eq!(items.reference.as_deref(), Some("#/$defs/item"));
            assert_eq!(properties["free"], JsonSchema::Boolean(true));
            let JsonSchema::Object(definition) = &schema.defs.as_ref().unwrap()["item"] else {
                panic!("expected schema definition")
            };
            assert_eq!(
                definition.any_of.as_ref().unwrap()[1],
                JsonSchema::Boolean(false)
            );
            assert_eq!(schema.extra["enum"], json!([{"custom":[1,null]}]));
        }
    }
}

#[rstest]
fn schema_maps_keep_wire_key_order() {
    let wire = r#"{"type":"object","properties":{"zeta":{"type":"string"},"alpha":{"type":"integer"}},"$defs":{"y":true,"b":false}}"#;
    let schema: JsonSchemaObject = serde_json::from_str(wire).unwrap();
    let names: Vec<&str> = schema
        .properties
        .as_ref()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    assert_eq!(names, ["zeta", "alpha"]);
    assert_eq!(serde_json::to_string(&schema).unwrap(), wire);
}

#[rstest]
fn schema_object_round_trips_supported_keywords() {
    let schema = round_trip::<JsonSchemaObject>(json!({
        "type":"object",
        "properties":{"name":{"type":"string"}},
        "required":["name"],
        "additionalProperties":false,
        "$ref":"#/$defs/value",
        "strict":true,
        "extension":{"nested":[1,null]}
    }));
    assert_eq!(
        schema.schema_type,
        Some(JsonSchemaType::Name("object".into()))
    );
    assert_eq!(schema.required.as_deref(), Some(["name".into()].as_slice()));
    assert_eq!(
        schema.additional_properties.as_deref(),
        Some(&JsonSchema::Boolean(false))
    );
    assert_eq!(schema.reference.as_deref(), Some("#/$defs/value"));
    assert_eq!(schema.strict, Some(true));
}

#[rstest]
#[case::scalar(json!(7))]
#[case::properties_shape(json!({"properties":[]}))]
#[case::nested_schema(json!({"properties":{"name":7}}))]
#[case::schema_type(json!({"type":["string",7]}))]
#[case::items_shape(json!({"items":7}))]
#[case::tuple_member(json!({"items":[{"type":"string"},7]}))]
#[case::prefix_items_shape(json!({"prefixItems":{"type":"string"}}))]
#[case::composite_shape(json!({"anyOf":["string"]}))]
fn schemas_reject_malformed_known_keywords(#[case] wire: Value) {
    assert!(serde_json::from_value::<JsonSchema>(wire).is_err());
}

#[rstest]
#[case::draft_07_tuple(json!({"type":"array","items":[{"type":"string"},true]}))]
#[case::draft_2020_12_tuple(json!({"type":"array","prefixItems":[{"type":"string"},true],"items":false}))]
fn tuple_schemas_expose_positional_members(#[case] wire: Value) {
    let schema = round_trip::<JsonSchemaObject>(wire);
    let members = match (&schema.items, &schema.prefix_items) {
        (Some(JsonSchemaItems::Tuple(members)), None) => members,
        (Some(JsonSchemaItems::Schema(rest)), Some(members)) => {
            assert_eq!(rest.as_ref(), &JsonSchema::Boolean(false));
            members
        }
        _ => panic!("expected positional members"),
    };
    let [JsonSchema::Object(first), JsonSchema::Boolean(true)] = members.as_slice() else {
        panic!("expected string schema then true");
    };
    assert_eq!(
        first.schema_type,
        Some(JsonSchemaType::Name("string".into()))
    );
    assert!(schema.extra.is_empty());
}

#[rstest]
fn empty_schema_omits_null_optionals_and_preserves_arbitrary_keywords() {
    let schema: JsonSchemaObject = serde_json::from_value(
        json!({"type":null,"properties":null,"const":{"arbitrary":[1,null]},"future":null}),
    )
    .unwrap();
    assert!(schema.schema_type.is_none());
    assert!(schema.properties.is_none());
    assert_eq!(
        serde_json::to_value(schema).unwrap(),
        json!({"const":{"arbitrary":[1,null]},"future":null})
    );
}
