use schemars::{JsonSchema, Schema};
use serde_json::{Map, Value, json};

#[derive(JsonSchema)]
#[schemars(rename = "ModelEntry")]
struct ModelEntry {
    #[serde(flatten)]
    _attributes: crate::Attributes,
    #[serde(flatten)]
    _pricing: crate::PricingFields,
}

/// JSON Schema for one model entry, including registry validation constraints.
pub fn model_entry_json_schema() -> Schema {
    let mut schema = serde_json::to_value(schemars::schema_for!(ModelEntry))
        .expect("derived model schema serializes");
    remove_nullable_optional_fields(&mut schema);
    decorate_model_entry(&mut schema);
    Schema::from(
        schema
            .as_object()
            .expect("derived schema is an object")
            .clone(),
    )
}

/// JSON Schema for the complete model prices registry document.
pub fn registry_json_schema() -> Schema {
    let mut entry = model_entry_json_schema().as_value().clone();
    let mut definitions = take_definitions(&mut entry);
    entry.as_object_mut().unwrap().remove("$schema");
    definitions.insert("modelEntry".into(), entry);

    let mut registry = serde_json::to_value(schemars::schema_for!(crate::catalog::Registry))
        .expect("derived registry schema serializes");
    remove_nullable_optional_fields(&mut registry);
    definitions.extend(take_definitions(&mut registry));
    let root = registry.as_object_mut().unwrap();
    root.insert(
        "title".into(),
        "LiteLLM model prices and context window registry".into(),
    );
    root.insert(
        "additionalProperties".into(),
        json!({"$ref": "#/$defs/modelEntry"}),
    );
    root.insert("$defs".into(), Value::Object(definitions));
    Schema::from(root.clone())
}

fn take_definitions(schema: &mut Value) -> Map<String, Value> {
    schema
        .as_object_mut()
        .unwrap()
        .remove("$defs")
        .and_then(|value| value.as_object().cloned())
        .unwrap_or_default()
}

fn remove_nullable_optional_fields(value: &mut Value) {
    match value {
        Value::Array(values) => values.iter_mut().for_each(remove_nullable_optional_fields),
        Value::Object(map) => {
            map.values_mut().for_each(remove_nullable_optional_fields);
            if let Some(Value::Array(types)) = map.get_mut("type") {
                types.retain(|value| value != "null");
                if types.len() == 1 {
                    let only = types[0].clone();
                    map.insert("type".into(), only);
                }
            }
            if let Some(Value::Array(branches)) = map.get_mut("anyOf") {
                branches.retain(|branch| branch.get("type") != Some(&Value::String("null".into())));
                if branches.len() == 1 {
                    let only = branches[0]
                        .as_object()
                        .expect("schema branch is an object")
                        .clone();
                    map.remove("anyOf");
                    map.extend(only);
                }
            }
        }
        _ => {}
    }
}

fn decorate_model_entry(schema: &mut Value) {
    let object = schema.as_object_mut().unwrap();
    object.insert("required".into(), json!(["litellm_provider"]));
    object.insert("additionalProperties".into(), Value::Bool(true));
    object.insert(
        "patternProperties".into(),
        json!({
            crate::rate::key_pattern(): {"type": "number", "minimum": 0},
            "^supports_": {"type": "boolean"},
        }),
    );
    let properties = object
        .get_mut("properties")
        .unwrap()
        .as_object_mut()
        .unwrap();
    properties.insert(
        "aliases".into(),
        json!({"type": "array", "items": {"type": "string"}}),
    );
    properties.iter_mut().for_each(|(name, property)| {
        if name.contains("dbu_cost") {
            property["minimum"] = json!(0);
        }
    });

    let rates = json!({crate::rate::key_pattern(): {"type": "number", "minimum": 0}});
    let definitions = object.get_mut("$defs").unwrap().as_object_mut().unwrap();
    for definition in ["OffPeakPricing", "TieredRate"] {
        definitions[definition]["patternProperties"] = rates.clone();
        definitions[definition]["additionalProperties"] = Value::Bool(false);
    }
    definitions["OffPeakPricing"]["anyOf"] = json!([
        {"required": ["hours_utc"]},
        {"required": ["windows"]}
    ]);
    definitions["OffPeakPricing"]["properties"]["windows"]["minItems"] = json!(1);
    definitions["OffPeakWindow"]["properties"]["weekdays"]["minItems"] = json!(1);
    definitions["TieredRate"]["properties"]["range"]["items"]["minimum"] = json!(0);
    definitions["TieredRate"]["properties"]["max_results_range"]["items"]["minimum"] = json!(0);
    definitions["UtcHours"]["anyOf"][1]["minItems"] = json!(1);
}
