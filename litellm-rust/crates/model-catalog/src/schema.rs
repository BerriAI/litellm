use schemars::Schema;
use serde_json::{Map, Value, json};

/// JSON Schema for one model entry, including registry validation constraints.
pub fn model_entry_json_schema() -> Schema {
    let mut schema = serde_json::to_value(schemars::schema_for!(crate::ModelInfo))
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

    let mut fallback = serde_json::to_value(schemars::schema_for!(crate::FallbackGeneralizations))
        .expect("derived fallback schema serializes");
    remove_nullable_optional_fields(&mut fallback);
    definitions.extend(take_definitions(&mut fallback));
    fallback.as_object_mut().unwrap().remove("$schema");

    let root = json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "LiteLLM model prices and context window registry",
        "type": "object",
        "properties": {
            "sample_spec": {"type": "object"},
            "fallback_generalizations": fallback
        },
        "additionalProperties": {"$ref": "#/$defs/modelEntry"},
        "$defs": definitions
    });
    Schema::from(root.as_object().unwrap().clone())
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
    let properties = object
        .get_mut("properties")
        .unwrap()
        .as_object_mut()
        .unwrap();
    properties.insert(
        "aliases".into(),
        json!({"type": "array", "items": {"type": "string"}}),
    );
    properties.get_mut("deprecation_date").unwrap()["format"] = json!("date");
    properties.get_mut("deprecation_date").unwrap()["pattern"] =
        json!(r"^\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])$");

    properties.iter_mut().for_each(|(name, property)| {
        if name.contains("cost") {
            property["minimum"] = json!(0);
        } else if name.contains("uplift_multiplier") {
            property["minimum"] = json!(1);
        }
    });
    properties.get_mut("guardrail_cost_per_unit").unwrap()["additionalProperties"]["minimum"] =
        json!(0);

    let definitions = object.get_mut("$defs").unwrap().as_object_mut().unwrap();
    for definition in ["OffPeakPricing", "TieredRate", "SearchContextCostPerQuery"] {
        let properties = definitions[definition]["properties"]
            .as_object_mut()
            .unwrap();
        properties.iter_mut().for_each(|(name, property)| {
            if name.contains("cost") || definition == "SearchContextCostPerQuery" {
                property["minimum"] = json!(0);
            }
        });
    }
    definitions["OffPeakPricing"]["anyOf"] = json!([
        {"required": ["hours_utc"]},
        {"required": ["windows"]}
    ]);
    definitions["OffPeakPricing"]["properties"]["windows"]["minItems"] = json!(1);
    definitions["OffPeakWindow"]["properties"]["weekdays"]["minItems"] = json!(1);
    definitions["TieredRate"]["properties"]["range"]["items"]["minimum"] = json!(0);
    definitions["TieredRate"]["properties"]["max_results_range"]["items"]["minimum"] = json!(0);
    definitions["Weekday"]["anyOf"][0]["minimum"] = json!(1);
    definitions["Weekday"]["anyOf"][0]["maximum"] = json!(7);
    definitions["Weekday"]["anyOf"][1]["pattern"] = json!(
        r"(?i)^(mon|monday|tue|tues|tuesday|wed|wednesday|thu|thur|thurs|thursday|fri|friday|sat|saturday|sun|sunday)$"
    );
    let window_pattern = json!(r"^([01]\d|2[0-3]):[0-5]\d-([01]\d|2[0-3]):[0-5]\d$");
    definitions["UtcHours"]["anyOf"][0]["pattern"] = window_pattern.clone();
    definitions["UtcHours"]["anyOf"][1]["items"]["pattern"] = window_pattern;
    definitions["UtcHours"]["anyOf"][1]["minItems"] = json!(1);
}
