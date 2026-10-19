//! Python's schema helpers in `litellm/llms/vertex_ai/common_utils.py` and `litellm/utils.py`:
//! a JSON schema rewritten into the OpenAPI subset Gemini's `Schema` accepts.

use serde_json::{Map, Value, json};

use crate::base_llm::chat::response_format::unpack_defs;

/// The fields of Python's `Schema` TypedDict, all a `responseSchema` or tool parameter keeps.
const SCHEMA_FIELDS: &[&str] = &[
    "type",
    "format",
    "title",
    "description",
    "nullable",
    "default",
    "items",
    "minItems",
    "maxItems",
    "enum",
    "properties",
    "propertyOrdering",
    "required",
    "minProperties",
    "maxProperties",
    "minimum",
    "maximum",
    "minLength",
    "maxLength",
    "pattern",
    "example",
    "anyOf",
];

/// The fields `_convert_schema_types` moves into the object or array branch of a type list.
const TYPE_SPECIFIC_FIELDS: &[&str] = &[
    "properties",
    "required",
    "additionalProperties",
    "items",
    "minItems",
    "maxItems",
    "minProperties",
    "maxProperties",
];

/// `remove_additional_properties`: every `additionalProperties: false` removed.
pub fn remove_additional_properties(schema: &mut Value) {
    match schema {
        Value::Object(object) => {
            if object.get("additionalProperties") == Some(&Value::Bool(false)) {
                object.remove("additionalProperties");
            }
            object.values_mut().for_each(remove_additional_properties);
        }
        Value::Array(items) => items.iter_mut().for_each(remove_additional_properties),
        _ => {}
    }
}

/// `remove_strict_from_schema`: every `strict` key removed, at any depth.
pub fn remove_strict_from_schema(schema: &mut Value) {
    match schema {
        Value::Object(object) => {
            object.remove("strict");
            object.values_mut().for_each(remove_strict_from_schema);
        }
        Value::Array(items) => items.iter_mut().for_each(remove_strict_from_schema),
        _ => {}
    }
}

/// `build_vertex_schema`: `$defs` unpacked, type lists turned into `anyOf`, null branches
/// turned into `nullable`, enums kept only on strings, arrays given `items`, objects typed,
/// every field `Schema` does not know dropped, and `propertyOrdering` added when asked.
pub fn build_vertex_schema(mut parameters: Value, add_property_ordering: bool) -> Value {
    let defs = match parameters
        .as_object_mut()
        .and_then(|object| object.remove("$defs"))
    {
        Some(Value::Object(defs)) => defs,
        _ => Map::new(),
    };
    unpack_defs(&mut parameters, &defs);
    convert_schema_types(&mut parameters);
    convert_anyof_null_to_nullable(&mut parameters);
    fix_enum_empty_strings(&mut parameters);
    fix_enum_types(&mut parameters);
    process_items(&mut parameters);
    add_object_type(&mut parameters);
    let mut parameters = filter_schema_fields(parameters);
    if add_property_ordering {
        set_schema_property_ordering(&mut parameters);
    }
    parameters
}

fn is_type(value: Option<&Value>, name: &str) -> bool {
    value
        .and_then(Value::as_str)
        .is_some_and(|kind| kind.eq_ignore_ascii_case(name))
}

/// `_convert_schema_types`: a type list of several entries becomes `anyOf`, a one-entry list
/// its only entry.
fn convert_schema_types(schema: &mut Value) {
    let Value::Object(object) = schema else {
        return;
    };
    match object.get("type") {
        Some(Value::Array(types)) if types.len() > 1 => {
            let types: Vec<String> = types
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_owned)
                .collect();
            let any_of: Vec<Value> = types
                .iter()
                .map(|kind| match kind.as_str() {
                    "object" | "array" => {
                        let mut branch = Map::from_iter([("type".to_owned(), json!(kind))]);
                        for field in TYPE_SPECIFIC_FIELDS {
                            if let Some(value) = object.get(*field) {
                                branch.insert((*field).to_owned(), value.clone());
                            }
                        }
                        Value::Object(branch)
                    }
                    _ => json!({"type": kind}),
                })
                .collect();
            if types.iter().any(|kind| kind == "object" || kind == "array") {
                for field in TYPE_SPECIFIC_FIELDS {
                    object.remove(*field);
                }
            }
            object.insert("anyOf".into(), Value::Array(any_of));
            object.remove("type");
        }
        Some(Value::Array(types)) if types.len() == 1 => {
            let only = types[0].clone();
            object.insert("type".into(), only);
        }
        _ => {}
    }
    if let Some(Value::Object(properties)) = object.get_mut("properties") {
        properties.values_mut().for_each(convert_schema_types);
    }
    if let Some(items) = object.get_mut("items") {
        convert_schema_types(items);
    }
    if let Some(Value::Array(any_of)) = object.get_mut("anyOf") {
        any_of.iter_mut().for_each(convert_schema_types);
    }
}

/// `convert_anyof_null_to_nullable`: a `null` branch of `anyOf` removed and every other
/// branch marked `nullable`.
fn convert_anyof_null_to_nullable(schema: &mut Value) {
    let Value::Object(object) = schema else {
        return;
    };
    if let Some(Value::Array(any_of)) = object.get_mut("anyOf") {
        let is_null = |branch: &Value| branch.get("type").and_then(Value::as_str) == Some("null");
        let contains_null = any_of.iter().any(is_null);
        let non_null: Vec<Value> = any_of
            .iter()
            .filter(|branch| !is_null(branch))
            .cloned()
            .collect();
        if non_null.is_empty() {
            any_of.truncate(1);
        } else {
            *any_of = non_null;
        }
        for branch in any_of.iter_mut() {
            let Value::Object(fields) = branch else {
                continue;
            };
            if fields.is_empty() {
                fields.insert("type".into(), json!("object"));
            }
            if contains_null {
                fields.insert("nullable".into(), json!(true));
            }
            convert_anyof_null_to_nullable(branch);
        }
    }
    if let Some(Value::Object(properties)) = object.get_mut("properties") {
        properties
            .values_mut()
            .for_each(convert_anyof_null_to_nullable);
    }
    if let Some(items) = object.get_mut("items") {
        convert_anyof_null_to_nullable(items);
    }
}

/// `_fix_enum_empty_strings`: Gemini rejects `""` in an enum, so it becomes `null`.
fn fix_enum_empty_strings(schema: &mut Value) {
    let Value::Object(object) = schema else {
        return;
    };
    if let Some(Value::Array(values)) = object.get_mut("enum") {
        for value in values.iter_mut() {
            if value.as_str() == Some("") {
                *value = Value::Null;
            }
        }
    }
    if let Some(Value::Object(properties)) = object.get_mut("properties") {
        properties.values_mut().for_each(fix_enum_empty_strings);
    }
    if let Some(items) = object.get_mut("items") {
        fix_enum_empty_strings(items);
    }
}

/// `_fix_enum_types`: an enum survives only on a string type, or an `anyOf` with one.
fn fix_enum_types(schema: &mut Value) {
    let Value::Object(object) = schema else {
        return;
    };
    if matches!(object.get("enum"), Some(Value::Array(_))) {
        let keep = is_type(object.get("type"), "string")
            || matches!(object.get("anyOf"), Some(Value::Array(branches))
                if branches.iter().any(|branch| is_type(branch.get("type"), "string")));
        if !keep {
            object.remove("enum");
        }
    }
    if let Some(Value::Object(properties)) = object.get_mut("properties") {
        properties.values_mut().for_each(fix_enum_types);
    }
    if let Some(items) = object.get_mut("items") {
        fix_enum_types(items);
    }
    if let Some(Value::Array(any_of)) = object.get_mut("anyOf") {
        any_of.iter_mut().for_each(fix_enum_types);
    }
}

/// `process_items`: every array, at any depth, gets `items`, `{"type": "object"}` when it has
/// none or an empty one.
fn process_items(schema: &mut Value) {
    let Value::Object(object) = schema else {
        return;
    };
    let empty_items = match object.get("items") {
        None => true,
        Some(Value::Object(items)) => items.is_empty(),
        Some(_) => false,
    };
    if is_type(object.get("type"), "array") && empty_items {
        object.insert("items".into(), json!({"type": "object"}));
    }
    for value in object.values_mut() {
        match value {
            Value::Object(_) => process_items(value),
            Value::Array(items) => items.iter_mut().for_each(process_items),
            _ => {}
        }
    }
}

/// `add_object_type`: an untyped schema is an object, as is anything with properties; empty
/// properties are dropped.
fn add_object_type(schema: &mut Value) {
    let Value::Object(object) = schema else {
        return;
    };
    if !["type", "anyOf", "oneOf", "allOf"]
        .iter()
        .any(|key| object.contains_key(*key))
    {
        object.insert("type".into(), json!("object"));
    }
    if let Some(properties) = object.get("properties") {
        let empty = match properties {
            Value::Object(properties) => properties.is_empty(),
            Value::Array(items) => items.is_empty(),
            Value::Null => true,
            Value::Bool(flag) => !flag,
            Value::String(text) => text.is_empty(),
            Value::Number(_) => false,
        };
        if object.get("required") == Some(&Value::Null) {
            object.remove("required");
        }
        if empty {
            object.remove("properties");
            object.remove("required");
        }
        object.insert("type".into(), json!("object"));
        if let Some(Value::Object(properties)) = object.get_mut("properties") {
            properties.values_mut().for_each(add_object_type);
        }
    }
    if let Some(items) = object.get_mut("items") {
        add_object_type(items);
    }
    for key in ["anyOf", "oneOf", "allOf"] {
        if let Some(Value::Array(branches)) = object.get_mut(key) {
            branches.iter_mut().for_each(add_object_type);
        }
    }
}

/// `filter_schema_fields` with `_filter_anyof_fields`: only `Schema`'s fields survive, `format`
/// only as `enum` or `date-time`, and a schema with `anyOf` keeps nothing else, its title and
/// description pushed into the branches.
fn filter_schema_fields(schema: Value) -> Value {
    let Value::Object(mut object) = schema else {
        return schema;
    };
    let any_of_set = object
        .get("anyOf")
        .is_some_and(crate::base_llm::chat::params::truthy);
    if any_of_set {
        let label = |key: &str| {
            object
                .get(key)
                .filter(|value| crate::base_llm::chat::params::truthy(value))
                .cloned()
        };
        let (title, description) = (label("title"), label("description"));
        let mut any_of = object.remove("anyOf").unwrap_or_default();
        if let Value::Array(branches) = &mut any_of
            && (title.is_some() || description.is_some())
            && branches.iter().all(Value::is_object)
        {
            for branch in branches.iter_mut() {
                if let Some(title) = &title {
                    branch["title"] = title.clone();
                }
                if let Some(description) = &description {
                    branch["description"] = description.clone();
                }
            }
        }
        object = Map::from_iter([("anyOf".to_owned(), any_of)]);
    }
    let filtered = object
        .into_iter()
        .filter(|(key, _)| SCHEMA_FIELDS.contains(&key.as_str()))
        .filter_map(|(key, value)| {
            let value = match (key.as_str(), value) {
                ("properties", Value::Object(properties)) => Value::Object(
                    properties
                        .into_iter()
                        .map(|(name, property)| (name, filter_schema_fields(property)))
                        .collect(),
                ),
                ("format", format) => match format.as_str() {
                    Some("enum" | "date-time") => format,
                    _ => return None,
                },
                ("items", items @ Value::Object(_)) => filter_schema_fields(items),
                ("anyOf", Value::Array(branches)) => {
                    Value::Array(branches.into_iter().map(filter_schema_fields).collect())
                }
                (_, value) => value,
            };
            Some((key, value))
        })
        .collect();
    Value::Object(filtered)
}

/// `set_schema_property_ordering`: each object lists its properties in the order given,
/// unless it already says.
fn set_schema_property_ordering(schema: &mut Value) {
    let Value::Object(object) = schema else {
        return;
    };
    if let Some(Value::Object(properties)) = object.get("properties") {
        let names: Vec<Value> = properties.keys().cloned().map(Value::String).collect();
        if !object.contains_key("propertyOrdering") {
            object.insert("propertyOrdering".into(), Value::Array(names));
        }
        if let Some(Value::Object(properties)) = object.get_mut("properties") {
            properties
                .values_mut()
                .for_each(set_schema_property_ordering);
        }
    }
    if let Some(items @ Value::Object(_)) = object.get_mut("items") {
        set_schema_property_ordering(items);
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    #[rstest]
    #[case::plain_object(
        json!({"type": "object", "properties": {"city": {"type": "string", "description": "c"}}, "required": ["city"], "additionalProperties": false}),
        json!({"type": "object", "properties": {"city": {"type": "string", "description": "c"}}, "required": ["city"]})
    )]
    #[case::type_list_with_null(
        json!({"type": "object", "properties": {"a": {"type": ["string", "null"]}}}),
        json!({"type": "object", "properties": {"a": {"anyOf": [{"type": "string", "nullable": true}]}}})
    )]
    #[case::anyof_null_keeps_only_anyof(
        json!({"type": "object", "properties": {"a": {"anyOf": [{"type": "integer"}, {"type": "null"}], "title": "A", "default": 1}}}),
        json!({"type": "object", "properties": {"a": {"anyOf": [{"type": "integer", "nullable": true, "title": "A"}]}}})
    )]
    #[case::refs_unpacked(
        json!({"type": "object", "properties": {"p": {"$ref": "#/$defs/P"}}, "$defs": {"P": {"type": "object", "properties": {"n": {"type": "string"}}}}}),
        json!({"type": "object", "properties": {"p": {"type": "object", "properties": {"n": {"type": "string"}}}}})
    )]
    #[case::enum_on_integer_dropped(
        json!({"type": "object", "properties": {"n": {"type": "integer", "enum": [1, 2]}, "s": {"type": "string", "enum": ["a", ""]}}}),
        json!({"type": "object", "properties": {"n": {"type": "integer"}, "s": {"type": "string", "enum": ["a", null]}}})
    )]
    #[case::array_without_items(
        json!({"type": "object", "properties": {"l": {"type": "array", "items": {}}}}),
        json!({"type": "object", "properties": {"l": {"type": "array", "items": {"type": "object"}}}})
    )]
    #[case::empty_properties(
        json!({"type": "object", "properties": {}, "required": []}),
        json!({"type": "object"})
    )]
    #[case::untyped(json!({}), json!({"type": "object"}))]
    #[case::format_filtered(
        json!({"type": "object", "properties": {"d": {"type": "string", "format": "date-time"}, "u": {"type": "string", "format": "uri"}}}),
        json!({"type": "object", "properties": {"d": {"type": "string", "format": "date-time"}, "u": {"type": "string"}}})
    )]
    fn tool_schemas_become_gemini_schemas(#[case] schema: Value, #[case] expected: Value) {
        assert_eq!(build_vertex_schema(schema, false), expected);
    }

    #[rstest]
    fn response_schemas_get_property_ordering() {
        assert_eq!(
            build_vertex_schema(
                json!({"type": "object", "properties": {"b": {"type": "string"}, "a": {"type": "object", "properties": {"y": {"type": "number"}, "x": {"type": "number"}}}}}),
                true
            ),
            json!({"type": "object", "properties": {"b": {"type": "string"}, "a": {"type": "object", "properties": {"y": {"type": "number"}, "x": {"type": "number"}}, "propertyOrdering": ["y", "x"]}}, "propertyOrdering": ["b", "a"]})
        );
    }

    #[rstest]
    #[case::strict(json!({"strict": true, "a": [{"strict": 1}]}), json!({"a": [{}]}))]
    fn strict_is_removed_everywhere(#[case] mut schema: Value, #[case] expected: Value) {
        remove_strict_from_schema(&mut schema);
        assert_eq!(schema, expected);
    }

    #[rstest]
    #[case::false_removed(json!({"additionalProperties": false, "p": {"additionalProperties": false}}), json!({"p": {}}))]
    #[case::true_kept(json!({"additionalProperties": true}), json!({"additionalProperties": true}))]
    fn additional_properties_false_is_removed(#[case] mut schema: Value, #[case] expected: Value) {
        remove_additional_properties(&mut schema);
        assert_eq!(schema, expected);
    }
}
