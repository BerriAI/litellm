use std::collections::{BTreeMap, BTreeSet};

use litellm_python_compat::{json as python_json, repr};
use serde_json::{Map, Value, json};

use crate::{Error, ErrorDetail, base_llm::messages::context::NativeOutputSupport};

use super::request::{Body, is_int, python_truthy};

const NATIVE_OUTPUT_MAX_PROPERTIES: usize = 64;
const NATIVE_OUTPUT_MAX_UNION_PARAMETERS: usize = 16;
const NATIVE_OUTPUT_MAX_OPTIONAL_PARAMETERS: usize = 24;
const NATIVE_OUTPUT_MAX_GRAMMAR_BRANCHES: u128 = 2048;
const MAX_SCHEMA_NESTING: usize = 1024;
const NATIVE_OUTPUT_STRING_FORMATS: [&str; 10] = [
    "date-time",
    "time",
    "date",
    "duration",
    "email",
    "hostname",
    "uri",
    "ipv4",
    "ipv6",
    "uuid",
];
const SUBSCHEMA_KEYWORDS: [&str; 11] = [
    "additionalItems",
    "additionalProperties",
    "contains",
    "else",
    "if",
    "items",
    "not",
    "propertyNames",
    "then",
    "unevaluatedItems",
    "unevaluatedProperties",
];
const SUBSCHEMA_LIST_KEYWORDS: [&str; 5] = ["allOf", "anyOf", "items", "oneOf", "prefixItems"];
const SUBSCHEMA_MAP_KEYWORDS: [&str; 5] = [
    "$defs",
    "definitions",
    "dependentSchemas",
    "patternProperties",
    "properties",
];
const CONSTRAINT_LABELS: [(&str, &str); 25] = [
    ("minItems", "minimum number of items: {}"),
    ("maxItems", "maximum number of items: {}"),
    ("uniqueItems", "all array items must be unique"),
    ("contains", "array must contain an item matching: {}"),
    ("minContains", "minimum number of matching items: {}"),
    ("maxContains", "maximum number of matching items: {}"),
    ("prefixItems", "leading items must match, in order: {}"),
    ("minimum", "minimum value: {}"),
    ("maximum", "maximum value: {}"),
    ("exclusiveMinimum", "exclusive minimum value: {}"),
    ("exclusiveMaximum", "exclusive maximum value: {}"),
    ("multipleOf", "must be a multiple of {}"),
    ("minLength", "minimum length: {}"),
    ("maxLength", "maximum length: {}"),
    ("minProperties", "minimum number of properties: {}"),
    ("maxProperties", "maximum number of properties: {}"),
    (
        "patternProperties",
        "properties whose names match each pattern must satisfy: {}",
    ),
    ("propertyNames", "property names must satisfy: {}"),
    ("dependentRequired", "dependent required properties: {}"),
    ("dependentSchemas", "dependent schemas: {}"),
    (
        "unevaluatedProperties",
        "unevaluated properties must satisfy: {}",
    ),
    ("if", "conditional (if): {}"),
    ("then", "conditional (then): {}"),
    ("else", "conditional (else): {}"),
    ("not", "must not match: {}"),
];

type Schema = Map<String, Value>;

fn subschemas(node: &Schema) -> impl Iterator<Item = &Schema> {
    node.iter().flat_map(|(key, value)| {
        let key = key.as_str();
        let children: Vec<&Schema> = match value {
            Value::Object(entries) if SUBSCHEMA_MAP_KEYWORDS.contains(&key) => {
                entries.values().filter_map(Value::as_object).collect()
            }
            Value::Array(entries) if SUBSCHEMA_LIST_KEYWORDS.contains(&key) => {
                entries.iter().filter_map(Value::as_object).collect()
            }
            Value::Object(child) if SUBSCHEMA_KEYWORDS.contains(&key) => vec![child],
            _ => Vec::new(),
        };
        children
    })
}

fn schema_nodes(schema: &Schema) -> Vec<&Schema> {
    let mut nodes = Vec::new();
    let mut frontier = vec![schema];
    let mut depth = 0;
    while !frontier.is_empty() && depth < MAX_SCHEMA_NESTING {
        let next: Vec<&Schema> = frontier.iter().flat_map(|node| subschemas(node)).collect();
        nodes.extend(frontier);
        frontier = next;
        depth += 1;
    }
    nodes
}

fn schema_at<'a>(node: &'a Schema, key: &str) -> Option<&'a Schema> {
    node.get(key).and_then(Value::as_object)
}

fn list_at<'a>(node: &'a Schema, key: &str) -> &'a [Value] {
    match node.get(key) {
        Some(Value::Array(values)) => values,
        _ => &[],
    }
}

fn schema_fragment_ref(reference: &str) -> String {
    if reference.starts_with("/$defs/") || reference.starts_with("/definitions/") {
        format!("#{reference}")
    } else {
        reference.to_string()
    }
}

fn refs_in(schema: &Schema) -> BTreeSet<String> {
    schema_nodes(schema)
        .into_iter()
        .filter_map(|node| node.get("$ref").and_then(Value::as_str))
        .map(schema_fragment_ref)
        .collect()
}

fn root_definitions(schema: &Schema) -> Vec<(String, &Schema)> {
    ["$defs", "definitions"]
        .into_iter()
        .filter_map(|defs_key| schema_at(schema, defs_key).map(|defs| (defs_key, defs)))
        .flat_map(|(defs_key, defs)| {
            defs.iter().filter_map(move |(name, body)| {
                body.as_object()
                    .map(|body| (format!("#/{defs_key}/{name}"), body))
            })
        })
        .collect()
}

fn has_cycle(graph: &BTreeMap<String, BTreeSet<String>>) -> bool {
    let mut remaining: BTreeMap<&String, BTreeSet<&String>> = graph
        .iter()
        .map(|(node, edges)| {
            (
                node,
                edges
                    .iter()
                    .filter(|edge| graph.contains_key(*edge))
                    .collect(),
            )
        })
        .collect();
    loop {
        let ready: Vec<&String> = remaining
            .iter()
            .filter(|(_, edges)| edges.is_empty())
            .map(|(node, _)| *node)
            .collect();
        if ready.is_empty() {
            return !remaining.is_empty();
        }
        ready.iter().for_each(|node| {
            remaining.remove(node);
        });
        remaining.values_mut().for_each(|edges| {
            ready.iter().for_each(|node| {
                edges.remove(node);
            });
        });
    }
}

fn schema_has_unresolvable_ref(schema: &Schema) -> bool {
    let definitions: BTreeMap<String, &Schema> = root_definitions(schema).into_iter().collect();
    if !refs_in(schema)
        .iter()
        .all(|reference| definitions.contains_key(reference))
    {
        return true;
    }
    has_cycle(
        &definitions
            .iter()
            .map(|(pointer, body)| (pointer.clone(), refs_in(body)))
            .collect(),
    )
}

fn is_object_schema(node: &Schema) -> bool {
    match node.get("type") {
        Some(Value::String(kind)) => kind == "object",
        Some(Value::Array(kinds)) => kinds.iter().any(|kind| kind.as_str() == Some("object")),
        _ => false,
    }
}

fn is_open_object(node: &Schema) -> bool {
    let additional = node.get("additionalProperties");
    if additional == Some(&Value::Bool(true))
        || additional.is_some_and(Value::is_object)
        || node.contains_key("patternProperties")
    {
        return true;
    }
    is_object_schema(node)
        && !node.contains_key("properties")
        && additional != Some(&Value::Bool(false))
}

fn uses_regex_lookaround(pattern: &str) -> bool {
    ["(?=", "(?!", "(?<=", "(?<!"]
        .iter()
        .any(|lookaround| pattern.contains(lookaround))
}

fn may_be_string(node: &Schema) -> bool {
    match node.get("type") {
        None => true,
        Some(Value::String(kind)) => kind == "string",
        Some(_) => list_at(node, "type")
            .iter()
            .any(|kind| kind.as_str() == Some("string")),
    }
}

fn optional_property_names(node: &Schema) -> BTreeSet<&str> {
    let Some(properties) = schema_at(node, "properties") else {
        return BTreeSet::new();
    };
    let required: BTreeSet<&str> = list_at(node, "required")
        .iter()
        .filter_map(Value::as_str)
        .collect();
    properties
        .keys()
        .map(String::as_str)
        .filter(|name| !required.contains(name))
        .collect()
}

fn is_union(node: &Schema) -> bool {
    node.get("anyOf").is_some_and(Value::is_array)
        || node.get("oneOf").is_some_and(Value::is_array)
        || list_at(node, "type").len() > 1
}

fn grammar_branches(node: &Schema) -> u128 {
    let Some(properties) = schema_at(node, "properties") else {
        return 0;
    };
    let optional = optional_property_names(node).len();
    let required = (properties.len() - optional.min(properties.len())) as u128 + 1;
    u32::try_from(optional)
        .ok()
        .and_then(|shift| 1u128.checked_shl(shift))
        .and_then(|branches| branches.checked_mul(required))
        .unwrap_or(u128::MAX)
}

pub(super) fn schema_property_count(schema: &Schema) -> usize {
    schema_nodes(schema)
        .into_iter()
        .filter_map(|node| schema_at(node, "properties"))
        .map(Map::len)
        .sum()
}

fn native_output_rejects(schema: &Schema) -> bool {
    let nodes = schema_nodes(schema);
    let string_formats_supported = nodes
        .iter()
        .filter(|node| may_be_string(node))
        .filter_map(|node| node.get("format").and_then(Value::as_str))
        .all(|format| NATIVE_OUTPUT_STRING_FORMATS.contains(&format));
    let union_properties = nodes
        .iter()
        .filter_map(|node| schema_at(node, "properties"))
        .flat_map(|properties| properties.values().filter_map(Value::as_object))
        .filter(|property| is_union(property))
        .count();
    let optional_properties: usize = nodes
        .iter()
        .map(|node| optional_property_names(node).len())
        .sum();
    let grammar_branches = nodes
        .iter()
        .map(|node| grammar_branches(node))
        .fold(0u128, u128::saturating_add);
    schema_has_unresolvable_ref(schema)
        || nodes.iter().any(|node| is_open_object(node))
        || nodes.iter().any(|node| {
            node.get("pattern")
                .and_then(Value::as_str)
                .is_some_and(uses_regex_lookaround)
        })
        || !string_formats_supported
        || union_properties > NATIVE_OUTPUT_MAX_UNION_PARAMETERS
        || optional_properties > NATIVE_OUTPUT_MAX_OPTIONAL_PARAMETERS
        || grammar_branches > NATIVE_OUTPUT_MAX_GRAMMAR_BRANCHES
}

fn enum_conflicts_with_declared_type(schema: &Schema) -> bool {
    let Some(Value::Array(values)) = schema.get("enum") else {
        return false;
    };
    let declared = match schema.get("type") {
        None | Some(Value::Null) => return false,
        Some(Value::Array(_)) => return true,
        Some(Value::String(kind)) => kind.as_str(),
        Some(_) => return false,
    };
    let check: fn(&Value) -> bool = match declared {
        "null" => Value::is_null,
        "boolean" => Value::is_boolean,
        "integer" => is_int,
        "number" => Value::is_number,
        "string" => Value::is_string,
        "array" => Value::is_array,
        "object" => Value::is_object,
        _ => return false,
    };
    !values.iter().all(check)
}

fn note_value(value: &Value) -> Result<String, Error> {
    let python = python_json::from_json(value.clone());
    match value {
        Value::Object(_) | Value::Array(_) => python_json::dumps(&python).map_err(|error| {
            Error::InvalidRequest(ErrorDetail::failed(
                "output schema note serialization",
                error,
            ))
        }),
        _ => Ok(repr::to_str(&python)),
    }
}

fn described_with_note(description: Option<&Value>, note: String) -> Result<Value, Error> {
    match description {
        Some(Value::String(text)) if !text.is_empty() => {
            Ok(Value::String(format!("{text} {note}")))
        }
        Some(other) if python_truthy(other) => {
            Err(Error::Declined(litellm_core_utils::decline::Decline::new(
                "a non-string output schema description next to a dropped constraint",
            )))
        }
        _ => Ok(Value::String(note)),
    }
}

fn filtered_subschema(value: &Value) -> Result<Value, Error> {
    match value {
        Value::Object(schema) => filter_output_schema(schema).map(Value::Object),
        other => Ok(other.clone()),
    }
}

fn filtered_subschemas(value: &Value) -> Result<Value, Error> {
    match value {
        Value::Object(entries) => entries
            .iter()
            .map(|(name, schema)| Ok((name.clone(), filtered_subschema(schema)?)))
            .collect::<Result<Map<_, _>, Error>>()
            .map(Value::Object),
        other => Ok(other.clone()),
    }
}

fn filtered_branches(branches: &[Value]) -> Result<Vec<Value>, Error> {
    branches.iter().map(filtered_subschema).collect()
}

fn extended_any_of(result: &Schema, branches: &[Value]) -> Result<Value, Error> {
    let filtered = filtered_branches(branches)?;
    match result.get("anyOf") {
        Some(Value::Array(existing)) => Ok(Value::Array(
            existing.iter().cloned().chain(filtered).collect(),
        )),
        Some(_) => Err(Error::Declined(litellm_core_utils::decline::Decline::new(
            "an output schema whose anyOf cannot be extended",
        ))),
        None => Ok(Value::Array(filtered)),
    }
}

pub(super) fn filter_output_schema(schema: &Schema) -> Result<Schema, Error> {
    let enforced_min_items = schema
        .get("minItems")
        .filter(|value| is_int(value))
        .and_then(Value::as_i64)
        .is_some_and(|count| count == 0 || count == 1);
    let unsupported = |field: &str| {
        CONSTRAINT_LABELS.iter().any(|(label, _)| *label == field)
            && !(field == "minItems" && enforced_min_items)
    };
    let notes = CONSTRAINT_LABELS
        .iter()
        .filter(|(field, _)| unsupported(field))
        .filter_map(|(field, label)| schema.get(*field).map(|value| (label, value)))
        .filter(|(_, value)| **value != Value::Bool(false))
        .map(|(label, value)| Ok(label.replacen("{}", &note_value(value)?, 1)))
        .collect::<Result<Vec<String>, Error>>()?;
    let mut result = Map::new();
    if !notes.is_empty() {
        let note = format!("Note: {}.", notes.join(", "));
        result.insert(
            "description".into(),
            described_with_note(schema.get("description"), note)?,
        );
    }
    let drops_conflicting_type = enum_conflicts_with_declared_type(schema);
    for (key, value) in schema {
        let key = key.as_str();
        if unsupported(key)
            || (key == "type" && drops_conflicting_type)
            || (key == "description" && result.contains_key("description"))
        {
            continue;
        }
        let filtered = match (key, value) {
            ("properties" | "$defs" | "definitions", _) => filtered_subschemas(value)?,
            ("items", _) => filtered_subschema(value)?,
            ("anyOf" | "oneOf", Value::Array(branches)) => {
                let extended = extended_any_of(&result, branches)?;
                result.insert("anyOf".into(), extended);
                continue;
            }
            ("allOf", Value::Array(branches)) => Value::Array(filtered_branches(branches)?),
            ("$ref", Value::String(reference)) => Value::String(schema_fragment_ref(reference)),
            _ => value.clone(),
        };
        result.insert(key.to_string(), filtered);
    }
    if is_object_schema(&result) {
        result.insert("additionalProperties".into(), Value::Bool(false));
    }
    Ok(result)
}

pub(super) fn native_output_schema(schema: &Schema) -> Result<Option<Schema>, Error> {
    if native_output_rejects(schema) {
        return Ok(None);
    }
    filter_output_schema(schema).map(Some)
}

pub(super) fn native_output_format(
    output_format: &Schema,
    native: NativeOutputSupport,
) -> Result<Option<Schema>, Error> {
    let Some(Value::Object(schema)) = output_format.get("schema") else {
        return Ok((native.enabled || native.opt_in).then(|| output_format.clone()));
    };
    let cleaned = if !native.mapped && schema_property_count(schema) > NATIVE_OUTPUT_MAX_PROPERTIES
    {
        None
    } else {
        native_output_schema(schema)?
    };
    let native_schema = if native.enabled {
        Some(cleaned.unwrap_or_else(|| schema.clone()))
    } else if native.opt_in {
        cleaned
    } else {
        None
    };
    Ok(native_schema.map(|native_schema| {
        let mut format = output_format.clone();
        format.insert("schema".into(), Value::Object(native_schema));
        format
    }))
}

fn pop_output_config_format(body: &mut Body) -> Option<Schema> {
    let Some(Value::Object(output_config)) = body.get_mut("output_config") else {
        return None;
    };
    let format = output_config.shift_remove("format");
    if output_config.is_empty() {
        body.shift_remove("output_config");
    }
    match format {
        Some(Value::Object(format)) => Some(format),
        _ => None,
    }
}

fn inline_schema(output_format: &Schema, body: &mut Body) -> Result<(), Error> {
    let Some(schema) = output_format
        .get("schema")
        .filter(|schema| python_truthy(schema))
    else {
        return Ok(());
    };
    let text = python_json::dumps(&python_json::from_json(schema.clone())).map_err(|error| {
        Error::InvalidRequest(ErrorDetail::failed("output schema serialization", error))
    })?;
    let schema_block = json!({"type": "text", "text": text});
    let Some(Value::Array(messages)) = body.get_mut("messages") else {
        return Ok(());
    };
    let Some(last_user) = messages
        .iter_mut()
        .rev()
        .find(|message| message.get("role").and_then(Value::as_str) == Some("user"))
        .and_then(Value::as_object_mut)
    else {
        return Ok(());
    };
    let content = match last_user.get("content") {
        None => Value::Array(vec![schema_block]),
        Some(Value::String(text)) => json!([{"type": "text", "text": text}, schema_block]),
        Some(Value::Array(blocks)) => {
            Value::Array(blocks.iter().cloned().chain([schema_block]).collect())
        }
        Some(_) => return Ok(()),
    };
    last_user.insert("content".into(), content);
    Ok(())
}

pub(super) fn apply_structured_output(
    body: &mut Body,
    model: &str,
    native: NativeOutputSupport,
) -> Result<(), Error> {
    let legacy = body.shift_remove("output_format");
    let output_config_format = pop_output_config_format(body);
    let schema_format = match legacy {
        Some(Value::Object(legacy)) => Some(legacy),
        _ => output_config_format,
    };
    let Some(schema_format) = schema_format else {
        return Ok(());
    };
    if let Some(native_format) = native_output_format(&schema_format, native)? {
        match body.get_mut("output_config") {
            Some(Value::Object(output_config)) => {
                output_config.insert("format".into(), Value::Object(native_format));
            }
            _ => {
                body.insert("output_config".into(), json!({"format": native_format}));
            }
        }
        return Ok(());
    }
    litellm_tracing::warn!(
        "Bedrock Invoke: model={model} does not take native structured output for this schema per model_prices_and_context_window.json, so the JSON schema was inlined into the last user message and is NOT enforced by the model."
    );
    inline_schema(&schema_format, body)
}

fn requests_output_config_format(params: &Map<String, Value>) -> bool {
    matches!(params.get("output_config"), Some(Value::Object(output_config)) if output_config.contains_key("format"))
}

pub(super) fn requested_output_format(params: &Map<String, Value>) -> Option<&Schema> {
    match params.get("output_format") {
        Some(Value::Object(legacy)) => Some(legacy),
        _ => params
            .get("output_config")
            .and_then(Value::as_object)
            .and_then(|output_config| output_config.get("format"))
            .and_then(Value::as_object),
    }
}

pub(super) fn native_output_retry_requested(
    params: &Map<String, Value>,
    native: NativeOutputSupport,
) -> bool {
    (params.get("output_format").is_some_and(Value::is_object)
        || requests_output_config_format(params))
        && native.retries()
}

pub(super) fn sent_output_config_format(body: &Value) -> bool {
    body.as_object().is_some_and(requests_output_config_format)
}

pub(super) const NATIVE_OUTPUT_UNSUPPORTED: &str =
    "output_config.format: Extra inputs are not permitted";

const NATIVE_OUTPUT_REJECTIONS: [&str; 4] = [
    NATIVE_OUTPUT_UNSUPPORTED,
    "compilation exceeded limits: Schema is too complex",
    "The compiled grammar is too large",
    "output_config.format.schema:",
];

pub(super) fn is_native_output_rejection(status: u16, error_text: &str) -> bool {
    status == 400
        && NATIVE_OUTPUT_REJECTIONS
            .iter()
            .any(|rejection| error_text.contains(rejection))
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    fn schema(value: Value) -> Schema {
        value.as_object().unwrap().clone()
    }

    fn legacy() -> NativeOutputSupport {
        NativeOutputSupport {
            enabled: true,
            mapped: true,
            ..NativeOutputSupport::default()
        }
    }

    fn bedrock_opt_in() -> NativeOutputSupport {
        NativeOutputSupport {
            opt_in: true,
            ..NativeOutputSupport::default()
        }
    }

    fn tree() -> Value {
        json!({
            "type": "object",
            "properties": {"root": {"$ref": "#/$defs/Node"}},
            "$defs": {"Node": {"type": "object", "properties": {"children": {"type": "array", "items": {"$ref": "#/$defs/Node"}}}}},
        })
    }

    fn open_object() -> Value {
        json!({
            "type": "object",
            "properties": {"counts": {"type": "object", "additionalProperties": {"type": "integer"}}},
        })
    }

    fn many_properties(count: usize) -> Value {
        let properties: Map<String, Value> = (0..count)
            .map(|index| (format!("p{index}"), json!({"type": "string"})))
            .collect();
        let required: Vec<Value> = (0..count).map(|index| json!(format!("p{index}"))).collect();
        json!({"type": "object", "properties": properties, "required": required, "additionalProperties": false})
    }

    fn sent(native: NativeOutputSupport, raw: Value) -> Body {
        let mut body = schema(json!({
            "messages": [{"role": "user", "content": "Hello"}],
            "output_config": {"format": {"type": "json_schema", "schema": raw}},
        }));
        apply_structured_output(&mut body, "model", native).unwrap();
        body
    }

    #[test]
    fn native_output_drops_schema_keywords_bedrock_rejects() {
        let body = sent(
            bedrock_opt_in(),
            json!({
                "type": "object",
                "properties": {
                    "count": {"type": "integer", "minimum": 1},
                    "ids": {"type": "array", "items": {"type": "string"}, "uniqueItems": true},
                },
                "required": ["count", "ids"],
            }),
        );
        assert_eq!(
            body["output_config"]["format"]["schema"],
            json!({
                "type": "object",
                "properties": {
                    "count": {"description": "Note: minimum value: 1.", "type": "integer"},
                    "ids": {"description": "Note: all array items must be unique.", "type": "array", "items": {"type": "string"}},
                },
                "required": ["count", "ids"],
                "additionalProperties": false,
            })
        );
    }

    #[rstest]
    #[case::recursive_on_a_bedrock_profile(bedrock_opt_in(), tree(), false)]
    #[case::open_object_on_a_bedrock_profile(bedrock_opt_in(), open_object(), false)]
    #[case::recursive_on_a_legacy_profile(legacy(), tree(), true)]
    #[case::open_object_on_a_legacy_profile(legacy(), open_object(), true)]
    fn schemas_native_output_cannot_hold(
        #[case] native: NativeOutputSupport,
        #[case] raw: Value,
        #[case] forwarded: bool,
    ) {
        let body = sent(native, raw.clone());
        if forwarded {
            assert_eq!(body["output_config"]["format"]["schema"], raw);
        } else {
            assert_eq!(body.get("output_config"), None);
            let text = body["messages"][0]["content"][1]["text"].as_str().unwrap();
            assert_eq!(serde_json::from_str::<Value>(text).unwrap(), raw);
        }
    }

    #[test]
    fn shared_defs_reach_native_output_once() {
        let party = json!({"$ref": "#/$defs/Party"});
        let body = sent(
            legacy(),
            json!({
                "type": "object",
                "properties": {"buyer": party, "seller": party},
                "required": ["buyer", "seller"],
                "additionalProperties": false,
                "$defs": {"Party": {"type": "object", "properties": {"email": {"anyOf": [{"type": "string"}, {"type": "null"}]}}, "required": ["email"], "additionalProperties": false}},
            }),
        );
        let sent_schema = &body["output_config"]["format"]["schema"];
        assert_eq!(sent_schema["properties"]["buyer"], party);
        assert_eq!(sent_schema.to_string().matches("\"anyOf\"").count(), 1);
    }

    #[rstest]
    #[case::unmapped_over_the_limit(NativeOutputSupport { opt_in: true, ..NativeOutputSupport::default() }, 65, false)]
    #[case::unmapped_at_the_limit(NativeOutputSupport { opt_in: true, ..NativeOutputSupport::default() }, 24, true)]
    #[case::mapped_over_the_limit(NativeOutputSupport { opt_in: true, mapped: true, ..NativeOutputSupport::default() }, 65, true)]
    fn property_count_rule_applies_to_unmapped_models(
        #[case] native: NativeOutputSupport,
        #[case] properties: usize,
        #[case] native_sent: bool,
    ) {
        assert_eq!(
            sent(native, many_properties(properties)).contains_key("output_config"),
            native_sent
        );
    }

    #[test]
    fn legacy_output_format_wins() {
        let mut body = schema(json!({
            "messages": [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}],
            "output_format": {"type": "json_schema", "schema": {"type": "object", "properties": {"legacy": {"type": "string"}}}},
            "output_config": {"effort": "high", "format": {"type": "json_schema", "schema": {"type": "object", "properties": {"new": {"type": "string"}}}}},
        }));
        apply_structured_output(&mut body, "model", NativeOutputSupport::default()).unwrap();
        assert_eq!(body["output_config"], json!({"effort": "high"}));
        assert_eq!(
            body["messages"][0]["content"][1]["text"],
            json!(r#"{"type": "object", "properties": {"legacy": {"type": "string"}}}"#)
        );
    }

    #[rstest]
    #[case::one_of_becomes_any_of(
        json!({"oneOf": [{"type": "string"}, {"type": "integer", "maximum": 3}]}),
        json!({"anyOf": [{"type": "string"}, {"description": "Note: maximum value: 3.", "type": "integer"}]}),
    )]
    #[case::small_min_items_is_kept(
        json!({"type": "array", "minItems": 1, "maxItems": 4}),
        json!({"description": "Note: maximum number of items: 4.", "type": "array", "minItems": 1}),
    )]
    #[case::false_constraints_add_no_note(
        json!({"type": "array", "uniqueItems": false}),
        json!({"type": "array"}),
    )]
    #[case::conflicting_enum_type_is_dropped(
        json!({"type": "string", "enum": [1, 2]}),
        json!({"enum": [1, 2]}),
    )]
    #[case::sub_schema_constraints_note_their_json(
        json!({"type": "array", "contains": {"type": "string"}, "description": "Items"}),
        json!({"description": "Items Note: array must contain an item matching: {\"type\": \"string\"}.", "type": "array"}),
    )]
    #[case::ref_becomes_a_fragment(
        json!({"$ref": "/$defs/Node"}),
        json!({"$ref": "#/$defs/Node"}),
    )]
    fn output_schema_filter_matches_the_anthropic_sdk(#[case] raw: Value, #[case] expected: Value) {
        assert_eq!(
            Value::Object(filter_output_schema(&schema(raw)).unwrap()),
            expected
        );
    }

    #[rstest]
    #[case::lookaround(json!({"type": "string", "pattern": "^(?!x)"}), true)]
    #[case::unsupported_format(json!({"type": "string", "format": "regex"}), true)]
    #[case::supported_format(json!({"type": "string", "format": "email"}), false)]
    #[case::unresolvable_ref(json!({"$ref": "#/$defs/Missing"}), true)]
    #[case::too_many_optional(json!({"type": "object", "properties": (0..25).map(|i| (format!("p{i}"), json!({"type": "string"}))).collect::<Map<String, Value>>(), "additionalProperties": false}), true)]
    fn native_output_rejects_what_the_grammar_cannot_compile(
        #[case] raw: Value,
        #[case] rejected: bool,
    ) {
        assert_eq!(native_output_rejects(&schema(raw)), rejected);
    }

    #[rstest]
    #[case::extra_inputs(400, "output_config.format: Extra inputs are not permitted", true)]
    #[case::too_complex(400, "compilation exceeded limits: Schema is too complex", true)]
    #[case::grammar_too_large(400, "The compiled grammar is too large", true)]
    #[case::schema_error(400, "output_config.format.schema: bad", true)]
    #[case::other_400(400, "messages: bad", false)]
    #[case::not_a_400(500, "output_config.format: Extra inputs are not permitted", false)]
    fn native_output_rejections_are_recognized(
        #[case] status: u16,
        #[case] text: &str,
        #[case] expected: bool,
    ) {
        assert_eq!(is_native_output_rejection(status, text), expected);
    }

    #[rstest]
    #[case::enabled(legacy(), true)]
    #[case::opt_in(bedrock_opt_in(), true)]
    #[case::replaces_unforceable_tool(NativeOutputSupport { replaces_unforceable_tool: true, ..NativeOutputSupport::default() }, true)]
    #[case::unsupported(NativeOutputSupport::default(), false)]
    fn native_output_retry_needs_native_support(
        #[case] native: NativeOutputSupport,
        #[case] expected: bool,
    ) {
        let params = schema(json!({"output_config": {"format": {"type": "json_schema"}}}));
        assert_eq!(native_output_retry_requested(&params, native), expected);
    }
}
