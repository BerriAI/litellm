//! `response_format` answers delivered through the `json_tool_call` tool, as Python's
//! `base_utils.py` reads them back: the schema's shape, the envelopes a model wraps the
//! answer in, and an answer split across several tool calls.

use std::collections::BTreeSet;

use serde_json::{Map, Value};

use crate::base_llm::chat::params::python_dumps;

/// `ResponseFormatSchema`: what the requested schema declares, enough to tell an answer
/// from an envelope around it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ResponseFormatSchema {
    pub property_names: BTreeSet<String>,
    pub required_names: BTreeSet<String>,
    pub allows_undeclared_keys: bool,
}

impl ResponseFormatSchema {
    /// `response_format_schema`: `None` unless `value` is an object whose `properties`, if
    /// present, is one too.
    pub fn from_schema(value: &Value) -> Option<Self> {
        let document = value.as_object()?;
        let properties = match document.get("properties") {
            None => Map::new(),
            Some(Value::Object(properties)) => properties.clone(),
            Some(_) => return None,
        };
        let required_names = match document.get("required") {
            Some(Value::Array(items)) if items.iter().all(Value::is_string) => items
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_owned)
                .collect(),
            _ => BTreeSet::new(),
        };
        let additional = document.get("additionalProperties");
        let allows_undeclared_keys = additional == Some(&Value::Bool(true))
            || additional.is_some_and(Value::is_object)
            || document
                .get("patternProperties")
                .and_then(Value::as_object)
                .is_some_and(|patterns| !patterns.is_empty());
        Some(Self {
            property_names: properties.keys().cloned().collect(),
            required_names,
            allows_undeclared_keys,
        })
    }

    fn accepts_a_lone_undeclared_key(&self) -> bool {
        self.allows_undeclared_keys && self.required_names.is_empty()
    }
}

const ENVELOPE_NAME_KEYS: [&str; 2] = ["name", "tool_name"];

/// `_json_object_or_encoded_object`.
fn object_or_encoded_object(value: &Value) -> Option<Map<String, Value>> {
    match value {
        Value::String(text) => match serde_json::from_str(text) {
            Ok(Value::Object(object)) => Some(object),
            _ => None,
        },
        Value::Object(object) => Some(object.clone()),
        _ => None,
    }
}

struct EnvelopeLayer {
    inner: Map<String, Value>,
    names_the_tool: bool,
}

fn envelope_layer(
    payload: &Map<String, Value>,
    schema: &ResponseFormatSchema,
    tool_name: &str,
) -> Option<EnvelopeLayer> {
    let name_keys: BTreeSet<&str> = ENVELOPE_NAME_KEYS
        .into_iter()
        .filter(|key| payload.get(*key).and_then(Value::as_str) == Some(tool_name))
        .collect();
    let label_keys: BTreeSet<&str> = match (name_keys.is_empty(), schema.required_names.is_empty())
    {
        (false, _) => name_keys.clone(),
        (true, true) => BTreeSet::new(),
        (true, false) => payload
            .iter()
            .filter(|(key, value)| {
                value.is_string()
                    && !schema.property_names.contains(*key)
                    && !schema.required_names.contains(*key)
            })
            .map(|(key, _)| key.as_str())
            .collect(),
    };
    let wrapper_keys: Vec<&String> = payload
        .keys()
        .filter(|key| !label_keys.contains(key.as_str()))
        .collect();
    let [wrapper] = wrapper_keys.as_slice() else {
        return None;
    };
    if schema.property_names.contains(*wrapper) {
        return None;
    }
    let wrapped = &payload[wrapper.as_str()];
    let inner = match name_keys.is_empty() {
        false => object_or_encoded_object(wrapped),
        true => wrapped.as_object().cloned(),
    }?;
    Some(EnvelopeLayer {
        inner,
        names_the_tool: !name_keys.is_empty(),
    })
}

fn holds_the_schema_object(layer: &EnvelopeLayer, schema: &ResponseFormatSchema) -> bool {
    layer.names_the_tool
        || (!schema.accepts_a_lone_undeclared_key()
            && layer
                .inner
                .keys()
                .any(|key| schema.property_names.contains(key)))
}

fn schema_object_inside_envelope(
    payload: &Map<String, Value>,
    schema: &ResponseFormatSchema,
    tool_name: &str,
) -> Option<Map<String, Value>> {
    let mut layer = envelope_layer(payload, schema, tool_name);
    let mut first = true;
    while let Some(current) = layer {
        if !first && schema.property_names.is_empty() {
            return None;
        }
        first = false;
        if holds_the_schema_object(&current, schema) {
            return Some(current.inner);
        }
        layer = envelope_layer(&current.inner, schema, tool_name);
    }
    None
}

/// `response_format_envelope_contents`.
pub fn envelope_contents(
    payload: &Value,
    schema: &ResponseFormatSchema,
    tool_name: &str,
) -> Option<Value> {
    let object = payload.as_object()?;
    if let Some(inner) = schema_object_inside_envelope(object, schema, tool_name) {
        return Some(Value::Object(inner));
    }
    if schema.property_names.contains("values") {
        return None;
    }
    object
        .get("values")
        .filter(|values| !values.is_null())
        .cloned()
}

/// `unwrap_response_format_envelope`.
pub fn unwrap_envelope(payload: Value, schema: &ResponseFormatSchema, tool_name: &str) -> Value {
    envelope_contents(&payload, schema, tool_name).unwrap_or(payload)
}

fn field_names(
    blocks: &[Map<String, Value>],
    key: &str,
    schema: &ResponseFormatSchema,
) -> Option<Vec<String>> {
    let names: Vec<String> = blocks
        .iter()
        .filter_map(|block| block[key].as_str())
        .filter(|name| schema.property_names.contains(*name))
        .map(str::to_owned)
        .collect();
    let distinct: BTreeSet<&String> = names.iter().collect();
    (distinct.len() == blocks.len()).then_some(names)
}

fn is_a_label_every_block_repeats(blocks: &[Map<String, Value>], key: &str) -> bool {
    blocks[0][key].is_string() && blocks.iter().all(|block| block[key] == blocks[0][key])
}

fn value_key(blocks: &[Map<String, Value>], keys: &[&String]) -> Option<String> {
    if let [only] = keys {
        return Some((*only).clone());
    }
    let varying: Vec<&&String> = keys
        .iter()
        .filter(|key| !is_a_label_every_block_repeats(blocks, key))
        .collect();
    match varying.as_slice() {
        [only] => Some((**only).clone()),
        _ => None,
    }
}

fn split_answer_fields(
    blocks: &[Map<String, Value>],
    schema: &ResponseFormatSchema,
) -> Option<Map<String, Value>> {
    let keys: BTreeSet<&String> = blocks[0].keys().collect();
    if keys.iter().any(|key| schema.property_names.contains(*key))
        || blocks
            .iter()
            .any(|block| block.keys().collect::<BTreeSet<_>>() != keys)
    {
        return None;
    }
    // Python iterates a frozenset here; the outcome only depends on there being one match.
    let named: Vec<(&String, Vec<String>)> = keys
        .iter()
        .filter_map(|key| Some((*key, field_names(blocks, key, schema)?)))
        .collect();
    let [(name_key, names)] = named.as_slice() else {
        return None;
    };
    let rest: Vec<&String> = keys.iter().copied().filter(|key| key != name_key).collect();
    let value_key = value_key(blocks, &rest)?;
    Some(
        names
            .iter()
            .cloned()
            .zip(blocks.iter().map(|block| block[value_key.as_str()].clone()))
            .collect(),
    )
}

fn merged_partial_answers(
    blocks: &[Map<String, Value>],
    schema: &ResponseFormatSchema,
) -> Option<Map<String, Value>> {
    let keys: BTreeSet<&String> = blocks.iter().flat_map(Map::keys).collect();
    if !keys.iter().all(|key| schema.property_names.contains(*key))
        || keys.len() != blocks.iter().map(Map::len).sum::<usize>()
    {
        return None;
    }
    Some(
        blocks
            .iter()
            .flat_map(|block| block.iter().map(|(k, v)| (k.clone(), v.clone())))
            .collect(),
    )
}

/// `split_response_format_answer`: one answer spread over several tool calls.
pub fn split_answer(
    encoded_blocks: &[String],
    schema: &ResponseFormatSchema,
) -> Option<Map<String, Value>> {
    let blocks: Vec<Map<String, Value>> = encoded_blocks
        .iter()
        .map(|encoded| object_or_encoded_object(&Value::String(encoded.clone())))
        .collect::<Option<_>>()?;
    if blocks.len() < 2 {
        return None;
    }
    split_answer_fields(&blocks, schema).or_else(|| merged_partial_answers(&blocks, schema))
}

/// Anthropic's `_convert_tool_response_to_message`: the message content the `json_tool_call`
/// calls' arguments become.
pub fn json_tool_answer(arguments: &[String], schema: Option<&ResponseFormatSchema>) -> String {
    let first = arguments.first().cloned().unwrap_or_default();
    if let Some(split) = schema.and_then(|schema| split_answer(arguments, schema)) {
        return python_dumps(&Value::Object(split));
    }
    let Ok(payload) = serde_json::from_str::<Value>(&first) else {
        return first;
    };
    match schema {
        Some(schema) => python_dumps(&unwrap_envelope(
            payload,
            schema,
            crate::base_llm::chat::params::RESPONSE_FORMAT_TOOL_NAME,
        )),
        None => {
            let values = payload
                .as_object()
                .and_then(|object| object.get("values"))
                .filter(|values| !values.is_null())
                .cloned();
            python_dumps(values.as_ref().unwrap_or(&payload))
        }
    }
}

/// `unpack_defs`: every `$ref` into `$defs` / `definitions` replaced by a copy of its
/// target, nested definitions shadowing outer ones, a circular reference left as it is.
pub fn unpack_defs(schema: &mut Value, defs: &Map<String, Value>) {
    fn scoped(node: &Map<String, Value>, outer: &Map<String, Value>) -> Map<String, Value> {
        let mut defs = outer.clone();
        for key in ["$defs", "definitions"] {
            if let Some(Value::Object(local)) = node.get(key) {
                defs.extend(local.iter().map(|(k, v)| (k.clone(), v.clone())));
            }
        }
        defs
    }
    fn walk(node: &mut Value, defs: &Map<String, Value>, chain: &BTreeSet<String>) {
        match node {
            Value::Object(object) => {
                if let Some(reference) = object.get("$ref").and_then(Value::as_str) {
                    let name = reference.rsplit('/').next().unwrap_or(reference).to_owned();
                    if chain.contains(&name) {
                        return;
                    }
                    let Some(target) = defs.get(&name).cloned() else {
                        return;
                    };
                    let child_defs = match &target {
                        Value::Object(target) => scoped(target, defs),
                        _ => defs.clone(),
                    };
                    *node = target;
                    let mut chain = chain.clone();
                    chain.insert(name);
                    walk(node, &child_defs, &chain);
                    return;
                }
                if object.contains_key("$ref") {
                    return;
                }
                let current = scoped(object, defs);
                for value in object.values_mut() {
                    walk(value, &current, chain);
                }
            }
            Value::Array(items) => {
                for item in items {
                    walk(item, defs, chain);
                }
            }
            _ => {}
        }
    }
    let root = match schema {
        Value::Object(object) => scoped(object, defs),
        _ => defs.clone(),
    };
    walk(schema, &root, &BTreeSet::new());
}

/// `unpack_legacy_defs`: refs into `definitions` or OpenAPI `components.schemas` inlined,
/// those blocks removed. `$defs` is left alone.
pub fn unpack_legacy_defs(schema: &mut Map<String, Value>) {
    let has_components = schema
        .get("components")
        .and_then(|components| components.get("schemas"))
        .is_some();
    if !schema.contains_key("definitions") && !has_components {
        return;
    }
    let mut defs = match schema.remove("components") {
        Some(Value::Object(mut components)) => match components.remove("schemas") {
            Some(Value::Object(schemas)) => schemas,
            _ => Map::new(),
        },
        _ => Map::new(),
    };
    if let Some(Value::Object(definitions)) = schema.remove("definitions") {
        defs.extend(definitions);
    }
    let mut value = Value::Object(std::mem::take(schema));
    unpack_defs(&mut value, &defs);
    if let Value::Object(object) = value {
        *schema = object;
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    fn schema(value: Value) -> ResponseFormatSchema {
        ResponseFormatSchema::from_schema(&value).unwrap()
    }

    #[rstest]
    #[case::bare(json!({"name": "a"}), json!({"name": "a"}))]
    #[case::named_envelope(json!({"name": "json_tool_call", "arguments": {"name": "a"}}), json!({"name": "a"}))]
    #[case::encoded_envelope(json!({"tool_name": "json_tool_call", "parameters": "{\"name\": \"a\"}"}), json!({"name": "a"}))]
    #[case::wrapper(json!({"answer": {"name": "a"}}), json!({"name": "a"}))]
    #[case::values(json!({"values": {"x": 1}}), json!({"x": 1}))]
    fn envelopes_are_unwrapped(#[case] payload: Value, #[case] expected: Value) {
        let schema = schema(
            json!({"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}),
        );
        assert_eq!(
            unwrap_envelope(payload, &schema, "json_tool_call"),
            expected
        );
    }

    #[rstest]
    fn an_answer_split_over_named_fields_is_joined() {
        let schema = schema(json!({"properties": {"a": {}, "b": {}}}));
        let blocks = vec![
            r#"{"field": "a", "value": 1}"#.to_owned(),
            r#"{"field": "b", "value": 2}"#.to_owned(),
        ];
        assert_eq!(
            split_answer(&blocks, &schema),
            Some(json!({"a": 1, "b": 2}).as_object().unwrap().clone())
        );
        let partial = vec![r#"{"a": 1}"#.to_owned(), r#"{"b": 2}"#.to_owned()];
        assert_eq!(
            json_tool_answer(&partial, Some(&schema)),
            r#"{"a": 1, "b": 2}"#
        );
    }

    #[rstest]
    fn an_undecodable_answer_is_returned_as_sent() {
        assert_eq!(json_tool_answer(&["{bad".to_owned()], None), "{bad");
        assert_eq!(
            json_tool_answer(&[r#"{"values": [1]}"#.to_owned()], None),
            "[1]"
        );
    }

    #[rstest]
    fn refs_are_inlined_and_cycles_left_alone() {
        let mut value =
            json!({"properties": {"a": {"$ref": "#/$defs/A"}, "n": {"$ref": "#/$defs/Node"}}});
        let defs = json!({"A": {"type": "string"}, "Node": {"properties": {"next": {"$ref": "#/$defs/Node"}}}});
        unpack_defs(&mut value, defs.as_object().unwrap());
        assert_eq!(
            value,
            json!({"properties": {"a": {"type": "string"}, "n": {"properties": {"next": {"$ref": "#/$defs/Node"}}}}})
        );
    }
}
