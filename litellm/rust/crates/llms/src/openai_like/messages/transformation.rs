use serde_json::{Map, Value};

pub fn complete_messages_url(base: &str) -> String {
    let base = base.trim_end_matches('/');
    if base.ends_with("/v1/messages") {
        return base.into();
    }
    format!("{}/v1/messages", base.strip_suffix("/v1").unwrap_or(base))
}

pub fn portable_cache_control(body: Value) -> Value {
    let Value::Object(fields) = body else {
        return body;
    };
    Value::Object(
        portable_block(fields)
            .into_iter()
            .map(|(key, value)| {
                let value = match key.as_str() {
                    "system" | "tools" => map_blocks(value, portable_value),
                    "messages" => map_blocks(value, portable_message),
                    _ => value,
                };
                (key, value)
            })
            .collect(),
    )
}

fn portable_block(fields: Map<String, Value>) -> Map<String, Value> {
    fields
        .into_iter()
        .filter_map(|(key, value)| {
            if key != "cache_control" {
                return Some((key, value));
            }
            let Value::Object(cache) = value else {
                return None;
            };
            let kind = cache
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or("ephemeral");
            Some((key, serde_json::json!({"type": kind})))
        })
        .collect()
}

fn portable_value(value: Value) -> Value {
    match value {
        Value::Object(fields) => Value::Object(portable_block(fields)),
        other => other,
    }
}

fn map_blocks(value: Value, map: fn(Value) -> Value) -> Value {
    match value {
        Value::Array(blocks) => Value::Array(blocks.into_iter().map(map).collect()),
        other => other,
    }
}

fn portable_message(value: Value) -> Value {
    let Value::Object(fields) = value else {
        return value;
    };
    Value::Object(
        fields
            .into_iter()
            .map(|(key, value)| {
                let value = if key == "content" {
                    map_blocks(value, portable_content)
                } else {
                    value
                };
                (key, value)
            })
            .collect(),
    )
}

fn portable_content(value: Value) -> Value {
    let Value::Object(fields) = value else {
        return value;
    };
    let tool_result = fields.get("type").and_then(Value::as_str) == Some("tool_result");
    Value::Object(
        portable_block(fields)
            .into_iter()
            .map(|(key, value)| {
                let value = if tool_result && key == "content" {
                    map_blocks(value, portable_value)
                } else {
                    value
                };
                (key, value)
            })
            .collect(),
    )
}
