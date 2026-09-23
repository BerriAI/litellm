use serde_json::{Value, json};

use super::reasoning::MessagesFeatures;

fn cache_breakpoints(value: &Value) -> usize {
    match value {
        Value::Object(fields) => {
            usize::from(
                fields
                    .get("cache_control")
                    .is_some_and(|value| !value.is_null()),
            ) + fields.values().map(cache_breakpoints).sum::<usize>()
        }
        Value::Array(values) => values.iter().map(cache_breakpoints).sum(),
        _ => 0,
    }
}

fn has_message_breakpoint(message: &Value) -> bool {
    cache_breakpoints(message) > 0
}

fn inject_message_breakpoint(message: &mut Value, control: Value) {
    let Some(fields) = message.as_object_mut() else {
        return;
    };
    match fields.get_mut("content") {
        Some(Value::String(_)) => {
            fields.insert("cache_control".into(), control);
        }
        Some(Value::Array(blocks)) => {
            if let Some(Value::Object(last)) = blocks.last_mut() {
                last.insert("cache_control".into(), control);
            }
        }
        _ => {}
    }
}

pub fn inject_cache_control(body: Value, capabilities: &MessagesFeatures) -> Value {
    let Value::Object(mut fields) = body else {
        return body;
    };
    let configured = fields.remove("cache_control_injection_points");
    let enabled = fields
        .remove("enable_prompt_caching")
        .and_then(|value| value.as_bool());
    let points = match configured {
        Some(Value::Array(points)) if !points.is_empty() => points,
        _ if (enabled == Some(true) || capabilities.prompt_cache_enabled)
            && capabilities.prompt_cache_supported
            && ["messages", "system", "tools", "cache_control"]
                .into_iter()
                .filter_map(|name| fields.get(name))
                .map(cache_breakpoints)
                .sum::<usize>()
                == 0 =>
        {
            let control = match capabilities.prompt_cache_ttl.as_deref() {
                Some("5m" | "1h") => {
                    json!({"type":"ephemeral","ttl":capabilities.prompt_cache_ttl})
                }
                _ => json!({"type":"ephemeral"}),
            };
            vec![
                json!({"location":"message","role":"system","control":control}),
                json!({"location":"message","index":-1,"control":control}),
            ]
        }
        _ => return Value::Object(fields),
    };
    if points.is_empty() {
        return Value::Object(fields);
    }
    let external = fields.get("tools").map(cache_breakpoints).unwrap_or(0)
        + fields
            .get("cache_control")
            .map(|value| usize::from(!value.is_null()))
            .unwrap_or(0);
    let mut used = fields.get("messages").map(cache_breakpoints).unwrap_or(0)
        + fields.get("system").map(cache_breakpoints).unwrap_or(0)
        + external;
    if let Some(system_point) = points
        .iter()
        .find(|point| point.get("role").and_then(Value::as_str) == Some("system"))
    {
        if used < 4 {
            let control = system_point
                .get("control")
                .filter(|value| !value.is_null())
                .cloned()
                .unwrap_or(json!({"type":"ephemeral"}));
            if let Some(system) = fields.get_mut("system") {
                match system {
                    Value::String(text) => {
                        *system = json!([{"type":"text","text":text,"cache_control":control}]);
                        used += 1;
                    }
                    Value::Array(blocks)
                        if !blocks.iter().any(|block| cache_breakpoints(block) > 0) =>
                    {
                        if let Some(Value::Object(last)) = blocks.last_mut() {
                            last.insert("cache_control".into(), control);
                            used += 1;
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    if let Some(Value::Array(messages)) = fields.get_mut("messages") {
        for message in messages.iter_mut() {
            if let Some(Value::String(content)) = message.get("content").cloned() {
                if let Some(fields) = message.as_object_mut() {
                    fields.insert("content".into(), json!([{"type":"text","text":content}]));
                }
            }
        }
        for point in points
            .iter()
            .filter(|point| point.get("role").and_then(Value::as_str) != Some("system"))
        {
            let control = point
                .get("control")
                .filter(|value| !value.is_null())
                .cloned()
                .unwrap_or(json!({"type":"ephemeral"}));
            let indices: Vec<usize> = if let Some(index) = point.get("index").and_then(|value| {
                value
                    .as_i64()
                    .or_else(|| value.as_str().and_then(|text| text.parse().ok()))
            }) {
                let resolved = if index < 0 {
                    messages.len() as i64 + index
                } else {
                    index
                };
                usize::try_from(resolved)
                    .ok()
                    .filter(|index| *index < messages.len())
                    .into_iter()
                    .collect()
            } else if let Some(role) = point.get("role").and_then(Value::as_str) {
                messages
                    .iter()
                    .enumerate()
                    .filter_map(|(index, message)| {
                        (message.get("role").and_then(Value::as_str) == Some(role)).then_some(index)
                    })
                    .collect()
            } else {
                Vec::new()
            };
            for index in indices {
                if used >= 4 {
                    break;
                }
                let message = &mut messages[index];
                if has_message_breakpoint(message) {
                    continue;
                }
                inject_message_breakpoint(message, control.clone());
                if has_message_breakpoint(message) {
                    used += 1;
                }
            }
        }
    }
    Value::Object(fields)
}
