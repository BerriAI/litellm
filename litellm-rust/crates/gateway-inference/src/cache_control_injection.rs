//! `cache_control_injection_points`, as Python's `AnthropicCacheControlHook` places its
//! `location: message` points on a chat request a provider translates: each marks the messages
//! it names, in config order, until the request holds Anthropic's limit of breakpoints.

use serde_json::{Map, Value, json};

pub(crate) const INJECTION_POINTS: &str = "cache_control_injection_points";

/// Anthropic rejects a request with more than four blocks marked `cache_control`.
const MAX_CACHE_CONTROL_BLOCKS: usize = 4;

/// Marks the messages of `body` that each point names. A message the client already marked
/// keeps its own mark, and the client's marks on messages and tools count toward the limit.
pub(crate) fn apply(points: &Value, body: &mut Map<String, Value>) {
    let Some(points) = points.as_array() else {
        return;
    };
    let reserved = body
        .get("tools")
        .and_then(Value::as_array)
        .map_or(0, |tools| tools.iter().filter(|tool| marked(tool)).count());
    let Some(Value::Array(messages)) = body.get_mut("messages") else {
        return;
    };
    let max_blocks = MAX_CACHE_CONTROL_BLOCKS.saturating_sub(reserved);
    let mut used = messages.iter().map(breakpoints).sum::<usize>();
    let message_points = points
        .iter()
        .filter_map(Value::as_object)
        .filter(|point| point.get("location").and_then(Value::as_str) == Some("message"));
    for point in message_points {
        let control = match point.get("control") {
            Some(control @ Value::Object(fields)) if !fields.is_empty() => control.clone(),
            _ => json!({"type": "ephemeral"}),
        };
        for index in targets(point, messages) {
            if used >= max_blocks {
                return;
            }
            let message = &mut messages[index];
            if breakpoints(message) > 0 {
                continue;
            }
            insert(message, &control);
            used += breakpoints(message);
        }
    }
}

/// The message a point's `index` names, negative counting from the end, or every message of
/// its `role`.
fn targets(point: &Map<String, Value>, messages: &[Value]) -> Vec<usize> {
    let index = match point.get("index") {
        Some(Value::Number(index)) => index.as_i64(),
        Some(Value::String(index)) => index.parse().ok(),
        _ => None,
    };
    if let Some(index) = index {
        let len = messages.len() as i64;
        let index = if index < 0 { index + len } else { index };
        return match (0..len).contains(&index) {
            true => vec![index as usize],
            false => Vec::new(),
        };
    }
    match point.get("role").and_then(Value::as_str) {
        Some(role) => messages
            .iter()
            .enumerate()
            .filter(|(_, message)| message.get("role").and_then(Value::as_str) == Some(role))
            .map(|(index, _)| index)
            .collect(),
        None => Vec::new(),
    }
}

/// Text content takes the mark on the message; a list of parts takes it on the last part.
fn insert(message: &mut Value, control: &Value) {
    let Some(fields) = message.as_object_mut() else {
        return;
    };
    match fields.get_mut("content") {
        Some(Value::String(_)) => {
            fields.insert("cache_control".into(), control.clone());
        }
        Some(Value::Array(parts)) => {
            if let Some(Value::Object(last)) = parts.last_mut() {
                last.insert("cache_control".into(), control.clone());
            }
        }
        _ => {}
    }
}

/// The marks a message carries: on itself, on its content parts and on its tool calls.
fn breakpoints(message: &Value) -> usize {
    let within = |key| {
        message
            .get(key)
            .and_then(Value::as_array)
            .map_or(0, |items| items.iter().filter(|item| marked(item)).count())
    };
    usize::from(marked(message)) + within("content") + within("tool_calls")
}

fn marked(value: &Value) -> bool {
    ["cache_control", "prompt_cache_breakpoint"]
        .iter()
        .any(|key| value.get(*key).is_some_and(|mark| !mark.is_null()))
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    fn marks(points: Value, body: Value) -> Value {
        let mut body = body.as_object().unwrap().clone();
        apply(&points, &mut body);
        body["messages"].clone()
    }

    #[rstest]
    fn a_role_point_marks_text_on_the_message_and_parts_on_the_last_part() {
        let messages = marks(
            json!([{"location": "message", "role": "system"}]),
            json!({"messages": [
                {"role": "system", "content": "rules"},
                {"role": "system", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]},
                {"role": "user", "content": "hi"},
            ]}),
        );
        assert_eq!(
            messages,
            json!([
                {"role": "system", "content": "rules", "cache_control": {"type": "ephemeral"}},
                {"role": "system", "content": [
                    {"type": "text", "text": "a"},
                    {"type": "text", "text": "b", "cache_control": {"type": "ephemeral"}},
                ]},
                {"role": "user", "content": "hi"},
            ])
        );
    }

    #[rstest]
    #[case::negative(json!(-1), Some(2))]
    #[case::text(json!("1"), Some(1))]
    #[case::out_of_bounds(json!(3), None)]
    fn an_index_point_marks_one_message(#[case] index: Value, #[case] marked_at: Option<usize>) {
        let messages = marks(
            json!([{"location": "message", "index": index, "control": {"type": "ephemeral", "ttl": "1h"}}]),
            json!({"messages": [
                {"role": "user", "content": "a"},
                {"role": "assistant", "content": "b"},
                {"role": "user", "content": "c"},
            ]}),
        );
        let marked: Vec<usize> = (0..3)
            .filter(|index| messages[index].get("cache_control").is_some())
            .collect();
        assert_eq!(marked, Vec::from_iter(marked_at));
        if let Some(index) = marked_at {
            assert_eq!(messages[index]["cache_control"]["ttl"], "1h");
        }
    }

    #[rstest]
    fn client_marks_are_kept_and_count_toward_the_limit() {
        let messages = marks(
            json!([{"location": "message", "role": "user"}]),
            json!({
                "tools": [{"type": "function", "function": {"name": "f"}, "cache_control": {"type": "ephemeral"}}],
                "messages": [
                    {"role": "user", "content": "a", "cache_control": {"type": "ephemeral", "ttl": "1h"}},
                    {"role": "user", "content": "b"},
                    {"role": "user", "content": "c"},
                    {"role": "user", "content": "d"},
                ],
            }),
        );
        assert_eq!(messages[0]["cache_control"]["ttl"], "1h");
        let marked = messages
            .as_array()
            .unwrap()
            .iter()
            .filter(|message| message.get("cache_control").is_some())
            .count();
        assert_eq!(marked, 3);
        assert!(messages[3].get("cache_control").is_none());
    }

    #[rstest]
    fn other_locations_mark_nothing() {
        let body = json!({"messages": [{"role": "user", "content": "a"}]});
        assert_eq!(
            marks(json!([{"location": "tool_config"}]), body.clone()),
            body["messages"]
        );
    }
}
