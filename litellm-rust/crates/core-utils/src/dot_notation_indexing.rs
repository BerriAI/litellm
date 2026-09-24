//! JSONPath-like field deletion for `additional_drop_params`: `field`, `parent.child`,
//! `array[*].field` and `array[0].field`, as Python's `delete_nested_value` reads them.

use serde_json::Value;

#[derive(Clone, Debug, PartialEq, Eq)]
enum Segment {
    Field(String),
    Every,
    Index(usize),
}

fn parse_segments(path: &str) -> Option<Vec<Segment>> {
    let mut segments = Vec::new();
    let mut rest = path;
    while !rest.is_empty() {
        if let Some(after_open) = rest.strip_prefix('[') {
            let (inside, after) = after_open.split_once(']')?;
            segments.push(match inside {
                "*" => Segment::Every,
                index => Segment::Index(index.trim().parse().ok()?),
            });
            rest = after.strip_prefix('.').unwrap_or(after);
            continue;
        }
        let end = rest.find(['.', '[']).unwrap_or(rest.len());
        let (field, after) = rest.split_at(end);
        if !field.is_empty() {
            segments.push(Segment::Field(field.to_string()));
        }
        rest = after.strip_prefix('.').unwrap_or(after);
    }
    Some(segments)
}

fn without_path(value: Value, segments: &[Segment]) -> Value {
    let Some((segment, tail)) = segments.split_first() else {
        return value;
    };
    match (segment, value) {
        (Segment::Field(name), Value::Object(object)) => Value::Object(
            object
                .into_iter()
                .filter_map(|(key, item)| {
                    if key != *name {
                        return Some((key, item));
                    }
                    (!tail.is_empty()).then(|| (key, without_path(item, tail)))
                })
                .collect(),
        ),
        (Segment::Every, Value::Array(items)) if !tail.is_empty() => Value::Array(
            items
                .into_iter()
                .map(|item| without_path(item, tail))
                .collect(),
        ),
        (Segment::Index(index), Value::Array(items)) if !tail.is_empty() => Value::Array(
            items
                .into_iter()
                .enumerate()
                .map(|(position, item)| {
                    if position == *index {
                        without_path(item, tail)
                    } else {
                        item
                    }
                })
                .collect(),
        ),
        (_, value) => value,
    }
}

/// The value with the field at `path` removed. An unparsable path leaves it untouched, and
/// array elements themselves are never removed, only fields inside them.
pub fn delete_nested_value(value: Value, path: &str) -> Value {
    match parse_segments(path) {
        Some(segments) => without_path(value, &segments),
        None => value,
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    const TOOLS: &str = r#"[{"name": "t1", "input_examples": ["a"]}, {"name": "t2", "keep": 1, "input_examples": ["b"]}]"#;

    fn body() -> Value {
        let tools: Value = serde_json::from_str(TOOLS).unwrap();
        json!({"tools": tools, "metadata": {"user_id": "u"}})
    }

    #[rstest]
    #[case("tools[*].input_examples", json!({"tools": [{"name": "t1"}, {"name": "t2", "keep": 1}], "metadata": {"user_id": "u"}}))]
    #[case("tools[0].input_examples", json!({"tools": [{"name": "t1"}, {"name": "t2", "keep": 1, "input_examples": ["b"]}], "metadata": {"user_id": "u"}}))]
    #[case("metadata.user_id", json!({"tools": body()["tools"], "metadata": {}}))]
    #[case("tools", json!({"metadata": {"user_id": "u"}}))]
    #[case("tools[*]", body())]
    #[case("missing.path", body())]
    #[case("tools[x].name", body())]
    #[case("tools[1", body())]
    fn deletes_like_python(#[case] path: &str, #[case] expected: Value) {
        assert_eq!(delete_nested_value(body(), path), expected);
    }
}
