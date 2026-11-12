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

fn remove_path(value: &mut Value, segments: &[Segment]) {
    let Some((segment, tail)) = segments.split_first() else {
        return;
    };
    match (segment, value) {
        (Segment::Field(name), Value::Object(object)) if tail.is_empty() => {
            if object.contains_key(name) {
                object.retain(|key, _| key != name);
            }
        }
        (Segment::Field(name), Value::Object(object)) => {
            if let Some(item) = object.get_mut(name) {
                remove_path(item, tail);
            }
        }
        (Segment::Every, Value::Array(items)) if !tail.is_empty() => {
            items.iter_mut().for_each(|item| remove_path(item, tail));
        }
        (Segment::Index(index), Value::Array(items)) => {
            if let Some(item) = items.get_mut(*index) {
                remove_path(item, tail);
            }
        }
        _ => {}
    }
}

pub fn delete_nested_value(mut value: Value, path: &str) -> Value {
    if let Some(segments) = parse_segments(path) {
        remove_path(&mut value, &segments);
    }
    value
}

#[cfg(test)]
mod tests {
    use rstest::{fixture, rstest};
    use serde_json::json;

    use super::*;

    #[fixture]
    fn body() -> Value {
        json!({
            "tools": [
                {"name": "t0", "examples": ["a"], "arr": [{"f": 1, "k": 1}, {"f": 2, "k": 2}]},
                {"name": "t1", "examples": ["b"], "arr": [{"f": 3, "k": 3}]}
            ],
            "meta": {"user": "u", "inner": {"drop": 1, "keep": 2}},
            "top": 0.7
        })
    }

    #[rstest]
    #[case::top_level_field("top", json!({
        "tools": [
            {"name": "t0", "examples": ["a"], "arr": [{"f": 1, "k": 1}, {"f": 2, "k": 2}]},
            {"name": "t1", "examples": ["b"], "arr": [{"f": 3, "k": 3}]}
        ],
        "meta": {"user": "u", "inner": {"drop": 1, "keep": 2}}
    }))]
    #[case::whole_object("meta", json!({
        "tools": [
            {"name": "t0", "examples": ["a"], "arr": [{"f": 1, "k": 1}, {"f": 2, "k": 2}]},
            {"name": "t1", "examples": ["b"], "arr": [{"f": 3, "k": 3}]}
        ],
        "top": 0.7
    }))]
    #[case::nested_field("meta.inner.drop", json!({
        "tools": [
            {"name": "t0", "examples": ["a"], "arr": [{"f": 1, "k": 1}, {"f": 2, "k": 2}]},
            {"name": "t1", "examples": ["b"], "arr": [{"f": 3, "k": 3}]}
        ],
        "meta": {"user": "u", "inner": {"keep": 2}},
        "top": 0.7
    }))]
    #[case::trailing_dot("meta.inner.drop.", json!({
        "tools": [
            {"name": "t0", "examples": ["a"], "arr": [{"f": 1, "k": 1}, {"f": 2, "k": 2}]},
            {"name": "t1", "examples": ["b"], "arr": [{"f": 3, "k": 3}]}
        ],
        "meta": {"user": "u", "inner": {"keep": 2}},
        "top": 0.7
    }))]
    #[case::leading_and_doubled_dots(".meta..inner.drop", json!({
        "tools": [
            {"name": "t0", "examples": ["a"], "arr": [{"f": 1, "k": 1}, {"f": 2, "k": 2}]},
            {"name": "t1", "examples": ["b"], "arr": [{"f": 3, "k": 3}]}
        ],
        "meta": {"user": "u", "inner": {"keep": 2}},
        "top": 0.7
    }))]
    #[case::field_in_every_element("tools[*].examples", json!({
        "tools": [
            {"name": "t0", "arr": [{"f": 1, "k": 1}, {"f": 2, "k": 2}]},
            {"name": "t1", "arr": [{"f": 3, "k": 3}]}
        ],
        "meta": {"user": "u", "inner": {"drop": 1, "keep": 2}},
        "top": 0.7
    }))]
    #[case::whole_array_field_in_every_element("tools[*].arr", json!({
        "tools": [
            {"name": "t0", "examples": ["a"]},
            {"name": "t1", "examples": ["b"]}
        ],
        "meta": {"user": "u", "inner": {"drop": 1, "keep": 2}},
        "top": 0.7
    }))]
    #[case::field_in_indexed_element("tools[1].examples", json!({
        "tools": [
            {"name": "t0", "examples": ["a"], "arr": [{"f": 1, "k": 1}, {"f": 2, "k": 2}]},
            {"name": "t1", "arr": [{"f": 3, "k": 3}]}
        ],
        "meta": {"user": "u", "inner": {"drop": 1, "keep": 2}},
        "top": 0.7
    }))]
    #[case::padded_index("tools[ 1 ].examples", json!({
        "tools": [
            {"name": "t0", "examples": ["a"], "arr": [{"f": 1, "k": 1}, {"f": 2, "k": 2}]},
            {"name": "t1", "arr": [{"f": 3, "k": 3}]}
        ],
        "meta": {"user": "u", "inner": {"drop": 1, "keep": 2}},
        "top": 0.7
    }))]
    #[case::field_right_after_bracket("tools[0]examples", json!({
        "tools": [
            {"name": "t0", "arr": [{"f": 1, "k": 1}, {"f": 2, "k": 2}]},
            {"name": "t1", "examples": ["b"], "arr": [{"f": 3, "k": 3}]}
        ],
        "meta": {"user": "u", "inner": {"drop": 1, "keep": 2}},
        "top": 0.7
    }))]
    #[case::index_then_wildcard("tools[0].arr[*].f", json!({
        "tools": [
            {"name": "t0", "examples": ["a"], "arr": [{"k": 1}, {"k": 2}]},
            {"name": "t1", "examples": ["b"], "arr": [{"f": 3, "k": 3}]}
        ],
        "meta": {"user": "u", "inner": {"drop": 1, "keep": 2}},
        "top": 0.7
    }))]
    #[case::wildcard_then_index_only_where_it_exists("tools[*].arr[1].f", json!({
        "tools": [
            {"name": "t0", "examples": ["a"], "arr": [{"f": 1, "k": 1}, {"k": 2}]},
            {"name": "t1", "examples": ["b"], "arr": [{"f": 3, "k": 3}]}
        ],
        "meta": {"user": "u", "inner": {"drop": 1, "keep": 2}},
        "top": 0.7
    }))]
    #[case::nested_wildcards("tools[*].arr[*].f", json!({
        "tools": [
            {"name": "t0", "examples": ["a"], "arr": [{"k": 1}, {"k": 2}]},
            {"name": "t1", "examples": ["b"], "arr": [{"k": 3}]}
        ],
        "meta": {"user": "u", "inner": {"drop": 1, "keep": 2}},
        "top": 0.7
    }))]
    fn deletes_the_addressed_field(body: Value, #[case] path: &str, #[case] expected: Value) {
        assert_eq!(delete_nested_value(body, path), expected);
    }

    #[rstest]
    #[case::empty_path("")]
    #[case::missing_field("missing")]
    #[case::missing_parent("missing.field")]
    #[case::field_through_a_scalar("top.value")]
    #[case::field_on_an_array("tools.name")]
    #[case::index_on_an_object("meta[0].user")]
    #[case::wildcard_on_an_object("meta[*].user")]
    #[case::wildcard_over_scalars("tools[*].examples[*].name")]
    #[case::index_out_of_range("tools[5].name")]
    #[case::every_element_itself("tools[*]")]
    #[case::indexed_element_itself("tools[0]")]
    #[case::nested_element_itself("tools[*].arr[0]")]
    #[case::negative_index("tools[-1].name")]
    #[case::non_numeric_index("tools[x].name")]
    #[case::empty_index("tools[].name")]
    #[case::unclosed_bracket("top[0")]
    fn leaves_the_value_untouched(body: Value, #[case] path: &str) {
        assert_eq!(delete_nested_value(body.clone(), path), body);
    }

    #[test]
    fn keeps_the_remaining_keys_in_their_original_order() {
        let schema = json!({"properties": {"a": 1, "b": 2, "c": 3, "d": 4}});
        let trimmed = delete_nested_value(schema, "properties.b");
        assert_eq!(
            serde_json::to_string(&trimmed).ok(),
            Some(r#"{"properties":{"a":1,"c":3,"d":4}}"#.to_string())
        );
    }

    #[test]
    fn many_absent_paths_over_a_wide_schema_finish_well_inside_one_request() {
        let width = 10_000;
        let properties: serde_json::Map<String, Value> = (0..width)
            .map(|n| (format!("p{n}"), json!({"type": "string"})))
            .collect();
        let body = json!({"tools": [{"name": "t", "input_schema": {"properties": properties}}]});
        let paths: Vec<String> = (0..width)
            .map(|n| format!("tools[*].input_schema.properties.absent{n}"))
            .collect();
        let started = std::time::Instant::now();
        let trimmed = paths
            .iter()
            .fold(body.clone(), |value, path| delete_nested_value(value, path));
        assert!(started.elapsed() < std::time::Duration::from_secs(1));
        assert_eq!(trimmed, body);
    }

    #[rstest]
    #[case::wildcards_indices_and_nesting(
        json!({"tools": [
            {"name": "t0", "configs": [{"id": "c0", "remove_me": 1, "keep": 1}, {"id": "c1", "remove_me": 2, "keep": 2}], "metadata": {"drop_this": 1, "preserve": 1}},
            {"name": "t1", "configs": [{"id": "c0", "remove_me": 3, "keep": 3}, {"id": "c1", "remove_me": 4, "keep": 4}], "metadata": {"drop_this": 2, "preserve": 2}},
            {"name": "t2", "configs": [{"id": "c0", "remove_me": 5, "keep": 5}], "metadata": {"drop_this": 3, "preserve": 3}}
        ]}),
        &["tools[*].configs[1].remove_me", "tools[1].metadata.drop_this", "tools[*].configs[*].id"],
        json!({"tools": [
            {"name": "t0", "configs": [{"remove_me": 1, "keep": 1}, {"keep": 2}], "metadata": {"drop_this": 1, "preserve": 1}},
            {"name": "t1", "configs": [{"remove_me": 3, "keep": 3}, {"keep": 4}], "metadata": {"preserve": 2}},
            {"name": "t2", "configs": [{"remove_me": 5, "keep": 5}], "metadata": {"drop_this": 3, "preserve": 3}}
        ]}),
    )]
    #[case::simple_and_wildcard_nesting(
        json!({
            "tools": [{"name": "t1", "simple_nested": {"remove": 1, "keep": 2}, "complex": [{"nested": {"remove": 3, "keep": 4}}]}],
            "top_level_remove": "should_go",
            "top_level_keep": "should_stay"
        }),
        &["tools[*].simple_nested.remove", "tools[*].complex[*].nested.remove"],
        json!({
            "tools": [{"name": "t1", "simple_nested": {"keep": 2}, "complex": [{"nested": {"keep": 4}}]}],
            "top_level_remove": "should_go",
            "top_level_keep": "should_stay"
        }),
    )]
    #[case::triple_nested_wildcards(
        json!({"tools": [{"name": "t1", "arr1": [
            {"arr2": [{"field": 1, "keep": 1}, {"field": 2, "keep": 2}]},
            {"arr2": [{"field": 3, "keep": 3}]}
        ]}]}),
        &["tools[*].arr1[*].arr2[*].field"],
        json!({"tools": [{"name": "t1", "arr1": [
            {"arr2": [{"keep": 1}, {"keep": 2}]},
            {"arr2": [{"keep": 3}]}
        ]}]}),
    )]
    fn applies_paths_in_sequence(
        #[case] value: Value,
        #[case] paths: &[&str],
        #[case] expected: Value,
    ) {
        let deleted = paths
            .iter()
            .fold(value, |value, path| delete_nested_value(value, path));
        assert_eq!(deleted, expected);
    }
}
