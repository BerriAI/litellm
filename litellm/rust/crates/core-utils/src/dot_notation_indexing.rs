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
        (Segment::Every, Value::Array(items)) => Value::Array(
            items
                .into_iter()
                .map(|item| without_path(item, tail))
                .collect(),
        ),
        (Segment::Index(index), Value::Array(items)) => Value::Array(
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

pub fn delete_nested_value(value: Value, path: &str) -> Value {
    match parse_segments(path) {
        Some(segments) => without_path(value, &segments),
        None => value,
    }
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
