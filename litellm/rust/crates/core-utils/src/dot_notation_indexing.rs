use std::collections::{HashMap, HashSet};

use serde_json::Value;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Segment<'a> {
    Field(&'a str),
    Every,
    Index(usize),
}

/// The first segment of `path` and what follows it, or `None` when `path` is spent or its first
/// segment does not parse.
fn next_segment(path: &str) -> Option<(Segment<'_>, &str)> {
    let path = path.trim_start_matches('.');
    if let Some(after_open) = path.strip_prefix('[') {
        let (inside, after) = after_open.split_once(']')?;
        let segment = match inside {
            "*" => Segment::Every,
            index => Segment::Index(index.trim().parse().ok()?),
        };
        return Some((segment, after.trim_start_matches('.')));
    }
    let end = path.find(['.', '[']).unwrap_or(path.len());
    if end == 0 {
        return None;
    }
    let (field, after) = path.split_at(end);
    Some((Segment::Field(field), after.trim_start_matches('.')))
}

/// Deletes what any of `paths` addresses under `value`, each path read one segment per level of
/// `value`. A field that ends a path is deleted from the object holding it, and a path that ends
/// on an element or `[*]` deletes nothing. Each part of `value` is visited once, with every path
/// that reaches it, so paths that overlap through `[*]` and an index never walk the same element
/// twice, and no path is read deeper than `value` goes.
fn prune(paths: &HashSet<&str>, value: &mut Value) {
    match value {
        Value::Object(object) => {
            let mut ending: HashSet<&str> = HashSet::new();
            let mut below: HashMap<&str, HashSet<&str>> = HashMap::new();
            paths
                .iter()
                .filter_map(|path| next_segment(path))
                .for_each(|(segment, rest)| match segment {
                    Segment::Field(name) if rest.is_empty() => {
                        ending.insert(name);
                    }
                    Segment::Field(name) => {
                        below.entry(name).or_default().insert(rest);
                    }
                    Segment::Every | Segment::Index(_) => {}
                });
            if ending.iter().any(|name| object.contains_key(*name)) {
                object.retain(|key, _| !ending.contains(key.as_str()));
            }
            below
                .iter()
                .filter(|(name, _)| !ending.contains(*name))
                .for_each(|(name, rests)| {
                    if let Some(item) = object.get_mut(*name) {
                        prune(rests, item);
                    }
                });
        }
        Value::Array(items) => {
            let mut every: HashSet<&str> = HashSet::new();
            let mut at: HashMap<usize, HashSet<&str>> = HashMap::new();
            paths
                .iter()
                .filter_map(|path| next_segment(path))
                .filter(|(_, rest)| !rest.is_empty())
                .for_each(|(segment, rest)| match segment {
                    Segment::Every => {
                        every.insert(rest);
                    }
                    Segment::Index(index) => {
                        at.entry(index).or_default().insert(rest);
                    }
                    Segment::Field(_) => {}
                });
            if every.is_empty() {
                at.iter().for_each(|(index, rests)| {
                    if let Some(item) = items.get_mut(*index) {
                        prune(rests, item);
                    }
                });
                return;
            }
            items
                .iter_mut()
                .enumerate()
                .for_each(|(index, item)| match at.get(&index) {
                    Some(rests) => prune(&every.union(rests).copied().collect(), item),
                    None => prune(&every, item),
                });
        }
        _ => {}
    }
}

/// `value` without what each of `paths` addresses, in one pass over `value` however many paths
/// there are. A path that does not parse deletes nothing.
pub fn delete_nested_values<'a>(
    mut value: Value,
    paths: impl IntoIterator<Item = &'a str>,
) -> Value {
    prune(&paths.into_iter().collect(), &mut value);
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
        assert_eq!(delete_nested_values(body, [path]), expected);
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
        assert_eq!(delete_nested_values(body.clone(), [path]), body);
    }

    #[rstest]
    #[case::absent_from_the_value(json!({"top": 0.7}))]
    #[case::through_fields_the_value_holds(json!({"a": {"a": {"keep": 1}}}))]
    fn a_path_far_deeper_than_the_value_leaves_it_untouched(#[case] body: Value) {
        let path = vec!["a"; 100_000].join(".");
        assert_eq!(delete_nested_values(body.clone(), [path.as_str()]), body);
    }

    #[rstest]
    #[case::first_key("properties.a", r#"{"properties":{"b":2,"c":3,"d":4}}"#)]
    #[case::middle_key("properties.b", r#"{"properties":{"a":1,"c":3,"d":4}}"#)]
    fn keeps_the_remaining_keys_in_their_original_order(
        #[case] path: &str,
        #[case] expected: &str,
    ) {
        let schema = json!({"properties": {"a": 1, "b": 2, "c": 3, "d": 4}});
        let trimmed = delete_nested_values(schema, [path]);
        assert_eq!(
            serde_json::to_string(&trimmed).ok().as_deref(),
            Some(expected)
        );
    }

    #[rstest]
    #[case::ten_thousand_properties(10_000)]
    fn many_absent_paths_over_a_wide_schema_finish_well_inside_one_request(#[case] width: usize) {
        let properties: serde_json::Map<String, Value> = (0..width)
            .map(|n| (format!("p{n}"), json!({"type": "string"})))
            .collect();
        let body = json!({"tools": [{"name": "t", "input_schema": {"properties": properties}}]});
        let paths: Vec<String> = (0..width)
            .map(|n| format!("tools[*].input_schema.properties.absent{n}"))
            .collect();
        let started = std::time::Instant::now();
        let trimmed = delete_nested_values(body.clone(), paths.iter().map(String::as_str));
        assert!(started.elapsed() < std::time::Duration::from_secs(5));
        assert_eq!(trimmed, body);
    }

    #[rstest]
    #[case::ten_thousand_properties(10_000)]
    fn many_present_paths_over_a_wide_schema_finish_well_inside_one_request(#[case] width: usize) {
        let properties: serde_json::Map<String, Value> = (0..width)
            .map(|n| (format!("p{n}"), json!({"type": "string"})))
            .collect();
        let body = json!({"tools": [{"name": "t", "input_schema": {"properties": properties}}]});
        let paths: Vec<String> = (0..width)
            .map(|n| format!("tools[*].input_schema.properties.p{n}"))
            .collect();
        let started = std::time::Instant::now();
        let trimmed = delete_nested_values(body, paths.iter().map(String::as_str));
        assert!(started.elapsed() < std::time::Duration::from_secs(5));
        assert_eq!(
            trimmed,
            json!({"tools": [{"name": "t", "input_schema": {"properties": {}}}]})
        );
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
    #[case::a_parent_and_a_field_inside_it(
        json!({"meta": {"inner": {"drop": 1}}, "top": 0.7}),
        &["meta.inner.drop", "meta"],
        json!({"top": 0.7}),
    )]
    #[case::every_element_and_one_element_of_the_same_list(
        json!({"tools": [{"a": 1, "b": 1, "c": 1}, {"a": 2, "b": 2, "c": 2}]}),
        &["tools[*].a", "tools[1].b"],
        json!({"tools": [{"b": 1, "c": 1}, {"c": 2}]}),
    )]
    #[case::every_element_and_one_element_each_going_deeper(
        json!({"tools": [
            {"arr": [{"f": 1, "k": 1, "x": 1}, {"f": 2, "k": 2, "x": 2}]},
            {"arr": [{"f": 3, "k": 3, "x": 3}]}
        ]}),
        &["tools[*].arr[0].f", "tools[0].arr[*].k"],
        json!({"tools": [
            {"arr": [{"x": 1}, {"f": 2, "x": 2}]},
            {"arr": [{"k": 3, "x": 3}]}
        ]}),
    )]
    #[case::one_path_that_does_not_parse_among_others(
        json!({"meta": {"user": "u"}, "top": 0.7}),
        &["top[0", "meta.user"],
        json!({"meta": {}, "top": 0.7}),
    )]
    fn applies_every_path(#[case] value: Value, #[case] paths: &[&str], #[case] expected: Value) {
        assert_eq!(delete_nested_values(value, paths.iter().copied()), expected);
    }

    #[rstest]
    #[case::twelve_levels_over_twenty_thousand_properties(12, 20_000)]
    fn paths_overlapping_through_every_and_an_index_finish_well_inside_one_request(
        #[case] depth: usize,
        #[case] width: usize,
    ) {
        let properties: serde_json::Map<String, Value> =
            (0..width).map(|n| (format!("p{n}"), json!(n))).collect();
        let nested = (0..depth).fold(Value::Object(properties), |inner, _| json!([inner]));
        let combos = 1usize << depth;
        let paths: Vec<String> = (0..combos)
            .map(|combo| {
                let steps: String = (0..depth)
                    .map(|level| {
                        if combo >> level & 1 == 1 {
                            "[*]"
                        } else {
                            "[0]"
                        }
                    })
                    .collect();
                format!("tools{steps}.p{combo}")
            })
            .collect();
        let started = std::time::Instant::now();
        let trimmed =
            delete_nested_values(json!({"tools": nested}), paths.iter().map(String::as_str));
        assert!(started.elapsed() < std::time::Duration::from_secs(5));
        let kept: serde_json::Map<String, Value> = (combos..width)
            .map(|n| (format!("p{n}"), json!(n)))
            .collect();
        let expected = (0..depth).fold(Value::Object(kept), |inner, _| json!([inner]));
        assert_eq!(trimmed, json!({"tools": expected}));
    }
}
