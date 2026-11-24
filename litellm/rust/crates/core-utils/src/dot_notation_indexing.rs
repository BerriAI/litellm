use std::collections::HashMap;

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

/// Every path to delete, as a tree of the segments they share. A field that ends a path is
/// deleted from the object holding it. A path that ends on an element or `[*]` deletes nothing.
#[derive(Default)]
struct Removals {
    ends: bool,
    fields: HashMap<String, Removals>,
    every: Option<Box<Removals>>,
    indexes: HashMap<usize, Removals>,
}

impl Removals {
    fn insert(&mut self, segments: &[Segment]) {
        let Some((segment, tail)) = segments.split_first() else {
            self.ends = true;
            return;
        };
        match segment {
            Segment::Field(name) => self.fields.entry(name.clone()).or_default().insert(tail),
            Segment::Every => self.every.get_or_insert_default().insert(tail),
            Segment::Index(index) => self.indexes.entry(*index).or_default().insert(tail),
        }
    }

    fn prune(&self, value: &mut Value) {
        match value {
            Value::Object(object) => {
                object.retain(|key, _| !self.fields.get(key).is_some_and(|field| field.ends));
                object.iter_mut().for_each(|(key, item)| {
                    if let Some(field) = self.fields.get(key) {
                        field.prune(item);
                    }
                });
            }
            Value::Array(items) => items.iter_mut().enumerate().for_each(|(index, item)| {
                if let Some(every) = &self.every {
                    every.prune(item);
                }
                if let Some(at) = self.indexes.get(&index) {
                    at.prune(item);
                }
            }),
            _ => {}
        }
    }
}

/// `value` without what each of `paths` addresses, in one pass over `value` however many paths
/// there are. A path that does not parse deletes nothing.
pub fn delete_nested_values<'a>(
    mut value: Value,
    paths: impl IntoIterator<Item = &'a str>,
) -> Value {
    let mut removals = Removals::default();
    paths
        .into_iter()
        .filter_map(parse_segments)
        .for_each(|segments| removals.insert(&segments));
    removals.prune(&mut value);
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
        assert!(started.elapsed() < std::time::Duration::from_secs(1));
        assert_eq!(trimmed, body);
    }

    #[rstest]
    #[case::forty_thousand_properties(40_000)]
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
        assert!(started.elapsed() < std::time::Duration::from_secs(1));
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
    #[case::one_path_that_does_not_parse_among_others(
        json!({"meta": {"user": "u"}, "top": 0.7}),
        &["top[0", "meta.user"],
        json!({"meta": {}, "top": 0.7}),
    )]
    fn applies_every_path(#[case] value: Value, #[case] paths: &[&str], #[case] expected: Value) {
        assert_eq!(delete_nested_values(value, paths.iter().copied()), expected);
    }
}
