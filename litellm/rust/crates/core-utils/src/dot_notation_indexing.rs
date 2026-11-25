use std::collections::{HashMap, HashSet};

use serde_json::{Map, Value};

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

type NodeId = usize;
type NameId = usize;

const ROOT: NodeId = 0;

/// The most nodes that may reach one part of the value. Paths reach a part through one node per
/// distinct run of `[*]` and index steps above it, and matching the part costs a lookup per node,
/// so the cap keeps the walk within a fixed multiple of the value's size however the paths overlap.
const MAX_NODES_AT_ONE_VALUE: usize = 64;

/// The walk stopped because more than `MAX_NODES_AT_ONE_VALUE` nodes reached one part of the value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OverlappingPaths;

impl OverlappingPaths {
    pub const EXPECTED: &'static str =
        "paths that reach no value through more than 64 different runs of `[*]` and index steps";
}

/// What the paths sharing one prefix address below it. `pending` holds the rest of each of those
/// paths until a part of the body reaches the prefix and `expand` reads one more segment of each,
/// so no path is read deeper than the body goes and each segment is read once.
#[derive(Default)]
struct Node<'a> {
    ends: bool,
    pending: Vec<&'a str>,
    fields: HashMap<NameId, NodeId>,
    field_name_bytes: usize,
    every: Option<NodeId>,
    indexes: HashMap<usize, NodeId>,
}

/// The paths as one tree of the segments they share, held in a flat list so that neither building
/// nor dropping it recurses, with each field name numbered once.
struct Removals<'a> {
    nodes: Vec<Node<'a>>,
    names: HashMap<&'a str, NameId>,
    spelled: Vec<&'a str>,
}

impl<'a> Removals<'a> {
    fn new(paths: impl IntoIterator<Item = &'a str>) -> Self {
        let root = Node {
            pending: paths.into_iter().collect(),
            ..Node::default()
        };
        Self {
            nodes: vec![root],
            names: HashMap::new(),
            spelled: Vec::new(),
        }
    }

    fn expand(&mut self, node: NodeId) {
        std::mem::take(&mut self.nodes[node].pending)
            .into_iter()
            .filter_map(next_segment)
            .for_each(|(segment, rest)| {
                let child = self.child(node, segment);
                match rest {
                    "" => self.nodes[child].ends = true,
                    rest => self.nodes[child].pending.push(rest),
                }
            });
    }

    fn child(&mut self, node: NodeId, segment: Segment<'a>) -> NodeId {
        let fresh = self.nodes.len();
        let child = match segment {
            Segment::Field(name) => {
                let id = self.number(name);
                let parent = &mut self.nodes[node];
                let child = *parent.fields.entry(id).or_insert(fresh);
                if child == fresh {
                    parent.field_name_bytes += name.len();
                }
                child
            }
            Segment::Every => *self.nodes[node].every.get_or_insert(fresh),
            Segment::Index(index) => *self.nodes[node].indexes.entry(index).or_insert(fresh),
        };
        if child == fresh {
            self.nodes.push(Node::default());
        }
        child
    }

    fn number(&mut self, name: &'a str) -> NameId {
        let fresh = self.spelled.len();
        let id = *self.names.entry(name).or_insert(fresh);
        if id == fresh {
            self.spelled.push(name);
        }
        id
    }

    /// Deletes what any of `nodes` addresses under `value`. A field that ends a path is deleted
    /// from the object holding it, and a path that ends on an element or `[*]` deletes nothing.
    /// Each part of `value` is visited once, with every node that reaches it, so paths that overlap
    /// through `[*]` and an index never walk the same element twice.
    fn prune(&mut self, nodes: &[NodeId], value: &mut Value) -> Result<(), OverlappingPaths> {
        if nodes.len() > MAX_NODES_AT_ONE_VALUE {
            return Err(OverlappingPaths);
        }
        match value {
            Value::Object(object) => self.prune_object(nodes, object),
            Value::Array(items) => self.prune_array(nodes, items),
            _ => Ok(()),
        }
    }

    fn prune_object(
        &mut self,
        nodes: &[NodeId],
        object: &mut Map<String, Value>,
    ) -> Result<(), OverlappingPaths> {
        nodes.iter().for_each(|node| self.expand(*node));
        let reached = self.fields_in(nodes, object);
        let ending: HashSet<NameId> = reached
            .iter()
            .filter(|(_, children)| children.iter().any(|child| self.nodes[*child].ends))
            .map(|(name, _)| *name)
            .collect();
        if !ending.is_empty() {
            object.retain(|key, _| {
                self.names
                    .get(key.as_str())
                    .is_none_or(|name| !ending.contains(name))
            });
        }
        reached
            .into_iter()
            .filter(|(name, _)| !ending.contains(name))
            .try_for_each(
                |(name, children)| match object.get_mut(self.spelled[name]) {
                    Some(item) => self.prune(&children, item),
                    None => Ok(()),
                },
            )
    }

    /// The fields of `object` that any of `nodes` names, each with the nodes below it. The match
    /// reads the names when they are shorter in total than `object` has keys, and the keys
    /// otherwise, so a part of the body never costs more to match than its own size, however long
    /// the names are or however many parts the same nodes reach.
    fn fields_in(
        &self,
        nodes: &[NodeId],
        object: &Map<String, Value>,
    ) -> HashMap<NameId, Vec<NodeId>> {
        let mut reached: HashMap<NameId, Vec<NodeId>> = HashMap::new();
        let name_bytes: usize = nodes
            .iter()
            .map(|node| self.nodes[*node].field_name_bytes)
            .sum();
        if name_bytes <= object.len() {
            nodes
                .iter()
                .flat_map(|node| &self.nodes[*node].fields)
                .filter(|(name, _)| object.contains_key(self.spelled[**name]))
                .for_each(|(name, child)| reached.entry(*name).or_default().push(*child));
            return reached;
        }
        object
            .keys()
            .filter_map(|key| self.names.get(key.as_str()))
            .for_each(|name| {
                nodes
                    .iter()
                    .filter_map(|node| self.nodes[*node].fields.get(name))
                    .for_each(|child| reached.entry(*name).or_default().push(*child));
            });
        reached
    }

    /// The elements of a list `len` long that any of `nodes` indexes, each with the nodes below it,
    /// read from whichever is smaller, the indexes or the list, so a short list costs no more to
    /// match than its own length however many indexes the paths name.
    fn indexes_in(&self, nodes: &[NodeId], len: usize) -> HashMap<usize, Vec<NodeId>> {
        let mut at: HashMap<usize, Vec<NodeId>> = HashMap::new();
        let named: usize = nodes
            .iter()
            .map(|node| self.nodes[*node].indexes.len())
            .sum();
        if named <= len {
            nodes
                .iter()
                .flat_map(|node| &self.nodes[*node].indexes)
                .filter(|(index, _)| **index < len)
                .for_each(|(index, child)| at.entry(*index).or_default().push(*child));
            return at;
        }
        (0..len).for_each(|index| {
            nodes
                .iter()
                .filter_map(|node| self.nodes[*node].indexes.get(&index))
                .for_each(|child| at.entry(index).or_default().push(*child));
        });
        at
    }

    fn prune_array(
        &mut self,
        nodes: &[NodeId],
        items: &mut [Value],
    ) -> Result<(), OverlappingPaths> {
        nodes.iter().for_each(|node| self.expand(*node));
        let every: Vec<NodeId> = nodes
            .iter()
            .filter_map(|node| self.nodes[*node].every)
            .collect();
        let at = self.indexes_in(nodes, items.len());
        if every.is_empty() {
            return at
                .into_iter()
                .try_for_each(|(index, indexed)| self.prune(&indexed, &mut items[index]));
        }
        items
            .iter_mut()
            .enumerate()
            .try_for_each(|(index, item)| match at.get(&index) {
                Some(indexed) => self.prune(&[every.as_slice(), indexed].concat(), item),
                None => self.prune(&every, item),
            })
    }
}

/// `value` without what each of `paths` addresses, in one pass over `value` however many paths
/// there are. A path that does not parse deletes nothing.
pub fn delete_nested_values<'a>(
    mut value: Value,
    paths: impl IntoIterator<Item = &'a str>,
) -> Result<Value, OverlappingPaths> {
    Removals::new(paths).prune(&[ROOT], &mut value)?;
    Ok(value)
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
        assert_eq!(delete_nested_values(body, [path]), Ok(expected));
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
        assert_eq!(delete_nested_values(body.clone(), [path]), Ok(body));
    }

    #[rstest]
    #[case::absent_from_the_value(json!({"top": 0.7}))]
    #[case::through_fields_the_value_holds(json!({"a": {"a": {"keep": 1}}}))]
    fn a_path_far_deeper_than_the_value_leaves_it_untouched(#[case] body: Value) {
        let path = vec!["a"; 100_000].join(".");
        assert_eq!(
            delete_nested_values(body.clone(), [path.as_str()]),
            Ok(body)
        );
    }

    #[rstest]
    #[case::long_rest_after_an_absent_field("tools[*].missing.")]
    #[case::long_field_name_in_every_element("tools[*].")]
    fn a_long_path_under_many_elements_finishes_well_inside_one_request(#[case] prefix: &str) {
        let body = json!({"tools": vec![json!({"name": "t"}); 2_000]});
        let path = format!("{prefix}{}", "x".repeat(1 << 20));
        let started = std::time::Instant::now();
        let trimmed = delete_nested_values(body.clone(), [path.as_str()]);
        assert!(started.elapsed() < std::time::Duration::from_secs(5));
        assert_eq!(trimmed, Ok(body));
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
            trimmed
                .ok()
                .and_then(|value| serde_json::to_string(&value).ok())
                .as_deref(),
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
        assert_eq!(trimmed, Ok(body));
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
            Ok(json!({"tools": [{"name": "t", "input_schema": {"properties": {}}}]}))
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
        assert_eq!(
            delete_nested_values(value, paths.iter().copied()),
            Ok(expected)
        );
    }

    fn every_and_first_element_runs(depth: usize) -> impl Iterator<Item = (usize, String)> {
        (0..1usize << depth).map(move |combo| {
            let steps = (0..depth)
                .map(|level| {
                    if combo >> level & 1 == 1 {
                        "[*]"
                    } else {
                        "[0]"
                    }
                })
                .collect();
            (combo, steps)
        })
    }

    #[rstest]
    #[case::six_levels_over_twenty_thousand_properties(6, 20_000)]
    fn paths_overlapping_through_every_and_an_index_finish_well_inside_one_request(
        #[case] depth: usize,
        #[case] width: usize,
    ) {
        let properties: serde_json::Map<String, Value> =
            (0..width).map(|n| (format!("p{n}"), json!(n))).collect();
        let nested = (0..depth).fold(Value::Object(properties), |inner, _| json!([inner]));
        let paths: Vec<String> = every_and_first_element_runs(depth)
            .map(|(combo, steps)| format!("tools{steps}.p{combo}"))
            .collect();
        let started = std::time::Instant::now();
        let trimmed =
            delete_nested_values(json!({"tools": nested}), paths.iter().map(String::as_str));
        assert!(started.elapsed() < std::time::Duration::from_secs(5));
        let kept: serde_json::Map<String, Value> = (paths.len()..width)
            .map(|n| (format!("p{n}"), json!(n)))
            .collect();
        let expected = (0..depth).fold(Value::Object(kept), |inner, _| json!([inner]));
        assert_eq!(trimmed, Ok(json!({"tools": expected})));
    }

    #[rstest]
    #[case::one_level_past_the_cap(7)]
    #[case::sixteen_levels(16)]
    fn paths_overlapping_past_the_cap_fail_before_walking_the_list_they_reach(
        #[case] depth: usize,
    ) {
        let elements = json!(vec![json!({"drop": 1, "keep": 1}); 1_000]);
        let nested = (0..depth).fold(elements, |inner, _| json!([inner]));
        let body = json!({"tools": [{"input_schema": {"x": nested}}]});
        let paths: Vec<String> = every_and_first_element_runs(depth)
            .map(|(_, steps)| format!("tools[0].input_schema.x{steps}[*].drop"))
            .collect();
        let started = std::time::Instant::now();
        let trimmed = delete_nested_values(body, paths.iter().map(String::as_str));
        assert!(started.elapsed() < std::time::Duration::from_secs(5));
        assert_eq!(trimmed, Err(OverlappingPaths));
    }

    #[rstest]
    #[case::two_hundred_thousand_indexes_over_five_thousand_short_lists(200_000, 5_000)]
    fn many_indexes_over_many_short_lists_finish_well_inside_one_request(
        #[case] indexes: usize,
        #[case] lists: usize,
    ) {
        let body = json!({"tools": vec![json!({"arr": [{"x": 1, "k": 1}]}); lists]});
        let paths: Vec<String> = (0..indexes)
            .map(|index| format!("tools[*].arr[{index}].x"))
            .collect();
        let started = std::time::Instant::now();
        let trimmed = delete_nested_values(body, paths.iter().map(String::as_str));
        assert!(started.elapsed() < std::time::Duration::from_secs(5));
        assert_eq!(
            trimmed,
            Ok(json!({"tools": vec![json!({"arr": [{"k": 1}]}); lists]}))
        );
    }
}
