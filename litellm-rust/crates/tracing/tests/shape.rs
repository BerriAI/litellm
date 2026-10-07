use litellm_tracing::{PayloadShape, ShapeLimits};
use rstest::rstest;
use serde::Deserialize;
use serde_json::{Value, json};

#[derive(Deserialize)]
struct Case {
    input: Value,
    expected: PayloadShape,
}

#[rstest]
#[case::provider_request(0)]
#[case::heterogeneous_arrays_and_empty_containers(1)]
#[case::dynamic_keys(2)]
#[case::escaped_keys(3)]
#[case::root_array(4)]
#[case::scalar(5)]
fn shapes_match_the_shared_language_contract(#[case] index: usize) {
    let cases: Vec<Case> =
        serde_json::from_str(include_str!("fixtures/payload-shapes.json")).unwrap();
    let case = &cases[index];
    assert_eq!(
        PayloadShape::extract(&case.input, ShapeLimits::default()),
        case.expected
    );
}

#[rstest]
#[case::nodes(ShapeLimits { nodes: 2, ..ShapeLimits::default() })]
#[case::depth(ShapeLimits { depth: 1, ..ShapeLimits::default() })]
#[case::paths(ShapeLimits { paths: 1, ..ShapeLimits::default() })]
#[case::bytes(ShapeLimits { bytes: 1, ..ShapeLimits::default() })]
fn exceeding_a_limit_discards_partial_paths(#[case] limits: ShapeLimits) {
    assert_eq!(
        PayloadShape::extract(&json!({"a": {"b": 1}, "c": 2}), limits),
        PayloadShape {
            field_paths: vec![],
            truncated: true
        }
    );
}

#[rstest]
fn changing_values_and_array_order_does_not_change_the_shape() {
    let first = json!({"items": [{"a": "private"}, {"b": 1}], "model": "private"});
    let second = json!({"model": false, "items": [{"b": null}, {"a": "different"}]});
    assert_eq!(
        PayloadShape::extract(&first, ShapeLimits::default()),
        PayloadShape::extract(&second, ShapeLimits::default())
    );
}

#[rstest]
fn oversized_keys_do_not_escape_the_byte_budget() {
    let value = json!({"k".repeat(129): null});
    assert_eq!(
        PayloadShape::extract(&value, ShapeLimits::default()),
        PayloadShape {
            field_paths: vec![],
            truncated: true
        }
    );
}

#[rstest]
fn stream_shapes_union_paths_and_remain_truncated_after_a_limit() {
    let mut shape = PayloadShape::extract(&json!({"a": "first"}), ShapeLimits::default());
    shape.merge(
        PayloadShape::extract(&json!({"b": "second"}), ShapeLimits::default()),
        ShapeLimits::default(),
    );
    assert_eq!(
        shape,
        PayloadShape {
            field_paths: vec!["$['a']".into(), "$['b']".into()],
            truncated: false
        }
    );
    shape.merge(
        PayloadShape::extract(&json!({"c": "third"}), ShapeLimits::default()),
        ShapeLimits {
            paths: 2,
            ..ShapeLimits::default()
        },
    );
    shape.merge(PayloadShape::default(), ShapeLimits::default());
    assert_eq!(
        shape,
        PayloadShape {
            field_paths: vec![],
            truncated: true
        }
    );
}

#[rstest]
#[case::provider_request(0)]
#[case::heterogeneous_arrays_and_empty_containers(1)]
#[case::dynamic_keys(2)]
#[case::escaped_keys(3)]
#[case::root_array(4)]
#[case::scalar(5)]
fn serde_source_matches_the_shared_contract(#[case] index: usize) {
    let cases: Vec<Case> =
        serde_json::from_str(include_str!("fixtures/payload-shapes.json")).unwrap();
    let case = &cases[index];
    assert_eq!(
        PayloadShape::extract_serialized(&case.input, ShapeLimits::default()),
        case.expected
    );
}

#[rstest]
fn serde_source_does_not_format_scalar_values() {
    struct Secret;
    impl serde::Serialize for Secret {
        fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            struct Private;
            impl std::fmt::Display for Private {
                fn fmt(&self, _: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                    panic!("must not format a payload scalar");
                }
            }
            serializer.collect_str(&Private)
        }
    }
    #[derive(serde::Serialize)]
    struct Response {
        content: Secret,
    }
    assert_eq!(
        PayloadShape::extract_serialized(&Response { content: Secret }, ShapeLimits::default())
            .field_paths,
        ["$['content']"]
    );
}

#[rstest]
#[case::nodes(ShapeLimits { nodes: 1, ..ShapeLimits::default() })]
#[case::depth(ShapeLimits { depth: 0, ..ShapeLimits::default() })]
fn serde_sources_obey_depth_and_node_limits(#[case] limits: ShapeLimits) {
    let value = serde_json::json!({"messages":[{"content":"private"}]});
    assert_eq!(
        PayloadShape::extract_serialized(&value, limits),
        PayloadShape {
            field_paths: vec![],
            truncated: true
        }
    );
}

#[rstest]
fn serde_source_respects_aliases_skips_and_flattening() {
    #[derive(serde::Serialize)]
    struct Response {
        #[serde(rename = "output")]
        content: String,
        #[serde(skip)]
        _private: String,
        #[serde(flatten)]
        extra: std::collections::BTreeMap<String, Value>,
    }
    let value = Response {
        content: "private".into(),
        _private: "secret".into(),
        extra: [("usage".into(), json!({"tokens": 1}))].into(),
    };
    assert_eq!(
        PayloadShape::extract_serialized(&value, ShapeLimits::default()),
        PayloadShape::extract(
            &serde_json::to_value(&value).unwrap(),
            ShapeLimits::default()
        )
    );
}

#[derive(serde::Serialize)]
enum ResponseVariant {
    Unit,
    Content(String),
    Tuple(String, Value),
    Object { content: String, usage: Value },
}

#[rstest]
#[case::unit(ResponseVariant::Unit)]
#[case::newtype(ResponseVariant::Content("private".into()))]
#[case::tuple(ResponseVariant::Tuple("private".into(), json!({"tokens": 1})))]
#[case::object(ResponseVariant::Object { content: "private".into(), usage: json!({"tokens": 1}) })]
fn serde_source_preserves_enum_container_paths(#[case] value: ResponseVariant) {
    assert_eq!(
        PayloadShape::extract_serialized(&value, ShapeLimits::default()),
        PayloadShape::extract(
            &serde_json::to_value(&value).unwrap(),
            ShapeLimits::default()
        )
    );
}
