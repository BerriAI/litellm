use std::collections::{BTreeSet, HashSet};
use std::path::{Path, PathBuf};

use indexmap::IndexMap;
use litellm_model_catalog::{
    Catalog, FallbackGeneralizations, ModelInfo, Provenance, model_entry_json_schema,
};
use rstest::{fixture, rstest};
use serde_json::{Map, Value};

#[fixture]
fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

fn json_eq(left: &Value, right: &Value) -> bool {
    match (left, right) {
        (Value::Number(left), Value::Number(right)) => left.as_f64() == right.as_f64(),
        (Value::Array(left), Value::Array(right)) => {
            left.len() == right.len() && left.iter().zip(right).all(|(a, b)| json_eq(a, b))
        }
        (Value::Object(left), Value::Object(right)) => {
            left.len() == right.len()
                && left
                    .iter()
                    .all(|(key, value)| right.get(key).is_some_and(|other| json_eq(value, other)))
        }
        _ => left == right,
    }
}

fn keys(value: &Map<String, Value>) -> BTreeSet<String> {
    value.keys().cloned().collect()
}

fn symmetric_difference(left: &BTreeSet<String>, right: &BTreeSet<String>) -> BTreeSet<String> {
    left.symmetric_difference(right).cloned().collect()
}

#[rstest]
#[case("model_prices_and_context_window.json")]
#[case("litellm/model_prices_and_context_window_backup.json")]
fn every_entry_round_trips_through_model_info(repo_root: PathBuf, #[case] filename: &str) {
    let body = std::fs::read(repo_root.join(filename)).unwrap();
    let document: IndexMap<String, Value> = serde_json::from_slice(&body).unwrap();
    for (model_name, value) in document {
        if matches!(
            model_name.as_str(),
            "sample_spec" | "fallback_generalizations"
        ) {
            continue;
        }
        let object = value
            .as_object()
            .unwrap_or_else(|| panic!("{model_name} is not an object"));
        let info: ModelInfo = serde_json::from_value(value.clone())
            .unwrap_or_else(|error| panic!("{model_name} does not deserialize: {error}"));
        let serialized = serde_json::to_value(info).unwrap();
        let serialized_object = serialized
            .as_object()
            .unwrap_or_else(|| panic!("{model_name} did not serialize as an object"));
        let mut expected = object.clone();
        expected.remove("aliases");
        let expected_keys = keys(&expected);
        let serialized_keys = keys(serialized_object);
        assert_eq!(
            expected_keys,
            serialized_keys,
            "{model_name} key difference: {:?}",
            symmetric_difference(&expected_keys, &serialized_keys)
        );
        assert!(
            json_eq(&Value::Object(expected), &serialized),
            "{model_name} changed during ModelInfo round-trip"
        );
    }
}

#[rstest]
fn fallback_generalizations_are_typed(repo_root: PathBuf) {
    let body = std::fs::read(repo_root.join("model_prices_and_context_window.json")).unwrap();
    let document: Map<String, Value> = serde_json::from_slice(&body).unwrap();
    let Some(raw_rules) = document.get("fallback_generalizations") else {
        return;
    };
    let _: FallbackGeneralizations = serde_json::from_value(raw_rules.clone()).unwrap();
    let catalog = Catalog::parse(&body, Provenance::default()).unwrap();
    assert!(
        catalog
            .fallback_rules()
            .is_some_and(|rules| !rules.is_empty())
    );
}

#[rstest]
fn generated_schema_properties_match_repo_schema(repo_root: PathBuf) {
    let body =
        std::fs::read(repo_root.join("model_prices_and_context_window.schema.json")).unwrap();
    let document: Value = serde_json::from_slice(&body).unwrap();
    let repo_entry_properties = document["$defs"]["modelEntry"]["properties"]
        .as_object()
        .unwrap();
    let generated = serde_json::to_value(model_entry_json_schema()).unwrap();
    let generated_properties = generated["properties"].as_object().unwrap();
    let expected = keys(repo_entry_properties);
    let actual = keys(generated_properties);
    assert_eq!(
        expected,
        actual,
        "modelEntry property difference: {:?}",
        symmetric_difference(&expected, &actual)
    );

    let repo_root_properties = document["properties"].as_object().unwrap();
    let actual_root: HashSet<String> = repo_root_properties.keys().cloned().collect();
    let expected_root: HashSet<String> = ["sample_spec", "fallback_generalizations"]
        .into_iter()
        .map(str::to_owned)
        .collect();
    assert_eq!(actual_root, expected_root);
}
