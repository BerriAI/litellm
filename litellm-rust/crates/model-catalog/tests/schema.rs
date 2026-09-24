#![cfg(feature = "schema")]

use std::collections::BTreeSet;
use std::path::Path;

use litellm_model_catalog::model_entry_json_schema;
use rstest::rstest;
use serde_json::Value;

fn schema() -> Value {
    serde_json::to_value(model_entry_json_schema()).expect("generated schema serializes")
}

#[rstest]
#[case("model_prices_and_context_window.json")]
#[case("litellm/model_prices_and_context_window_backup.json")]
fn generated_schema_covers_catalog_fields(#[case] path: &str) {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..");
    let catalog: Value = serde_json::from_slice(&std::fs::read(root.join(path)).unwrap()).unwrap();
    let schema = schema();
    let properties = schema["properties"]
        .as_object()
        .expect("ModelInfo schema has properties");
    let fields: BTreeSet<&str> = catalog
        .as_object()
        .expect("catalog is an object")
        .iter()
        .filter(|(name, _)| *name != "sample_spec" && *name != "fallback_generalizations")
        .flat_map(|(_, entry)| entry.as_object().expect("model entry is an object").keys())
        .map(String::as_str)
        .filter(|name| *name != "aliases")
        .collect();
    let missing: Vec<_> = fields
        .into_iter()
        .filter(|name| !properties.contains_key(*name))
        .collect();

    assert!(
        missing.is_empty(),
        "{path}: fields missing from schema: {missing:?}"
    );
}

#[rstest]
#[case("Mode", "chat")]
#[case("ReasoningEffort", "high")]
#[case("InputModality", "image")]
fn generated_schema_includes_enum_values(#[case] definition: &str, #[case] value: &str) {
    let schema = schema();
    let variants = schema["$defs"][definition]["enum"]
        .as_array()
        .expect("enum definition has variants");

    assert!(variants.iter().any(|variant| variant == value));
}

#[test]
fn generated_schema_includes_nested_pricing_types() {
    let schema = schema();
    let definitions = schema["$defs"].as_object().expect("schema has definitions");

    assert!(definitions.contains_key("OffPeakPricing"));
    assert!(definitions.contains_key("TieredRate"));
    assert!(definitions.contains_key("UtcHours"));
}
