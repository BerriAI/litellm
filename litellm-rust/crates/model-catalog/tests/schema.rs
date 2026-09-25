#![cfg(feature = "schema")]

use std::collections::BTreeSet;
use std::path::Path;

use litellm_model_catalog::{model_entry_json_schema, registry_json_schema};
use rstest::rstest;
use serde_json::{Value, json};

fn schema() -> Value {
    serde_json::to_value(model_entry_json_schema()).expect("generated schema serializes")
}

fn registry_validator() -> jsonschema::Validator {
    let schema = serde_json::to_value(registry_json_schema()).unwrap();
    jsonschema::options()
        .should_validate_formats(true)
        .build(&schema)
        .expect("generated registry schema is valid")
}

#[rstest]
#[case("model_prices_and_context_window.json")]
#[case("litellm/model_prices_and_context_window_backup.json")]
fn generated_registry_schema_validates_checked_in_catalog(#[case] path: &str) {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..");
    let catalog: Value = serde_json::from_slice(&std::fs::read(root.join(path)).unwrap()).unwrap();
    let validator = registry_validator();
    let errors: Vec<_> = validator
        .iter_errors(&catalog)
        .map(|error| error.to_string())
        .collect();
    assert!(errors.is_empty(), "{path}: {errors:?}");
}

#[rstest]
#[case(json!({"example": {"litellm_provider": "test"}}))]
#[case(json!({"example": {"litellm_provider": "test", "future_field": true}}))]
#[case(json!({"sample_spec": {"litellm_provider": "placeholder"}}))]
fn generated_registry_schema_keeps_reader_compatibility(#[case] document: Value) {
    assert!(registry_validator().is_valid(&document));
}

#[rstest]
#[case::missing_provider(json!({"mode": "chat"}))]
#[case::negative_cost(json!({"litellm_provider": "test", "input_cost_per_token": -1}))]
#[case::negative_guardrail_cost(json!({"litellm_provider": "test", "guardrail_cost_per_unit": {"unit": -1}}))]
#[case::negative_search_cost(json!({"litellm_provider": "test", "search_context_cost_per_query": {"search_context_size_low": -1}}))]
#[case::negative_tier_cost(json!({"litellm_provider": "test", "tiered_pricing": [{"input_cost_per_token": -1}]}))]
#[case::negative_tier_range(json!({"litellm_provider": "test", "tiered_pricing": [{"range": [-1, 2]}]}))]
#[case::low_uplift(json!({"litellm_provider": "test", "regional_endpoint_uplift_multiplier": 0.5}))]
#[case::nullable_cost(json!({"litellm_provider": "test", "input_cost_per_token": null}))]
#[case::invalid_mode(json!({"litellm_provider": "test", "mode": "telepathy"}))]
#[case::invalid_date(json!({"litellm_provider": "test", "deprecation_date": "2026-02-31"}))]
#[case::invalid_hours(json!({"litellm_provider": "test", "off_peak_pricing": {"hours_utc": "25:00-01:00"}}))]
#[case::empty_windows(json!({"litellm_provider": "test", "off_peak_pricing": {"windows": []}}))]
#[case::invalid_weekday(json!({"litellm_provider": "test", "off_peak_pricing": {"windows": [{"hours_utc": "00:00-01:00", "weekdays": [0]}]}}))]
#[case::invalid_aliases(json!({"litellm_provider": "test", "aliases": "wrong"}))]
#[case::non_object_model(json!(4))]
fn generated_registry_schema_rejects_invalid_entries(#[case] entry: Value) {
    assert!(!registry_validator().is_valid(&json!({"example": entry})));
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
