use std::path::{Path, PathBuf};

use litellm_model_catalog::{
    Catalog, FallbackGeneralizations, Provenance, validate_model_entry, validate_registry,
};
use rstest::{fixture, rstest};
use serde_json::{Map, Value};

#[fixture]
fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

#[rstest]
#[case("model_prices_and_context_window.json")]
#[case("litellm/model_prices_and_context_window_backup.json")]
fn checked_in_registry_passes_strict_validation(repo_root: PathBuf, #[case] filename: &str) {
    let body = std::fs::read(repo_root.join(filename)).unwrap();
    let catalog = Catalog::parse(&body, Provenance::default()).unwrap();
    validate_registry(&catalog).unwrap();
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
#[case::missing_provider(serde_json::json!({"mode": "chat"}), "litellm_provider")]
#[case::unknown_field(serde_json::json!({"litellm_provider": "test", "typo": true}), "unknown field")]
#[case::negative_price(serde_json::json!({"litellm_provider": "test", "input_cost_per_token": -1}), "nonnegative")]
#[case::negative_nested_price(serde_json::json!({"litellm_provider": "test", "guardrail_cost_per_unit": {"unit": -1}}), "nonnegative")]
#[case::invalid_mode(serde_json::json!({"litellm_provider": "test", "mode": "invalid"}), "unknown variant")]
#[case::invalid_date(serde_json::json!({"litellm_provider": "test", "deprecation_date": "2026-02-31"}), "deprecation_date")]
#[case::invalid_hours(serde_json::json!({"litellm_provider": "test", "off_peak_pricing": {"hours_utc": "25:00-01:00"}}), "hours_utc")]
#[case::empty_windows(serde_json::json!({"litellm_provider": "test", "off_peak_pricing": {"windows": []}}), "windows is empty")]
#[case::invalid_weekday(serde_json::json!({"litellm_provider": "test", "off_peak_pricing": {"windows": [{"hours_utc": "00:00-01:00", "weekdays": [0]}]}}), "weekdays is invalid")]
#[case::invalid_aliases(serde_json::json!({"litellm_provider": "test", "aliases": ["good", 7]}), "aliases must contain strings")]
#[case::null_aliases(serde_json::json!({"litellm_provider": "test", "aliases": null}), "aliases must be an array")]
fn registry_validation_rejects_malformed_entries(#[case] entry: Value, #[case] expected: &str) {
    assert!(
        validate_model_entry("test", &entry)
            .unwrap_err()
            .to_string()
            .contains(expected)
    );
}

#[test]
fn checked_in_catalog_and_backup_match() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..");
    let current = std::fs::read(root.join("model_prices_and_context_window.json")).unwrap();
    let backup =
        std::fs::read(root.join("litellm/model_prices_and_context_window_backup.json")).unwrap();
    assert_eq!(current, backup);
    let catalog = Catalog::parse(&current, Provenance::default()).unwrap();
    assert!(
        catalog.alias_issues().is_empty(),
        "invalid registry aliases"
    );
}
