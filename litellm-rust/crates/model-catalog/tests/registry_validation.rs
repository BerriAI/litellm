use std::path::{Path, PathBuf};

use litellm_model_catalog::{Catalog, FallbackGeneralizations, Provenance};
use rstest::{fixture, rstest};
use serde_json::{Map, Value, json};

#[fixture]
fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

#[rstest]
#[ignore]
fn checked_in_registry_parses_with_every_key_understood(repo_root: PathBuf) {
    let body = std::fs::read(repo_root.join("model_prices_and_context_window.json")).unwrap();
    let catalog = Catalog::parse(&body, Provenance::default()).unwrap();
    let unrecognized: Vec<_> = catalog.unrecognized().collect();
    assert!(
        unrecognized.is_empty(),
        "keys no type reads: {unrecognized:#?}"
    );
}

#[rstest]
#[ignore]
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
#[case::input("input_cost_per_token_above_100k_tokens")]
#[case::input_batches("input_cost_per_token_above_100k_tokens_batches")]
#[case::output("output_cost_per_token_above_100k_tokens")]
#[case::output_batches("output_cost_per_token_above_100k_tokens_batches")]
#[case::cache_creation("cache_creation_input_token_cost_above_100k_tokens")]
#[case::cache_creation_batches("cache_creation_input_token_cost_above_100k_tokens_batches")]
#[case::cache_creation_1hr("cache_creation_input_token_cost_above_1hr_above_100k_tokens")]
#[case::cache_read("cache_read_input_token_cost_above_100k_tokens")]
#[case::cache_read_batches("cache_read_input_token_cost_above_100k_tokens_batches")]
fn above_100k_tier_rates_are_prices(#[case] field: &str) {
    let body = json!({"test": {"litellm_provider": "anthropic", field: 5e-7}}).to_string();
    let catalog = Catalog::parse(body.as_bytes(), Provenance::default()).unwrap();
    assert_eq!(catalog.unrecognized().count(), 0);
    assert_eq!(
        catalog
            .lookup("test")
            .unwrap()
            .entry
            .model()
            .prices
            .iter()
            .count(),
        1
    );
}
