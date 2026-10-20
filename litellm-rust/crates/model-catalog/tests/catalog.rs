use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use litellm_model_catalog::{Catalog, Error, IntegrityLimits, Mode, Provenance};
use rstest::{fixture, rstest};
use serde_json::json;

const ALPHA_FIXTURE: &[u8] = br#"{
    "sample_spec":{"explanation":"example"},
    "fallback_generalizations":{"rules":[{"name":"family","pattern":"^new-","model_info":{"mode":"chat"}}]},
    "Alpha":{"litellm_provider":"test","aliases":["short"],"price":0,"enabled":false,
             "optional":null,"unknown":{"nested":[1,{"x":true}]}}
}"#;

#[fixture]
fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

#[fixture]
fn current_catalog(repo_root: PathBuf) -> Catalog {
    let body = std::fs::read(repo_root.join("model_prices_and_context_window.json")).unwrap();
    Catalog::parse(&body, Provenance::default()).unwrap()
}

#[fixture]
fn fixture_catalog() -> Catalog {
    Catalog::parse(
        ALPHA_FIXTURE,
        Provenance {
            source: Some("fixture".into()),
            revision: Some("rev".into()),
            etag: None,
        },
    )
    .unwrap()
}

fn rejection(body: &[u8]) -> String {
    Catalog::parse(body, Provenance::default())
        .unwrap_err()
        .to_string()
}

#[rstest]
#[case::decisions("decisions", Mode::Decisions)]
#[case::legacy_evaluation("evaluation", Mode::Evaluation)]
fn parses_decision_modes(#[case] mode: &str, #[case] expected: Mode) {
    let source = json!({"Test":{"litellm_provider":"test","mode":mode}}).to_string();
    let catalog = Catalog::parse(source.as_bytes(), Provenance::default()).unwrap();

    assert_eq!(
        catalog
            .lookup("Test")
            .unwrap()
            .entry
            .model()
            .attributes
            .mode,
        Some(expected)
    );
    assert_eq!(serde_json::to_value(expected).unwrap(), json!(mode));
}

#[rstest]
fn unknown_keys_are_served_raw_and_listed_as_unrecognized(fixture_catalog: Catalog) {
    let entry = fixture_catalog.lookup("Alpha").unwrap().entry;

    assert_eq!(
        entry.field("unknown"),
        Some(&json!({"nested":[1,{"x":true}]}))
    );
    assert_eq!(
        fixture_catalog.unrecognized().collect::<BTreeSet<_>>(),
        BTreeSet::from([
            ("Alpha", "enabled"),
            ("Alpha", "price"),
            ("Alpha", "unknown")
        ])
    );
}

#[rstest]
fn a_bad_known_value_rejects_the_document_naming_every_problem() {
    let rejected = rejection(
        json!({
            "Good": {"litellm_provider": "test"},
            "Bad": {"litellm_provider": "test", "mode": "chatt", "input_cost_per_token": "free"},
            "Worse": {"mode": "chat"}
        })
        .to_string()
        .as_bytes(),
    );

    assert!(
        rejected.contains("Bad.mode: unknown variant `chatt`"),
        "{rejected}"
    );
    assert!(rejected.contains("Bad.input_cost_per_token"), "{rejected}");
    assert!(
        rejected.contains("Worse.litellm_provider is required"),
        "{rejected}"
    );
}

#[rstest]
fn preserves_fields_and_metadata(fixture_catalog: Catalog) {
    let catalog = fixture_catalog;
    let entry = catalog.lookup("SHORT").unwrap();
    assert_eq!(entry.canonical_key, "Alpha");
    assert_eq!(entry.matched_key, "short");
    assert_eq!(entry.entry.field("price"), Some(&json!(0)));
    assert_eq!(entry.entry.field("enabled"), Some(&json!(false)));
    assert_eq!(entry.entry.field("optional"), Some(&json!(null)));
    assert_eq!(entry.entry.field("missing"), None);
    assert_eq!(entry.entry.field("aliases"), None);
    assert_eq!(
        entry.entry.model().attributes.litellm_provider.as_deref(),
        Some("test")
    );
    assert_eq!(
        catalog.sample_spec(),
        json!({"explanation":"example"}).as_object()
    );
    assert_eq!(catalog.fallback_rules().unwrap().len(), 1);
    assert_eq!(catalog.provenance().revision.as_deref(), Some("rev"));
    assert_eq!(catalog.model_count(), 1);
}

#[rstest]
fn snapshot_does_not_borrow_source() {
    let mut source = ALPHA_FIXTURE.to_vec();
    let catalog = Catalog::parse(&source, Provenance::default()).unwrap();
    source.fill(b' ');

    let entry = catalog.lookup("short").unwrap();
    assert_eq!(entry.canonical_key, "Alpha");
    assert_eq!(entry.entry.field("price"), Some(&json!(0)));
}

#[rstest]
#[case::alias_names_a_model(
    br#"{"First":{"litellm_provider":"t","aliases":["Second"]},"Second":{"litellm_provider":"t"}}"#,
    r#"First: alias "Second" is already a model name"#
)]
#[case::alias_claimed_twice(
    br#"{"First":{"litellm_provider":"t","aliases":["shared"]},"Second":{"litellm_provider":"t","aliases":["shared"]}}"#,
    r#"Second: alias "shared" already belongs to First"#
)]
#[case::aliases_not_a_list(
    br#"{"a":{"litellm_provider":"t","aliases":"bad"}}"#,
    "a.aliases must be an array of strings"
)]
#[case::alias_not_a_string(
    br#"{"b":{"litellm_provider":"t","aliases":[9,"ok"]}}"#,
    "b.aliases must be an array of strings"
)]
#[case::aliases_null(
    br#"{"c":{"litellm_provider":"t","aliases":null}}"#,
    "c.aliases must be an array of strings"
)]
#[case::entry_not_object(br#"{"a":1}"#, "a must be a JSON object")]
fn alias_and_shape_problems_reject_the_document(#[case] body: &[u8], #[case] expected: &str) {
    let rejected = rejection(body);
    assert!(rejected.contains(expected), "{rejected}");
}

#[rstest]
#[case("foo", "fOO")]
#[case("FOO", "fOO")]
#[case("Foo", "Foo")]
fn case_insensitive_lookup_prefers_the_later_entry(#[case] lookup: &str, #[case] expected: &str) {
    let catalog = Catalog::parse(
        br#"{"Foo":{"litellm_provider":"t"},"fOO":{"litellm_provider":"t"}}"#,
        Provenance::default(),
    )
    .unwrap();
    assert_eq!(catalog.lookup(lookup).unwrap().canonical_key, expected);
}

#[derive(Debug)]
enum FetchOutcome {
    Ok,
    Shrunk,
    BelowMinimum,
    InvalidRatio,
}

fn limits(min_model_count: usize, min_reference_ratio: f64) -> IntegrityLimits {
    IntegrityLimits {
        min_model_count,
        min_reference_ratio,
    }
}

#[rstest]
#[case::half_of_two(2, limits(1, 0.5), FetchOutcome::Ok)]
#[case::under_half_of_three(3, limits(1, 0.5), FetchOutcome::Shrunk)]
#[case::below_minimum(1, limits(2, 0.5), FetchOutcome::BelowMinimum)]
#[case::nan_ratio(1, limits(0, f64::NAN), FetchOutcome::InvalidRatio)]
fn a_fetched_registry_counts_canonical_models_against_the_current_one(
    #[case] current_models: usize,
    #[case] limits: IntegrityLimits,
    #[case] expected: FetchOutcome,
) {
    let current_body = serde_json::to_vec(
        &(0..current_models)
            .map(|index| (format!("m{index}"), json!({"litellm_provider": "t"})))
            .collect::<serde_json::Map<_, _>>(),
    )
    .unwrap();
    let current = Catalog::parse(&current_body, Provenance::default()).unwrap();
    let fetched = Catalog::fetched(
        br#"{"sample_spec":{},"fallback_generalizations":{"rules":[]},"a":{"litellm_provider":"t","aliases":["b","c"]}}"#,
        Provenance::default(),
        &current,
        limits,
    );
    match expected {
        FetchOutcome::Ok => assert!(fetched.is_ok()),
        FetchOutcome::Shrunk => {
            assert!(matches!(fetched, Err(Error::Shrunk { actual: 1, .. })))
        }
        FetchOutcome::BelowMinimum => {
            assert!(matches!(
                fetched,
                Err(Error::BelowMinimum { actual: 1, .. })
            ))
        }
        FetchOutcome::InvalidRatio => assert!(matches!(fetched, Err(Error::InvalidRatio))),
    }
}

#[derive(Debug)]
enum MalformedOutcome {
    Empty,
    Json,
}

#[rstest]
#[case::empty(b"{}", MalformedOutcome::Empty)]
#[case::invalid_json(b"{", MalformedOutcome::Json)]
#[case::fallback_rules_missing(
    br#"{"fallback_generalizations":{},"a":{"litellm_provider":"t"}}"#,
    MalformedOutcome::Json
)]
fn malformed_documents_have_typed_outcomes(
    #[case] body: &[u8],
    #[case] expected: MalformedOutcome,
) {
    let actual = Catalog::parse(body, Provenance::default());
    match expected {
        MalformedOutcome::Empty => assert!(matches!(actual, Err(Error::Empty))),
        MalformedOutcome::Json => assert!(matches!(actual, Err(Error::Json(_)))),
    }
}

#[rstest]
#[ignore]
fn parses_current_catalog(current_catalog: Catalog) {
    assert!(current_catalog.sample_spec().is_some());
    // Snapshot from 2026-09-23, an independent baseline that catches catalog shrinkage.
    const REFERENCE_MODEL_COUNT: usize = 4303;
    assert!(current_catalog.model_count() as f64 >= REFERENCE_MODEL_COUNT as f64 * 0.9);
}
