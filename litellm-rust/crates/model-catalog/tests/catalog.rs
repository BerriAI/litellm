use std::path::{Path, PathBuf};

use litellm_model_catalog::{AliasIssue, Catalog, Error, IntegrityLimits, Provenance};
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
fn backup_catalog(repo_root: PathBuf) -> Catalog {
    let body = std::fs::read(repo_root.join("litellm/model_prices_and_context_window_backup.json"))
        .unwrap();
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

#[rstest]
#[ignore]
fn preserves_fields_and_metadata(fixture_catalog: Catalog) {
    let catalog = fixture_catalog;
    let entry = catalog.lookup("SHORT").unwrap();
    assert_eq!(entry.canonical_key, "Alpha");
    assert_eq!(entry.matched_key, "short");
    assert_eq!(entry.entry.field("price"), Some(&json!(0)));
    assert_eq!(entry.entry.field("enabled"), Some(&json!(false)));
    assert_eq!(entry.entry.field("optional"), Some(&json!(null)));
    assert_eq!(entry.entry.field("missing"), None);
    assert_eq!(
        entry.entry.field("unknown"),
        Some(&json!({"nested":[1,{"x":true}]}))
    );
    assert_eq!(entry.entry.field("aliases"), None);
    assert_eq!(entry.entry.info().litellm_provider.as_deref(), Some("test"));
    assert_eq!(
        catalog.sample_spec(),
        Some(&json!({"explanation":"example"}))
    );
    assert_eq!(catalog.fallback_rules().unwrap().len(), 1);
    assert_eq!(catalog.provenance().revision.as_deref(), Some("rev"));
    assert_eq!(catalog.model_count(), 1);
}

#[rstest]
#[ignore]
fn snapshot_does_not_borrow_source() {
    let mut source = ALPHA_FIXTURE.to_vec();
    let catalog = Catalog::parse(&source, Provenance::default()).unwrap();
    source.fill(b' ');

    let entry = catalog.lookup("short").unwrap();
    assert_eq!(entry.canonical_key, "Alpha");
    assert_eq!(entry.entry.field("price"), Some(&json!(0)));
}

#[rstest]
#[case("Shared", "First")]
#[case("Second", "Second")]
#[case("shared", "Second")]
#[case("FIRST", "First")]
#[case("sHaReD", "Second")]
#[ignore]
fn alias_collisions_and_case_fallback_follow_entry_order(
    #[case] lookup: &str,
    #[case] expected: &str,
) {
    let catalog = Catalog::parse(
        br#"{
            "First":{"aliases":["Shared","Second","first"],"value":1},
            "Second":{"aliases":["Shared","sHaReD"],"value":2},
            "SHARED":{"value":3}
        }"#,
        Provenance::default(),
    )
    .unwrap();
    assert_eq!(catalog.lookup(lookup).unwrap().canonical_key, expected);
    assert_eq!(catalog.alias_count(), 3);
    assert!(
        catalog
            .alias_issues()
            .contains(&AliasIssue::CanonicalCollision {
                model: "First".into(),
                alias: "Second".into(),
            })
    );
    assert!(
        catalog
            .alias_issues()
            .contains(&AliasIssue::AliasCollision {
                model: "Second".into(),
                alias: "Shared".into(),
            })
    );
}

#[test]
#[ignore]
fn json_entry_order_controls_alias_ownership_and_case_fallback() {
    let forward = Catalog::parse(
        br#"{
            "Alpha":{"aliases":["shared"]},
            "Beta":{"aliases":["shared"]},
            "Foo":{},
            "fOO":{}
        }"#,
        Provenance::default(),
    )
    .unwrap();
    let reversed = Catalog::parse(
        br#"{
            "fOO":{},
            "Foo":{},
            "Beta":{"aliases":["shared"]},
            "Alpha":{"aliases":["shared"]}
        }"#,
        Provenance::default(),
    )
    .unwrap();

    assert_eq!(forward.lookup("shared").unwrap().canonical_key, "Alpha");
    assert_eq!(reversed.lookup("shared").unwrap().canonical_key, "Beta");
    assert_eq!(forward.lookup("foo").unwrap().canonical_key, "fOO");
    assert_eq!(reversed.lookup("foo").unwrap().canonical_key, "Foo");
}

#[derive(Debug)]
enum ValidationOutcome {
    Ok,
    Shrunk,
    BelowMinimum,
    InvalidRatio,
}

#[rstest]
#[case(
    IntegrityLimits {
        reference_model_count: 2,
        min_model_count: 1,
        min_reference_ratio: 0.5,
    },
    ValidationOutcome::Ok
)]
#[case(
    IntegrityLimits {
        reference_model_count: 3,
        min_model_count: 1,
        min_reference_ratio: 0.5,
    },
    ValidationOutcome::Shrunk
)]
#[case(
    IntegrityLimits {
        reference_model_count: 0,
        min_model_count: 2,
        min_reference_ratio: 0.5,
    },
    ValidationOutcome::BelowMinimum
)]
#[case(
    IntegrityLimits {
        reference_model_count: 0,
        min_model_count: 0,
        min_reference_ratio: f64::NAN,
    },
    ValidationOutcome::InvalidRatio
)]
#[ignore]
fn integrity_uses_canonical_count_and_strict_shrink_boundary(
    #[case] limits: IntegrityLimits,
    #[case] expected: ValidationOutcome,
) {
    let catalog = Catalog::parse(
        br#"{"sample_spec":{},"fallback_generalizations":{"rules":[]},"a":{"aliases":["b","c"]}}"#,
        Provenance::default(),
    )
    .unwrap();
    let actual = catalog.validate(limits);
    match expected {
        ValidationOutcome::Ok => assert!(actual.is_ok()),
        ValidationOutcome::Shrunk => {
            assert!(matches!(actual, Err(Error::Shrunk { actual: 1, .. })))
        }
        ValidationOutcome::BelowMinimum => {
            assert!(matches!(actual, Err(Error::BelowMinimum { actual: 1, .. })))
        }
        ValidationOutcome::InvalidRatio => assert!(matches!(actual, Err(Error::InvalidRatio))),
    }
}

#[derive(Debug)]
enum MalformedOutcome {
    Empty,
    Json,
    EntryNotObject,
}

#[rstest]
#[case::empty(b"{}", MalformedOutcome::Empty)]
#[case::invalid_json(b"{", MalformedOutcome::Json)]
#[case::entry_not_object(br#"{"a":1}"#, MalformedOutcome::EntryNotObject)]
#[case::fallback_rules_missing(
    br#"{"fallback_generalizations":{},"a":{}}"#,
    MalformedOutcome::Json
)]
#[ignore]
fn malformed_input_and_aliases_have_typed_outcomes(
    #[case] body: &[u8],
    #[case] expected: MalformedOutcome,
) {
    let actual = Catalog::parse(body, Provenance::default());
    match expected {
        MalformedOutcome::Empty => assert!(matches!(actual, Err(Error::Empty))),
        MalformedOutcome::Json => assert!(matches!(actual, Err(Error::Json(_)))),
        MalformedOutcome::EntryNotObject => {
            assert!(matches!(actual, Err(Error::EntryNotObject { .. })))
        }
    }
}

#[rstest]
#[ignore]
fn invalid_aliases_are_reported_not_fatal() {
    let catalog = Catalog::parse(
        br#"{"a":{"aliases":"bad"},"b":{"aliases":[9,"ok"]}}"#,
        Provenance::default(),
    )
    .unwrap();
    assert_eq!(
        catalog.alias_issues(),
        &[
            AliasIssue::InvalidList { model: "a".into() },
            AliasIssue::InvalidName { model: "b".into() },
        ]
    );
    assert_eq!(catalog.lookup("ok").unwrap().canonical_key, "b");
    assert!(catalog.lookup("missing").is_none());
}

#[rstest]
#[ignore]
fn parses_current_and_packaged_catalogs_against_independent_baseline(
    current_catalog: Catalog,
    backup_catalog: Catalog,
) {
    assert!(current_catalog.model_count() > 0);
    assert!(backup_catalog.model_count() > 0);
    assert!(current_catalog.sample_spec().is_some());
    assert!(backup_catalog.sample_spec().is_some());
    // Snapshot from 2026-09-23; the backup file mirrors the current file and cannot detect shrinkage.
    const REFERENCE_MODEL_COUNT: usize = 4303;
    current_catalog
        .validate(IntegrityLimits {
            reference_model_count: REFERENCE_MODEL_COUNT,
            min_model_count: 50,
            min_reference_ratio: 0.9,
        })
        .unwrap();
    assert!(current_catalog.model_names().all(|name| {
        let entry = current_catalog.lookup(name).unwrap().entry;
        entry.info().litellm_provider.is_some() == entry.field("litellm_provider").is_some()
    }));
}
