use litellm_model_catalog::{AliasIssue, Catalog, CatalogError, IntegrityLimits, Provenance};
use serde_json::json;

fn parse(body: &str) -> Catalog {
    Catalog::parse(body.as_bytes(), Provenance::default()).unwrap()
}

#[test]
fn preserves_fields_metadata_and_snapshot_isolation() {
    let mut source = br#"{
        "sample_spec":{"explanation":"example"},
        "fallback_generalizations":{"rules":[{"name":"family","pattern":"^new-","model_info":{"mode":"chat"}}]},
        "Alpha":{"litellm_provider":"test","aliases":["short"],"price":0,"enabled":false,
                 "optional":null,"unknown":{"nested":[1,{"x":true}]}}
    }"#.to_vec();
    let catalog = Catalog::parse(
        &source,
        Provenance {
            source: Some("fixture".into()),
            revision: Some("rev".into()),
            etag: None,
        },
    )
    .unwrap();
    source.fill(b' ');

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
    assert_eq!(
        catalog.sample_spec(),
        Some(&json!({"explanation":"example"}))
    );
    assert_eq!(catalog.fallback_rules().unwrap().len(), 1);
    assert_eq!(catalog.provenance().revision.as_deref(), Some("rev"));
    assert_eq!(catalog.model_count(), 1);
}

#[test]
fn alias_collisions_and_case_fallback_follow_python_order() {
    let catalog = parse(
        r#"{
        "First":{"aliases":["Shared","Second","first"],"value":1},
        "Second":{"aliases":["Shared","sHaReD"],"value":2},
        "SHARED":{"value":3}
    }"#,
    );
    assert_eq!(catalog.lookup("Shared").unwrap().canonical_key, "First");
    assert_eq!(catalog.lookup("Second").unwrap().canonical_key, "Second");
    assert_eq!(catalog.lookup("shared").unwrap().canonical_key, "Second");
    assert_eq!(catalog.lookup("FIRST").unwrap().canonical_key, "First");
    assert_eq!(catalog.lookup("sHaReD").unwrap().canonical_key, "Second");
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
fn integrity_uses_canonical_count_and_strict_shrink_boundary() {
    let catalog = parse(
        r#"{"sample_spec":{},"fallback_generalizations":{},
        "a":{"aliases":["b","c"]}}"#,
    );
    assert!(
        catalog
            .validate(IntegrityLimits {
                backup_model_count: 2,
                min_model_count: 1,
                min_backup_ratio: 0.5,
            })
            .is_ok()
    );
    assert!(matches!(
        catalog.validate(IntegrityLimits {
            backup_model_count: 3,
            min_model_count: 1,
            min_backup_ratio: 0.5,
        }),
        Err(CatalogError::Shrunk { actual: 1, .. })
    ));
    assert!(matches!(
        catalog.validate(IntegrityLimits {
            backup_model_count: 0,
            min_model_count: 2,
            min_backup_ratio: 0.5,
        }),
        Err(CatalogError::BelowMinimum { actual: 1, .. })
    ));
    assert!(matches!(
        catalog.validate(IntegrityLimits {
            backup_model_count: 0,
            min_model_count: 0,
            min_backup_ratio: f64::NAN,
        }),
        Err(CatalogError::InvalidRatio)
    ));
}

#[test]
fn malformed_input_and_aliases_have_typed_outcomes() {
    assert!(matches!(
        Catalog::parse(b"{}", Provenance::default()),
        Err(CatalogError::Empty)
    ));
    assert!(matches!(
        Catalog::parse(b"{", Provenance::default()),
        Err(CatalogError::Json(_))
    ));
    assert!(matches!(
        Catalog::parse(b"{\"a\":1}", Provenance::default()),
        Err(CatalogError::EntryNotObject { .. })
    ));
    let catalog = parse(r#"{"a":{"aliases":"bad"},"b":{"aliases":[9,"ok"]}}"#);
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

#[test]
fn parses_current_and_packaged_catalogs_without_pinning_counts() {
    let current = Catalog::parse(
        include_bytes!("../../../../model_prices_and_context_window.json"),
        Provenance::default(),
    )
    .unwrap();
    let backup = Catalog::parse(
        include_bytes!("../../../../litellm/model_prices_and_context_window_backup.json"),
        Provenance::default(),
    )
    .unwrap();
    assert!(current.model_count() > 0);
    assert!(backup.model_count() > 0);
    assert!(current.sample_spec().is_some());
    assert!(backup.sample_spec().is_some());
    assert!(
        current
            .validate(IntegrityLimits::python_defaults(backup.model_count()))
            .is_ok()
    );
}
