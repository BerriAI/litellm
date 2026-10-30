use std::collections::BTreeSet;

use litellm_model_catalog::{
    BUNDLED_REGISTRY, Catalog, Error, ErrorKind, IntegrityLimits, Mode, Rejects, Written,
};
use rstest::{fixture, rstest};
use serde_json::{Map, Value, json};

const ALPHA_FIXTURE: &[u8] = br#"{
    "sample_spec":{"explanation":"example","input_cost_per_token":"not checked"},
    "fallback_generalizations":{"rules":[{"name":"family","pattern":"^new-","model_info":{"mode":"chat"}}]},
    "Alpha":{"litellm_provider":"test","aliases":["short"],"price":0,"enabled":false,
             "optional":null,"unknown":{"nested":[1,{"x":true}]}}
}"#;

#[fixture]
fn fixture_catalog() -> Catalog {
    Catalog::parse(ALPHA_FIXTURE).unwrap().value
}

fn loaded(document: Value) -> litellm_model_catalog::Loaded<Catalog> {
    Catalog::parse(document.to_string().as_bytes()).unwrap()
}

fn catalog(document: Value) -> Catalog {
    loaded(document).value
}

fn written(catalog: &Catalog) -> String {
    let document: Map<String, Value> = catalog
        .written()
        .map(|(key, value)| {
            let value = match value {
                Written::Entry(fields) => serde_json::to_value(fields).unwrap(),
                Written::Section(value) => value.clone(),
            };
            (key.to_owned(), value)
        })
        .collect();
    serde_json::to_string(&document).unwrap()
}

fn report(error: &Error) -> String {
    litellm_error::Report::new(error).to_string()
}

fn error_text(document: Value) -> Vec<String> {
    loaded(document).errors.iter().map(report).collect()
}

#[rstest]
#[case::decisions("decisions", Mode::Decisions)]
#[case::legacy_evaluation("evaluation", Mode::Evaluation)]
fn parses_decision_modes(#[case] mode: &str, #[case] expected: Mode) {
    let catalog = catalog(json!({"Test": {"litellm_provider": "test", "mode": mode}}));
    assert_eq!(
        catalog
            .lookup("Test")
            .unwrap()
            .entry
            .model()
            .unwrap()
            .attributes
            .mode,
        Some(expected)
    );
}

#[rstest]
fn unknown_keys_are_kept_raw_and_listed_as_unrecognized(fixture_catalog: Catalog) {
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
fn sections_aliases_and_raw_fields_are_read_in_their_document_views(fixture_catalog: Catalog) {
    let found = fixture_catalog.lookup("SHORT").unwrap();
    assert_eq!(found.canonical_key, "Alpha");
    assert_eq!(found.matched_key, "short");
    assert_eq!(found.entry.field("optional"), Some(&json!(null)));
    assert_eq!(found.entry.field("aliases"), Some(&json!(["short"])));
    assert_eq!(
        fixture_catalog.sample_spec(),
        Some(&json!({"explanation":"example","input_cost_per_token":"not checked"}))
    );
    assert_eq!(fixture_catalog.fallback_rules().unwrap()[0].name, "family");
    assert_eq!(fixture_catalog.model_count(), 1);
    assert_eq!(
        fixture_catalog
            .served()
            .map(|(key, _)| key)
            .collect::<Vec<_>>(),
        ["Alpha", "short"]
    );
}

#[rstest]
fn a_bad_entry_is_reported_by_model_and_key_and_kept_raw() {
    let parsed = loaded(json!({
        "Good": {"litellm_provider": "test"},
        "Bad": {"litellm_provider": "test", "mode": "chatt", "input_cost_per_token": "free"},
        "Worse": {"mode": "chat"}
    }));
    let catalog = parsed.value;
    let problems: Vec<String> = parsed.errors.iter().map(report).collect();
    assert_eq!(problems.len(), 3, "{problems:?}");
    assert!(problems[0].starts_with("Bad.mode: invalid value: unknown variant `chatt`"));
    assert!(problems[1].starts_with("Bad.input_cost_per_token: invalid value"));
    assert_eq!(problems[2], "Worse.litellm_provider: required");
    assert!(catalog.lookup("Good").unwrap().entry.model().is_some());
    assert!(catalog.lookup("Bad").unwrap().entry.model().is_none());
    assert_eq!(catalog.model_count(), 3);
    assert!(
        parsed
            .errors
            .iter()
            .find(|error| error.rejects() == Rejects::Document)
            .is_none()
    );
}

#[rstest]
#[case::alias_names_a_model(
    json!({"First": {"litellm_provider": "t", "aliases": ["Second"]}, "Second": {"litellm_provider": "t"}}),
    r#"First: alias "Second" is already a model name"#
)]
#[case::alias_claimed_twice(
    json!({"First": {"litellm_provider": "t", "aliases": ["shared"]}, "Second": {"litellm_provider": "t", "aliases": ["shared"]}}),
    r#"Second: alias "shared" already belongs to First"#
)]
#[case::aliases_not_a_list(
    json!({"a": {"litellm_provider": "t", "aliases": "bad"}}),
    "a.aliases: invalid value: invalid type: string \"bad\", expected a sequence"
)]
#[case::entry_not_object(
    json!({"a": 1, "valid": {"litellm_provider": "test"}}),
    "a: must be a JSON object"
)]
#[case::fallback_rules_missing(
    json!({"fallback_generalizations": {}, "a": {"litellm_provider": "t"}}),
    "fallback_generalizations: invalid value: missing field `rules`"
)]
#[case::unknown_timezone(
    json!({"a": {"litellm_provider": "t", "off_peak_pricing": {
        "windows": [{"hours_utc": "00:00-01:00"}],
        "weekday_timezone": "Not/AZone",
        "input_cost_per_token": 1e-6
    }}}),
    r#"a.off_peak_pricing.weekday_timezone: "Not/AZone" is not an IANA time zone"#
)]
fn document_level_problems_are_reported(#[case] document: Value, #[case] expected: &str) {
    assert_eq!(error_text(document), [expected]);
}

#[rstest]
fn escaped_row_paths_and_row_validation_share_the_parse_rules() {
    let document = json!({
        "open/router~v1": {"litellm_provider": "test", "input_cost_per_token": -1}
    });
    let parsed = loaded(document);
    let fields = parsed
        .value
        .lookup("open/router~v1")
        .unwrap()
        .entry
        .fields();
    let row_errors = Catalog::row_errors("open/router~v1", fields);
    assert_eq!(
        row_errors
            .iter()
            .map(|error| (error.path(), report(error)))
            .collect::<Vec<_>>(),
        parsed
            .errors
            .iter()
            .map(|error| (error.path(), report(error)))
            .collect::<Vec<_>>()
    );
    assert_eq!(
        row_errors[0].path(),
        "/open~1router~0v1/input_cost_per_token"
    );
    assert!(row_errors[0].kind() == ErrorKind::Invalid);
}

#[rstest]
fn malformed_fallback_is_collected_while_models_are_read() {
    let parsed = loaded(json!({
        "fallback_generalizations": {"rules": "bad"},
        "a": {"litellm_provider": "test"}
    }));
    assert_eq!(parsed.value.model_count(), 1);
    assert_eq!(parsed.errors.len(), 1);
    assert_eq!(parsed.errors[0].path(), "/fallback_generalizations");
}

#[rstest]
fn rejected_rows_and_their_aliases_are_removed_without_disturbing_written_order() {
    let body = r#"{"before":{"litellm_provider":"test","aliases":["before-alias"]},"bad":{"litellm_provider":"test","aliases":["bad-alias"],"mode":"chatt"},"not-object":4,"sample_spec":{"keep":true},"after":{"litellm_provider":"test"}}"#;
    let parsed = Catalog::parse(body.as_bytes()).unwrap();
    let (catalog, removed) = parsed.value.without_rejected(&parsed.errors);

    assert_eq!(removed.len(), 2);
    assert_eq!(
        catalog.lookup("before-alias").unwrap().canonical_key,
        "before"
    );
    assert_eq!(
        catalog.lookup("BEFORE-ALIAS").unwrap().matched_key,
        "before-alias"
    );
    assert!(catalog.lookup("bad").is_none());
    assert!(catalog.lookup("BAD").is_none());
    assert!(catalog.lookup("Bad-Alias").is_none());
    assert_eq!(catalog.model_count(), 2);
    assert_eq!(
        written(&catalog),
        r#"{"before":{"litellm_provider":"test","aliases":["before-alias"]},"sample_spec":{"keep":true},"after":{"litellm_provider":"test"}}"#
    );
}

#[rstest]
fn invalid_empty_key_rows_are_removed() {
    let parsed = loaded(json!({
        "": {"litellm_provider": "test", "mode": "chatt"},
        "good": {"litellm_provider": "test"}
    }));
    assert_eq!(parsed.errors[0].path(), "//mode");

    let (catalog, removed) = parsed.value.without_rejected(&parsed.errors);
    assert_eq!(catalog.model_count(), 1);
    assert!(catalog.lookup("").is_none());
    assert!(catalog.lookup("good").is_some());
    assert_eq!(removed.len(), 1);
}

#[rstest]
fn timezone_errors_reject_providerless_rows_without_rejecting_other_providerless_rows() {
    let parsed = loaded(json!({
        "good": {"mode": "chat"},
        "bad": {"off_peak_pricing": {
            "windows": [{"hours_utc": "00:00-01:00"}],
            "weekday_timezone": "Not/AZone",
            "input_cost_per_token": 1e-6
        }}
    }));
    assert_eq!(parsed.errors.len(), 3);
    assert!(
        parsed
            .errors
            .iter()
            .any(|error| error.kind() == ErrorKind::UnknownTimezone)
    );

    let (catalog, removed) = parsed.value.without_rejected(&parsed.errors);
    assert_eq!(catalog.model_count(), 1);
    assert!(catalog.lookup("good").is_some());
    assert!(catalog.lookup("bad").is_none());
    assert_eq!(removed.len(), 2);
}

#[rstest]
fn bundled_per_row_errors_match_document_errors() {
    let parsed = Catalog::parse(BUNDLED_REGISTRY).unwrap();
    let row_errors: Vec<_> = parsed
        .value
        .written()
        .flat_map(|(key, written)| match written {
            Written::Entry(fields) => Catalog::row_errors(key, fields),
            Written::Section(_) => Vec::new(),
        })
        .map(|error| (error.path().to_owned(), report(&error)))
        .collect();
    let document_errors: Vec<_> = parsed
        .errors
        .iter()
        .map(|error| (error.path().to_owned(), report(error)))
        .collect();

    assert_eq!(row_errors, document_errors);
}

#[rstest]
fn three_hundred_rows_filter_one_row_with_two_invalid_prices() {
    let valid_rows: Map<String, Value> = (0..299)
        .map(|index| {
            (
                format!("valid-{index:03}"),
                json!({"litellm_provider": "test"}),
            )
        })
        .collect();
    let mut document = valid_rows;
    document.insert(
        "bad".to_owned(),
        json!({
            "litellm_provider": "test",
            "input_cost_per_token": -1,
            "output_cost_per_token": -1
        }),
    );
    let parsed = Catalog::parse(serde_json::to_vec(&document).unwrap().as_slice()).unwrap();
    assert_eq!(parsed.value.model_count(), 300);
    assert_eq!(parsed.errors.len(), 2);

    let (catalog, removed) = parsed.value.without_rejected(&parsed.errors);
    assert_eq!(catalog.model_count(), 299);
    assert_eq!(removed.len(), 2);
}

#[rstest]
fn integrity_limits_count_canonical_models_not_aliases() {
    let fetched = catalog(json!({"a": {"litellm_provider": "t", "aliases": ["b", "c"]}}));
    assert!(
        fetched
            .check_size(2, IntegrityLimits::new(1, 0.5).unwrap())
            .is_ok()
    );
    assert_eq!(
        fetched
            .check_size(3, IntegrityLimits::new(1, 0.5).unwrap())
            .unwrap_err()
            .kind(),
        ErrorKind::Shrunk
    );
    assert_eq!(
        fetched
            .check_size(2, IntegrityLimits::new(2, 0.5).unwrap())
            .unwrap_err()
            .kind(),
        ErrorKind::BelowMinimum
    );
}

#[rstest]
#[case::empty(b"{}", "empty")]
#[case::only_sections(br#"{"sample_spec":{}}"#, "empty")]
#[case::invalid_json(b"{", "json")]
#[case::array_root_is_json_error(b"[]", "json")]
fn malformed_documents_are_rejected_whole(#[case] body: &[u8], #[case] kind: &str) {
    let error = Catalog::parse(body).unwrap_err();
    assert_eq!(error.rejects(), Rejects::Document);
    assert_eq!(<&str>::from(error.kind()), kind);
}

#[rstest]
fn written_is_the_document_in_order_including_sections_and_bad_entries() {
    let body = concat!(
        r#"{"b":{"litellm_provider":"t","z":1,"a":null},"sample_spec":{"x":"y"},"n":4,"#,
        r#""fallback_generalizations":{"rules":"bad"},"a":{"mode":"chatt","price":1.0}}"#
    );
    let catalog = Catalog::parse(body.as_bytes()).unwrap().value;
    assert_eq!(written(&catalog), body);
    assert_eq!(catalog.model_count(), 2);
}

#[rstest]
fn the_bundled_registry_has_no_problems() {
    let problems: Vec<_> = Catalog::problems(BUNDLED_REGISTRY)
        .iter()
        .map(report)
        .collect();
    assert!(problems.is_empty(), "{problems:#?}");
    const REFERENCE_MODEL_COUNT: usize = 4303;
    let catalog = Catalog::parse(BUNDLED_REGISTRY).unwrap().value;
    assert!(catalog.model_count() as f64 >= REFERENCE_MODEL_COUNT as f64 * 0.9);
}

#[rstest]
fn the_bundled_registry_is_written_back_as_it_was_read() {
    let document: Value = serde_json::from_slice(BUNDLED_REGISTRY).unwrap();
    assert_eq!(
        written(&Catalog::parse(BUNDLED_REGISTRY).unwrap().value),
        serde_json::to_string(&document).unwrap()
    );
}

#[rstest]
fn a_bad_value_keeps_its_json_error_as_the_source_and_names_its_key_and_field() {
    let parsed =
        loaded(json!({"open/router": {"litellm_provider": "t", "input_cost_per_token": "free"}}));
    let error = &parsed.errors[0];
    let cause = litellm_error::find::<serde_json::Error>(error)
        .expect("the JSON error is in the chain")
        .to_string();
    assert_eq!(report(error).matches(&cause).count(), 1);
    assert!(!error.to_string().contains(&cause));
    assert_eq!(error.key().as_deref(), Some("open/router"));
    assert_eq!(error.field().as_deref(), Some("input_cost_per_token"));
}
