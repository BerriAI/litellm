use std::collections::BTreeSet;

use litellm_model_catalog::{
    BUNDLED_REGISTRY, Catalog, IntegrityLimits, IntegrityViolation, InvalidDocument, Mode, Problem,
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
    Catalog::parse(ALPHA_FIXTURE).unwrap()
}

fn catalog(document: Value) -> Catalog {
    Catalog::parse(document.to_string().as_bytes()).unwrap()
}

fn written(catalog: &Catalog) -> String {
    let document: Map<String, Value> = catalog
        .written()
        .map(|(key, value)| (key.to_owned(), serde_json::to_value(value).unwrap()))
        .collect();
    serde_json::to_string(&document).unwrap()
}

fn problems(catalog: &Catalog) -> Vec<String> {
    catalog.problems().iter().map(ToString::to_string).collect()
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
fn fields_sample_spec_and_fallback_rules_pass_through_as_written(fixture_catalog: Catalog) {
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
    assert!(fixture_catalog.problems().is_empty());
}

#[rstest]
fn a_bad_entry_is_reported_by_model_and_key_and_kept_raw() {
    let catalog = catalog(json!({
        "Good": {"litellm_provider": "test"},
        "Bad": {"litellm_provider": "test", "mode": "chatt", "input_cost_per_token": "free"},
        "Worse": {"mode": "chat"}
    }));
    let problems = problems(&catalog);

    assert_eq!(problems.len(), 3, "{problems:?}");
    assert!(problems[0].starts_with("Bad.mode: unknown variant `chatt`"));
    assert!(problems[1].starts_with("Bad.input_cost_per_token: invalid value"));
    assert_eq!(problems[2], "Worse.litellm_provider is required");
    assert!(catalog.lookup("Good").unwrap().entry.model().is_some());
    let bad = catalog.lookup("Bad").unwrap().entry;
    assert!(bad.model().is_none());
    assert_eq!(bad.field("mode"), Some(&json!("chatt")));
    assert_eq!(catalog.model_count(), 3);
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
    "a.aliases: invalid type: string \"bad\", expected a sequence"
)]
#[case::entry_not_object(json!({"a": 1}), "a must be a JSON object")]
#[case::fallback_rules_missing(
    json!({"fallback_generalizations": {}, "a": {"litellm_provider": "t"}}),
    "fallback_generalizations: missing field `rules`"
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
    assert_eq!(problems(&catalog(document)), [expected]);
}

#[rstest]
fn the_first_owner_keeps_a_contested_alias() {
    let catalog = catalog(json!({
        "First": {"litellm_provider": "t", "aliases": ["shared"]},
        "Second": {"litellm_provider": "t", "aliases": ["shared"]}
    }));
    assert_eq!(catalog.lookup("shared").unwrap().canonical_key, "First");
}

#[rstest]
fn a_known_timezone_is_not_a_problem_and_keeps_the_model() {
    let catalog = catalog(json!({"a": {"litellm_provider": "t", "off_peak_pricing": {
        "windows": [{"hours_utc": "00:00-01:00", "weekdays": [6, 7]}],
        "weekday_timezone": " Asia/Shanghai ",
        "input_cost_per_token": 1e-6
    }}}));
    assert!(catalog.problems().is_empty());
    assert!(catalog.lookup("a").unwrap().entry.model().is_some());
}

#[rstest]
#[case("foo", "fOO")]
#[case("FOO", "fOO")]
#[case("Foo", "Foo")]
fn case_insensitive_lookup_prefers_the_later_entry(#[case] lookup: &str, #[case] expected: &str) {
    let catalog =
        catalog(json!({"Foo": {"litellm_provider": "t"}, "fOO": {"litellm_provider": "t"}}));
    assert_eq!(catalog.lookup(lookup).unwrap().canonical_key, expected);
}

#[rstest]
#[case::half_of_two(2, 1, 0.5, Ok(()))]
#[case::under_half_of_three(
    3,
    1,
    0.5,
    Err(IntegrityViolation::Shrunk { actual: 1, reference: 3, ratio: 0.5 })
)]
#[case::below_minimum(
    1,
    2,
    0.5,
    Err(IntegrityViolation::BelowMinimum { actual: 1, minimum: 2 })
)]
fn integrity_limits_count_canonical_models_not_aliases(
    #[case] reference: usize,
    #[case] min_model_count: usize,
    #[case] min_reference_ratio: f64,
    #[case] expected: Result<(), IntegrityViolation>,
) {
    let fetched = catalog(json!({"a": {"litellm_provider": "t", "aliases": ["b", "c"]}}));
    let limits = IntegrityLimits::new(min_model_count, min_reference_ratio).unwrap();

    assert_eq!(fetched.check_size(reference, limits), expected);
}

#[rstest]
#[case::nan(f64::NAN)]
#[case::above_one(1.5)]
#[case::negative(-0.1)]
fn a_reference_ratio_outside_zero_to_one_is_not_a_limit(#[case] ratio: f64) {
    assert_eq!(IntegrityLimits::new(0, ratio), None);
}

#[derive(Debug)]
enum Malformed {
    Empty,
    Json,
}

#[rstest]
#[case::empty(b"{}", Malformed::Empty)]
#[case::only_sections(br#"{"sample_spec":{}}"#, Malformed::Empty)]
#[case::invalid_json(b"{", Malformed::Json)]
#[case::not_an_object(b"[]", Malformed::Json)]
fn malformed_documents_are_rejected_whole(#[case] body: &[u8], #[case] expected: Malformed) {
    let actual = Catalog::parse(body);
    match expected {
        Malformed::Empty => assert!(matches!(actual, Err(InvalidDocument::Empty))),
        Malformed::Json => assert!(matches!(actual, Err(InvalidDocument::Json(_)))),
    }
}

#[rstest]
fn written_is_the_document_in_order_including_sections_and_bad_entries() {
    let body = concat!(
        r#"{"b":{"litellm_provider":"t","z":1,"a":null},"sample_spec":{"x":"y"},"n":4,"#,
        r#""fallback_generalizations":{"rules":"bad"},"a":{"mode":"chatt","price":1.0}}"#
    );
    let catalog = Catalog::parse(body.as_bytes()).unwrap();

    assert_eq!(written(&catalog), body);
    assert_eq!(catalog.model_count(), 2);
}

#[rstest]
fn the_bundled_registry_parses_with_no_problems_and_every_key_understood() {
    let catalog = Catalog::parse(BUNDLED_REGISTRY).unwrap();
    let problems: Vec<&Problem> = catalog.problems().iter().collect();
    let unrecognized: Vec<_> = catalog.unrecognized().collect();

    assert!(problems.is_empty(), "{problems:#?}");
    assert!(
        unrecognized.is_empty(),
        "keys no type reads: {unrecognized:#?}"
    );
    assert!(
        catalog
            .fallback_rules()
            .is_some_and(|rules| !rules.is_empty())
    );
    // Snapshot from 2026-09-23, an independent baseline that catches catalog shrinkage.
    const REFERENCE_MODEL_COUNT: usize = 4303;
    assert!(catalog.model_count() as f64 >= REFERENCE_MODEL_COUNT as f64 * 0.9);
}

#[rstest]
fn the_bundled_registry_is_written_back_as_it_was_read() {
    let document: Value = serde_json::from_slice(BUNDLED_REGISTRY).unwrap();

    assert_eq!(
        written(&Catalog::parse(BUNDLED_REGISTRY).unwrap()),
        serde_json::to_string(&document).unwrap()
    );
}
