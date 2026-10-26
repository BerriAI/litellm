use std::collections::BTreeSet;
use std::mem::{Discriminant, discriminant};

use litellm_model_catalog::{
    BUNDLED_REGISTRY, Catalog, Error, IntegrityLimits, Located, Mode, Written, fatal, rejects_row,
};
use proptest::prelude::*;
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
    let invalid = discriminant(&Error::Invalid(json_error()));
    assert_eq!(
        located(&parsed.errors),
        [
            ("/Bad/mode", invalid),
            ("/Bad/input_cost_per_token", invalid),
            ("/Worse", discriminant(&Error::MissingProvider)),
        ]
    );
    assert!(catalog.lookup("Good").unwrap().entry.model().is_some());
    assert!(catalog.lookup("Bad").unwrap().entry.model().is_none());
    assert_eq!(catalog.model_count(), 3);
    assert!(
        parsed
            .errors
            .iter()
            .find(|error| fatal(&error.error))
            .is_none()
    );
}

fn json_error() -> serde_json::Error {
    serde_json::from_str::<()>("x").unwrap_err()
}

fn located(errors: &[Located]) -> Vec<(&str, Discriminant<Error>)> {
    errors
        .iter()
        .map(|error| (error.path.as_str(), discriminant(&error.error)))
        .collect()
}

#[rstest]
#[case::alias_names_a_model(
    json!({"First": {"litellm_provider": "t", "aliases": ["Second"]}, "Second": {"litellm_provider": "t"}}),
    "/First", Error::AliasShadowsModel { alias: "Second".into() }
)]
#[case::alias_claimed_twice(
    json!({"First": {"litellm_provider": "t", "aliases": ["shared"]}, "Second": {"litellm_provider": "t", "aliases": ["shared"]}}),
    "/Second", Error::AliasTaken { alias: "shared".into(), taken_by: "First".into() }
)]
#[case::aliases_not_a_list(json!({"a": {"litellm_provider": "t", "aliases": "bad"}}), "/a/aliases", Error::Invalid(json_error()))]
#[case::entry_not_object(json!({"a": 1, "valid": {"litellm_provider": "test"}}), "/a", Error::NotAnObject)]
#[case::fallback_rules_missing(
    json!({"fallback_generalizations": {}, "a": {"litellm_provider": "t"}}),
    "/fallback_generalizations", Error::Invalid(json_error())
)]
#[case::unknown_timezone(
    json!({"a": {"litellm_provider": "t", "off_peak_pricing": {
        "windows": [{"hours_utc": "00:00-01:00"}],
        "weekday_timezone": "Not/AZone",
        "input_cost_per_token": 1e-6
    }}}),
    "/a/off_peak_pricing/weekday_timezone", Error::UnknownTimezone { zone: "Not/AZone".into() }
)]
#[case::missing_provider(json!({"openrouter/deepseek/deepseek-chat-v3.1": {"input_cost_per_token": 2e-7}}), "/openrouter~1deepseek~1deepseek-chat-v3.1", Error::MissingProvider)]
fn document_level_problems_are_reported(
    #[case] document: Value,
    #[case] path: &str,
    #[case] expected: Error,
) {
    let parsed = loaded(document);
    assert_eq!(located(&parsed.errors), [(path, discriminant(&expected))]);
    if !matches!(expected, Error::Invalid(_)) {
        assert_eq!(parsed.errors[0].error.to_string(), expected.to_string());
    }
}

#[rstest]
#[case::kaggle_no_provider(
    "openrouter/deepseek/deepseek-chat-v3.1",
    json!({"input_cost_per_token": 2e-7, "output_cost_per_token": 8e-7}),
    vec!["/openrouter~1deepseek~1deepseek-chat-v3.1"],
    false
)]
#[case::langwatch_negative_rates(
    "typesafe/jev-router",
    json!({"litellm_provider": "typesafe", "input_cost_per_token": -1, "output_cost_per_token": -1}),
    vec!["/typesafe~1jev-router/input_cost_per_token", "/typesafe~1jev-router/output_cost_per_token"],
    true
)]
#[case::sample_spec_shape(
    "spec-shaped",
    json!({"litellm_provider": "test", "input_cost_per_token": "USD per token"}),
    vec!["/spec-shaped/input_cost_per_token"],
    true
)]
fn row_errors_locate_each_problem_and_decide_the_row(
    #[case] model: &str,
    #[case] row: Value,
    #[case] paths: Vec<&str>,
    #[case] rejected: bool,
) {
    let Value::Object(fields) = row else {
        unreachable!()
    };
    let errors = Catalog::row_errors(model, &fields);
    assert_eq!(
        errors
            .iter()
            .map(|error| error.path.as_str())
            .collect::<Vec<_>>(),
        paths
    );
    assert_eq!(
        errors
            .iter()
            .map(|error| rejects_row(&error.error))
            .collect::<Vec<_>>(),
        vec![rejected; paths.len()]
    );
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
            .map(|error| (error.path.as_str(), error.to_string()))
            .collect::<Vec<_>>(),
        parsed
            .errors
            .iter()
            .map(|error| (error.path.as_str(), error.to_string()))
            .collect::<Vec<_>>()
    );
    assert_eq!(row_errors[0].path, "/open~1router~0v1/input_cost_per_token");
    assert!(matches!(&row_errors[0].error, Error::Invalid(_)));
}

#[rstest]
fn malformed_fallback_is_collected_while_models_are_read() {
    let parsed = loaded(json!({
        "fallback_generalizations": {"rules": "bad"},
        "a": {"litellm_provider": "test"}
    }));
    assert_eq!(parsed.value.model_names().collect::<Vec<_>>(), ["a"]);
    assert!(parsed.value.fallback_rules().is_none());
    assert_eq!(
        located(&parsed.errors),
        [(
            "/fallback_generalizations",
            discriminant(&Error::Invalid(json_error()))
        )]
    );
    assert!(!fatal(&parsed.errors[0].error));
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
    assert!(catalog.lookup("bad").is_none());
    assert!(catalog.lookup("bad-alias").is_none());
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
    assert_eq!(parsed.errors[0].path, "//mode");

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
            .any(|error| matches!(&error.error, Error::UnknownTimezone { .. }))
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
        .map(|error| (error.path.clone(), error.to_string()))
        .collect();
    let document_errors: Vec<_> = parsed
        .errors
        .iter()
        .map(|error| (error.path.clone(), error.to_string()))
        .collect();

    assert_eq!(row_errors, document_errors);
}

#[rstest]
fn without_rejected_leaves_out_the_one_row_with_negative_rates() {
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
        "typesafe/jev-router".to_owned(),
        json!({
            "litellm_provider": "typesafe",
            "input_cost_per_token": -1,
            "output_cost_per_token": -1
        }),
    );
    let parsed = Catalog::parse(serde_json::to_vec(&document).unwrap().as_slice()).unwrap();
    assert_eq!(parsed.value.model_count(), 300);
    assert_eq!(parsed.errors.len(), 2);

    let (catalog, removed) = parsed.value.without_rejected(&parsed.errors);
    assert_eq!(catalog.model_count(), 299);
    assert!(catalog.lookup("typesafe/jev-router").is_none());
    assert_eq!(
        removed
            .iter()
            .map(|error| error.path.as_str())
            .collect::<Vec<_>>(),
        [
            "/typesafe~1jev-router/input_cost_per_token",
            "/typesafe~1jev-router/output_cost_per_token"
        ]
    );
}

#[rstest]
fn integrity_limits_count_canonical_models_not_aliases() {
    let fetched = catalog(json!({"a": {"litellm_provider": "t", "aliases": ["b", "c"]}}));
    assert!(
        fetched
            .check_size(2, IntegrityLimits::new(1, 0.5).unwrap())
            .is_ok()
    );
    assert!(matches!(
        fetched.check_size(3, IntegrityLimits::new(1, 0.5).unwrap()),
        Err(Located {
            error: Error::Shrunk { .. },
            ..
        })
    ));
    assert!(matches!(
        fetched.check_size(2, IntegrityLimits::new(2, 0.5).unwrap()),
        Err(Located {
            error: Error::BelowMinimum { .. },
            ..
        })
    ));
}

#[rstest]
#[case::empty(b"{}", "empty")]
#[case::only_sections(br#"{"sample_spec":{}}"#, "empty")]
#[case::invalid_json(b"{", "json")]
#[case::array_root_is_json_error(b"[]", "json")]
fn malformed_documents_are_rejected_whole(#[case] body: &[u8], #[case] kind: &str) {
    let actual = Catalog::parse(body);
    assert!(matches!(&actual, Err(error) if fatal(&error.error)));
    match kind {
        "empty" => assert!(matches!(
            actual,
            Err(Located {
                error: Error::Empty,
                ..
            })
        )),
        "json" => assert!(matches!(
            actual,
            Err(Located {
                error: Error::Json(_),
                ..
            })
        )),
        _ => unreachable!(),
    }
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
fn the_bundled_registry_parses_with_no_problems_and_every_key_understood() {
    let loaded = Catalog::parse(BUNDLED_REGISTRY).unwrap();
    let catalog = loaded.value;
    assert!(loaded.errors.is_empty(), "{:#?}", loaded.errors);
    assert_eq!(catalog.unrecognized().collect::<Vec<_>>(), []);
    assert!(
        catalog
            .fallback_rules()
            .is_some_and(|rules| !rules.is_empty())
    );
    const REFERENCE_MODEL_COUNT: usize = 4303;
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
fn the_bundled_registry_serves_models_and_aliases_and_writes_its_sections() {
    let catalog = Catalog::parse(BUNDLED_REGISTRY).unwrap().value;
    let written: BTreeSet<&str> = catalog.written().map(|(key, _)| key).collect();
    let served: BTreeSet<&str> = catalog.served().map(|(key, _)| key).collect();
    let sections = BTreeSet::from(["fallback_generalizations", "sample_spec"]);
    assert!(sections.is_subset(&written));
    assert!(served.is_disjoint(&sections));
    let models_and_aliases: BTreeSet<&str> = catalog
        .model_names()
        .chain(catalog.aliases().keys().map(String::as_str))
        .collect();
    assert_eq!(served, models_and_aliases);
    assert_eq!(
        written
            .difference(&sections)
            .copied()
            .collect::<BTreeSet<_>>(),
        catalog.model_names().collect::<BTreeSet<_>>()
    );
}

fn row_strategy() -> impl Strategy<Value = Value> {
    (
        prop::option::of(prop::sample::select(vec![json!("test"), json!(4)])),
        prop::option::of(prop::sample::select(vec![json!("chat"), json!("chatt")])),
        prop::option::of(prop::sample::select(vec![
            json!(1e-6),
            json!(-1),
            json!("free"),
            json!(0),
        ])),
        prop::option::of(prop::sample::select(vec![
            json!(["m0"]),
            json!(["shared"]),
            json!("bad"),
        ])),
        prop::option::of(prop::sample::select(vec![
            json!(1),
            json!({"nested": true}),
        ])),
    )
        .prop_map(|(provider, mode, price, aliases, unknown)| {
            let fields = [
                ("litellm_provider", provider),
                ("mode", mode),
                ("input_cost_per_token", price),
                ("aliases", aliases),
                ("not_a_known_key", unknown),
            ];
            Value::Object(
                fields
                    .into_iter()
                    .filter_map(|(key, value)| Some((key.to_owned(), value?)))
                    .collect(),
            )
        })
}

fn is_cross_entry(error: &Error) -> bool {
    matches!(
        error,
        Error::AliasShadowsModel { .. } | Error::AliasTaken { .. }
    )
}

proptest! {
    #[test]
    fn parse_reports_row_errors_per_entry_plus_the_cross_entry_errors(rows in prop::collection::vec(row_strategy(), 1..8)) {
        let document: Map<String, Value> = rows
            .into_iter()
            .enumerate()
            .map(|(index, row)| (format!("m{index}"), row))
            .collect();
        let parsed = Catalog::parse(serde_json::to_vec(&document).unwrap().as_slice()).unwrap();
        let per_entry: Vec<(String, String)> = document
            .iter()
            .flat_map(|(key, row)| Catalog::row_errors(key, row.as_object().unwrap()))
            .map(|error| (error.path.clone(), error.to_string()))
            .collect();
        let (cross_entry, row_level): (Vec<&Located>, Vec<&Located>) =
            parsed.errors.iter().partition(|error| is_cross_entry(&error.error));
        prop_assert_eq!(
            row_level.iter().map(|error| (error.path.clone(), error.to_string())).collect::<Vec<_>>(),
            per_entry
        );
        let cross_entry_rows: Vec<&str> = cross_entry.iter().map(|error| error.path.trim_start_matches('/')).collect();
        prop_assert!(cross_entry_rows.iter().all(|row| document.contains_key(*row)));
    }
}
