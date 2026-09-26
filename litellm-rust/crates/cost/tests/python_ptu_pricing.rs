#![allow(clippy::disallowed_types)]

// mirrors: test_litellm/litellm_core_utils/test_ptu_pricing.py::test_a_complete_reservation_is_accepted

use litellm_cost::ptu_pricing::{
    MAX_COST_PER_PTU_PER_HOUR, MAX_PTU_COUNT, PTU_ZEROED_PRICING_FIELDS, SEARCH_CONTEXT_SIZES,
    azure_spillover, declares_ptu, is_spilled_over_ptu_request, ptu_config_error,
    ptu_identity_error, ptu_terms, zeroed_ptu_pricing,
};
use rstest::rstest;
use serde_json::{Map, Value, json};

fn valid_model_info() -> Value {
    json!({
        "team_id": "team-alpha",
        "ptu_count": 100,
        "cost_per_ptu_per_hour": 0.02,
        "ptu_effective_from": "2026-01-01T00:00:00Z",
    })
}

fn zeroed_with_flag(
    model_info: &Value,
    declared: &Value,
    enabled: bool,
) -> Option<Map<String, Value>> {
    zeroed_ptu_pricing(model_info, declared, enabled)
}

#[rstest]
fn a_complete_reservation_is_accepted() {
    let terms = ptu_terms(&valid_model_info()).expect("terms");

    assert_eq!(terms.team_id, "team-alpha");
    assert_eq!(terms.ptu_count, 100);
    assert_eq!(
        terms
            .effective_from
            .to_zoned(jiff::tz::TimeZone::UTC)
            .datetime(),
        jiff::civil::date(2026, 1, 1).at(0, 0, 0, 0)
    );
    assert_eq!(terms.effective_to, None);
}

#[rstest]
#[case::no_team(json!({"team_id": Value::Null}))]
#[case::blank_team(json!({"team_id": ""}))]
#[case::no_count(json!({"ptu_count": Value::Null}))]
#[case::no_rate(json!({"cost_per_ptu_per_hour": Value::Null}))]
#[case::zero_count(json!({"ptu_count": 0}))]
#[case::negative_count(json!({"ptu_count": -1}))]
#[case::count_over_the_cap(json!({"ptu_count": MAX_PTU_COUNT + 1}))]
#[case::negative_rate(json!({"cost_per_ptu_per_hour": -0.01}))]
#[case::rate_over_the_cap(json!({"cost_per_ptu_per_hour": MAX_COST_PER_PTU_PER_HOUR + 1.0}))]
#[case::count_not_a_number(json!({"ptu_count": "not-a-number"}))]
#[case::no_start(json!({"ptu_effective_from": Value::Null}))]
#[case::unparseable_start(json!({"ptu_effective_from": "not-a-date"}))]
#[case::unparseable_end(json!({"ptu_effective_to": "not-a-date"}))]
#[case::end_before_start(json!({"ptu_effective_to": "2025-01-01T00:00:00Z"}))]
#[case::end_equal_to_start(json!({"ptu_effective_to": "2026-01-01T00:00:00Z"}))]
fn an_incomplete_reservation_accrues_nothing(#[case] override_value: Value) {
    let model_info = merge(&valid_model_info(), &override_value);

    assert_eq!(ptu_terms(&model_info), None);
    assert_eq!(zeroed_with_flag(&model_info, &json!({}), true), None);
}

fn merge(base: &Value, override_value: &Value) -> Value {
    let mut merged = base.as_object().cloned().unwrap_or_default();
    for (key, value) in override_value.as_object().into_iter().flatten() {
        merged.insert(key.clone(), value.clone());
    }
    Value::Object(merged)
}

#[rstest]
fn a_naive_start_is_read_as_utc() {
    let model_info = merge(
        &valid_model_info(),
        &json!({"ptu_effective_from": "2026-05-01T12:00:00"}),
    );

    let terms = ptu_terms(&model_info).expect("terms");
    assert_eq!(
        terms
            .effective_from
            .to_zoned(jiff::tz::TimeZone::UTC)
            .datetime(),
        jiff::civil::date(2026, 5, 1).at(12, 0, 0, 0)
    );
}

#[rstest]
fn an_offset_start_is_converted_rather_than_relabelled() {
    let model_info = merge(
        &valid_model_info(),
        &json!({"ptu_effective_from": "2026-05-01T12:00:00-05:00"}),
    );

    let terms = ptu_terms(&model_info).expect("terms");
    assert_eq!(
        terms
            .effective_from
            .to_zoned(jiff::tz::TimeZone::UTC)
            .datetime(),
        jiff::civil::date(2026, 5, 1).at(17, 0, 0, 0)
    );
}

#[rstest]
fn nothing_is_zeroed_while_the_feature_is_off() {
    assert_eq!(
        zeroed_with_flag(&valid_model_info(), &json!({}), false),
        None
    );
}

#[rstest]
fn the_standing_rates_are_all_zeroed() {
    let override_pricing =
        zeroed_with_flag(&valid_model_info(), &json!({}), true).expect("pricing");

    assert!(
        PTU_ZEROED_PRICING_FIELDS
            .iter()
            .all(|field| override_pricing.get(*field) == Some(&json!(0.0)))
    );
}

#[rstest]
fn tiered_pricing_is_emptied_rather_than_zeroed() {
    let declared = json!({"tiered_pricing": [{"range": [0, 1000], "input_cost_per_token": 0.003}]});

    let override_pricing = zeroed_with_flag(&valid_model_info(), &declared, true).expect("pricing");

    assert_eq!(override_pricing.get("tiered_pricing"), Some(&json!([])));
}

#[rstest]
fn the_search_context_table_is_zeroed_in_place_on_every_deployment() {
    let override_pricing =
        zeroed_with_flag(&valid_model_info(), &json!({}), true).expect("pricing");

    let table = override_pricing
        .get("search_context_cost_per_query")
        .expect("table");
    assert!(
        SEARCH_CONTEXT_SIZES
            .iter()
            .all(|size| table.get(*size) == Some(&json!(0.0)))
    );
}

#[rstest]
fn the_maps_grounding_rate_is_zeroed_on_every_deployment() {
    let override_pricing =
        zeroed_with_flag(&valid_model_info(), &json!({}), true).expect("pricing");

    assert_eq!(
        override_pricing.get("google_maps_grounding_cost_per_query"),
        Some(&json!(0.0))
    );
}

#[rstest]
fn a_declared_table_does_not_become_a_scalar() {
    let declared = json!({"search_context_cost_per_query": {"search_context_size_medium": 0.05}});

    let override_pricing = zeroed_with_flag(&valid_model_info(), &declared, true).expect("pricing");

    let table = override_pricing
        .get("search_context_cost_per_query")
        .expect("table");
    assert!(
        SEARCH_CONTEXT_SIZES
            .iter()
            .all(|size| table.get(*size) == Some(&json!(0.0)))
    );
}

#[rstest]
fn a_rate_the_deployment_declares_itself_is_zeroed_too() {
    let extra = "input_cost_per_token_above_200k_tokens";
    let declared = json!({extra: 9e-06});

    let override_pricing = zeroed_with_flag(&valid_model_info(), &declared, true).expect("pricing");

    assert_eq!(override_pricing.get(extra), Some(&json!(0.0)));
}

#[rstest]
fn a_setting_that_is_not_a_charge_is_left_alone() {
    let declared = json!({"output_vector_size": 1536});

    let override_pricing = zeroed_with_flag(&valid_model_info(), &declared, true).expect("pricing");

    assert_eq!(override_pricing.get("output_vector_size"), None);
}

#[rstest]
fn a_complete_reservation_has_no_error() {
    assert_eq!(ptu_config_error(&valid_model_info(), None), None);
}

#[rstest]
fn a_deployment_with_no_ptu_fields_is_not_a_ptu_deployment() {
    assert_eq!(
        ptu_config_error(&json!({"team_id": "team-alpha"}), None),
        None
    );
    assert_eq!(ptu_config_error(&json!({}), None), None);
    assert!(!declares_ptu(&json!({"team_id": "team-alpha"})));
}

#[rstest]
#[case::no_team(json!({"team_id": Value::Null}), "team_id is required when PTU fields are set (one model maps to one team)")]
#[case::blank_team(json!({"team_id": ""}), "team_id is required when PTU fields are set (one model maps to one team)")]
#[case::count_without_rate(json!({"cost_per_ptu_per_hour": Value::Null}), "ptu_count and cost_per_ptu_per_hour must be set together")]
#[case::rate_without_count(json!({"ptu_count": Value::Null}), "ptu_count and cost_per_ptu_per_hour must be set together")]
#[case::inverted_window(json!({"ptu_effective_to": "2025-01-01T00:00:00Z"}), "ptu_effective_to must be after ptu_effective_from")]
fn an_incoherent_reservation_names_its_reason(
    #[case] override_value: Value,
    #[case] expected: &str,
) {
    let model_info = merge(&valid_model_info(), &override_value);

    assert_eq!(
        ptu_config_error(&model_info, None),
        Some(expected.to_string())
    );
}

#[rstest]
fn a_missing_start_is_explained_rather_than_inferred() {
    let model_info = json!({
        "team_id": "team-alpha",
        "ptu_count": 100,
        "cost_per_ptu_per_hour": 0.02,
    });

    let error = ptu_config_error(&model_info, None).expect("error");
    assert!(error.starts_with("ptu_effective_from is required when PTU fields are set"));
}

#[rstest]
fn an_inverted_window_is_caught_before_the_count_and_rate_gate() {
    let window_only = json!({
        "ptu_effective_from": "2026-01-01T00:00:00Z",
        "ptu_effective_to": "2025-01-01T00:00:00Z",
    });

    assert_eq!(
        ptu_config_error(&window_only, None),
        Some("ptu_effective_to must be after ptu_effective_from".to_string())
    );
}

#[rstest]
fn a_declared_unique_id_is_accepted() {
    assert_eq!(
        ptu_identity_error(Some("azure-ptu-eastus"), false, None, None),
        None
    );
}

#[rstest]
#[case::absent(None)]
#[case::blank(Some(""))]
fn a_reservation_without_an_id_is_refused(#[case] missing: Option<&str>) {
    let error = ptu_identity_error(missing, false, None, None).expect("error");

    assert!(error.starts_with("model_info.id is required when PTU fields are set"));
}

#[rstest]
fn the_refusal_names_the_id_the_deployment_already_uses() {
    let error = ptu_identity_error(None, false, Some("0ba149287615"), None).expect("error");

    assert!(error.contains("0ba149287615"));
}

#[rstest]
fn the_refusal_points_at_the_model_info_route_when_the_current_id_is_unknown() {
    let error = ptu_identity_error(None, false, None, None).expect("error");

    assert!(error.contains("GET /model/info"));
}

#[rstest]
fn an_id_declared_twice_is_refused() {
    let error = ptu_identity_error(Some("azure-ptu-eastus"), true, None, None).expect("error");

    assert!(error.contains("declared on more than one deployment"));
}

#[rstest]
fn the_deployment_is_named_when_the_caller_supplies_one() {
    let error = ptu_identity_error(None, false, None, Some("azure-ptu")).expect("error");

    assert!(error.starts_with("PTU configuration on model 'azure-ptu' is invalid:"));
}

#[rstest]
fn a_bare_yaml_date_bound_is_read_as_that_day_opening() {
    let model_info = merge(
        &valid_model_info(),
        &json!({"ptu_effective_to": "2027-01-01"}),
    );

    let terms = ptu_terms(&model_info).expect("terms");
    assert_eq!(
        terms
            .effective_to
            .expect("to")
            .to_zoned(jiff::tz::TimeZone::UTC)
            .datetime(),
        jiff::civil::date(2027, 1, 1).at(0, 0, 0, 0)
    );
}

#[rstest]
fn a_bare_yaml_date_start_is_read_as_that_day_opening() {
    let model_info = merge(
        &valid_model_info(),
        &json!({"ptu_effective_from": "2026-05-01"}),
    );

    let terms = ptu_terms(&model_info).expect("terms");
    assert_eq!(
        terms
            .effective_from
            .to_zoned(jiff::tz::TimeZone::UTC)
            .datetime(),
        jiff::civil::date(2026, 5, 1).at(0, 0, 0, 0)
    );
}

#[rstest]
fn the_string_zero_is_a_declared_id() {
    assert_eq!(ptu_identity_error(Some("0"), false, None, None), None);
}

#[rstest]
fn an_empty_id_is_no_id() {
    let error = ptu_identity_error(Some(""), false, None, None).expect("error");

    assert!(error.starts_with("model_info.id is required"));
}

#[rstest]
fn the_spillover_header_marks_the_request_as_pay_as_you_go() {
    assert!(is_spilled_over_ptu_request(
        &valid_model_info(),
        Some(&json!({"x-ms-is-spilled-over": "True"})),
        None,
        true,
    ));
}

#[rstest]
fn no_spillover_marker_keeps_the_zeroed_ptu_rates() {
    assert!(!is_spilled_over_ptu_request(
        &valid_model_info(),
        Some(&json!({"x-ms-is-spilled-over": "false"})),
        None,
        true,
    ));
    assert!(!is_spilled_over_ptu_request(
        &valid_model_info(),
        None,
        Some(&json!({"llm_provider-x-ms-is-spilled-over": "absent"})),
        true,
    ));
}

#[rstest]
fn azure_spillover_carries_the_source_deployment_from_raw_headers() {
    assert_eq!(
        azure_spillover(
            Some(&json!({
                "x-ms-is-spilled-over": "true",
                "x-ms-spillover-from-deployment": "my-ptu",
            })),
            None,
        ),
        Some(litellm_cost::ptu_pricing::AzureSpillover {
            from_deployment: Some("my-ptu".to_string()),
        })
    );
}

#[rstest]
fn azure_spillover_from_processed_headers_has_no_source_when_absent() {
    assert_eq!(
        azure_spillover(
            None,
            Some(&json!({"llm_provider-x-ms-is-spilled-over": "true"})),
        ),
        Some(litellm_cost::ptu_pricing::AzureSpillover {
            from_deployment: None,
        })
    );
}

#[rstest]
fn no_spillover_marker_returns_none() {
    assert_eq!(
        azure_spillover(Some(&json!({"x-ms-is-spilled-over": "false"})), None),
        None
    );
    assert_eq!(azure_spillover(None, None), None);
}

#[rstest]
fn a_boolean_spillover_marker_matches_pythons_string_coercion() {
    assert!(is_spilled_over_ptu_request(
        &valid_model_info(),
        Some(&json!({"x-ms-is-spilled-over": true})),
        None,
        true,
    ));
    assert!(!is_spilled_over_ptu_request(
        &valid_model_info(),
        Some(&json!({"x-ms-is-spilled-over": false})),
        None,
        true,
    ));
}

#[rstest]
fn spillover_requires_both_the_terms_and_the_feature_flag() {
    let headers = Some(&json!({"x-ms-is-spilled-over": "true"}));

    assert!(!is_spilled_over_ptu_request(
        &valid_model_info(),
        headers,
        None,
        false
    ));
    assert!(!is_spilled_over_ptu_request(
        &json!({"team_id": "t"}),
        headers,
        None,
        true
    ));
}

#[rstest]
#[case::numeric_string_count(json!({"ptu_count": "100"}), Some(100))]
#[case::numeric_string_rate(json!({"cost_per_ptu_per_hour": "0.02"}), Some(100))]
#[case::float_count_truncates_like_python(json!({"ptu_count": 3.7}), Some(3))]
#[case::float_count_truncating_to_zero_is_rejected(json!({"ptu_count": 0.5}), None)]
#[case::at_the_caps(
    json!({"ptu_count": MAX_PTU_COUNT, "cost_per_ptu_per_hour": MAX_COST_PER_PTU_PER_HOUR}),
    Some(MAX_PTU_COUNT),
)]
fn coercions_and_bounds_match_python(
    #[case] override_value: Value,
    #[case] expected_count: Option<i64>,
) {
    let model_info = merge(&valid_model_info(), &override_value);
    let terms = ptu_terms(&model_info);

    match expected_count {
        Some(count) => assert_eq!(terms.expect("terms").ptu_count, count),
        None => assert_eq!(terms, None),
    }
}

#[rstest]
fn a_numeric_team_id_is_stringified_like_python() {
    let model_info = merge(&valid_model_info(), &json!({"team_id": 7}));

    assert_eq!(ptu_terms(&model_info).expect("terms").team_id, "7");
}

#[rstest]
fn a_declared_512k_cache_read_rate_is_zeroed_too() {
    let extra = "cache_read_input_token_cost_above_512k_tokens";
    let declared = json!({extra: 1e-06});

    let override_pricing = zeroed_with_flag(&valid_model_info(), &declared, true).expect("pricing");

    assert_eq!(override_pricing.get(extra), Some(&json!(0.0)));
}

#[rstest]
#[case(json!({"ptu_effective_to": Value::Null}))]
#[case::missing(json!({}))]
fn a_null_or_absent_effective_to_leaves_the_window_open(#[case] override_value: Value) {
    let model_info = merge(&valid_model_info(), &override_value);

    assert_eq!(ptu_terms(&model_info).expect("terms").effective_to, None);
}

#[rstest]
fn a_blank_current_id_falls_back_to_the_model_info_route() {
    let error = ptu_identity_error(None, false, Some(""), None).expect("error");

    assert!(error.contains("shown by GET /model/info"));
    assert!(!error.contains("uses, ,"));
}

#[rstest]
fn a_null_spillover_source_is_absent_not_the_string_none() {
    assert_eq!(
        azure_spillover(
            Some(&json!({
                "x-ms-is-spilled-over": "true",
                "x-ms-spillover-from-deployment": Value::Null,
            })),
            None,
        ),
        Some(litellm_cost::ptu_pricing::AzureSpillover {
            from_deployment: None,
        })
    );
}

#[rstest]
fn a_config_error_names_the_model_when_supplied() {
    let model_info = merge(&valid_model_info(), &json!({"team_id": ""}));

    let error = ptu_config_error(&model_info, Some("azure-ptu")).expect("error");
    assert!(
        error.starts_with("PTU configuration on model 'azure-ptu' is invalid: team_id is required")
    );
}

#[rstest]
fn the_zeroed_pricing_map_carries_exactly_the_python_fields() {
    let override_pricing =
        zeroed_with_flag(&valid_model_info(), &json!({}), true).expect("pricing");

    let mut names: Vec<&str> = override_pricing.keys().map(String::as_str).collect();
    names.sort_unstable();
    assert_eq!(
        names,
        [
            "cache_creation_input_token_cost",
            "cache_creation_input_token_cost_above_1hr",
            "cache_creation_input_token_cost_above_200k_tokens",
            "cache_read_input_token_cost",
            "cache_read_input_token_cost_above_200k_tokens",
            "google_maps_grounding_cost_per_query",
            "input_cost_per_character",
            "input_cost_per_token",
            "output_cost_per_character",
            "output_cost_per_token",
            "search_context_cost_per_query",
            "tiered_pricing",
        ]
    );
}

#[rstest]
#[case::string(json!("team-alpha"), "team-alpha")]
#[case::integer(json!(7), "7")]
#[case::integral_float(json!(7.0), "7.0")]
#[case::fractional_float(json!(7.5), "7.5")]
#[case::boolean(json!(true), "True")]
fn a_team_id_is_rendered_like_python_str(#[case] team_id: Value, #[case] expected: &str) {
    let model_info = merge(&valid_model_info(), &json!({"team_id": team_id}));

    assert_eq!(ptu_terms(&model_info).expect("terms").team_id, expected);
}

#[rstest]
#[case::padded_string(json!(" 12 "), Some(12))]
#[case::float_truncates(json!(3.9), Some(3))]
#[case::boolean_true(json!(true), Some(1))]
#[case::decimal_string_is_not_an_int(json!("3.5"), None)]
#[case::list_is_not_a_count(json!([3]), None)]
fn a_ptu_count_is_coerced_like_python_int(#[case] count: Value, #[case] expected: Option<i64>) {
    let model_info = merge(&valid_model_info(), &json!({"ptu_count": count}));

    assert_eq!(
        ptu_terms(&model_info).map(|terms| terms.ptu_count),
        expected
    );
}

#[rstest]
#[case::padded_string(json!(" 0.5 "), Some(0.5))]
#[case::boolean_false(json!(false), Some(0.0))]
#[case::integer(json!(2), Some(2.0))]
#[case::not_a_number(json!("NaN"), None)]
#[case::object_is_not_a_rate(json!({"rate": 1}), None)]
fn a_ptu_rate_is_coerced_like_python_float(#[case] rate: Value, #[case] expected: Option<f64>) {
    let model_info = merge(&valid_model_info(), &json!({"cost_per_ptu_per_hour": rate}));

    assert_eq!(
        ptu_terms(&model_info).map(|terms| terms.cost_per_ptu_per_hour),
        expected
    );
}

#[rstest]
#[case::date_only("2026-05-01", jiff::civil::date(2026, 5, 1).at(0, 0, 0, 0))]
#[case::space_separated("2026-05-01 06:30:00", jiff::civil::date(2026, 5, 1).at(6, 30, 0, 0))]
fn a_start_without_a_time_or_t_separator_is_read_as_utc(
    #[case] start: &str,
    #[case] expected: jiff::civil::DateTime,
) {
    let model_info = merge(&valid_model_info(), &json!({"ptu_effective_from": start}));

    let terms = ptu_terms(&model_info).expect("terms");
    assert_eq!(
        terms
            .effective_from
            .to_zoned(jiff::tz::TimeZone::UTC)
            .datetime(),
        expected
    );
}
