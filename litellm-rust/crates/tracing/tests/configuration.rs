use litellm_tracing::{DestinationConfig, DiagnosticPolicy, DiagnosticsConfig, Level};
use rstest::rstest;
use serde_json::{Value, json};

#[rstest]
fn rust_configuration_matches_the_shared_python_wire_contract() {
    let value: Value = serde_json::from_str(include_str!("fixtures/diagnostics.json")).unwrap();
    let config: DiagnosticsConfig = serde_json::from_value(value.clone()).unwrap();
    assert_eq!(serde_json::to_value(config).unwrap(), value);
}

#[rstest]
#[case::trace("TRACE", Level::TRACE)]
#[case::debug("DEBUG", Level::DEBUG)]
#[case::info("INFO", Level::INFO)]
#[case::warn("WARN", Level::WARN)]
#[case::error("ERROR", Level::ERROR)]
fn diagnostic_levels_preserve_the_wire_contract(#[case] value: &str, #[case] level: Level) {
    let policy: DiagnosticPolicy = serde_json::from_value(json!({"minimum_level": value})).unwrap();
    assert_eq!(policy.minimum_level, level);
    assert_eq!(
        serde_json::to_value(policy).unwrap()["minimum_level"],
        value
    );
}

#[rstest]
fn omitted_diagnostic_level_defaults_to_info() {
    let policy: DiagnosticPolicy = serde_json::from_value(json!({})).unwrap();
    assert_eq!(policy.minimum_level, Level::INFO);
    assert_eq!(
        serde_json::to_value(policy).unwrap()["minimum_level"],
        "INFO"
    );
}

#[rstest]
fn environment_replaces_yaml_and_resolves_destination_secrets() {
    let config = DiagnosticsConfig::from_sources(
        Some(json!({"enabled": false, "service_name": "yaml-service"})),
        |name| match name {
            "LITELLM_DIAGNOSTICS" => Some(include_str!("fixtures/diagnostics.json").to_owned()),
            "TEST_LOGS_ENDPOINT" => Some("http://localhost/v1/logs".into()),
            "TEST_LOGS_AUTH" => Some("synthetic-secret".into()),
            _ => None,
        },
    )
    .unwrap();
    assert!(config.enabled);
    assert_eq!(config.service_name, "diagnostic-contract");
    let DestinationConfig::Otlp {
        endpoint, headers, ..
    } = &config.destinations[0]
    else {
        panic!("OTLP destination")
    };
    assert_eq!(endpoint, "http://localhost/v1/logs");
    assert_eq!(headers["Authorization"], "synthetic-secret");
}

#[rstest]
#[case::shape_boolean_string(json!({"payload_shapes": "true"}))]
#[case::shape_boolean_number(json!({"payload_shapes": 1}))]
#[case::boolean_string(json!({"enabled": "false"}))]
#[case::unknown_level(json!({"policy": {"minimum_level": "quiet"}}))]
#[case::lowercase_level(json!({"policy": {"minimum_level": "info"}}))]
#[case::mixed_case_level(json!({"policy": {"minimum_level": "Info"}}))]
#[case::numeric_level_string(json!({"policy": {"minimum_level": "3"}}))]
#[case::numeric_level(json!({"policy": {"minimum_level": 3}}))]
#[case::padded_level(json!({"policy": {"minimum_level": " INFO "}}))]
#[case::off_level(json!({"policy": {"minimum_level": "OFF"}}))]
#[case::unknown_field(json!({"unexpected": true}))]
#[case::negative_sample(json!({"policy": {"sample_rate": -0.1}}))]
#[case::large_sample(json!({"policy": {"sample_rate": 1.1}}))]
#[case::string_sample(json!({"policy": {"sample_rate": "0.5"}}))]
#[case::duplicate_destinations(json!({"enabled": true, "destinations": [{"transport": "otlp", "name": "same", "endpoint": "http://localhost/v1/logs"}, {"transport": "otlp", "name": "same", "endpoint": "http://localhost/v1/logs"}]}))]
fn malformed_diagnostic_configuration_is_rejected(#[case] value: Value) {
    assert!(DiagnosticsConfig::from_sources(Some(value), |_| None).is_err());
}

#[rstest]
fn missing_secret_errors_exclude_the_reference_and_value() {
    let error = DiagnosticsConfig::from_sources(Some(json!({"enabled": true, "destinations": [{"transport": "otlp", "name": "logs", "endpoint": "os.environ/SYNTHETIC_SECRET_NAME"}]})), |_| None).err().unwrap();
    assert_eq!(
        error.to_string(),
        "diagnostic configuration requires a value for endpoint"
    );
}
