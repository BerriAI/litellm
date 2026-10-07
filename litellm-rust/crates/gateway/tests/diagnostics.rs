use litellm_config::Config;
use litellm_gateway::diagnostics_configuration;
use litellm_tracing::DestinationConfig;
use rstest::rstest;

#[rstest]
fn gateway_yaml_projects_policy_and_secret_references_into_the_shared_runtime() {
    let config = Config::from_yaml(
        r#"
general_settings:
  diagnostics:
    enabled: true
    service_name: native-gateway
    policy:
      minimum_level: WARN
      sample_rate: 0.25
    destinations:
      - transport: otlp
        name: logs
        endpoint: os.environ/LOCAL_DIAGNOSTIC_ENDPOINT
        headers:
          Authorization: os.environ/LOCAL_DIAGNOSTIC_AUTH
environment_variables:
  LOCAL_DIAGNOSTIC_ENDPOINT: http://localhost/v1/logs
  LOCAL_DIAGNOSTIC_AUTH: synthetic-secret
"#,
    )
    .unwrap();
    let diagnostics = diagnostics_configuration(&config).unwrap();
    assert!(diagnostics.enabled);
    assert_eq!(diagnostics.service_name, "native-gateway");
    assert_eq!(diagnostics.policy.sample_rate, 0.25);
    let DestinationConfig::Otlp {
        endpoint, headers, ..
    } = &diagnostics.destinations[0]
    else {
        panic!("OTLP destination")
    };
    assert_eq!(endpoint, "http://localhost/v1/logs");
    assert_eq!(headers["Authorization"], "synthetic-secret");
}
