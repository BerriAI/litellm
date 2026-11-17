use std::{ffi::OsString, time::Duration};

use litellm_gateway_forward::{Error, Settings};
use rstest::rstest;

const COMPLETE: [(&str, &str); 5] = [
    ("LITELLM_FORWARD_UPSTREAM", "http://127.0.0.1:53211"),
    ("LITELLM_FORWARD_HOST", "0.0.0.0"),
    ("LITELLM_FORWARD_PORT", "4000"),
    ("LITELLM_FORWARD_UPSTREAM_KEEPALIVE", "2"),
    ("LITELLM_FORWARD_SHUTDOWN_DRAIN", "40"),
];

fn parse(variables: &[(&str, &str)]) -> Result<Option<Settings>, Error> {
    Settings::from_vars(
        variables
            .iter()
            .map(|(key, value)| (OsString::from(key), OsString::from(value))),
    )
}

fn without(variable: &str) -> Vec<(&'static str, &'static str)> {
    COMPLETE
        .into_iter()
        .filter(|(key, _)| *key != variable)
        .collect()
}

fn with<'a>(variable: &'a str, value: &'a str) -> Vec<(&'a str, &'a str)> {
    without(variable)
        .into_iter()
        .chain([(variable, value)])
        .collect()
}

#[rstest]
fn reads_every_variable() {
    assert_eq!(
        parse(&COMPLETE).unwrap(),
        Some(Settings {
            upstream: "http://127.0.0.1:53211".parse().unwrap(),
            host: "0.0.0.0".into(),
            port: 4000,
            upstream_keepalive: Duration::from_secs(2),
            shutdown_drain: Duration::from_secs(40),
        })
    );
}

#[rstest]
fn without_an_upstream_the_gateway_is_not_forwarding() {
    assert_eq!(parse(&without("LITELLM_FORWARD_UPSTREAM")).unwrap(), None);
}

#[rstest]
#[case::host("LITELLM_FORWARD_HOST")]
#[case::port("LITELLM_FORWARD_PORT")]
#[case::upstream_keepalive("LITELLM_FORWARD_UPSTREAM_KEEPALIVE")]
#[case::shutdown_drain("LITELLM_FORWARD_SHUTDOWN_DRAIN")]
fn a_missing_variable_fails_naming_it(#[case] variable: &str) {
    let error = parse(&without(variable)).unwrap_err();

    assert!(error.to_string().contains(variable), "{error}");
}

#[rstest]
#[case::port_overflow("LITELLM_FORWARD_PORT", "65536")]
#[case::port_text("LITELLM_FORWARD_PORT", "http")]
#[case::keepalive_text("LITELLM_FORWARD_UPSTREAM_KEEPALIVE", "soon")]
#[case::keepalive_negative("LITELLM_FORWARD_UPSTREAM_KEEPALIVE", "-1")]
#[case::keepalive_zero("LITELLM_FORWARD_UPSTREAM_KEEPALIVE", "0")]
#[case::drain_text("LITELLM_FORWARD_SHUTDOWN_DRAIN", "soon")]
#[case::drain_negative("LITELLM_FORWARD_SHUTDOWN_DRAIN", "-1")]
#[case::drain_infinite("LITELLM_FORWARD_SHUTDOWN_DRAIN", "inf")]
fn an_invalid_variable_fails_naming_it(#[case] variable: &str, #[case] value: &str) {
    let error = parse(&with(variable, value)).unwrap_err();

    assert!(error.to_string().contains(variable), "{error}");
}

#[rstest]
fn fractional_drain_seconds_are_used_as_given() {
    let settings = parse(&with("LITELLM_FORWARD_SHUTDOWN_DRAIN", "12.5"))
        .unwrap()
        .unwrap();

    assert_eq!(settings.shutdown_drain, Duration::from_millis(12_500));
}

#[rstest]
#[case::keepalive("KEEPALIVE_TIMEOUT")]
#[case::graceful_shutdown("GRACEFUL_SHUTDOWN_TIMEOUT")]
#[case::host("HOST")]
#[case::port("PORT")]
fn python_settings_are_never_read(#[case] variable: &str) {
    let mut variables = COMPLETE.to_vec();
    variables.push((variable, "9"));

    assert_eq!(parse(&variables).unwrap(), parse(&COMPLETE).unwrap());
}

#[rstest]
fn a_non_loopback_upstream_fails_startup() {
    let error = parse(&with("LITELLM_FORWARD_UPSTREAM", "http://192.0.2.1:4000")).unwrap_err();

    assert!(
        error.to_string().contains("LITELLM_FORWARD_UPSTREAM"),
        "{error}"
    );
}
