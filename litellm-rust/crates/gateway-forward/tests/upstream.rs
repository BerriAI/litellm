use std::time::Duration;

use litellm_gateway_forward::Forward;
use rstest::rstest;

#[rstest]
#[case::https("https://127.0.0.1:4000")]
#[case::public("http://192.0.2.1:4000")]
#[case::hostname("http://localhost:4000")]
#[case::ipv6("http://[::1]:4000")]
#[case::path("http://127.0.0.1:4000/api")]
#[case::query("http://127.0.0.1:4000/?x=y")]
#[case::credentials("http://user@127.0.0.1:4000")]
fn rejects_anything_but_a_loopback_http_origin(#[case] upstream: &str) {
    assert!(Forward::new(upstream, Duration::from_secs(5)).is_err());
}

#[rstest]
#[case::standard("http://127.0.0.1:4000")]
#[case::loopback_range("http://127.0.0.2:4000/")]
fn accepts_a_loopback_http_origin(#[case] upstream: &str) {
    assert!(Forward::new(upstream, Duration::from_secs(5)).is_ok());
}
