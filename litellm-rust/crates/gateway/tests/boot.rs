use std::sync::mpsc;

use rstest::rstest;

use litellm_config::Config;
use litellm_config_load::Environment;
use litellm_tracing::{Logger, Metadata, Record, Sink};

#[rstest]
fn applies_configured_http_environment_before_building_clients() {
    let directory = tempfile::tempdir().unwrap();
    let missing = directory.path().join("missing-ca.pem");
    let config = Config::from_yaml(&format!(
        "environment_variables: {{SSL_VERIFY: '{}'}}\nlitellm_settings: {{ssl_verify: false}}",
        missing.display(),
    ))
    .unwrap();
    let error = litellm_gateway::build_inference(&config, &Environment::default())
        .err()
        .unwrap();
    assert!(matches!(
        error,
        litellm_gateway::Error::Http(litellm_http::Error::Read { path, .. })
            if path == missing
    ));
}

struct LogSink(mpsc::Sender<String>);

impl Sink for LogSink {
    fn enabled(&self, _: &Metadata<'_>) -> bool {
        true
    }

    fn emit(&self, record: &Record) {
        self.0.send(record.message.clone()).unwrap();
    }
}

#[rstest]
fn warns_about_tls_settings_the_native_client_cannot_honor() {
    let config = Config::from_yaml(
        "litellm_settings: {ssl_security_level: 'DEFAULT@SECLEVEL=1', ssl_ecdh_curve: secp521r1}",
    )
    .unwrap();
    let (sender, receiver) = mpsc::channel();
    let logger = Logger::new(LogSink(sender));
    logger
        .scope(|| litellm_gateway::build_inference(&config, &Environment::default()))
        .unwrap();
    let messages: Vec<String> = receiver.try_iter().collect();
    assert!(
        messages
            .iter()
            .any(|message| message.contains("ssl_ecdh_curve \"secp521r1\""))
    );
    assert!(
        messages
            .iter()
            .any(|message| message.contains("ssl_security_level \"@SECLEVEL=1\""))
    );
}
