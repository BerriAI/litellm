use std::path::PathBuf;

use litellm_core_utils::{serde_compat::parse_str_bool, settings::Lookup};

use litellm_config::{LiteLlmSettings, Spelled};

use crate::Environment;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum SslVerify {
    #[default]
    Enabled,
    Disabled,
    CaBundle(PathBuf),
}

impl SslVerify {
    fn parse(value: &str) -> Self {
        match parse_str_bool(value) {
            Some(true) => Self::Enabled,
            Some(false) => Self::Disabled,
            None => Self::CaBundle(PathBuf::from(value)),
        }
    }

    fn written(value: &Spelled<bool>) -> Self {
        match value {
            Spelled::Value(true) => Self::Enabled,
            Spelled::Value(false) => Self::Disabled,
            Spelled::Text(text) => Self::parse(text),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Transport {
    pub ssl_verify: SslVerify,
    pub ssl_certificate: Option<PathBuf>,
    pub ssl_security_level: Option<String>,
    pub ssl_ecdh_curve: Option<String>,
    pub force_ipv4: bool,
    pub http2: bool,
    pub aiohttp_trust_env: bool,
    pub disable_aiohttp_trust_env: bool,
    pub disable_aiohttp_transport: bool,
}

impl Transport {
    pub fn lower(settings: &LiteLlmSettings, environment: &Environment) -> Self {
        let text = |name: &str, written: &Option<String>| {
            environment
                .get(name)
                .or_else(|| written.clone())
                .filter(|value| !value.is_empty())
        };
        let flag = |name: &str, written: Option<bool>| {
            environment.enabled(name).or(written).unwrap_or(false)
        };
        Self {
            ssl_verify: environment
                .get("SSL_VERIFY")
                .map(|value| SslVerify::parse(&value))
                .or_else(|| settings.tls.ssl_verify.as_ref().map(SslVerify::written))
                .unwrap_or_default(),
            ssl_certificate: environment
                .get("SSL_CERTIFICATE")
                .or_else(|| settings.client_certificate.ssl_certificate.clone())
                .map(PathBuf::from),
            ssl_security_level: text("SSL_SECURITY_LEVEL", &settings.tls.ssl_security_level),
            ssl_ecdh_curve: text("SSL_ECDH_CURVE", &settings.tls.ssl_ecdh_curve),
            force_ipv4: settings.socket.force_ipv4.unwrap_or(false),
            http2: flag("LITELLM_HTTP2", settings.http2.http2),
            aiohttp_trust_env: flag("AIOHTTP_TRUST_ENV", settings.aiohttp.aiohttp_trust_env),
            disable_aiohttp_trust_env: flag(
                "DISABLE_AIOHTTP_TRUST_ENV",
                settings.aiohttp.disable_aiohttp_trust_env,
            ),
            disable_aiohttp_transport: flag(
                "DISABLE_AIOHTTP_TRANSPORT",
                settings.aiohttp.disable_aiohttp_transport,
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use rstest::rstest;

    use super::{SslVerify, Transport};
    use litellm_config::Config;

    use crate::Environment;

    fn lowered(yaml: &str, variables: &[(&str, &str)]) -> Transport {
        let config = Config::from_yaml(yaml).unwrap();
        let environment = variables
            .iter()
            .map(|(name, value)| ((*name).to_owned(), (*value).to_owned()))
            .collect();
        Transport::lower(&config.litellm_settings, &environment)
    }

    #[rstest]
    fn absent_everywhere_is_the_default() {
        assert_eq!(lowered("{}", &[]), Transport::default());
    }

    #[rstest]
    #[case::default("{}", &[], SslVerify::Enabled)]
    #[case::written("litellm_settings: {ssl_verify: false}", &[], SslVerify::Disabled)]
    #[case::written_bundle("litellm_settings: {ssl_verify: /etc/roots.pem}", &[], SslVerify::CaBundle(PathBuf::from("/etc/roots.pem")))]
    #[case::environment("{}", &[("SSL_VERIFY", "False")], SslVerify::Disabled)]
    #[case::environment_over_file("litellm_settings: {ssl_verify: false}", &[("SSL_VERIFY", "true")], SslVerify::Enabled)]
    fn ssl_verify_takes_environment_then_file_then_default(
        #[case] yaml: &str,
        #[case] variables: &[(&str, &str)],
        #[case] expected: SslVerify,
    ) {
        assert_eq!(lowered(yaml, variables).ssl_verify, expected);
    }

    #[rstest]
    #[case::file("litellm_settings: {http2: true}", &[], true)]
    #[case::environment("{}", &[("LITELLM_HTTP2", "TRUE")], true)]
    #[case::environment_false_does_not_override_file("litellm_settings: {http2: true}", &[("LITELLM_HTTP2", "false")], true)]
    fn flags_are_only_turned_on_by_the_environment(
        #[case] yaml: &str,
        #[case] variables: &[(&str, &str)],
        #[case] expected: bool,
    ) {
        assert_eq!(lowered(yaml, variables).http2, expected);
    }

    #[rstest]
    #[case::file("litellm_settings: {ssl_security_level: '1'}", &[], Some("1"))]
    #[case::environment_over_file("litellm_settings: {ssl_security_level: '1'}", &[("SSL_SECURITY_LEVEL", "2")], Some("2"))]
    #[case::empty_is_unset("{}", &[("SSL_SECURITY_LEVEL", "")], None)]
    fn text_settings_skip_empty_values(
        #[case] yaml: &str,
        #[case] variables: &[(&str, &str)],
        #[case] expected: Option<&str>,
    ) {
        assert_eq!(
            lowered(yaml, variables).ssl_security_level.as_deref(),
            expected
        );
    }

    #[rstest]
    fn later_overlay_wins_and_unrelated_variables_are_ignored() {
        let environment = Environment::from_iter([("SSL_VERIFY".to_owned(), "true".to_owned())])
            .overlaid([
                ("SSL_VERIFY".to_owned(), "false".to_owned()),
                ("HTTP2".to_owned(), "true".to_owned()),
            ]);
        let transport = Transport::lower(
            &Config::from_yaml("{}").unwrap().litellm_settings,
            &environment,
        );
        assert_eq!(transport.ssl_verify, SslVerify::Disabled);
        assert!(!transport.http2);
    }
}
