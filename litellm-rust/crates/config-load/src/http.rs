use litellm_config::{LiteLlmSettings, Spelled};
use litellm_http_types::{HttpSettings, HttpSettingsLayer, SslVerify};

use crate::Environment;

pub fn http_settings(settings: &LiteLlmSettings, environment: &Environment) -> HttpSettings {
    HttpSettings::from_layers([
        HttpSettingsLayer::from_environment(environment),
        written(settings),
    ])
}

fn written(settings: &LiteLlmSettings) -> HttpSettingsLayer {
    HttpSettingsLayer {
        ssl_verify: settings.tls.ssl_verify.as_ref().map(ssl_verify),
        ssl_certificate: settings
            .client_certificate
            .ssl_certificate
            .clone()
            .map(Into::into),
        ssl_security_level: settings.tls.ssl_security_level.clone(),
        ssl_ecdh_curve: settings.tls.ssl_ecdh_curve.clone(),
        force_ipv4: settings.socket.force_ipv4,
        http2: settings.http2.http2,
        aiohttp_trust_env: settings.aiohttp.aiohttp_trust_env,
        disable_aiohttp_trust_env: settings.aiohttp.disable_aiohttp_trust_env,
        disable_aiohttp_transport: settings.aiohttp.disable_aiohttp_transport,
        ..HttpSettingsLayer::default()
    }
}

fn ssl_verify(value: &Spelled<bool>) -> SslVerify {
    match value {
        Spelled::Value(true) => SslVerify::Enabled,
        Spelled::Value(false) => SslVerify::Disabled,
        Spelled::Text(text) => SslVerify::parse(text),
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use rstest::rstest;

    use litellm_config::Config;
    use litellm_http_types::{HttpSettings, SslVerify};

    use super::http_settings;
    use crate::Environment;

    fn lowered(yaml: &str, variables: &[(&str, &str)]) -> HttpSettings {
        let config = Config::from_yaml(yaml).unwrap();
        let environment = variables
            .iter()
            .map(|(name, value)| ((*name).to_owned(), (*value).to_owned()))
            .collect();
        http_settings(&config.litellm_settings, &environment)
    }

    #[test]
    fn absent_everywhere_is_the_default() {
        assert_eq!(lowered("{}", &[]), HttpSettings::default());
    }

    #[rstest]
    #[case::default("{}", &[], None)]
    #[case::written("litellm_settings: {ssl_verify: false}", &[], Some(SslVerify::Disabled))]
    #[case::written_text("litellm_settings: {ssl_verify: 'False'}", &[], Some(SslVerify::Disabled))]
    #[case::written_bundle("litellm_settings: {ssl_verify: /etc/roots.pem}", &[], Some(SslVerify::CaBundle(PathBuf::from("/etc/roots.pem"))))]
    #[case::environment("{}", &[("SSL_VERIFY", "False")], Some(SslVerify::Disabled))]
    #[case::environment_over_file("litellm_settings: {ssl_verify: false}", &[("SSL_VERIFY", "true")], Some(SslVerify::Enabled))]
    fn ssl_verify_takes_environment_then_file(
        #[case] yaml: &str,
        #[case] variables: &[(&str, &str)],
        #[case] expected: Option<SslVerify>,
    ) {
        assert_eq!(lowered(yaml, variables).ssl_verify, expected);
    }

    #[rstest]
    #[case::file("litellm_settings: {ssl_certificate: /etc/client.pem}", &[], "/etc/client.pem")]
    #[case::environment_over_file("litellm_settings: {ssl_certificate: /etc/client.pem}", &[("SSL_CERTIFICATE", "/env/client.pem")], "/env/client.pem")]
    fn client_certificate_takes_environment_then_file(
        #[case] yaml: &str,
        #[case] variables: &[(&str, &str)],
        #[case] expected: &str,
    ) {
        assert_eq!(
            lowered(yaml, variables).ssl_certificate,
            Some(PathBuf::from(expected))
        );
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

    #[test]
    fn force_ipv4_comes_from_the_file_only() {
        assert!(lowered("litellm_settings: {force_ipv4: true}", &[]).force_ipv4);
        assert!(!lowered("{}", &[("FORCE_IPV4", "true")]).force_ipv4);
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

    #[test]
    fn aiohttp_flags_from_the_file_decide_proxy_trust() {
        assert!(
            !lowered("litellm_settings: {disable_aiohttp_trust_env: true}", &[]).trust_proxy_env
        );
        assert!(
            lowered(
                "litellm_settings: {disable_aiohttp_trust_env: true, aiohttp_trust_env: true}",
                &[]
            )
            .trust_proxy_env
        );
    }

    #[test]
    fn later_overlay_wins_and_unrelated_variables_are_ignored() {
        let environment = Environment::from_iter([("SSL_VERIFY".to_owned(), "true".to_owned())])
            .overlaid([
                ("SSL_VERIFY".to_owned(), "false".to_owned()),
                ("HTTP2".to_owned(), "true".to_owned()),
            ]);
        let settings = http_settings(
            &Config::from_yaml("{}").unwrap().litellm_settings,
            &environment,
        );
        assert_eq!(settings.ssl_verify, Some(SslVerify::Disabled));
        assert!(!settings.http2);
    }
}
