use litellm_config::{LiteLlmSettings, Spelled};
use litellm_http::{HttpKeysPartial, HttpSettings, SslVerify};
use litellm_layered::{Lookup, merge};

pub(crate) fn http_settings(
    env: &(impl Lookup + ?Sized),
    settings: &LiteLlmSettings,
) -> Result<HttpSettings, litellm_layered::Error> {
    Ok(HttpSettings::from(merge([
        HttpKeysPartial::from_env(env)?,
        written(settings),
    ])))
}

fn written(settings: &LiteLlmSettings) -> HttpKeysPartial {
    HttpKeysPartial {
        ssl_verify: settings.ssl_verify.as_ref().map(|value| match value {
            Spelled::Value(true) => SslVerify::Enabled,
            Spelled::Value(false) => SslVerify::Disabled,
            Spelled::Text(value) => SslVerify::parse(value),
        }),
        ssl_cert_file: None,
        ssl_certificate: settings.ssl_certificate.as_ref().map(Into::into),
        ssl_security_level: settings.ssl_security_level.clone(),
        ssl_ecdh_curve: settings.ssl_ecdh_curve.clone(),
        force_ipv4: settings.force_ipv4,
        http2: settings.http2,
        aiohttp_trust_env: settings.aiohttp_trust_env,
        disable_aiohttp_trust_env: settings.disable_aiohttp_trust_env,
        disable_aiohttp_transport: settings.disable_aiohttp_transport,
        user_agent: None,
        tcp_keepalive: None,
        pool_idle_timeout: None,
        proxies: None,
    }
}

#[cfg(test)]
mod tests {
    use litellm_layered::Env;
    use rstest::rstest;

    use super::*;

    fn file(yaml: &str) -> LiteLlmSettings {
        serde_yaml_ng::from_str(yaml).unwrap()
    }

    #[rstest]
    #[case::env_beats_file(&[("SSL_VERIFY", "false")], "ssl_verify: true", Some(SslVerify::Disabled))]
    #[case::file_fills_unset_env(&[], "ssl_verify: false", Some(SslVerify::Disabled))]
    #[case::neither(&[], "{}", None)]
    fn env_is_stacked_above_the_config_file(
        #[case] env: &[(&str, &str)],
        #[case] yaml: &str,
        #[case] expected: Option<SslVerify>,
    ) {
        let settings = http_settings(&Env::of(env), &file(yaml)).unwrap();
        assert_eq!(settings.ssl_verify, expected);
    }

    #[rstest]
    #[case::bool_false("false", "false", SslVerify::Disabled)]
    #[case::text_false("'False'", "False", SslVerify::Disabled)]
    #[case::text_true("'true'", "true", SslVerify::Enabled)]
    #[case::bundle("/etc/ca.pem", "/etc/ca.pem", SslVerify::CaBundle("/etc/ca.pem".into()))]
    fn ssl_verify_spellings_read_the_same_from_the_file_and_the_environment(
        #[case] yaml: &str,
        #[case] env: &str,
        #[case] expected: SslVerify,
    ) {
        let from_file =
            http_settings(&Env::default(), &file(&format!("ssl_verify: {yaml}"))).unwrap();
        let from_env = http_settings(&Env::of(&[("SSL_VERIFY", env)]), &file("{}")).unwrap();
        assert_eq!(from_file.ssl_verify, Some(expected.clone()));
        assert_eq!(from_env.ssl_verify, Some(expected));
    }

    #[rstest]
    fn flags_are_only_turned_on_by_the_environment() {
        let settings = http_settings(
            &Env::of(&[("LITELLM_HTTP2", "false")]),
            &file("http2: true"),
        )
        .unwrap();
        assert!(settings.http2);
    }
}
