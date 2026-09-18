use std::{path::PathBuf, time::Duration};

/// `litellm.ssl_verify` / `SSL_VERIFY`: a bool or a CA bundle path.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum SslVerify {
    Enabled,
    Disabled,
    CaBundle(PathBuf),
}

impl SslVerify {
    pub fn parse(value: &str) -> Self {
        match value.trim().to_ascii_lowercase().as_str() {
            "true" => Self::Enabled,
            "false" => Self::Disabled,
            _ => Self::CaBundle(PathBuf::from(value)),
        }
    }
}

/// The plain inputs `http_handler.py` reads from `litellm.*` globals and the environment.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HttpSettings {
    pub ssl_verify: Option<SslVerify>,
    pub ssl_cert_file: Option<PathBuf>,
    pub ssl_certificate: Option<PathBuf>,
    pub ssl_security_level: Option<String>,
    pub ssl_ecdh_curve: Option<String>,
    pub force_ipv4: bool,
    pub http2: bool,
    pub user_agent: Option<String>,
    pub trust_proxy_env: bool,
    pub connect_timeout: Duration,
    pub request_timeout: Option<Duration>,
}

impl Default for HttpSettings {
    fn default() -> Self {
        Self {
            ssl_verify: None,
            ssl_cert_file: None,
            ssl_certificate: None,
            ssl_security_level: None,
            ssl_ecdh_curve: None,
            force_ipv4: false,
            http2: false,
            user_agent: None,
            trust_proxy_env: false,
            connect_timeout: Duration::from_secs(5),
            request_timeout: None,
        }
    }
}

impl HttpSettings {
    /// Overlay the environment variables `http_handler.py` consults, with the same precedence:
    /// `SSL_VERIFY`, `SSL_CERTIFICATE`, `SSL_SECURITY_LEVEL`, `SSL_ECDH_CURVE` and
    /// `LITELLM_USER_AGENT` win over the configured value; `SSL_CERT_FILE` only applies when
    /// verification is on without an explicit bundle; `LITELLM_HTTP2` and `AIOHTTP_TRUST_ENV`
    /// can only turn their switch on.
    pub fn with_environment(self, env: &(dyn Fn(&str) -> Option<String> + Sync)) -> Self {
        let enabled =
            |name: &str| env(name).is_some_and(|value| value.trim().eq_ignore_ascii_case("true"));
        Self {
            ssl_verify: env("SSL_VERIFY")
                .map(|value| SslVerify::parse(&value))
                .or(self.ssl_verify),
            ssl_cert_file: env("SSL_CERT_FILE")
                .map(PathBuf::from)
                .or(self.ssl_cert_file),
            ssl_certificate: env("SSL_CERTIFICATE")
                .map(PathBuf::from)
                .or(self.ssl_certificate),
            ssl_security_level: env("SSL_SECURITY_LEVEL").or(self.ssl_security_level),
            ssl_ecdh_curve: env("SSL_ECDH_CURVE").or(self.ssl_ecdh_curve),
            http2: self.http2 || enabled("LITELLM_HTTP2"),
            user_agent: env("LITELLM_USER_AGENT").or(self.user_agent),
            trust_proxy_env: self.trust_proxy_env || enabled("AIOHTTP_TRUST_ENV"),
            ..self
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    fn no_env(_: &str) -> Option<String> {
        None
    }

    fn env_of(
        values: &'static [(&'static str, &'static str)],
    ) -> impl Fn(&str) -> Option<String> + Sync {
        move |name| {
            values
                .iter()
                .find(|(key, _)| *key == name)
                .map(|(_, value)| value.to_string())
        }
    }

    #[rstest]
    #[case("true", SslVerify::Enabled)]
    #[case(" True ", SslVerify::Enabled)]
    #[case("FALSE", SslVerify::Disabled)]
    #[case("/etc/ssl/bundle.pem", SslVerify::CaBundle("/etc/ssl/bundle.pem".into()))]
    fn ssl_verify_parses_bools_and_treats_anything_else_as_a_bundle_path(
        #[case] value: &str,
        #[case] expected: SslVerify,
    ) {
        assert_eq!(SslVerify::parse(value), expected);
    }

    #[test]
    fn environment_overrides_configured_ssl_values() {
        let settings = HttpSettings {
            ssl_verify: Some(SslVerify::Enabled),
            ssl_certificate: Some("/configured/client.pem".into()),
            ssl_security_level: Some("configured".into()),
            user_agent: Some("configured/1".into()),
            ..HttpSettings::default()
        }
        .with_environment(&env_of(&[
            ("SSL_VERIFY", "false"),
            ("SSL_CERT_FILE", "/env/roots.pem"),
            ("SSL_CERTIFICATE", "/env/client.pem"),
            ("SSL_SECURITY_LEVEL", "DEFAULT@SECLEVEL=1"),
            ("SSL_ECDH_CURVE", "X25519"),
            ("LITELLM_USER_AGENT", "env/2"),
        ]));
        assert_eq!(settings.ssl_verify, Some(SslVerify::Disabled));
        assert_eq!(settings.ssl_cert_file, Some("/env/roots.pem".into()));
        assert_eq!(settings.ssl_certificate, Some("/env/client.pem".into()));
        assert_eq!(
            settings.ssl_security_level.as_deref(),
            Some("DEFAULT@SECLEVEL=1")
        );
        assert_eq!(settings.ssl_ecdh_curve.as_deref(), Some("X25519"));
        assert_eq!(settings.user_agent.as_deref(), Some("env/2"));
    }

    #[test]
    fn missing_environment_keeps_configured_values() {
        let configured = HttpSettings {
            ssl_verify: Some(SslVerify::CaBundle("/configured/roots.pem".into())),
            http2: true,
            trust_proxy_env: true,
            user_agent: Some("configured/1".into()),
            ..HttpSettings::default()
        };
        assert_eq!(configured.clone().with_environment(&no_env), configured);
    }

    #[rstest]
    #[case("true", true)]
    #[case("True", true)]
    #[case("false", false)]
    #[case("1", false)]
    fn boolean_switches_only_turn_on_for_true(#[case] value: &'static str, #[case] expected: bool) {
        let env = move |name: &str| match name {
            "LITELLM_HTTP2" | "AIOHTTP_TRUST_ENV" => Some(value.to_string()),
            _ => None,
        };
        let settings = HttpSettings::default().with_environment(&env);
        assert_eq!(settings.http2, expected);
        assert_eq!(settings.trust_proxy_env, expected);
    }
}
