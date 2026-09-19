use std::{
    path::{Path, PathBuf},
    time::Duration,
};

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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HttpSettings {
    pub ssl_verify: Option<SslVerify>,
    pub ssl_cert_file: Option<PathBuf>,
    pub ssl_certificate: Option<PathBuf>,
    pub ssl_security_level: Option<String>,
    pub ssl_ecdh_curve: Option<String>,
    pub force_ipv4: bool,
    pub http2: bool,
    pub httpx_transport: bool,
    pub user_agent: Option<String>,
    pub trust_proxy_env: bool,
    pub connect_timeout: Duration,
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
            httpx_transport: false,
            user_agent: None,
            trust_proxy_env: false,
            connect_timeout: Duration::from_secs(5),
        }
    }
}

impl HttpSettings {
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
                .or(self.ssl_certificate)
                .filter(|path| !path.as_os_str().is_empty()),
            ssl_security_level: env("SSL_SECURITY_LEVEL")
                .or(self.ssl_security_level)
                .filter(|level| !level.is_empty()),
            ssl_ecdh_curve: env("SSL_ECDH_CURVE")
                .or(self.ssl_ecdh_curve)
                .filter(|curve| !curve.is_empty()),
            http2: self.http2 || enabled("LITELLM_HTTP2"),
            httpx_transport: self.httpx_transport || enabled("DISABLE_AIOHTTP_TRANSPORT"),
            user_agent: env("LITELLM_USER_AGENT").or(self.user_agent),
            trust_proxy_env: self.trust_proxy_env || enabled("AIOHTTP_TRUST_ENV"),
            ..self
        }
    }

    pub fn without_missing_files(self, exists: &dyn Fn(&Path) -> bool) -> Self {
        Self {
            ssl_verify: match self.ssl_verify {
                Some(SslVerify::CaBundle(path)) if !exists(&path) => Some(SslVerify::Enabled),
                other => other,
            },
            ssl_cert_file: self.ssl_cert_file.filter(|path| exists(path)),
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

    #[test]
    fn empty_environment_values_clear_the_setting_like_python_truthiness() {
        let settings = HttpSettings {
            ssl_certificate: Some("/configured/client.pem".into()),
            ssl_security_level: Some("configured".into()),
            ssl_ecdh_curve: Some("X25519".into()),
            ..HttpSettings::default()
        }
        .with_environment(&env_of(&[
            ("SSL_CERTIFICATE", ""),
            ("SSL_SECURITY_LEVEL", ""),
            ("SSL_ECDH_CURVE", ""),
        ]));
        assert_eq!(settings.ssl_certificate, None);
        assert_eq!(settings.ssl_security_level, None);
        assert_eq!(settings.ssl_ecdh_curve, None);
    }

    #[test]
    fn missing_files_fall_back_to_default_verification() {
        let settings = HttpSettings {
            ssl_verify: Some(SslVerify::CaBundle("/absent/roots.pem".into())),
            ssl_cert_file: Some("/absent/env.pem".into()),
            ..HttpSettings::default()
        }
        .without_missing_files(&|_| false);
        assert_eq!(settings.ssl_verify, Some(SslVerify::Enabled));
        assert_eq!(settings.ssl_cert_file, None);
    }

    #[test]
    fn existing_files_are_kept() {
        let settings = HttpSettings {
            ssl_verify: Some(SslVerify::CaBundle("/present/roots.pem".into())),
            ssl_cert_file: Some("/present/env.pem".into()),
            ..HttpSettings::default()
        };
        assert_eq!(settings.clone().without_missing_files(&|_| true), settings);
    }

    #[rstest]
    #[case("true", true)]
    #[case("True", true)]
    #[case("false", false)]
    #[case("1", false)]
    fn boolean_switches_only_turn_on_for_true(#[case] value: &'static str, #[case] expected: bool) {
        let env = move |name: &str| match name {
            "LITELLM_HTTP2" | "AIOHTTP_TRUST_ENV" | "DISABLE_AIOHTTP_TRANSPORT" => {
                Some(value.to_string())
            }
            _ => None,
        };
        let settings = HttpSettings::default().with_environment(&env);
        assert_eq!(settings.http2, expected);
        assert_eq!(settings.httpx_transport, expected);
        assert_eq!(settings.trust_proxy_env, expected);
    }
}
