use std::{
    path::{Path, PathBuf},
    time::Duration,
};

use litellm_core_utils::{
    serde_compat::parse_str_bool,
    settings::{Layer, Lookup, merge},
};

use crate::proxy::EnvironmentProxies;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum SslVerify {
    Enabled,
    Disabled,
    CaBundle(PathBuf),
}

impl SslVerify {
    pub fn parse(value: &str) -> Self {
        match parse_str_bool(value) {
            Some(true) => Self::Enabled,
            Some(false) => Self::Disabled,
            _ => Self::CaBundle(PathBuf::from(value)),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TcpKeepalive {
    pub idle: Duration,
    pub interval: Duration,
    pub retries: u32,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct HttpSettingsLayer {
    pub ssl_verify: Option<SslVerify>,
    pub ssl_cert_file: Option<PathBuf>,
    pub ssl_certificate: Option<PathBuf>,
    pub ssl_security_level: Option<String>,
    pub ssl_ecdh_curve: Option<String>,
    pub force_ipv4: Option<bool>,
    pub http2: Option<bool>,
    pub aiohttp_trust_env: Option<bool>,
    pub disable_aiohttp_trust_env: Option<bool>,
    pub disable_aiohttp_transport: Option<bool>,
    pub user_agent: Option<String>,
    pub tcp_keepalive: Option<TcpKeepalive>,
    pub pool_idle_timeout: Option<Duration>,
    pub proxies: Option<EnvironmentProxies>,
}

impl HttpSettingsLayer {
    pub fn from_environment(env: &impl Lookup) -> Self {
        let seconds = |name: &str, default: u32| {
            Duration::from_secs(u64::from(env.parsed::<u32>(name).unwrap_or(default)))
        };
        Self {
            ssl_verify: env.get("SSL_VERIFY").map(|value| SslVerify::parse(&value)),
            ssl_cert_file: env.truthy("SSL_CERT_FILE").map(PathBuf::from),
            ssl_certificate: env.get("SSL_CERTIFICATE").map(PathBuf::from),
            ssl_security_level: env.get("SSL_SECURITY_LEVEL"),
            ssl_ecdh_curve: env.get("SSL_ECDH_CURVE"),
            force_ipv4: None,
            http2: env.enabled("LITELLM_HTTP2"),
            aiohttp_trust_env: env.enabled("AIOHTTP_TRUST_ENV"),
            disable_aiohttp_trust_env: env.enabled("DISABLE_AIOHTTP_TRUST_ENV"),
            disable_aiohttp_transport: env.enabled("DISABLE_AIOHTTP_TRANSPORT"),
            user_agent: env.get("LITELLM_USER_AGENT"),
            tcp_keepalive: env.enabled("AIOHTTP_SO_KEEPALIVE").map(|_| TcpKeepalive {
                idle: seconds("AIOHTTP_TCP_KEEPIDLE", 60),
                interval: seconds("AIOHTTP_TCP_KEEPINTVL", 30),
                retries: env.parsed("AIOHTTP_TCP_KEEPCNT").unwrap_or(5),
            }),
            pool_idle_timeout: env
                .parsed::<u32>("AIOHTTP_KEEPALIVE_TIMEOUT")
                .map(|timeout| Duration::from_secs(u64::from(timeout))),
            proxies: Some(EnvironmentProxies::from_environment(env))
                .filter(|proxies| *proxies != EnvironmentProxies::default()),
        }
    }
}

impl Layer for HttpSettingsLayer {
    fn or(self, lower: Self) -> Self {
        Self {
            ssl_verify: self.ssl_verify.or(lower.ssl_verify),
            ssl_cert_file: self.ssl_cert_file.or(lower.ssl_cert_file),
            ssl_certificate: self.ssl_certificate.or(lower.ssl_certificate),
            ssl_security_level: self.ssl_security_level.or(lower.ssl_security_level),
            ssl_ecdh_curve: self.ssl_ecdh_curve.or(lower.ssl_ecdh_curve),
            force_ipv4: self.force_ipv4.or(lower.force_ipv4),
            http2: self.http2.or(lower.http2),
            aiohttp_trust_env: self.aiohttp_trust_env.or(lower.aiohttp_trust_env),
            disable_aiohttp_trust_env: self
                .disable_aiohttp_trust_env
                .or(lower.disable_aiohttp_trust_env),
            disable_aiohttp_transport: self
                .disable_aiohttp_transport
                .or(lower.disable_aiohttp_transport),
            user_agent: self.user_agent.or(lower.user_agent),
            tcp_keepalive: self.tcp_keepalive.or(lower.tcp_keepalive),
            pool_idle_timeout: self.pool_idle_timeout.or(lower.pool_idle_timeout),
            proxies: self.proxies.or(lower.proxies),
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
    pub user_agent: Option<String>,
    pub trust_proxy_env: bool,
    pub proxies: EnvironmentProxies,
    pub connect_timeout: Duration,
    pub tcp_keepalive: Option<TcpKeepalive>,
    pub pool_idle_timeout: Duration,
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
            trust_proxy_env: true,
            proxies: EnvironmentProxies::default(),
            connect_timeout: Duration::from_secs(10),
            tcp_keepalive: None,
            pool_idle_timeout: Duration::from_secs(120),
        }
    }
}

impl HttpSettings {
    pub fn from_layers(
        highest_precedence_first: impl IntoIterator<Item = HttpSettingsLayer>,
    ) -> Self {
        let merged = merge(highest_precedence_first);
        let defaults = Self::default();
        let http2 = merged.http2.unwrap_or(defaults.http2);
        Self {
            ssl_verify: merged.ssl_verify,
            ssl_cert_file: merged.ssl_cert_file,
            ssl_certificate: merged.ssl_certificate,
            ssl_security_level: merged.ssl_security_level.filter(|level| !level.is_empty()),
            ssl_ecdh_curve: merged.ssl_ecdh_curve.filter(|curve| !curve.is_empty()),
            force_ipv4: merged.force_ipv4.unwrap_or(defaults.force_ipv4),
            http2,
            user_agent: merged.user_agent,
            trust_proxy_env: !merged.disable_aiohttp_trust_env.unwrap_or(false)
                || merged.aiohttp_trust_env.unwrap_or(false)
                || merged.disable_aiohttp_transport.unwrap_or(false)
                || http2,
            tcp_keepalive: merged.tcp_keepalive,
            pool_idle_timeout: merged
                .pool_idle_timeout
                .unwrap_or(defaults.pool_idle_timeout),
            proxies: merged.proxies.unwrap_or_default(),
            ..defaults
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

    fn env_of(values: &'static [(&'static str, &'static str)]) -> impl Fn(&str) -> Option<String> {
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
    fn higher_layers_override_lower_ones() {
        let configured = HttpSettingsLayer {
            ssl_verify: Some(SslVerify::Enabled),
            ssl_certificate: Some("/configured/client.pem".into()),
            ssl_security_level: Some("configured".into()),
            user_agent: Some("configured/1".into()),
            ..HttpSettingsLayer::default()
        };
        let environment = HttpSettingsLayer::from_environment(&env_of(&[
            ("SSL_VERIFY", "false"),
            ("SSL_CERT_FILE", "/env/roots.pem"),
            ("SSL_CERTIFICATE", "/env/client.pem"),
            ("SSL_SECURITY_LEVEL", "DEFAULT@SECLEVEL=1"),
            ("SSL_ECDH_CURVE", "X25519"),
            ("LITELLM_USER_AGENT", "env/2"),
        ]));
        let settings = HttpSettings::from_layers([environment, configured]);
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
    fn an_explicit_false_in_a_higher_layer_beats_a_lower_true() {
        let higher = HttpSettingsLayer {
            http2: Some(false),
            force_ipv4: Some(false),
            ..HttpSettingsLayer::default()
        };
        let lower = HttpSettingsLayer {
            http2: Some(true),
            force_ipv4: Some(true),
            ..HttpSettingsLayer::default()
        };
        let settings = HttpSettings::from_layers([higher, lower]);
        assert!(!settings.http2);
        assert!(!settings.force_ipv4);
    }

    #[test]
    fn an_empty_environment_is_an_empty_layer_so_lower_layers_and_defaults_apply() {
        assert_eq!(
            HttpSettingsLayer::from_environment(&no_env),
            HttpSettingsLayer::default()
        );
        let configured = HttpSettingsLayer {
            ssl_verify: Some(SslVerify::CaBundle("/configured/roots.pem".into())),
            http2: Some(true),
            user_agent: Some("configured/1".into()),
            ..HttpSettingsLayer::default()
        };
        assert_eq!(
            HttpSettings::from_layers([HttpSettingsLayer::default(), configured]),
            HttpSettings {
                ssl_verify: Some(SslVerify::CaBundle("/configured/roots.pem".into())),
                http2: true,
                user_agent: Some("configured/1".into()),
                ..HttpSettings::default()
            }
        );
        assert_eq!(HttpSettings::from_layers([]), HttpSettings::default());
    }

    #[test]
    fn empty_certificate_is_retained_for_validation_while_empty_tuning_is_absent() {
        let configured = HttpSettingsLayer {
            ssl_certificate: Some("/configured/client.pem".into()),
            ssl_security_level: Some("configured".into()),
            ssl_ecdh_curve: Some("X25519".into()),
            ..HttpSettingsLayer::default()
        };
        let environment = HttpSettingsLayer::from_environment(&env_of(&[
            ("SSL_CERTIFICATE", ""),
            ("SSL_SECURITY_LEVEL", ""),
            ("SSL_ECDH_CURVE", ""),
        ]));
        let settings = HttpSettings::from_layers([environment, configured]);
        assert_eq!(settings.ssl_certificate, Some(PathBuf::new()));
        assert_eq!(settings.ssl_security_level, None);
        assert_eq!(settings.ssl_ecdh_curve, None);
    }

    #[test]
    fn socket_keepalive_follows_the_aiohttp_variables_with_python_defaults() {
        let tuned = HttpSettings::from_layers([HttpSettingsLayer::from_environment(&env_of(&[
            ("AIOHTTP_SO_KEEPALIVE", "True"),
            ("AIOHTTP_TCP_KEEPIDLE", "45"),
            ("AIOHTTP_KEEPALIVE_TIMEOUT", "30"),
        ]))]);
        assert_eq!(
            tuned.tcp_keepalive,
            Some(TcpKeepalive {
                idle: Duration::from_secs(45),
                interval: Duration::from_secs(30),
                retries: 5,
            })
        );
        assert_eq!(tuned.pool_idle_timeout, Duration::from_secs(30));
    }

    #[test]
    fn socket_keepalive_stays_off_unless_enabled() {
        let settings = HttpSettings::from_layers([HttpSettingsLayer::from_environment(&env_of(
            &[("AIOHTTP_TCP_KEEPIDLE", "45")],
        ))]);
        assert_eq!(settings.tcp_keepalive, None);
        assert_eq!(settings.pool_idle_timeout, Duration::from_secs(120));
    }

    fn proxy_flags(
        aiohttp_trust_env: bool,
        disable_aiohttp_trust_env: bool,
        disable_aiohttp_transport: bool,
        http2: bool,
    ) -> HttpSettingsLayer {
        HttpSettingsLayer {
            aiohttp_trust_env: Some(aiohttp_trust_env),
            disable_aiohttp_trust_env: Some(disable_aiohttp_trust_env),
            disable_aiohttp_transport: Some(disable_aiohttp_transport),
            http2: Some(http2),
            ..HttpSettingsLayer::default()
        }
    }

    #[rstest]
    #[case::aiohttp_default(proxy_flags(false, false, false, false), true)]
    #[case::aiohttp_opted_out(proxy_flags(false, true, false, false), false)]
    #[case::session_trust_env_beats_opt_out(proxy_flags(true, true, false, false), true)]
    #[case::http2_uses_httpx(proxy_flags(false, true, false, true), true)]
    #[case::aiohttp_disabled(proxy_flags(false, true, true, false), true)]
    fn environment_proxies_apply_unless_the_aiohttp_transport_opts_out(
        #[case] layer: HttpSettingsLayer,
        #[case] expected: bool,
    ) {
        assert_eq!(HttpSettings::from_layers([layer]).trust_proxy_env, expected);
    }

    #[test]
    fn a_proxy_opt_out_in_one_source_still_yields_to_trust_env_from_another() {
        let environment =
            HttpSettingsLayer::from_environment(&env_of(&[("DISABLE_AIOHTTP_TRUST_ENV", "true")]));
        let configured = HttpSettingsLayer {
            aiohttp_trust_env: Some(true),
            ..HttpSettingsLayer::default()
        };
        assert!(!HttpSettings::from_layers([environment.clone()]).trust_proxy_env);
        assert!(HttpSettings::from_layers([environment, configured]).trust_proxy_env);
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
    #[case("true", Some(true))]
    #[case("True", Some(true))]
    #[case("false", None)]
    #[case("1", None)]
    fn boolean_switches_only_turn_on_for_true(
        #[case] value: &'static str,
        #[case] expected: Option<bool>,
    ) {
        let env = move |name: &str| match name {
            "LITELLM_HTTP2"
            | "AIOHTTP_TRUST_ENV"
            | "DISABLE_AIOHTTP_TRANSPORT"
            | "DISABLE_AIOHTTP_TRUST_ENV" => Some(value.to_string()),
            _ => None,
        };
        let layer = HttpSettingsLayer::from_environment(&env);
        assert_eq!(layer.http2, expected);
        assert_eq!(layer.aiohttp_trust_env, expected);
        assert_eq!(layer.disable_aiohttp_transport, expected);
        assert_eq!(layer.disable_aiohttp_trust_env, expected);
    }
}
