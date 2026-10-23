use std::{
    path::{Path, PathBuf},
    time::Duration,
};

use litellm_core_utils::serde_compat::parse_str_bool;
use litellm_layered::{Invalid, Layered, Lookup, read, read_env};

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

    pub fn parse_env(value: &str) -> Result<Option<Self>, Invalid> {
        Ok(Some(Self::parse(value)))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TcpKeepalive {
    pub idle: Duration,
    pub interval: Duration,
    pub retries: u32,
}

impl TcpKeepalive {
    pub fn from_env(env: &(impl Lookup + ?Sized)) -> Result<Option<Self>, litellm_layered::Error> {
        if read_env(env, &["AIOHTTP_SO_KEEPALIVE"], false, read::enabled)?.is_none() {
            return Ok(None);
        }
        let seconds = |names: &'static [&'static str], default: u64| {
            Ok::<_, litellm_layered::Error>(
                read_env(env, names, false, read::seconds)?.unwrap_or(Duration::from_secs(default)),
            )
        };
        Ok(Some(Self {
            idle: seconds(&["AIOHTTP_TCP_KEEPIDLE"], 60)?,
            interval: seconds(&["AIOHTTP_TCP_KEEPINTVL"], 30)?,
            retries: read_env(env, &["AIOHTTP_TCP_KEEPCNT"], false, read::parsed)?.unwrap_or(5),
        }))
    }
}

#[derive(Layered)]
pub struct HttpKeys {
    #[layer(env = "SSL_VERIFY", read = SslVerify::parse_env, litellm_setting = "ssl_verify")]
    pub ssl_verify: SslVerify,
    #[layer(env = "SSL_CERT_FILE", read = read::truthy)]
    pub ssl_cert_file: PathBuf,
    #[layer(env = "SSL_CERTIFICATE", read = read::text, litellm_setting = "ssl_certificate")]
    pub ssl_certificate: PathBuf,
    #[layer(env = "SSL_SECURITY_LEVEL", read = read::text, litellm_setting = "ssl_security_level")]
    pub ssl_security_level: String,
    #[layer(env = "SSL_ECDH_CURVE", read = read::text, litellm_setting = "ssl_ecdh_curve")]
    pub ssl_ecdh_curve: String,
    #[layer(litellm_setting = "force_ipv4")]
    pub force_ipv4: bool,
    #[layer(env = "LITELLM_HTTP2", read = read::enabled, litellm_setting = "http2")]
    pub http2: bool,
    #[layer(env = "AIOHTTP_TRUST_ENV", read = read::enabled, litellm_setting = "aiohttp_trust_env")]
    pub aiohttp_trust_env: bool,
    #[layer(
        env = "DISABLE_AIOHTTP_TRUST_ENV",
        read = read::enabled,
        litellm_setting = "disable_aiohttp_trust_env"
    )]
    pub disable_aiohttp_trust_env: bool,
    #[layer(
        env = "DISABLE_AIOHTTP_TRANSPORT",
        read = read::enabled,
        litellm_setting = "disable_aiohttp_transport"
    )]
    pub disable_aiohttp_transport: bool,
    #[layer(env = "LITELLM_USER_AGENT", read = read::text, keep_blank)]
    pub user_agent: String,
    #[layer(from_env = TcpKeepalive::from_env)]
    pub tcp_keepalive: TcpKeepalive,
    #[layer(env = "AIOHTTP_KEEPALIVE_TIMEOUT", read = read::seconds)]
    pub pool_idle_timeout: Duration,
    #[layer(from_env = EnvironmentProxies::from_env)]
    pub proxies: EnvironmentProxies,
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
            pool_idle_timeout: Duration::from_secs(15),
        }
    }
}

impl From<HttpKeysPartial> for HttpSettings {
    fn from(merged: HttpKeysPartial) -> Self {
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
}

impl HttpSettings {
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

    use litellm_layered::{Env, merge};

    use super::*;

    fn stack(highest_first: impl IntoIterator<Item = HttpKeysPartial>) -> HttpSettings {
        HttpSettings::from(merge(highest_first))
    }

    fn from_env(values: &[(&str, &str)]) -> HttpKeysPartial {
        HttpKeysPartial::from_env(&Env::of(values)).unwrap()
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
        let configured = HttpKeysPartial {
            ssl_verify: Some(SslVerify::Enabled),
            ssl_certificate: Some("/configured/client.pem".into()),
            ssl_security_level: Some("configured".into()),
            user_agent: Some("configured/1".into()),
            ..HttpKeysPartial::default()
        };
        let environment = from_env(&[
            ("SSL_VERIFY", "false"),
            ("SSL_CERT_FILE", "/env/roots.pem"),
            ("SSL_CERTIFICATE", "/env/client.pem"),
            ("SSL_SECURITY_LEVEL", "DEFAULT@SECLEVEL=1"),
            ("SSL_ECDH_CURVE", "X25519"),
            ("LITELLM_USER_AGENT", "env/2"),
        ]);
        let settings = stack([environment, configured]);
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
        let higher = HttpKeysPartial {
            http2: Some(false),
            force_ipv4: Some(false),
            ..HttpKeysPartial::default()
        };
        let lower = HttpKeysPartial {
            http2: Some(true),
            force_ipv4: Some(true),
            ..HttpKeysPartial::default()
        };
        let settings = stack([higher, lower]);
        assert!(!settings.http2);
        assert!(!settings.force_ipv4);
    }

    #[test]
    fn an_empty_environment_is_an_empty_layer_so_lower_layers_and_defaults_apply() {
        assert_eq!(from_env(&[]), HttpKeysPartial::default());
        let configured = HttpKeysPartial {
            ssl_verify: Some(SslVerify::CaBundle("/configured/roots.pem".into())),
            http2: Some(true),
            user_agent: Some("configured/1".into()),
            ..HttpKeysPartial::default()
        };
        assert_eq!(
            stack([HttpKeysPartial::default(), configured]),
            HttpSettings {
                ssl_verify: Some(SslVerify::CaBundle("/configured/roots.pem".into())),
                http2: true,
                user_agent: Some("configured/1".into()),
                ..HttpSettings::default()
            }
        );
        assert_eq!(stack([]), HttpSettings::default());
    }

    #[test]
    fn blank_environment_values_fall_through_to_the_configured_settings() {
        let configured = HttpKeysPartial {
            ssl_certificate: Some("/configured/client.pem".into()),
            ssl_security_level: Some("configured".into()),
            ssl_ecdh_curve: Some("X25519".into()),
            ..HttpKeysPartial::default()
        };
        let environment = from_env(&[
            ("SSL_CERTIFICATE", ""),
            ("SSL_SECURITY_LEVEL", " "),
            ("SSL_ECDH_CURVE", ""),
        ]);
        let settings = stack([environment, configured]);
        assert_eq!(
            settings.ssl_certificate,
            Some(PathBuf::from("/configured/client.pem"))
        );
        assert_eq!(settings.ssl_security_level.as_deref(), Some("configured"));
        assert_eq!(settings.ssl_ecdh_curve.as_deref(), Some("X25519"));
    }

    #[test]
    fn an_empty_user_agent_environment_value_beats_the_configured_one() {
        let configured = HttpKeysPartial {
            user_agent: Some("configured/1".into()),
            ..HttpKeysPartial::default()
        };
        let settings = stack([from_env(&[("LITELLM_USER_AGENT", "")]), configured]);
        assert_eq!(settings.user_agent.as_deref(), Some(""));
    }

    #[test]
    fn socket_keepalive_follows_the_aiohttp_variables_with_python_defaults() {
        let tuned = stack([from_env(&[
            ("AIOHTTP_SO_KEEPALIVE", "True"),
            ("AIOHTTP_TCP_KEEPIDLE", "45"),
            ("AIOHTTP_KEEPALIVE_TIMEOUT", "30"),
        ])]);
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
        let settings = stack([from_env(&[("AIOHTTP_TCP_KEEPIDLE", "45")])]);
        assert_eq!(settings.tcp_keepalive, None);
        assert_eq!(settings.pool_idle_timeout, Duration::from_secs(15));
    }

    fn proxy_flags(
        aiohttp_trust_env: bool,
        disable_aiohttp_trust_env: bool,
        disable_aiohttp_transport: bool,
        http2: bool,
    ) -> HttpKeysPartial {
        HttpKeysPartial {
            aiohttp_trust_env: Some(aiohttp_trust_env),
            disable_aiohttp_trust_env: Some(disable_aiohttp_trust_env),
            disable_aiohttp_transport: Some(disable_aiohttp_transport),
            http2: Some(http2),
            ..HttpKeysPartial::default()
        }
    }

    #[rstest]
    #[case::aiohttp_default(proxy_flags(false, false, false, false), true)]
    #[case::aiohttp_opted_out(proxy_flags(false, true, false, false), false)]
    #[case::session_trust_env_beats_opt_out(proxy_flags(true, true, false, false), true)]
    #[case::http2_uses_httpx(proxy_flags(false, true, false, true), true)]
    #[case::aiohttp_disabled(proxy_flags(false, true, true, false), true)]
    fn environment_proxies_apply_unless_the_aiohttp_transport_opts_out(
        #[case] layer: HttpKeysPartial,
        #[case] expected: bool,
    ) {
        assert_eq!(stack([layer]).trust_proxy_env, expected);
    }

    #[test]
    fn a_proxy_opt_out_in_one_source_still_yields_to_trust_env_from_another() {
        let environment = from_env(&[("DISABLE_AIOHTTP_TRUST_ENV", "true")]);
        let configured = HttpKeysPartial {
            aiohttp_trust_env: Some(true),
            ..HttpKeysPartial::default()
        };
        assert!(!stack([environment.clone()]).trust_proxy_env);
        assert!(stack([environment, configured]).trust_proxy_env);
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
        let layer = HttpKeysPartial::from_env(&env).unwrap();
        assert_eq!(layer.http2, expected);
        assert_eq!(layer.aiohttp_trust_env, expected);
        assert_eq!(layer.disable_aiohttp_transport, expected);
        assert_eq!(layer.disable_aiohttp_trust_env, expected);
    }
}
