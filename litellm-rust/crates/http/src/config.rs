use std::{
    net::{IpAddr, Ipv4Addr},
    path::PathBuf,
    time::Duration,
};

use crate::{
    error::Error,
    settings::{HttpSettings, SslVerify, TcpKeepalive},
    tls::{CipherSelection, KeyExchangeGroup, Tls12CipherSuite, Unsupported},
};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Verify {
    Disabled,
    CaBundle(PathBuf),
    BuiltInRoots,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct HttpClientConfig {
    pub verify: Verify,
    pub client_certificate: Option<PathBuf>,
    pub key_exchange_group: Option<KeyExchangeGroup>,
    pub tls12_cipher_suites: Option<Vec<Tls12CipherSuite>>,
    pub force_ipv4: bool,
    pub http2: bool,
    pub user_agent: Option<String>,
    pub trust_proxy_env: bool,
    pub connect_timeout: Duration,
    pub tcp_keepalive: Option<TcpKeepalive>,
    pub pool_idle_timeout: Duration,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Resolution {
    pub config: HttpClientConfig,
    pub unsupported: Vec<Unsupported>,
}

impl HttpClientConfig {
    pub fn resolve(settings: &HttpSettings) -> Resolution {
        let (key_exchange_group, unsupported_curve) = match settings
            .ssl_ecdh_curve
            .as_deref()
            .map(str::parse::<KeyExchangeGroup>)
        {
            None => (None, None),
            Some(Ok(group)) => (Some(group), None),
            Some(Err(unsupported)) => (None, Some(unsupported)),
        };
        let ciphers = settings
            .ssl_security_level
            .as_deref()
            .map(CipherSelection::from);
        let verify = match &settings.ssl_verify {
            Some(SslVerify::Disabled) => Verify::Disabled,
            Some(SslVerify::CaBundle(path)) => Verify::CaBundle(path.clone()),
            Some(SslVerify::Enabled) | None => settings
                .ssl_cert_file
                .clone()
                .map_or(Verify::BuiltInRoots, Verify::CaBundle),
        };
        let (tls12_cipher_suites, unsupported_ciphers) = ciphers
            .map_or((None, Vec::new()), |ciphers| {
                (ciphers.tls12_cipher_suites, ciphers.unsupported)
            });
        Resolution {
            config: Self {
                verify,
                client_certificate: settings.ssl_certificate.clone(),
                key_exchange_group,
                tls12_cipher_suites,
                force_ipv4: settings.force_ipv4,
                http2: settings.http2,
                user_agent: settings.user_agent.clone(),
                trust_proxy_env: !settings.ignore_proxy_env
                    || settings.trust_proxy_env
                    || settings.http2
                    || settings.httpx_transport,
                connect_timeout: settings.connect_timeout,
                tcp_keepalive: settings.tcp_keepalive,
                pool_idle_timeout: settings.pool_idle_timeout,
            },
            unsupported: unsupported_curve
                .into_iter()
                .chain(unsupported_ciphers)
                .collect(),
        }
    }

    pub fn client_builder(&self) -> Result<reqwest::ClientBuilder, Error> {
        let base = reqwest::Client::builder()
            .use_preconfigured_tls(rustls::ClientConfig::try_from(self)?)
            .connect_timeout(self.connect_timeout)
            .pool_idle_timeout(self.pool_idle_timeout);
        let with_keepalive = match self.tcp_keepalive {
            None => base,
            Some(keepalive) => base
                .tcp_keepalive(keepalive.idle)
                .tcp_keepalive_interval(keepalive.interval)
                .tcp_keepalive_retries(keepalive.retries),
        };
        let with_address = if self.force_ipv4 {
            with_keepalive.local_address(IpAddr::V4(Ipv4Addr::UNSPECIFIED))
        } else {
            with_keepalive
        };
        let with_protocol = if self.http2 {
            with_address
        } else {
            with_address.http1_only()
        };
        let with_agent = match &self.user_agent {
            Some(agent) => with_protocol.user_agent(agent),
            None => with_protocol,
        };
        Ok(if self.trust_proxy_env {
            with_agent
        } else {
            with_agent.no_proxy()
        })
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    fn no_env(_: &str) -> Option<String> {
        None
    }

    fn settings(ssl_verify: Option<SslVerify>, ssl_cert_file: Option<&str>) -> HttpSettings {
        HttpSettings {
            ssl_verify,
            ssl_cert_file: ssl_cert_file.map(PathBuf::from),
            ..HttpSettings::default()
        }
        .with_environment(&no_env)
    }

    #[rstest]
    #[case::default(settings(None, None), Verify::BuiltInRoots)]
    #[case::setting_disables(
        settings(Some(SslVerify::Disabled), Some("/env/roots.pem")),
        Verify::Disabled
    )]
    #[case::setting_bundle(
        settings(Some(SslVerify::CaBundle("/configured.pem".into())), Some("/env/roots.pem")),
        Verify::CaBundle("/configured.pem".into())
    )]
    #[case::enabled_uses_cert_file(
        settings(Some(SslVerify::Enabled), Some("/env/roots.pem")),
        Verify::CaBundle("/env/roots.pem".into())
    )]
    #[case::unset_uses_cert_file(settings(None, Some("/env/roots.pem")), Verify::CaBundle("/env/roots.pem".into()))]
    fn verify_follows_setting_then_cert_file(
        #[case] settings: HttpSettings,
        #[case] expected: Verify,
    ) {
        let config = HttpClientConfig::resolve(&settings).config;
        assert_eq!(config.verify, expected);
    }

    #[test]
    fn ssl_verify_environment_variable_beats_the_configured_setting() {
        let settings = HttpSettings {
            ssl_verify: Some(SslVerify::Disabled),
            ..HttpSettings::default()
        }
        .with_environment(&|name: &str| (name == "SSL_VERIFY").then(|| "true".to_string()));
        let config = HttpClientConfig::resolve(&settings).config;
        assert_eq!(config.verify, Verify::BuiltInRoots);
    }

    #[rstest]
    #[case::x25519("X25519", Some(KeyExchangeGroup::X25519))]
    #[case::openssl_p256("prime256v1", Some(KeyExchangeGroup::Secp256r1))]
    #[case::p384("secp384r1", Some(KeyExchangeGroup::Secp384r1))]
    fn ecdh_curve_selects_the_single_key_exchange_group(
        #[case] curve: &str,
        #[case] expected: Option<KeyExchangeGroup>,
    ) {
        let settings = HttpSettings {
            ssl_ecdh_curve: Some(curve.into()),
            ..HttpSettings::default()
        };
        let resolution = HttpClientConfig::resolve(&settings);
        assert_eq!(resolution.config.key_exchange_group, expected);
        assert_eq!(resolution.unsupported, []);
    }

    #[test]
    fn unsupported_ecdh_curve_keeps_the_defaults_and_is_reported() {
        let settings = HttpSettings {
            ssl_ecdh_curve: Some("secp521r1".into()),
            ..HttpSettings::default()
        };
        let resolution = HttpClientConfig::resolve(&settings);
        assert_eq!(resolution.config.key_exchange_group, None);
        assert_eq!(
            resolution.unsupported,
            [Unsupported::EcdhCurve("secp521r1".into())]
        );
    }

    #[test]
    fn legacy_security_level_keeps_every_suite_and_is_reported_unsupported() {
        let settings = HttpSettings {
            ssl_security_level: Some("DEFAULT@SECLEVEL=1".into()),
            ..HttpSettings::default()
        };
        let resolution = HttpClientConfig::resolve(&settings);
        assert_eq!(resolution.config.tls12_cipher_suites, None);
        assert_eq!(
            resolution.unsupported,
            [Unsupported::SecurityLevel("@SECLEVEL=1".into())]
        );
    }

    #[test]
    fn named_suites_restrict_tls12_and_unsupported_entries_are_reported() {
        let settings = HttpSettings {
            ssl_security_level: Some(
                "ECDHE-RSA-AES256-GCM-SHA384:ECDHE-ECDSA-AES128-GCM-SHA256:!aNULL:AES256-SHA@SECLEVEL=2"
                    .into(),
            ),
            ..HttpSettings::default()
        };
        let resolution = HttpClientConfig::resolve(&settings);
        assert_eq!(
            resolution.config.tls12_cipher_suites,
            Some(vec![
                Tls12CipherSuite::EcdheEcdsaAes128Gcm,
                Tls12CipherSuite::EcdheRsaAes256Gcm
            ])
        );
        assert_eq!(
            resolution.unsupported,
            [
                Unsupported::CipherToken("!aNULL".into()),
                Unsupported::CipherToken("AES256-SHA".into())
            ]
        );
    }

    #[test]
    fn connection_settings_carry_over_unchanged() {
        let keepalive = TcpKeepalive {
            idle: Duration::from_secs(60),
            interval: Duration::from_secs(30),
            retries: 5,
        };
        let settings = HttpSettings {
            ssl_certificate: Some("/client.pem".into()),
            force_ipv4: true,
            http2: true,
            user_agent: Some("litellm/1.0".into()),
            trust_proxy_env: true,
            connect_timeout: Duration::from_secs(7),
            tcp_keepalive: Some(keepalive),
            pool_idle_timeout: Duration::from_secs(45),
            ..HttpSettings::default()
        };
        let config = HttpClientConfig::resolve(&settings).config;
        assert_eq!(
            config,
            HttpClientConfig {
                verify: Verify::BuiltInRoots,
                client_certificate: Some("/client.pem".into()),
                key_exchange_group: None,
                tls12_cipher_suites: None,
                force_ipv4: true,
                http2: true,
                user_agent: Some("litellm/1.0".into()),
                trust_proxy_env: true,
                connect_timeout: Duration::from_secs(7),
                tcp_keepalive: Some(keepalive),
                pool_idle_timeout: Duration::from_secs(45),
            }
        );
    }

    #[rstest]
    #[case::aiohttp_default(HttpSettings::default(), true)]
    #[case::aiohttp_opted_out(HttpSettings { ignore_proxy_env: true, ..HttpSettings::default() }, false)]
    #[case::session_trust_env_beats_opt_out(
        HttpSettings { ignore_proxy_env: true, trust_proxy_env: true, ..HttpSettings::default() },
        true
    )]
    #[case::http2_uses_httpx(
        HttpSettings { ignore_proxy_env: true, http2: true, ..HttpSettings::default() },
        true
    )]
    #[case::aiohttp_disabled(
        HttpSettings { ignore_proxy_env: true, httpx_transport: true, ..HttpSettings::default() },
        true
    )]
    fn environment_proxies_apply_unless_the_aiohttp_transport_opts_out(
        #[case] settings: HttpSettings,
        #[case] expected: bool,
    ) {
        let config = HttpClientConfig::resolve(&settings).config;
        assert_eq!(config.trust_proxy_env, expected);
    }

    #[test]
    fn missing_ca_bundle_is_a_read_error() {
        let path = std::env::temp_dir().join("litellm-http-missing-bundle.pem");
        let config = HttpClientConfig {
            verify: Verify::CaBundle(path.clone()),
            ..HttpClientConfig::resolve(&HttpSettings::default()).config
        };
        assert!(matches!(
            config.client_builder(),
            Err(Error::Read { path: reported, .. }) if reported == path
        ));
    }

    #[test]
    fn non_pem_ca_bundle_is_an_invalid_pem_error() {
        let path =
            std::env::temp_dir().join(format!("litellm-http-not-pem-{}.pem", std::process::id()));
        std::fs::write(&path, b"not a certificate").unwrap();
        let config = HttpClientConfig {
            verify: Verify::CaBundle(path.clone()),
            ..HttpClientConfig::resolve(&HttpSettings::default()).config
        };
        let result = config.client_builder().map(drop);
        std::fs::remove_file(&path).unwrap();
        assert!(matches!(
            result,
            Err(Error::InvalidPem { path: reported, .. }) if reported == path
        ));
    }
}
