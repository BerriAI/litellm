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

impl From<&HttpSettings> for Verify {
    fn from(settings: &HttpSettings) -> Self {
        match &settings.ssl_verify {
            Some(SslVerify::Disabled) => Self::Disabled,
            Some(SslVerify::CaBundle(path)) => Self::CaBundle(path.clone()),
            Some(SslVerify::Enabled) | None => settings
                .ssl_cert_file
                .clone()
                .map_or(Self::BuiltInRoots, Self::CaBundle),
        }
    }
}

impl From<&HttpSettings> for Resolution {
    fn from(settings: &HttpSettings) -> Self {
        let curve = settings
            .ssl_ecdh_curve
            .as_deref()
            .map(str::parse::<KeyExchangeGroup>)
            .transpose();
        let ciphers = settings
            .ssl_security_level
            .as_deref()
            .map(CipherSelection::from)
            .unwrap_or_default();
        Self {
            config: HttpClientConfig {
                verify: Verify::from(settings),
                client_certificate: settings.ssl_certificate.clone(),
                key_exchange_group: curve.clone().ok().flatten(),
                tls12_cipher_suites: ciphers.tls12_cipher_suites,
                force_ipv4: settings.force_ipv4,
                http2: settings.http2,
                user_agent: settings.user_agent.clone(),
                trust_proxy_env: settings.trust_proxy_env,
                connect_timeout: settings.connect_timeout,
                tcp_keepalive: settings.tcp_keepalive,
                pool_idle_timeout: settings.pool_idle_timeout,
            },
            unsupported: curve.err().into_iter().chain(ciphers.unsupported).collect(),
        }
    }
}

impl TryFrom<&HttpClientConfig> for reqwest::ClientBuilder {
    type Error = Error;

    fn try_from(config: &HttpClientConfig) -> Result<Self, Self::Error> {
        let base = reqwest::Client::builder()
            .use_preconfigured_tls(rustls::ClientConfig::try_from(config)?)
            .connect_timeout(config.connect_timeout)
            .pool_idle_timeout(config.pool_idle_timeout);
        let with_keepalive = match config.tcp_keepalive {
            None => base,
            Some(keepalive) => base
                .tcp_keepalive(keepalive.idle)
                .tcp_keepalive_interval(keepalive.interval)
                .tcp_keepalive_retries(keepalive.retries),
        };
        let with_address = if config.force_ipv4 {
            with_keepalive.local_address(IpAddr::V4(Ipv4Addr::UNSPECIFIED))
        } else {
            with_keepalive
        };
        let with_protocol = if config.http2 {
            with_address
        } else {
            with_address.http1_only()
        };
        let with_agent = match &config.user_agent {
            Some(agent) => with_protocol.user_agent(agent),
            None => with_protocol,
        };
        Ok(if config.trust_proxy_env {
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

    fn settings(ssl_verify: Option<SslVerify>, ssl_cert_file: Option<&str>) -> HttpSettings {
        HttpSettings {
            ssl_verify,
            ssl_cert_file: ssl_cert_file.map(PathBuf::from),
            ..HttpSettings::default()
        }
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
        let config = Resolution::from(&settings).config;
        assert_eq!(config.verify, expected);
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
        let resolution = Resolution::from(&settings);
        assert_eq!(resolution.config.key_exchange_group, expected);
        assert_eq!(resolution.unsupported, []);
    }

    #[test]
    fn unsupported_ecdh_curve_keeps_the_defaults_and_is_reported() {
        let settings = HttpSettings {
            ssl_ecdh_curve: Some("secp521r1".into()),
            ..HttpSettings::default()
        };
        let resolution = Resolution::from(&settings);
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
        let resolution = Resolution::from(&settings);
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
        let resolution = Resolution::from(&settings);
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
        let config = Resolution::from(&settings).config;
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

    #[test]
    fn missing_ca_bundle_is_a_read_error() {
        let path = std::env::temp_dir().join("litellm-http-missing-bundle.pem");
        let config = HttpClientConfig {
            verify: Verify::CaBundle(path.clone()),
            ..Resolution::from(&HttpSettings::default()).config
        };
        assert!(matches!(
            reqwest::ClientBuilder::try_from(&config),
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
            ..Resolution::from(&HttpSettings::default()).config
        };
        let result = reqwest::ClientBuilder::try_from(&config).map(drop);
        std::fs::remove_file(&path).unwrap();
        assert!(matches!(
            result,
            Err(Error::InvalidPem { path: reported, .. }) if reported == path
        ));
    }
}
