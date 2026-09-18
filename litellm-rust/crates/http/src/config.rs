use std::{
    net::{IpAddr, Ipv4Addr},
    path::{Path, PathBuf},
    time::Duration,
};

use crate::settings::{HttpSettings, SslVerify};

#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
pub enum Error {
    #[error("{setting} cannot be expressed with rustls: {reason}")]
    Unsupported {
        setting: &'static str,
        reason: String,
    },
    #[error("could not read {}: {message}", path.display())]
    Read { path: PathBuf, message: String },
    #[error("{} is not a PEM file: {message}", path.display())]
    InvalidPem { path: PathBuf, message: String },
    #[error("could not build the HTTP client: {0}")]
    Client(String),
}

impl From<reqwest::Error> for Error {
    fn from(error: reqwest::Error) -> Self {
        Self::Client(error.without_url().to_string())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Verify {
    Disabled,
    CaBundle(PathBuf),
    BuiltInRoots,
}

/// One fully resolved client configuration. Every field is a plain value so the pool can
/// key cached clients on it.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct HttpClientConfig {
    pub verify: Verify,
    pub client_certificate: Option<PathBuf>,
    pub force_ipv4: bool,
    pub http2: bool,
    pub user_agent: Option<String>,
    pub trust_proxy_env: bool,
    pub connect_timeout: Duration,
    pub request_timeout: Option<Duration>,
}

impl HttpClientConfig {
    /// Port of `get_ssl_verify` + `get_ssl_configuration`: the per-call value wins, then the
    /// configured (environment-overlaid) `ssl_verify`, then `SSL_CERT_FILE`, then the built-in
    /// roots. Settings rustls has no equivalent for are an error instead of a silent no-op.
    pub fn resolve(
        settings: &HttpSettings,
        per_call_ssl_verify: Option<&SslVerify>,
    ) -> Result<Self, Error> {
        if let Some(level) = &settings.ssl_security_level {
            return Err(Error::Unsupported {
                setting: "ssl_security_level",
                reason: format!("OpenSSL cipher string {level:?} has no rustls equivalent"),
            });
        }
        if let Some(curve) = &settings.ssl_ecdh_curve {
            return Err(Error::Unsupported {
                setting: "ssl_ecdh_curve",
                reason: format!("key exchange group {curve:?} is fixed by the rustls provider"),
            });
        }
        let verify = match per_call_ssl_verify.or(settings.ssl_verify.as_ref()) {
            Some(SslVerify::Disabled) => Verify::Disabled,
            Some(SslVerify::CaBundle(path)) => Verify::CaBundle(path.clone()),
            Some(SslVerify::Enabled) | None => settings
                .ssl_cert_file
                .clone()
                .map_or(Verify::BuiltInRoots, Verify::CaBundle),
        };
        Ok(Self {
            verify,
            client_certificate: settings.ssl_certificate.clone(),
            force_ipv4: settings.force_ipv4,
            http2: settings.http2,
            user_agent: settings.user_agent.clone(),
            trust_proxy_env: settings.trust_proxy_env,
            connect_timeout: settings.connect_timeout,
            request_timeout: settings.request_timeout,
        })
    }

    /// A builder carrying every shared setting; variants add their own policy on top.
    pub fn client_builder(&self) -> Result<reqwest::ClientBuilder, Error> {
        let base = reqwest::Client::builder().connect_timeout(self.connect_timeout);
        let with_roots = match &self.verify {
            Verify::Disabled => base.danger_accept_invalid_certs(true),
            Verify::BuiltInRoots => base,
            Verify::CaBundle(path) => {
                let pem = read(path)?;
                let certificates =
                    reqwest::Certificate::from_pem_bundle(&pem).map_err(|error| {
                        Error::InvalidPem {
                            path: path.clone(),
                            message: error.without_url().to_string(),
                        }
                    })?;
                if certificates.is_empty() {
                    return Err(Error::InvalidPem {
                        path: path.clone(),
                        message: "no certificates found".into(),
                    });
                }
                certificates.into_iter().fold(
                    base.tls_built_in_root_certs(false),
                    |builder, certificate| builder.add_root_certificate(certificate),
                )
            }
        };
        let with_identity = match &self.client_certificate {
            None => with_roots,
            Some(path) => {
                let identity = reqwest::Identity::from_pem(&read(path)?).map_err(|error| {
                    Error::InvalidPem {
                        path: path.clone(),
                        message: error.without_url().to_string(),
                    }
                })?;
                with_roots.identity(identity)
            }
        };
        let with_address = if self.force_ipv4 {
            with_identity.local_address(IpAddr::V4(Ipv4Addr::UNSPECIFIED))
        } else {
            with_identity
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
        let with_proxy = if self.trust_proxy_env {
            with_agent
        } else {
            with_agent.no_proxy()
        };
        Ok(match self.request_timeout {
            Some(timeout) => with_proxy.timeout(timeout),
            None => with_proxy,
        })
    }
}

fn read(path: &Path) -> Result<Vec<u8>, Error> {
    std::fs::read(path).map_err(|error| Error::Read {
        path: path.to_path_buf(),
        message: error.to_string(),
    })
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
    #[case::default(settings(None, None), None, Verify::BuiltInRoots)]
    #[case::setting_disables(
        settings(Some(SslVerify::Disabled), Some("/env/roots.pem")),
        None,
        Verify::Disabled
    )]
    #[case::setting_bundle(
        settings(Some(SslVerify::CaBundle("/configured.pem".into())), Some("/env/roots.pem")),
        None,
        Verify::CaBundle("/configured.pem".into())
    )]
    #[case::enabled_uses_cert_file(
        settings(Some(SslVerify::Enabled), Some("/env/roots.pem")),
        None,
        Verify::CaBundle("/env/roots.pem".into())
    )]
    #[case::unset_uses_cert_file(settings(None, Some("/env/roots.pem")), None, Verify::CaBundle("/env/roots.pem".into()))]
    #[case::per_call_beats_setting(
        settings(Some(SslVerify::Disabled), None),
        Some(SslVerify::Enabled),
        Verify::BuiltInRoots
    )]
    #[case::per_call_disables(
        settings(Some(SslVerify::CaBundle("/configured.pem".into())), Some("/env/roots.pem")),
        Some(SslVerify::Disabled),
        Verify::Disabled
    )]
    #[case::per_call_bundle(
        settings(None, Some("/env/roots.pem")),
        Some(SslVerify::CaBundle("/call.pem".into())),
        Verify::CaBundle("/call.pem".into())
    )]
    #[case::per_call_enabled_still_honours_cert_file(
        settings(Some(SslVerify::Disabled), Some("/env/roots.pem")),
        Some(SslVerify::Enabled),
        Verify::CaBundle("/env/roots.pem".into())
    )]
    fn verify_follows_per_call_then_setting_then_cert_file(
        #[case] settings: HttpSettings,
        #[case] per_call: Option<SslVerify>,
        #[case] expected: Verify,
    ) {
        let config = HttpClientConfig::resolve(&settings, per_call.as_ref()).unwrap();
        assert_eq!(config.verify, expected);
    }

    #[test]
    fn ssl_verify_environment_variable_beats_the_configured_setting() {
        let settings = HttpSettings {
            ssl_verify: Some(SslVerify::Disabled),
            ..HttpSettings::default()
        }
        .with_environment(&|name: &str| (name == "SSL_VERIFY").then(|| "true".to_string()));
        let config = HttpClientConfig::resolve(&settings, None).unwrap();
        assert_eq!(config.verify, Verify::BuiltInRoots);
    }

    #[test]
    fn cipher_strings_are_rejected_rather_than_ignored() {
        let settings = HttpSettings {
            ssl_security_level: Some("DEFAULT@SECLEVEL=1".into()),
            ..HttpSettings::default()
        };
        assert!(matches!(
            HttpClientConfig::resolve(&settings, None),
            Err(Error::Unsupported {
                setting: "ssl_security_level",
                ..
            })
        ));
    }

    #[test]
    fn ecdh_curves_are_rejected_rather_than_ignored() {
        let settings = HttpSettings {
            ssl_ecdh_curve: Some("X25519".into()),
            ..HttpSettings::default()
        };
        assert!(matches!(
            HttpClientConfig::resolve(&settings, None),
            Err(Error::Unsupported {
                setting: "ssl_ecdh_curve",
                ..
            })
        ));
    }

    #[test]
    fn connection_settings_carry_over_unchanged() {
        let settings = HttpSettings {
            ssl_certificate: Some("/client.pem".into()),
            force_ipv4: true,
            http2: true,
            user_agent: Some("litellm/1.0".into()),
            trust_proxy_env: true,
            connect_timeout: Duration::from_secs(7),
            request_timeout: Some(Duration::from_secs(70)),
            ..HttpSettings::default()
        };
        let config = HttpClientConfig::resolve(&settings, None).unwrap();
        assert_eq!(
            config,
            HttpClientConfig {
                verify: Verify::BuiltInRoots,
                client_certificate: Some("/client.pem".into()),
                force_ipv4: true,
                http2: true,
                user_agent: Some("litellm/1.0".into()),
                trust_proxy_env: true,
                connect_timeout: Duration::from_secs(7),
                request_timeout: Some(Duration::from_secs(70)),
            }
        );
    }

    #[test]
    fn missing_ca_bundle_is_a_read_error() {
        let path = std::env::temp_dir().join("litellm-http-missing-bundle.pem");
        let config = HttpClientConfig {
            verify: Verify::CaBundle(path.clone()),
            ..HttpClientConfig::resolve(&HttpSettings::default(), None).unwrap()
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
            ..HttpClientConfig::resolve(&HttpSettings::default(), None).unwrap()
        };
        let result = config.client_builder().map(drop);
        std::fs::remove_file(&path).unwrap();
        assert!(matches!(
            result,
            Err(Error::InvalidPem { path: reported, .. }) if reported == path
        ));
    }
}
