use std::{fmt, path::Path, str::FromStr, sync::Arc};

use rustls::{
    CipherSuite, ClientConfig, DigitallySignedStruct, RootCertStore, SignatureScheme,
    client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier},
    crypto::{CryptoProvider, SupportedKxGroup, ring},
    pki_types::{CertificateDer, PrivateKeyDer, ServerName, UnixTime, pem::PemObject},
};

use crate::{
    config::{HttpClientConfig, Verify},
    error::Error,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum KeyExchangeGroup {
    X25519,
    Secp256r1,
    Secp384r1,
}

impl FromStr for KeyExchangeGroup {
    type Err = Unsupported;

    fn from_str(name: &str) -> Result<Self, Self::Err> {
        match name.trim().to_ascii_lowercase().as_str() {
            "x25519" => Ok(Self::X25519),
            "prime256v1" | "secp256r1" | "p-256" => Ok(Self::Secp256r1),
            "secp384r1" | "p-384" => Ok(Self::Secp384r1),
            _ => Err(Unsupported::EcdhCurve(name.to_owned())),
        }
    }
}

impl KeyExchangeGroup {
    fn supported(self) -> &'static dyn SupportedKxGroup {
        match self {
            Self::X25519 => ring::kx_group::X25519,
            Self::Secp256r1 => ring::kx_group::SECP256R1,
            Self::Secp384r1 => ring::kx_group::SECP384R1,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Tls12CipherSuite {
    EcdheEcdsaAes128Gcm,
    EcdheEcdsaAes256Gcm,
    EcdheEcdsaChacha20,
    EcdheRsaAes128Gcm,
    EcdheRsaAes256Gcm,
    EcdheRsaChacha20,
}

impl FromStr for Tls12CipherSuite {
    type Err = Unsupported;

    fn from_str(name: &str) -> Result<Self, Self::Err> {
        match name {
            "ECDHE-ECDSA-AES128-GCM-SHA256" => Ok(Self::EcdheEcdsaAes128Gcm),
            "ECDHE-ECDSA-AES256-GCM-SHA384" => Ok(Self::EcdheEcdsaAes256Gcm),
            "ECDHE-ECDSA-CHACHA20-POLY1305" => Ok(Self::EcdheEcdsaChacha20),
            "ECDHE-RSA-AES128-GCM-SHA256" => Ok(Self::EcdheRsaAes128Gcm),
            "ECDHE-RSA-AES256-GCM-SHA384" => Ok(Self::EcdheRsaAes256Gcm),
            "ECDHE-RSA-CHACHA20-POLY1305" => Ok(Self::EcdheRsaChacha20),
            _ => Err(Unsupported::CipherToken(name.to_owned())),
        }
    }
}

impl Tls12CipherSuite {
    fn suite(self) -> CipherSuite {
        match self {
            Self::EcdheEcdsaAes128Gcm => CipherSuite::TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256,
            Self::EcdheEcdsaAes256Gcm => CipherSuite::TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384,
            Self::EcdheEcdsaChacha20 => CipherSuite::TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305_SHA256,
            Self::EcdheRsaAes128Gcm => CipherSuite::TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256,
            Self::EcdheRsaAes256Gcm => CipherSuite::TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384,
            Self::EcdheRsaChacha20 => CipherSuite::TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, thiserror::Error)]
pub enum Unsupported {
    #[error(
        "ssl_ecdh_curve {0:?} is not supported: rustls with ring only offers X25519, prime256v1 and secp384r1, so the default key exchange groups are used"
    )]
    EcdhCurve(String),
    #[error(
        "ssl_security_level {0:?} is not supported: rustls has one fixed security level, comparable to OpenSSL level 2, so legacy servers that need a lower level cannot be reached"
    )]
    SecurityLevel(String),
    #[error(
        "ssl_security_level entry {0:?} is not supported: rustls only offers ECDHE AEAD cipher suites, so the entry is ignored"
    )]
    CipherToken(String),
}

#[derive(Default)]
pub(crate) struct CipherSelection {
    pub(crate) tls12_cipher_suites: Option<Vec<Tls12CipherSuite>>,
    pub(crate) unsupported: Vec<Unsupported>,
}

enum CipherToken {
    Suite(Tls12CipherSuite),
    EverySuite,
    Ordering,
    Unsupported(Unsupported),
}

impl From<&str> for CipherToken {
    fn from(token: &str) -> Self {
        match token {
            "DEFAULT" | "ALL" | "HIGH" => Self::EverySuite,
            "@STRENGTH" | "@SECLEVEL=2" => Self::Ordering,
            level if level.starts_with("@SECLEVEL=") => {
                Self::Unsupported(Unsupported::SecurityLevel(level.to_owned()))
            }
            name => name.parse().map_or_else(Self::Unsupported, Self::Suite),
        }
    }
}

impl From<&str> for CipherSelection {
    fn from(value: &str) -> Self {
        let tokens: Vec<CipherToken> = tokenize(value)
            .iter()
            .map(|token| CipherToken::from(token.as_str()))
            .collect();
        let every_suite = tokens
            .iter()
            .any(|token| matches!(token, CipherToken::EverySuite));
        let mut suites: Vec<Tls12CipherSuite> = tokens
            .iter()
            .filter_map(|token| match token {
                CipherToken::Suite(suite) => Some(*suite),
                _ => None,
            })
            .collect();
        suites.sort_unstable();
        suites.dedup();
        CipherSelection {
            tls12_cipher_suites: (!every_suite && !suites.is_empty()).then_some(suites),
            unsupported: tokens
                .into_iter()
                .filter_map(|token| match token {
                    CipherToken::Unsupported(unsupported) => Some(unsupported),
                    _ => None,
                })
                .collect(),
        }
    }
}

fn tokenize(value: &str) -> Vec<String> {
    value
        .split([':', ',', ' '])
        .flat_map(|entry| match entry.split_once('@') {
            Some((name, command)) => vec![name.to_owned(), format!("@{command}")],
            None => vec![entry.to_owned()],
        })
        .filter(|token| !token.is_empty())
        .collect()
}

impl TryFrom<&HttpClientConfig> for ClientConfig {
    type Error = Error;

    fn try_from(config: &HttpClientConfig) -> Result<Self, Self::Error> {
        let base = ring::default_provider();
        let provider = Arc::new(CryptoProvider {
            kx_groups: config
                .key_exchange_group
                .map_or_else(|| base.kx_groups.clone(), |group| vec![group.supported()]),
            cipher_suites: base
                .cipher_suites
                .iter()
                .copied()
                .filter(|suite| {
                    suite.tls13().is_some()
                        || config.tls12_cipher_suites.as_ref().is_none_or(|allowed| {
                            allowed.iter().any(|a| a.suite() == suite.suite())
                        })
                })
                .collect(),
            ..base
        });
        let builder = ClientConfig::builder_with_provider(Arc::clone(&provider))
            .with_safe_default_protocol_versions()
            .map_err(|error| Error::Client(error.to_string()))?;
        let verified = match &config.verify {
            Verify::Disabled => builder
                .dangerous()
                .with_custom_certificate_verifier(Arc::new(NoVerification(provider))),
            Verify::BuiltInRoots => builder.with_root_certificates(RootCertStore {
                roots: webpki_roots::TLS_SERVER_ROOTS.to_vec(),
            }),
            Verify::CaBundle(path) => builder.with_root_certificates(bundle_roots(path)?),
        };
        let mut tls = match &config.client_certificate {
            None => verified.with_no_client_auth(),
            Some(path) => {
                let (chain, key) = identity(path)?;
                verified
                    .with_client_auth_cert(chain, key)
                    .map_err(|error| invalid_pem(path, error))?
            }
        };
        tls.alpn_protocols = if config.http2 {
            vec![b"h2".to_vec(), b"http/1.1".to_vec()]
        } else {
            vec![b"http/1.1".to_vec()]
        };
        Ok(tls)
    }
}

fn bundle_roots(path: &Path) -> Result<RootCertStore, Error> {
    let certificates = certificates(path)?;
    if certificates.is_empty() {
        return Err(invalid_pem(path, "no certificates found"));
    }
    let mut store = RootCertStore::empty();
    for certificate in certificates {
        store
            .add(certificate)
            .map_err(|error| invalid_pem(path, error))?;
    }
    Ok(store)
}

fn identity(path: &Path) -> Result<(Vec<CertificateDer<'static>>, PrivateKeyDer<'static>), Error> {
    let chain = certificates(path)?;
    if chain.is_empty() {
        return Err(invalid_pem(path, "no certificates found"));
    }
    let key =
        PrivateKeyDer::from_pem_slice(&read(path)?).map_err(|error| invalid_pem(path, error))?;
    Ok((chain, key))
}

fn certificates(path: &Path) -> Result<Vec<CertificateDer<'static>>, Error> {
    CertificateDer::pem_slice_iter(&read(path)?)
        .collect::<Result<_, _>>()
        .map_err(|error| invalid_pem(path, error))
}

fn read(path: &Path) -> Result<Vec<u8>, Error> {
    std::fs::read(path).map_err(|error| Error::Read {
        path: path.to_path_buf(),
        message: error.to_string(),
    })
}

fn invalid_pem(path: &Path, message: impl fmt::Display) -> Error {
    Error::InvalidPem {
        path: path.to_path_buf(),
        message: message.to_string(),
    }
}

#[derive(Debug)]
struct NoVerification(Arc<CryptoProvider>);

impl ServerCertVerifier for NoVerification {
    fn verify_server_cert(
        &self,
        _end_entity: &CertificateDer<'_>,
        _intermediates: &[CertificateDer<'_>],
        _server_name: &ServerName<'_>,
        _ocsp_response: &[u8],
        _now: UnixTime,
    ) -> Result<ServerCertVerified, rustls::Error> {
        Ok(ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        _message: &[u8],
        _cert: &CertificateDer<'_>,
        _dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, rustls::Error> {
        Ok(HandshakeSignatureValid::assertion())
    }

    fn verify_tls13_signature(
        &self,
        _message: &[u8],
        _cert: &CertificateDer<'_>,
        _dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, rustls::Error> {
        Ok(HandshakeSignatureValid::assertion())
    }

    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        self.0.signature_verification_algorithms.supported_schemes()
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use rustls::NamedGroup;

    use super::*;
    use crate::{HttpSettings, Resolution};

    fn config(settings: HttpSettings) -> HttpClientConfig {
        Resolution::from(&settings).config
    }

    fn offered_groups(tls: &ClientConfig) -> Vec<NamedGroup> {
        tls.crypto_provider()
            .kx_groups
            .iter()
            .map(|group| group.name())
            .collect()
    }

    fn offered_tls12_suites(tls: &ClientConfig) -> Vec<CipherSuite> {
        tls.crypto_provider()
            .cipher_suites
            .iter()
            .filter(|suite| suite.tls13().is_none())
            .map(|suite| suite.suite())
            .collect()
    }

    #[rstest]
    #[case("X25519", NamedGroup::X25519)]
    #[case("prime256v1", NamedGroup::secp256r1)]
    #[case("secp384r1", NamedGroup::secp384r1)]
    fn ecdh_curve_is_the_only_key_exchange_group_offered(
        #[case] curve: &str,
        #[case] expected: NamedGroup,
    ) {
        let tls = ClientConfig::try_from(&config(HttpSettings {
            ssl_ecdh_curve: Some(curve.into()),
            ..HttpSettings::default()
        }))
        .unwrap();
        assert_eq!(offered_groups(&tls), [expected]);
    }

    #[test]
    fn default_settings_offer_every_group_and_suite_of_the_provider() {
        let tls = ClientConfig::try_from(&config(HttpSettings::default())).unwrap();
        let provider = ring::default_provider();
        assert_eq!(offered_groups(&tls).len(), provider.kx_groups.len());
        assert_eq!(
            tls.crypto_provider().cipher_suites.len(),
            provider.cipher_suites.len()
        );
    }

    #[test]
    fn named_suites_are_the_only_tls12_suites_offered_and_tls13_stays() {
        let tls = ClientConfig::try_from(&config(HttpSettings {
            ssl_security_level: Some("ECDHE-RSA-AES256-GCM-SHA384".into()),
            ..HttpSettings::default()
        }))
        .unwrap();
        assert_eq!(
            offered_tls12_suites(&tls),
            [CipherSuite::TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384]
        );
        assert!(
            tls.crypto_provider()
                .cipher_suites
                .iter()
                .any(|suite| suite.tls13().is_some())
        );
    }

    #[rstest]
    #[case(true, &[b"h2".as_slice(), b"http/1.1".as_slice()])]
    #[case(false, &[b"http/1.1".as_slice()])]
    fn alpn_offers_h2_only_when_http2_is_on(#[case] http2: bool, #[case] expected: &[&[u8]]) {
        let tls = ClientConfig::try_from(&config(HttpSettings {
            http2,
            ..HttpSettings::default()
        }))
        .unwrap();
        assert_eq!(tls.alpn_protocols, expected);
    }

    #[test]
    fn client_certificate_without_a_private_key_is_an_invalid_pem_error() {
        let path = std::env::temp_dir().join(format!(
            "litellm-http-cert-without-key-{}.pem",
            std::process::id()
        ));
        std::fs::write(
            &path,
            b"-----BEGIN CERTIFICATE-----\nAA==\n-----END CERTIFICATE-----\n",
        )
        .unwrap();
        let result = ClientConfig::try_from(&HttpClientConfig {
            client_certificate: Some(path.clone()),
            ..config(HttpSettings::default())
        })
        .map(drop);
        std::fs::remove_file(&path).unwrap();
        assert!(matches!(
            result,
            Err(Error::InvalidPem { path: reported, .. }) if reported == path
        ));
    }
}
