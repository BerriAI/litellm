//! Outbound WebSocket dials, with the rustls crypto provider settled first.
//!
//! `reqwest/rustls-tls` enables `rustls/ring` and `litellm-core`'s `bedrock-auth`
//! enables `rustls/aws-lc-rs`, so `ClientConfig::builder()` — which is how
//! `tokio-tungstenite` builds its TLS config — panics rather than guess between
//! them. reqwest and the AWS SDK pick a provider explicitly and never panic.
//!
//! Installing from the dial rather than from a `main` also covers the `cdylib`
//! the Python bridge loads, the tests, and the benches, none of which have one.
//! ring is what reqwest already falls back to, so installing it changes no
//! working path, and whoever installs into this rustls build first still wins.

use std::sync::Once;

use tokio::net::TcpStream;
use tokio_tungstenite::tungstenite::Error;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::handshake::client::Response;
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream, connect_async};

static INSTALL_CRYPTO_PROVIDER: Once = Once::new();

pub(crate) fn ensure_crypto_provider() {
    INSTALL_CRYPTO_PROVIDER.call_once(|| {
        let _ = rustls::crypto::ring::default_provider().install_default();
    });
}

pub(crate) async fn connect_upstream<R>(
    request: R,
) -> Result<(WebSocketStream<MaybeTlsStream<TcpStream>>, Response), Box<Error>>
where
    R: IntoClientRequest + Unpin,
{
    ensure_crypto_provider();
    connect_async(request).await.map_err(Box::new)
}

#[cfg(test)]
mod tests {
    use rustls::crypto::CryptoProvider;

    use super::ensure_crypto_provider;

    fn fingerprint(
        provider: &CryptoProvider,
    ) -> (Vec<rustls::CipherSuite>, Vec<rustls::NamedGroup>) {
        (
            provider
                .cipher_suites
                .iter()
                .map(|suite| suite.suite())
                .collect(),
            provider
                .kx_groups
                .iter()
                .map(|group| group.name())
                .collect(),
        )
    }

    #[test]
    fn client_config_builder_works_with_both_provider_features_enabled() {
        ensure_crypto_provider();

        assert!(CryptoProvider::get_default().is_some());

        let config = rustls::ClientConfig::builder()
            .with_root_certificates(rustls::RootCertStore::empty())
            .with_no_client_auth();

        assert!(!config.crypto_provider().cipher_suites.is_empty());
    }

    #[test]
    fn installs_ring_rather_than_aws_lc_rs() {
        ensure_crypto_provider();

        let installed = CryptoProvider::get_default().expect("a provider is installed");

        assert_eq!(
            fingerprint(installed),
            fingerprint(&rustls::crypto::ring::default_provider())
        );
        assert_ne!(
            fingerprint(installed),
            fingerprint(&rustls::crypto::aws_lc_rs::default_provider())
        );
    }

    #[test]
    fn ensure_crypto_provider_is_idempotent() {
        ensure_crypto_provider();
        let first = CryptoProvider::get_default().cloned();

        ensure_crypto_provider();
        let second = CryptoProvider::get_default().cloned();

        assert!(first.is_some());
        assert!(std::sync::Arc::ptr_eq(&first.unwrap(), &second.unwrap()));
    }
}
