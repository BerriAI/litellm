//! Outbound WebSocket dials over a TLS config this crate builds once and owns.
//!
//! `reqwest/rustls-tls` enables `rustls/ring` and `litellm-core`'s `bedrock-auth`
//! enables `rustls/aws-lc-rs`, so the bare `ClientConfig::builder()` that
//! `tokio-tungstenite` uses when handed no connector panics rather than guess
//! between them. Naming ring on a connector of our own settles that for these
//! dials without touching the process-wide default, and building the config
//! once keeps the platform trust store, which `tokio-tungstenite` would
//! otherwise re-read on every dial, off the dial path.

use std::io;
use std::sync::{Arc, OnceLock};

use rustls::{ClientConfig, RootCertStore};
use tokio::net::TcpStream;
use tokio_tungstenite::tungstenite::Error;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::error::TlsError;
use tokio_tungstenite::tungstenite::handshake::client::Response;
use tokio_tungstenite::{
    Connector, MaybeTlsStream, WebSocketStream, connect_async_tls_with_config,
};

static TLS_CONFIG: OnceLock<Arc<ClientConfig>> = OnceLock::new();

fn build_config() -> Result<ClientConfig, Box<Error>> {
    let native = rustls_native_certs::load_native_certs();
    let roots = {
        let mut store = RootCertStore::empty();
        let (added, _ignored) = store.add_parsable_certificates(native.certs);
        if added == 0 {
            return Err(Box::new(Error::Io(io::Error::other(format!(
                "no usable native root certificates: {:?}",
                native.errors
            )))));
        }
        store
    };

    ClientConfig::builder_with_provider(Arc::new(rustls::crypto::ring::default_provider()))
        .with_safe_default_protocol_versions()
        .map(|builder| builder.with_root_certificates(roots).with_no_client_auth())
        .map_err(|error| Box::new(Error::Tls(TlsError::Rustls(error))))
}

fn tls_config() -> Result<Arc<ClientConfig>, Box<Error>> {
    if let Some(config) = TLS_CONFIG.get() {
        return Ok(Arc::clone(config));
    }
    let built = Arc::new(build_config()?);
    Ok(Arc::clone(TLS_CONFIG.get_or_init(|| built)))
}

pub(crate) async fn connect_upstream<R>(
    request: R,
) -> Result<(WebSocketStream<MaybeTlsStream<TcpStream>>, Response), Box<Error>>
where
    R: IntoClientRequest + Unpin,
{
    let request = request.into_client_request().map_err(Box::new)?;
    let connector = match request.uri().scheme_str() {
        Some("wss") => Some(Connector::Rustls(tls_config()?)),
        _ => None,
    };
    connect_async_tls_with_config(request, None, false, connector)
        .await
        .map_err(Box::new)
}

#[cfg(test)]
mod tests {
    use super::build_config;

    #[test]
    fn builds_a_usable_config_with_both_provider_features_enabled() {
        let config = build_config().expect("a client config");

        assert!(!config.crypto_provider().cipher_suites.is_empty());
    }
}
