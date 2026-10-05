use std::{
    io,
    sync::{Arc, OnceLock},
};

use rustls::{ClientConfig, RootCertStore};
use tokio::net::TcpStream;
use tokio_tungstenite::{
    Connector, MaybeTlsStream, WebSocketStream, connect_async_tls_with_config,
    tungstenite::{client::IntoClientRequest, error::TlsError, handshake::client::Response},
};

pub type UpstreamWebSocket = WebSocketStream<MaybeTlsStream<TcpStream>>;

static TLS_CONFIG: OnceLock<Arc<ClientConfig>> = OnceLock::new();

fn build_tls_config() -> Result<ClientConfig, Box<tokio_tungstenite::tungstenite::Error>> {
    let native = rustls_native_certs::load_native_certs();
    let mut store = RootCertStore::empty();
    let (added, _ignored) = store.add_parsable_certificates(native.certs);
    if added == 0 {
        return Err(Box::new(tokio_tungstenite::tungstenite::Error::Io(
            io::Error::other(format!(
                "no usable native root certificates: {:?}",
                native.errors
            )),
        )));
    }
    ClientConfig::builder_with_provider(Arc::new(rustls::crypto::ring::default_provider()))
        .with_safe_default_protocol_versions()
        .map(|builder| builder.with_root_certificates(store).with_no_client_auth())
        .map_err(|error| {
            Box::new(tokio_tungstenite::tungstenite::Error::Tls(
                TlsError::Rustls(error),
            ))
        })
}

fn tls_config() -> Result<Arc<ClientConfig>, Box<tokio_tungstenite::tungstenite::Error>> {
    if let Some(config) = TLS_CONFIG.get() {
        return Ok(Arc::clone(config));
    }
    let built = Arc::new(build_tls_config()?);
    Ok(Arc::clone(TLS_CONFIG.get_or_init(|| built)))
}

#[tracing::instrument(name = "litellm.websocket.handshake", level = "debug", skip_all)]
pub async fn connect_upstream<R>(
    request: R,
) -> Result<(UpstreamWebSocket, Response), Box<tokio_tungstenite::tungstenite::Error>>
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
