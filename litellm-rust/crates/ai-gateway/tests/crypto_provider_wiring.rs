//! Guards the wiring, not just the helper: a `wss://` dial through the public
//! API has to resolve its own crypto provider, in a test binary where nothing
//! has installed a process-wide one, and has to leave it uninstalled.

use std::collections::HashMap;
use std::time::Duration;

use litellm_ai_gateway::io::responses_ws::ResponsesWebSocketConnection;
use tokio::net::TcpListener;

async fn dead_tls_server() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind a loopback port");
    let port = listener
        .local_addr()
        .expect("read the bound address")
        .port();

    tokio::spawn(async move {
        while let Ok((stream, _peer)) = listener.accept().await {
            drop(stream);
        }
    });

    port
}

#[tokio::test]
async fn dialing_wss_returns_an_error_instead_of_panicking() {
    let port = dead_tls_server().await;

    let result = ResponsesWebSocketConnection::connect_url(
        &format!("wss://127.0.0.1:{port}/"),
        &HashMap::new(),
        Some(Duration::from_secs(10)),
    )
    .await;

    assert!(
        result.is_err(),
        "a plain TCP server cannot finish a TLS handshake"
    );
    assert!(
        rustls::crypto::CryptoProvider::get_default().is_none(),
        "the dial settles its provider on its own connector, not process-wide"
    );
}
