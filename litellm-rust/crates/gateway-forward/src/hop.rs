use axum::http::{HeaderMap, HeaderValue, header};

// Per-connection headers (RFC 9110 section 7.6.1) describe the client's connection to Rust, not
// Rust's connection to Python, so a proxy must drop them instead of forwarding them.
pub(crate) fn strip(headers: &mut HeaderMap) {
    let connection_headers: Vec<_> = headers
        .get_all(header::CONNECTION)
        .iter()
        .filter_map(|value| value.to_str().ok())
        .flat_map(|value| value.split(','))
        .filter_map(|name| header::HeaderName::from_bytes(name.trim().as_bytes()).ok())
        .collect();
    for name in connection_headers {
        headers.remove(name);
    }
    for name in [
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
        "proxy-connection",
    ] {
        headers.remove(name);
    }
}

// Stripping drops the handshake's Connection and Upgrade, so each hop gets them back.
pub(crate) fn restore_websocket_handshake(headers: &mut HeaderMap) {
    headers.insert(header::CONNECTION, HeaderValue::from_static("upgrade"));
    headers.insert(header::UPGRADE, HeaderValue::from_static("websocket"));
}
