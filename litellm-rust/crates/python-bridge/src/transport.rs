use std::sync::OnceLock;
use std::time::Duration;

use litellm_core::error::{Error, TransportError};

const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

struct BridgeTransport {
    /// Automatic redirects must stay disabled: OCR validates each document
    /// redirect target before following it.
    ocr: reqwest::Client,
}

impl BridgeTransport {
    fn new() -> Result<Self, reqwest::Error> {
        Ok(Self {
            ocr: reqwest::Client::builder()
                .connect_timeout(CONNECT_TIMEOUT)
                .redirect(reqwest::redirect::Policy::none())
                .build()?,
        })
    }
}

/// Borrow the Python SDK's native OCR transport at the Rust/Python boundary.
///
/// `reqwest::Client` is internally reference counted, so this clone lets a
/// `'static` bridge future share the boundary-owned connection pool.
pub(crate) fn ocr_http_client() -> Result<reqwest::Client, Error> {
    static TRANSPORT: OnceLock<Result<BridgeTransport, String>> = OnceLock::new();
    TRANSPORT
        .get_or_init(|| BridgeTransport::new().map_err(|error| error.to_string()))
        .as_ref()
        .map(|transport| transport.ocr.clone())
        .map_err(|error| TransportError::Network(error.clone()).into())
}
