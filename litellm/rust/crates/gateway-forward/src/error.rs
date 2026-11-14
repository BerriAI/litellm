use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid forwarding upstream URI")]
    UpstreamUri(#[source] axum::http::uri::InvalidUri),
    #[error(
        "LITELLM_FORWARD_UPSTREAM must be a loopback IPv4 http origin without a path, query or credentials"
    )]
    Upstream,
    #[error("{0} must be set in forward mode")]
    MissingSetting(String),
    #[error("invalid forwarding setting: {0}")]
    InvalidSetting(String),
    #[error("forwarding gateway I/O failed: {0}")]
    Io(#[from] std::io::Error),
}

/// Why a request got an answer from Rust instead of from Python.
#[derive(Debug, thiserror::Error)]
pub(crate) enum Rejection {
    #[error("HTTP/1.1 request without a Host header")]
    MissingHost,
    #[error("request target has no path to forward")]
    NoPath,
    #[error("request uses a transfer coding other than chunked")]
    UnsupportedTransferCoding,
    #[error("client request body failed: {0}")]
    ClientBody(#[source] hyper_util::client::legacy::Error),
    #[error("python proxy unreachable: {0}")]
    Unreachable(#[source] hyper_util::client::legacy::Error),
    #[error("python proxy switched protocols without being asked to")]
    UnrequestedUpgrade,
    #[error("python proxy websocket upgrade failed: {0}")]
    UpgradeFailed(#[source] hyper::Error),
    #[error("forwarding gateway is shutting down")]
    ShuttingDown,
    #[error("could not build the request for the python proxy")]
    Internal,
}

impl Rejection {
    /// A client body that ends early or breaks its framing fails the upstream request too, but
    /// the fault is the client's, not Python's.
    pub fn from_upstream(error: hyper_util::client::legacy::Error) -> Self {
        let client_body = std::error::Error::source(&error)
            .and_then(|source| source.downcast_ref::<hyper::Error>())
            .is_some_and(hyper::Error::is_user);
        match client_body {
            true => Self::ClientBody(error),
            false => Self::Unreachable(error),
        }
    }

    fn status(&self) -> StatusCode {
        match self {
            Self::MissingHost
            | Self::NoPath
            | Self::UnsupportedTransferCoding
            | Self::ClientBody(_) => StatusCode::BAD_REQUEST,
            Self::Unreachable(_) | Self::UnrequestedUpgrade | Self::UpgradeFailed(_) => {
                StatusCode::BAD_GATEWAY
            }
            Self::ShuttingDown => StatusCode::SERVICE_UNAVAILABLE,
            Self::Internal => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    fn is_gateway_fault(&self) -> bool {
        matches!(
            self,
            Self::Unreachable(_)
                | Self::UnrequestedUpgrade
                | Self::UpgradeFailed(_)
                | Self::Internal
        )
    }
}

impl IntoResponse for Rejection {
    fn into_response(self) -> Response {
        match self.is_gateway_fault() {
            true => tracing::warn!(error = %self, "request not forwarded"),
            false => tracing::debug!(error = %self, "request not forwarded"),
        }
        self.status().into_response()
    }
}
