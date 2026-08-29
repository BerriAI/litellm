use thiserror::Error;

pub type CoreResult<T> = Result<T, CoreError>;

/// Top-level error for every core entrypoint. The layer is the contract:
///
/// * [`CoreError::Request`] — the provider call never went out. Nothing was
///   billed, so a host that keeps a reference implementation may serve or
///   retry the request itself.
/// * [`CoreError::Upstream`] — the provider call was (or may have been)
///   issued. It must never be blindly retried.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum CoreError {
    #[error(transparent)]
    Request(#[from] RequestError),
    #[error(transparent)]
    Upstream(#[from] UpstreamError),
}

/// Failures before the provider HTTP call is issued: request validation,
/// provider resolution, auth setup, routing, or connection establishment.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum RequestError {
    #[error("expected {expected}, got {actual}")]
    InvalidType {
        expected: &'static str,
        actual: &'static str,
    },
    #[error("missing required field: {0}")]
    MissingField(&'static str),
    #[error("invalid provider: {0}")]
    InvalidProvider(String),
    #[error("invalid request: {0}")]
    InvalidRequest(String),
    #[error("{0}")]
    Auth(String),
    /// The provider was never reached: DNS, TCP, TLS or proxy setup failed
    /// before any byte of the request went out. Nothing was billed, so a host
    /// that keeps a reference implementation can serve the request itself.
    /// A timeout is deliberately not this, since the provider may have received
    /// and answered the request already.
    #[error("could not reach the provider: {0}")]
    Connect(String),
    #[error("routing error: {0}")]
    Routing(String),
    /// The request is outside the surface this route covers in Rust. Hosts that
    /// keep a reference implementation treat this as "fall back", not "fail".
    #[error("unsupported by the rust path: {0}")]
    Unsupported(&'static str),
}

/// Failures after the provider HTTP call was issued: the request went over
/// the wire, so the provider may have received and billed it.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum UpstreamError {
    #[error("upstream request failed with status {status}: {body}")]
    Http {
        status: u16,
        /// Already truncated to `UPSTREAM_ERROR_BODY_MAX_CHARS` by the caller.
        body: String,
    },
    /// A timeout is deliberately this, not [`RequestError::Connect`], since
    /// the provider may have received and answered the request already.
    #[error("upstream network error: {0}")]
    Network(String),
    #[error("invalid response: {0}")]
    InvalidResponse(String),
}

impl CoreError {
    pub fn invalid_type(expected: &'static str, actual: &'static str) -> Self {
        Self::Request(RequestError::InvalidType { expected, actual })
    }

    pub fn missing_field(field: &'static str) -> Self {
        Self::Request(RequestError::MissingField(field))
    }

    pub fn invalid_provider(provider: impl Into<String>) -> Self {
        Self::Request(RequestError::InvalidProvider(provider.into()))
    }

    pub fn invalid_request(message: impl Into<String>) -> Self {
        Self::Request(RequestError::InvalidRequest(message.into()))
    }

    pub fn auth(message: impl Into<String>) -> Self {
        Self::Request(RequestError::Auth(message.into()))
    }

    pub fn connect(message: impl Into<String>) -> Self {
        Self::Request(RequestError::Connect(message.into()))
    }

    pub fn routing(message: impl Into<String>) -> Self {
        Self::Request(RequestError::Routing(message.into()))
    }

    pub fn unsupported(what: &'static str) -> Self {
        Self::Request(RequestError::Unsupported(what))
    }

    pub fn http(status: u16, body: impl Into<String>) -> Self {
        Self::Upstream(UpstreamError::Http {
            status,
            body: body.into(),
        })
    }

    pub fn network(message: impl Into<String>) -> Self {
        Self::Upstream(UpstreamError::Network(message.into()))
    }

    pub fn invalid_response(message: impl Into<String>) -> Self {
        Self::Upstream(UpstreamError::InvalidResponse(message.into()))
    }

    /// True when the provider call was (or may have been) issued and the
    /// request must not be blindly retried.
    pub fn is_upstream(&self) -> bool {
        matches!(self, Self::Upstream(_))
    }

    /// Stable machine-readable name for logging and metrics.
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Request(error) => error.kind(),
            Self::Upstream(error) => error.kind(),
        }
    }
}

impl RequestError {
    pub fn kind(&self) -> &'static str {
        match self {
            Self::InvalidType { .. } => "InvalidType",
            Self::MissingField(_) => "MissingField",
            Self::InvalidProvider(_) => "InvalidProvider",
            Self::InvalidRequest(_) => "InvalidRequest",
            Self::Auth(_) => "AuthError",
            Self::Connect(_) => "ConnectError",
            Self::Routing(_) => "RoutingError",
            Self::Unsupported(_) => "UnsupportedRequest",
        }
    }
}

impl UpstreamError {
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Http { .. } => "HttpError",
            Self::Network(_) => "NetworkError",
            Self::InvalidResponse(_) => "InvalidResponse",
        }
    }

    /// A required field is absent from a body the provider already returned.
    pub fn missing_field(field: &'static str) -> Self {
        Self::InvalidResponse(format!("missing required field: {field}"))
    }

    /// The body the provider returned is not the shape this route translates.
    pub fn invalid_type(expected: &'static str, actual: &'static str) -> Self {
        Self::InvalidResponse(format!("expected {expected}, got {actual}"))
    }

    /// The provider returned something this route never asked for, so the
    /// response cannot be normalized.
    pub fn unsupported(what: &'static str) -> Self {
        Self::InvalidResponse(format!("unsupported by the rust path: {what}"))
    }
}

pub fn json_type_name(value: &serde_json::Value) -> &'static str {
    match value {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "bool",
        serde_json::Value::Number(_) => "number",
        serde_json::Value::String(_) => "string",
        serde_json::Value::Array(_) => "array",
        serde_json::Value::Object(_) => "object",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layers_carry_the_billing_contract() {
        let request = CoreError::invalid_request("nope");
        let upstream = CoreError::http(429, "slow down");
        assert!(!request.is_upstream());
        assert!(upstream.is_upstream());
    }

    #[test]
    fn kinds_are_stable_names_for_logging() {
        let kinds = [
            (CoreError::invalid_type("object", "string"), "InvalidType"),
            (CoreError::missing_field("model"), "MissingField"),
            (CoreError::invalid_provider("openai"), "InvalidProvider"),
            (CoreError::invalid_request("bad"), "InvalidRequest"),
            (CoreError::auth("no key"), "AuthError"),
            (CoreError::connect("refused"), "ConnectError"),
            (CoreError::routing("no route"), "RoutingError"),
            (CoreError::unsupported("streaming"), "UnsupportedRequest"),
            (CoreError::http(500, "boom"), "HttpError"),
            (CoreError::network("timed out"), "NetworkError"),
            (CoreError::invalid_response("bad json"), "InvalidResponse"),
        ];
        for (error, kind) in kinds {
            assert_eq!(error.kind(), kind);
        }
    }

    #[test]
    fn display_matches_the_previous_flat_enum_messages() {
        assert_eq!(
            CoreError::missing_field("model").to_string(),
            "missing required field: model"
        );
        assert_eq!(
            CoreError::http(429, "slow down").to_string(),
            "upstream request failed with status 429: slow down"
        );
        assert_eq!(
            CoreError::connect("refused").to_string(),
            "could not reach the provider: refused"
        );
    }
}
