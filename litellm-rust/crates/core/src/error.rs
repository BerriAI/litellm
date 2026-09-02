use thiserror::Error as ThisError;

#[derive(Debug, ThisError, PartialEq, Eq)]
pub enum Error {
    #[error("expected {expected}, got {actual}")]
    InvalidType {
        expected: &'static str,
        actual: &'static str,
    },
    #[error("missing required field: {0}")]
    MissingField(&'static str),
    #[error("invalid response: {0}")]
    InvalidResponse(String),
    #[error("invalid provider: {0}")]
    InvalidProvider(String),
    #[error("invalid request: {0}")]
    InvalidRequest(String),
    #[error("{0}")]
    Auth(String),
    #[error("upstream request failed with status {status}: {body}")]
    Http { status: u16, body: String },
    #[error("upstream network error: {0}")]
    Network(String),
    #[error("routing error: {0}")]
    Routing(String),
    /// The request is outside the surface this route covers in Rust. Hosts that
    /// keep a reference implementation treat this as "fall back", not "fail".
    #[error("unsupported by the rust path: {0}")]
    Unsupported(&'static str),
}

/// Re-tag an error raised after the provider has already returned a response.
pub(crate) fn as_response_error(err: Error) -> Error {
    match err {
        already @ (Error::InvalidResponse(_) | Error::Http { .. }) => already,
        other => Error::InvalidResponse(other.to_string()),
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
    fn response_errors_collapse_to_one_non_retryable_variant() {
        for original in [
            Error::MissingField("usage"),
            Error::Unsupported("non-text response content block"),
            Error::InvalidRequest("whatever".to_string()),
            Error::Auth("whatever".to_string()),
        ] {
            assert!(matches!(
                as_response_error(original),
                Error::InvalidResponse(_)
            ));
        }
    }

    #[test]
    fn response_errors_preserve_an_upstream_status() {
        assert!(matches!(
            as_response_error(Error::Http {
                status: 500,
                body: "boom".to_string()
            }),
            Error::Http { status: 500, .. }
        ));
    }
}
