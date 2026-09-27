//! One error for every route in this crate. OCR still carries its own, richer enum.
//!
//! A variant is declared by the layer that produces it and nested here as is:
//! credentials by `litellm_auth` (AWS folds into it at that crate's boundary), the wire by
//! `litellm_http`, secrets by `litellm_secrets`. The transformation layer's [`LlmError`]
//! maps onto the same-named variants once, here, so no route re-declares them.

use std::sync::Arc;

use litellm_http::transport::Error as TransportError;
use litellm_llms::Error as LlmError;

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum RouteError {
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
    /// The call's projection was consumed already.
    #[error("request was already projected")]
    AlreadyProjected,
    #[error("invalid Anthropic messages request: {0}")]
    RequestDecoding(#[source] JsonError),
    #[error("failed to serialize Anthropic messages request: {0}")]
    RequestEncoding(#[source] JsonError),
    #[error("invalid response: {0}")]
    InvalidResponse(String),
    #[error("invalid messages response JSON: {0}")]
    ResponseDecoding(#[source] JsonError),
    #[error("unsupported by the rust path: {0}")]
    Unsupported(&'static str),
    #[error(transparent)]
    Auth(#[from] litellm_auth::Error),
    #[error(transparent)]
    Transport(#[from] TransportError),
    #[error(transparent)]
    Headers(#[from] litellm_http::request::HeaderError),
    #[error(transparent)]
    Http(#[from] litellm_http::Error),
    #[error(transparent)]
    Secret(#[from] SecretError),
    #[error(transparent)]
    HostFault(#[from] litellm_host::MachineFault),
}

/// Whether the provider had already been called when the route failed. Before the send, a
/// host may retry on another path; after it, the provider has done the work and billed for it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    BeforeSend,
    AfterSend,
}

impl RouteError {
    pub fn phase(&self) -> Phase {
        match self {
            Self::InvalidResponse(_)
            | Self::ResponseDecoding(_)
            | Self::Transport(TransportError::Http { .. } | TransportError::Network(_)) => {
                Phase::AfterSend
            }
            Self::Transport(TransportError::Connect(_))
            | Self::InvalidType { .. }
            | Self::MissingField(_)
            | Self::InvalidProvider(_)
            | Self::InvalidRequest(_)
            | Self::AlreadyProjected
            | Self::RequestDecoding(_)
            | Self::RequestEncoding(_)
            | Self::Unsupported(_)
            | Self::Auth(_)
            | Self::Headers(_)
            | Self::Http(_)
            | Self::Secret(_)
            | Self::HostFault(_) => Phase::BeforeSend,
        }
    }

    /// The caller's request is what is wrong, as opposed to the environment, the wire, or
    /// the provider's answer.
    pub fn is_request(&self) -> bool {
        match self {
            Self::InvalidType { .. }
            | Self::MissingField(_)
            | Self::InvalidProvider(_)
            | Self::InvalidRequest(_)
            | Self::AlreadyProjected
            | Self::RequestDecoding(_)
            | Self::RequestEncoding(_)
            | Self::Unsupported(_)
            | Self::Headers(_) => true,
            Self::Auth(error) => !matches!(error, litellm_auth::Error::MissingApiKey { .. }),
            Self::InvalidResponse(_)
            | Self::ResponseDecoding(_)
            | Self::Transport(_)
            | Self::Http(_)
            | Self::Secret(_)
            | Self::HostFault(_) => false,
        }
    }

    /// The provider's answer is what is wrong, as opposed to the request or the wire.
    pub fn is_response(&self) -> bool {
        matches!(self, Self::InvalidResponse(_) | Self::ResponseDecoding(_))
    }
}

impl From<LlmError> for RouteError {
    fn from(error: LlmError) -> Self {
        match error {
            LlmError::InvalidType { expected, actual } => Self::InvalidType { expected, actual },
            LlmError::MissingField(field) => Self::MissingField(field),
            LlmError::InvalidRequest(message) => Self::InvalidRequest(message),
            LlmError::InvalidResponse(message) => Self::InvalidResponse(message),
            LlmError::Unsupported(reason) => Self::Unsupported(reason),
            LlmError::Auth(error) => Self::Auth(error),
        }
    }
}

#[derive(Clone, Debug, thiserror::Error)]
#[error(transparent)]
pub struct SecretError(Arc<litellm_secrets::Error>);

impl SecretError {
    pub fn source_error(&self) -> &litellm_secrets::Error {
        &self.0
    }
}

impl From<litellm_secrets::Error> for RouteError {
    fn from(error: litellm_secrets::Error) -> Self {
        Self::Secret(SecretError(Arc::new(error)))
    }
}

impl PartialEq for SecretError {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for SecretError {}

#[derive(Clone, Debug, thiserror::Error)]
#[error(transparent)]
pub struct JsonError(Arc<serde_json::Error>);

impl JsonError {
    pub fn source_error(&self) -> &serde_json::Error {
        &self.0
    }
}

impl From<serde_json::Error> for JsonError {
    fn from(error: serde_json::Error) -> Self {
        Self(Arc::new(error))
    }
}

impl PartialEq for JsonError {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for JsonError {}

#[cfg(test)]
mod tests {
    use super::{Phase, RouteError};
    use litellm_http::transport::Error as TransportError;

    #[test]
    fn only_a_provider_answer_or_a_lost_connection_counts_as_after_send() {
        let after = [
            RouteError::InvalidResponse("bad json".into()),
            RouteError::Transport(TransportError::Http {
                status: 500,
                body: "boom".into(),
            }),
            RouteError::Transport(TransportError::Network("reset".into())),
        ];
        for error in after {
            assert_eq!(error.phase(), Phase::AfterSend, "{error:?}");
        }
        let before = [
            RouteError::Transport(TransportError::Connect("refused".into())),
            RouteError::Unsupported("streaming"),
            RouteError::Auth(litellm_auth::Error::InvalidHeader),
        ];
        for error in before {
            assert_eq!(error.phase(), Phase::BeforeSend, "{error:?}");
        }
    }

    #[test]
    fn a_missing_api_key_is_the_environment_not_the_request() {
        assert!(
            !RouteError::Auth(litellm_auth::Error::MissingApiKey {
                provider: "Anthropic",
                environment_variable: "ANTHROPIC_API_KEY",
            })
            .is_request()
        );
        assert!(RouteError::Auth(litellm_auth::Error::InvalidHeader).is_request());
        assert!(RouteError::InvalidRequest("top_k".into()).is_request());
        assert!(!RouteError::InvalidResponse("bad json".into()).is_request());
    }
}
