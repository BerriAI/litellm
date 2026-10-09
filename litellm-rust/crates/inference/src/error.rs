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
    InvalidRequest(#[source] litellm_llms::ErrorDetail),
    #[error("invalid response: {0}")]
    InvalidResponse(#[source] litellm_llms::ErrorDetail),
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
    Hook(#[from] litellm_host::error::HookError),
    #[error("post-call hook failed: {0}")]
    PostCallHook(#[source] Arc<RouteError>),
}

impl From<litellm_host::machine::MachineFault> for RouteError {
    fn from(fault: litellm_host::machine::MachineFault) -> Self {
        use litellm_host::machine::MachineFault;
        Self::InvalidRequest(match fault {
            MachineFault::Abandoned => "host driver was abandoned".into(),
            MachineFault::Protocol(message) => format!("host {message}").into(),
        })
    }
}

impl RouteError {
    pub fn post_call(error: Self) -> Self {
        Self::PostCallHook(Arc::new(error))
    }

    /// The caller's request is what is wrong, as opposed to the environment, the wire, or
    /// the provider's answer.
    pub fn is_request(&self) -> bool {
        match self {
            Self::InvalidType { .. }
            | Self::MissingField(_)
            | Self::InvalidProvider(_)
            | Self::InvalidRequest(_)
            | Self::Unsupported(_)
            | Self::Headers(_)
            | Self::Hook(_) => true,
            Self::Auth(error) => !matches!(error, litellm_auth::Error::MissingApiKey { .. }),
            Self::InvalidResponse(_)
            | Self::Transport(_)
            | Self::Http(_)
            | Self::Secret(_)
            | Self::PostCallHook(_) => false,
        }
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

#[cfg(test)]
mod tests {
    use super::RouteError;
    use litellm_llms::{Error as LlmError, ErrorDetail};
    use rstest::rstest;

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
    #[rstest]
    #[case::request(true)]
    #[case::response(false)]
    fn contextual_errors_preserve_sources_and_route_classification(#[case] request: bool) {
        let source = serde_json::from_str::<serde_json::Value>("{").unwrap_err();
        let source_message = source.to_string();
        let detail = ErrorDetail::invalid("test payload", source);
        let error = RouteError::from(if request {
            LlmError::InvalidRequest(detail)
        } else {
            LlmError::InvalidResponse(detail)
        });
        assert_eq!(error.is_request(), request);
        let category = if request { "request" } else { "response" };
        assert_eq!(
            error.to_string(),
            format!("invalid {category}: invalid test payload: {source_message}")
        );
        let source = std::iter::successors(Some(&error as &dyn std::error::Error), |error| {
            error.source()
        })
        .find_map(|error| error.downcast_ref::<serde_json::Error>())
        .expect("the original JSON error remains available");
        assert_eq!(source.to_string(), source_message);
    }
}
