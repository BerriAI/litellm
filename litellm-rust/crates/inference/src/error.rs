//! One error for every route in this crate. OCR still carries its own, richer enum.
//!
//! A variant is declared by the layer that produces it and nested here as is:
//! credentials by `litellm_auth` (AWS folds into it at that crate's boundary), the wire by
//! `litellm_http`, secrets by `litellm_secrets`. The transformation layer's [`LlmError`]
//! maps onto the same-named variants once, here, so no route re-declares them.

use std::{borrow::Cow, sync::Arc};

use litellm_http::transport::Error as TransportError;
use litellm_llms::Error as LlmError;

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum Error {
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
    #[error("post-call hook failed: {0}")]
    PostCallHook(#[source] Arc<Error>),
}

impl From<litellm_host::machine::MachineFault> for Error {
    fn from(fault: litellm_host::machine::MachineFault) -> Self {
        use litellm_host::machine::MachineFault;
        Self::InvalidRequest(match fault {
            MachineFault::Abandoned => "host driver was abandoned".into(),
            MachineFault::Protocol(message) => format!("host {message}").into(),
        })
    }
}

impl Error {
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
            | Self::Headers(_) => true,
            Self::Auth(error) => !matches!(
                error,
                litellm_auth::Error::MissingApiKey { .. }
                    | litellm_auth::Error::CredentialRequest { .. }
                    | litellm_auth::Error::InvalidCredentialResponse(_)
            ),
            Self::InvalidResponse(_)
            | Self::Transport(_)
            | Self::Http(_)
            | Self::Secret(_)
            | Self::PostCallHook(_) => false,
        }
    }

    pub fn http_status(&self) -> Option<u16> {
        match self {
            Self::Transport(TransportError::Http { status, .. }) => Some(*status),
            Self::Auth(litellm_auth::Error::CredentialRequest { status, .. }) => *status,
            _ => None,
        }
    }

    pub fn http_failure(&self) -> Option<(u16, Cow<'_, str>)> {
        match self {
            Self::Transport(TransportError::Http { status, body }) => {
                Some((*status, Cow::Borrowed(body)))
            }
            Self::Auth(litellm_auth::Error::CredentialRequest {
                status: Some(status),
                ..
            }) => Some((*status, Cow::Borrowed("credential request failed"))),
            _ => None,
        }
    }
}

impl From<LlmError> for Error {
    fn from(error: LlmError) -> Self {
        match error {
            LlmError::InvalidType { expected, actual } => Self::InvalidType { expected, actual },
            LlmError::MissingField(field) => Self::MissingField(field),
            LlmError::InvalidRequest(message) => Self::InvalidRequest(message),
            LlmError::InvalidResponse(message) => Self::InvalidResponse(message),
            LlmError::Unsupported(reason) => Self::Unsupported(reason),
            LlmError::Auth(error) => Self::from(error),
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

impl From<litellm_secrets::Error> for Error {
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
    use std::borrow::Cow;

    use super::Error;
    use litellm_llms::{Error as LlmError, ErrorDetail};
    use rstest::rstest;

    #[rstest]
    #[case::legacy_acquisition(litellm_auth::Error::CredentialAcquisition("legacy".into()), true)]
    #[case::invalid_header(litellm_auth::Error::InvalidHeader, true)]
    #[case::missing_key(litellm_auth::Error::MissingApiKey {
        provider: "provider",
        environment_variable: "API_KEY",
    }, false)]
    fn existing_auth_errors_keep_their_variants_and_classification(
        #[case] auth: litellm_auth::Error,
        #[case] is_request: bool,
    ) {
        let error = Error::from(auth.clone());
        assert_eq!(error, Error::Auth(auth));
        assert_eq!(error.is_request(), is_request);
    }

    #[rstest]
    #[case::unavailable(Some(503))]
    #[case::unauthorized(Some(401))]
    #[case::rate_limited(Some(429))]
    #[case::network(None)]
    fn credential_request_errors_keep_the_auth_variant_and_status(#[case] status: Option<u16>) {
        let auth = litellm_auth::Error::CredentialRequest {
            status,
            source: "credential endpoint failed".into(),
        };
        let error = Error::from(auth.clone());
        assert_eq!(error, Error::Auth(auth));
        assert!(!error.is_request());
        assert_eq!(error.http_status(), status);
        assert_eq!(
            error.http_failure(),
            status.map(|status| (status, Cow::Borrowed("credential request failed")))
        );
    }

    #[test]
    fn malformed_credential_responses_remain_provider_failures() {
        let error = Error::from(litellm_auth::Error::InvalidCredentialResponse(
            "missing token".into(),
        ));
        assert!(!error.is_request());
        assert!(matches!(
            error,
            Error::Auth(litellm_auth::Error::InvalidCredentialResponse(_))
        ));
    }

    #[test]
    fn credential_request_causes_remain_traversable_and_sanitized() {
        let error = Error::Auth(litellm_auth::Error::CredentialRequest {
            status: None,
            source: litellm_auth::ErrorDetail::failed(
                "credential endpoint",
                std::io::Error::other("connection refused"),
            ),
        });
        let source = std::iter::successors(Some(&error as &dyn std::error::Error), |error| {
            error.source()
        })
        .find_map(|error| error.downcast_ref::<std::io::Error>())
        .expect("the underlying transport error remains available");
        assert_eq!(source.to_string(), "connection refused");
        assert!(!format!("{error:?}").contains("https://"));
        assert!(!error.to_string().contains("response body"));
    }

    #[test]
    fn a_missing_api_key_is_the_environment_not_the_request() {
        assert!(
            !Error::Auth(litellm_auth::Error::MissingApiKey {
                provider: "Anthropic",
                environment_variable: "ANTHROPIC_API_KEY",
            })
            .is_request()
        );
        assert!(Error::Auth(litellm_auth::Error::InvalidHeader).is_request());
        assert!(Error::InvalidRequest("top_k".into()).is_request());
        assert!(!Error::InvalidResponse("bad json".into()).is_request());
    }
    #[rstest]
    #[case::request(true)]
    #[case::response(false)]
    fn contextual_errors_preserve_sources_and_route_classification(#[case] request: bool) {
        let source = serde_json::from_str::<serde_json::Value>("{").unwrap_err();
        let source_message = source.to_string();
        let detail = ErrorDetail::invalid("test payload", source);
        let error = Error::from(if request {
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
