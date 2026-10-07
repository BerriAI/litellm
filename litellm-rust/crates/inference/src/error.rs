//! One error for every route in this crate. OCR still carries its own, richer enum.
//!
//! A variant is declared by the layer that produces it and nested here as is:
//! credentials by `litellm_auth` (AWS folds into it at that crate's boundary), the wire by
//! `litellm_http`, a provider's answer by `litellm_host::failure`, secrets by
//! `litellm_secrets`. The transformation layer's [`LlmError`] maps onto the same-named
//! variants once, here, so no route re-declares them. Where in the call a variant surfaced
//! is the [`litellm_host::failure::Failure`] around it, never the variant.

use std::sync::Arc;

use litellm_host::failure::{Classify, Kind, UpstreamResponse};
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
    Upstream(#[from] UpstreamResponse),
    #[error(transparent)]
    Headers(#[from] litellm_http::request::HeaderError),
    #[error(transparent)]
    Http(#[from] litellm_http::Error),
    #[error(transparent)]
    Secret(#[from] SecretError),
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

impl Classify for RouteError {
    fn kind(&self) -> Kind {
        match self {
            Self::Upstream(response) => Kind::Upstream(response.clone()),
            Self::Unsupported(_) => Kind::Unsupported,
            Self::Auth(litellm_auth::Error::MissingApiKey { .. }) => Kind::Auth,
            Self::Auth(_)
            | Self::InvalidType { .. }
            | Self::MissingField(_)
            | Self::InvalidProvider(_)
            | Self::InvalidRequest(_)
            | Self::Headers(_)
            | Self::Http(_) => Kind::Request,
            Self::Transport(TransportError::Timeout(_)) => Kind::Timeout,
            Self::Transport(_) => Kind::Connection,
            Self::InvalidResponse(_) => Kind::Response,
            Self::Secret(_) => Kind::Internal,
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
    use litellm_host::failure::{Classify, Kind, UpstreamResponse};
    use litellm_http::transport::Error as TransportError;
    use litellm_llms::{Error as LlmError, ErrorDetail};
    use rstest::rstest;

    use super::RouteError;

    fn upstream() -> UpstreamResponse {
        UpstreamResponse {
            status: 429,
            headers: Vec::new(),
            body: "slow down".into(),
            url: None,
        }
    }

    #[rstest]
    #[case::missing_key_is_auth(
        RouteError::Auth(litellm_auth::Error::MissingApiKey {
            provider: "Anthropic",
            environment_variable: "ANTHROPIC_API_KEY",
        }),
        Kind::Auth,
    )]
    #[case::malformed_header_is_the_request(
        RouteError::Auth(litellm_auth::Error::InvalidHeader),
        Kind::Request
    )]
    #[case::rejected_parameter(RouteError::InvalidRequest("top_k".into()), Kind::Request)]
    #[case::unsupported_capability(RouteError::Unsupported("streaming"), Kind::Unsupported)]
    #[case::provider_answer(RouteError::Upstream(upstream()), Kind::Upstream(upstream()))]
    #[case::timeout(RouteError::Transport(TransportError::Timeout("slow".into())), Kind::Timeout)]
    #[case::lost_connection(RouteError::Transport(TransportError::Network("reset".into())), Kind::Connection)]
    #[case::unreachable(RouteError::Transport(TransportError::Connect("refused".into())), Kind::Connection)]
    #[case::undecodable_answer(RouteError::InvalidResponse("bad json".into()), Kind::Response)]
    fn each_variant_has_one_kind(#[case] error: RouteError, #[case] kind: Kind) {
        assert_eq!(error.kind(), kind);
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
        assert_eq!(
            error.kind(),
            if request {
                Kind::Request
            } else {
                Kind::Response
            }
        );
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
