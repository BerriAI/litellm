use std::sync::Arc;

use litellm_host::machine::MachineFault;
use litellm_llms::{Error as LlmError, ErrorDetail};

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum Error {
    #[error("invalid provider: {0}")]
    InvalidProvider(String),
    #[error("invalid provider: unable to resolve custom_llm_provider for {route} request")]
    UnresolvedProvider { route: &'static str },
    #[error("unsupported by the rust path: {0}")]
    Unsupported(&'static str),
    #[error(transparent)]
    Llm(LlmError),
    #[error(transparent)]
    Auth(#[from] litellm_auth::Error),
    #[error(transparent)]
    Headers(#[from] litellm_http::request::HeaderError),
    #[error(transparent)]
    Http(litellm_http::Error),
    #[error(transparent)]
    Secret(#[from] SecretError),
    #[error("the host driver failed")]
    Host(#[source] MachineFault),
    #[error("post-call hook failed")]
    PostCallHook(#[source] Arc<Error>),
}

impl From<MachineFault> for Error {
    fn from(fault: MachineFault) -> Self {
        Self::Host(fault)
    }
}

impl Error {
    pub fn invalid_request(detail: ErrorDetail) -> Self {
        Self::Llm(LlmError::InvalidRequest(detail))
    }

    pub fn invalid_response(detail: ErrorDetail) -> Self {
        Self::Llm(LlmError::InvalidResponse(detail))
    }

    pub fn post_call(error: Self) -> Self {
        Self::PostCallHook(Arc::new(error))
    }

    /// The caller's request is what is wrong, as opposed to the environment, the wire, or
    /// the provider's answer.
    pub fn is_request(&self) -> bool {
        match self {
            Self::InvalidProvider(_)
            | Self::UnresolvedProvider { .. }
            | Self::Unsupported(_)
            | Self::Headers(_) => true,
            Self::Llm(error) => error.is_request(),
            Self::Auth(error) => !error.is_missing_credential(),
            Self::Http(error) => error.is_request(),
            Self::Secret(_) | Self::Host(_) | Self::PostCallHook(_) => false,
        }
    }

    pub fn is_response(&self) -> bool {
        match self {
            Self::Llm(error) => error.is_response(),
            Self::Http(error) => error.is_response(),
            Self::InvalidProvider(_)
            | Self::UnresolvedProvider { .. }
            | Self::Unsupported(_)
            | Self::Auth(_)
            | Self::Headers(_)
            | Self::Secret(_)
            | Self::Host(_)
            | Self::PostCallHook(_) => false,
        }
    }

    pub fn is_missing_credential(&self) -> bool {
        match self {
            Self::Llm(error) => error.is_missing_credential(),
            Self::Auth(error) => error.is_missing_credential(),
            Self::InvalidProvider(_)
            | Self::UnresolvedProvider { .. }
            | Self::Unsupported(_)
            | Self::Headers(_)
            | Self::Http(_)
            | Self::Secret(_)
            | Self::Host(_)
            | Self::PostCallHook(_) => false,
        }
    }

    pub fn is_unsupported(&self) -> bool {
        match self {
            Self::Unsupported(_) => true,
            Self::InvalidProvider(_)
            | Self::UnresolvedProvider { .. }
            | Self::Llm(_)
            | Self::Auth(_)
            | Self::Headers(_)
            | Self::Http(_)
            | Self::Secret(_)
            | Self::Host(_)
            | Self::PostCallHook(_) => false,
        }
    }

    pub fn dispatched(&self) -> bool {
        match self {
            Self::Llm(error) => error.dispatched(),
            Self::Http(error) => error.dispatched(),
            Self::Host(_) | Self::PostCallHook(_) => true,
            Self::InvalidProvider(_)
            | Self::UnresolvedProvider { .. }
            | Self::Unsupported(_)
            | Self::Auth(_)
            | Self::Headers(_)
            | Self::Secret(_) => false,
        }
    }

    pub fn upstream_response(&self) -> Option<(u16, &str)> {
        match self {
            Self::Llm(error) => error.upstream_response(),
            Self::Http(error) => error.upstream_response(),
            Self::InvalidProvider(_)
            | Self::UnresolvedProvider { .. }
            | Self::Unsupported(_)
            | Self::Auth(_)
            | Self::Headers(_)
            | Self::Secret(_)
            | Self::Host(_)
            | Self::PostCallHook(_) => None,
        }
    }
}

impl From<litellm_http::Error> for Error {
    fn from(error: litellm_http::Error) -> Self {
        // Python drops the caller's copy and prefers a forwarded Authorization
        // over the signature, so leave the request to it.
        if error.is_computed_header() {
            return Self::Unsupported("request forwards a header AWS SigV4 computes");
        }
        Self::Http(error)
    }
}

impl From<LlmError> for Error {
    fn from(error: LlmError) -> Self {
        match error {
            LlmError::Unsupported(reason) => Self::Unsupported(reason),
            LlmError::Http(error) => Self::from(error),
            LlmError::Auth(error) => Self::Auth(error),
            error @ (LlmError::InvalidType { .. }
            | LlmError::MissingField(_)
            | LlmError::InvalidRequest(_)
            | LlmError::InvalidResponse(_)) => Self::Llm(error),
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
    use super::Error;
    use litellm_host::machine::MachineFault;
    use litellm_llms::{Error as LlmError, ErrorDetail};
    use rstest::rstest;

    #[rstest]
    #[case::missing_api_key(
        litellm_auth::Error::MissingApiKey {
            provider: "Anthropic",
            environment_variable: "ANTHROPIC_API_KEY",
        }
        .into(),
        false
    )]
    #[case::invalid_auth_header(litellm_auth::Error::InvalidHeader.into(), true)]
    #[case::transformation_missing_api_key(
        LlmError::Auth(litellm_auth::Error::MissingApiKey {
            provider: "Anthropic",
            environment_variable: "ANTHROPIC_API_KEY",
        })
        .into(),
        false
    )]
    #[case::invalid_request(
        LlmError::InvalidRequest(ErrorDetail::Expected {
            subject: "top_k",
            expected: "an integer",
        })
        .into(),
        true
    )]
    #[case::invalid_response(
        LlmError::InvalidResponse(ErrorDetail::MissingField {
            subject: "response",
            field: "usage",
        })
        .into(),
        false
    )]
    #[case::host_driver(MachineFault::Abandoned.into(), false)]
    #[case::post_call_hook_over_a_request_fault(
        Error::post_call(LlmError::MissingField("usage").into()),
        false
    )]
    fn only_the_callers_input_is_a_request_fault(#[case] error: Error, #[case] is_request: bool) {
        assert_eq!(error.is_request(), is_request);
    }

    #[rstest]
    #[case::own_auth(litellm_auth::Error::MissingApiKey {
        provider: "Anthropic",
        environment_variable: "ANTHROPIC_API_KEY",
    }
    .into())]
    #[case::transformation_auth(
        LlmError::Auth(litellm_auth::Error::MissingApiKey {
            provider: "Anthropic",
            environment_variable: "ANTHROPIC_API_KEY",
        })
        .into()
    )]
    fn a_missing_api_key_is_a_missing_credential_wherever_it_is_raised(#[case] error: Error) {
        assert!(error.is_missing_credential());
    }

    #[rstest]
    #[case::direct(litellm_http::Error::computed_header("authorization").into())]
    #[case::through_a_transformation(
        LlmError::Http(litellm_http::Error::computed_header("authorization")).into()
    )]
    fn a_header_the_signer_computes_is_left_to_the_host_on_every_route(#[case] error: Error) {
        assert!(error.is_unsupported(), "{error:?}");
        assert!(error.is_request());
    }

    #[test]
    fn a_transformation_unsupported_stays_a_decline() {
        assert!(Error::from(LlmError::Unsupported("test capability")).is_unsupported());
    }

    #[test]
    fn an_upstream_failure_through_a_transformation_keeps_the_provider_answer() {
        let error = Error::from(LlmError::Http(litellm_http::Error::upstream(
            429,
            "slow down".into(),
        )));
        assert_eq!(error.upstream_response(), Some((429, "slow down")));
        assert!(error.dispatched());
    }

    #[test]
    fn a_post_call_hook_failure_is_not_the_providers_answer() {
        let error =
            Error::post_call(litellm_http::Error::upstream(500, "hook backend down".into()).into());
        assert_eq!(error.upstream_response(), None);
        assert!(!error.is_request());
        let report = litellm_error::Report::new(&error).to_string();
        assert!(report.starts_with("post-call hook failed: "), "{report}");
        assert!(report.contains("hook backend down"), "{report}");
    }

    #[test]
    fn a_host_driver_fault_keeps_the_fault_as_its_source() {
        let error = Error::from(MachineFault::Abandoned);
        assert_eq!(
            litellm_error::find::<MachineFault>(&error),
            Some(&MachineFault::Abandoned)
        );
        assert!(error.dispatched());
    }

    #[rstest]
    #[case::request(true)]
    #[case::response(false)]
    fn contextual_errors_keep_their_cause_and_route_classification(#[case] request: bool) {
        let source = serde_json::from_str::<serde_json::Value>("{").unwrap_err();
        let source_message = source.to_string();
        let detail = ErrorDetail::invalid("test payload", source);
        let error = Error::from(if request {
            LlmError::InvalidRequest(detail)
        } else {
            LlmError::InvalidResponse(detail)
        });
        assert_eq!(error.is_request(), request);
        assert_eq!(error.is_response(), !request);
        let report = litellm_error::Report::new(&error).to_string();
        assert!(report.contains("test payload"), "{report}");
        assert_eq!(report.matches(&source_message).count(), 1, "{report}");
        assert!(litellm_error::find::<serde_json::Error>(&error).is_some());
    }
}
