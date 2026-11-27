/// The public LiteLLM exception classes a Rust route failure can become. Python builds the
/// class; Rust decides which one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PublicError {
    BadRequest,
    ContextWindowExceeded,
    ContentPolicyViolation,
    Authentication,
    PermissionDenied,
    NotFound,
    Timeout { status: u16 },
    RateLimit,
    InternalServer,
    BadGateway,
    ServiceUnavailable,
    ApiConnection,
    Api { status: u16 },
}

impl PublicError {
    /// The `status_code` the Python class carries.
    pub const fn status_code(self) -> u16 {
        match self {
            Self::BadRequest | Self::ContextWindowExceeded | Self::ContentPolicyViolation => 400,
            Self::Authentication => 401,
            Self::PermissionDenied => 403,
            Self::NotFound => 404,
            Self::RateLimit => 429,
            Self::InternalServer | Self::ApiConnection => 500,
            Self::BadGateway => 502,
            Self::ServiceUnavailable => 503,
            Self::Timeout { status } | Self::Api { status } => status,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UpstreamResponse {
    pub status: u16,
    pub body: String,
    pub headers: Vec<(String, String)>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MappedFailure {
    pub error: PublicError,
    pub message: String,
    pub upstream: Option<UpstreamResponse>,
    pub debug_info: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[rstest::rstest]
    #[case::bad_request(PublicError::BadRequest, 400)]
    #[case::context_window(PublicError::ContextWindowExceeded, 400)]
    #[case::content_policy(PublicError::ContentPolicyViolation, 400)]
    #[case::authentication(PublicError::Authentication, 401)]
    #[case::permission_denied(PublicError::PermissionDenied, 403)]
    #[case::not_found(PublicError::NotFound, 404)]
    #[case::request_timeout(PublicError::Timeout { status: 408 }, 408)]
    #[case::gateway_timeout(PublicError::Timeout { status: 504 }, 504)]
    #[case::rate_limit(PublicError::RateLimit, 429)]
    #[case::internal_server(PublicError::InternalServer, 500)]
    #[case::api_connection(PublicError::ApiConnection, 500)]
    #[case::bad_gateway(PublicError::BadGateway, 502)]
    #[case::service_unavailable(PublicError::ServiceUnavailable, 503)]
    #[case::api(PublicError::Api { status: 501 }, 501)]
    fn status_codes_are_the_ones_the_python_classes_set(
        #[case] error: PublicError,
        #[case] status: u16,
    ) {
        assert_eq!(error.status_code(), status);
    }
}
