use std::time::TryFromFloatSecsError;

use axum::{
    Json,
    extract::rejection::{JsonRejection, PathRejection, QueryRejection},
    http::StatusCode,
    response::{IntoResponse, Response},
};
use http::header::{InvalidHeaderName, InvalidHeaderValue, ToStrError};
use litellm_error::{Report, Source};
use rmcp::{
    model::{ErrorCode, ErrorData},
    service::{ClientInitializeError, ServiceError},
};
use serde_json::json;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{0} is required in request body")]
    MissingField(&'static str),
    #[error("Gateway listings do not accept an upstream cursor")]
    UpstreamCursor,
    #[error("read_resource requires exactly one allowed MCP server")]
    AmbiguousResourceServer,
    #[error("Invalid x-mcp-servers header")]
    InvalidServersHeader(#[source] ToStrError),
    #[error("invalid JSON body")]
    Json(#[source] JsonRejection),
    #[error("invalid query string")]
    Query(#[source] QueryRejection),
    #[error("invalid path")]
    Path(#[source] PathRejection),
    #[error("request body is too large")]
    BodyTooLarge(#[source] axum::Error),
    #[error("User not allowed to access this MCP server or operation.")]
    Forbidden,
    #[error("request host is not allowed")]
    HostNotAllowed,
    #[error("request origin is not allowed")]
    OriginNotAllowed,
    #[error("MCP session belongs to another caller")]
    ForeignSession,
    #[error("MCP session not found")]
    UnknownSession,
    #[error("MCP session is busy")]
    SessionBusy,
    #[error("MCP upstream request failed")]
    Upstream(#[from] ServiceError),
    #[error("MCP upstream returned invalid pagination")]
    InvalidPagination,
    #[error("MCP operation returned an unexpected result")]
    UnexpectedResult,
    #[error("MCP authorization unavailable")]
    AuthorizationUnavailable(#[source] Source),
    #[error("MCP request cancelled")]
    Cancelled,
}

impl Error {
    pub fn status(&self) -> StatusCode {
        match self {
            Self::MissingField(_)
            | Self::UpstreamCursor
            | Self::AmbiguousResourceServer
            | Self::InvalidServersHeader(_) => StatusCode::BAD_REQUEST,
            Self::Json(rejection) => rejection.status(),
            Self::Query(rejection) => rejection.status(),
            Self::Path(rejection) => rejection.status(),
            Self::BodyTooLarge(_) => StatusCode::PAYLOAD_TOO_LARGE,
            Self::Forbidden
            | Self::HostNotAllowed
            | Self::OriginNotAllowed
            | Self::ForeignSession => StatusCode::FORBIDDEN,
            Self::UnknownSession => StatusCode::NOT_FOUND,
            Self::SessionBusy => StatusCode::TOO_MANY_REQUESTS,
            Self::Upstream(_) | Self::InvalidPagination => StatusCode::BAD_GATEWAY,
            Self::Cancelled => StatusCode::REQUEST_TIMEOUT,
            Self::UnexpectedResult | Self::AuthorizationUnavailable(_) => {
                StatusCode::INTERNAL_SERVER_ERROR
            }
        }
    }

    fn is_request(&self) -> bool {
        match self {
            Self::MissingField(_)
            | Self::UpstreamCursor
            | Self::AmbiguousResourceServer
            | Self::InvalidServersHeader(_)
            | Self::Json(_)
            | Self::Query(_)
            | Self::Path(_)
            | Self::BodyTooLarge(_) => true,
            Self::Forbidden
            | Self::HostNotAllowed
            | Self::OriginNotAllowed
            | Self::ForeignSession
            | Self::UnknownSession
            | Self::SessionBusy
            | Self::Upstream(_)
            | Self::InvalidPagination
            | Self::UnexpectedResult
            | Self::AuthorizationUnavailable(_)
            | Self::Cancelled => false,
        }
    }

    fn detail(&self) -> String {
        match self {
            Self::Upstream(_) | Self::AuthorizationUnavailable(_) => self.to_string(),
            Self::MissingField(_)
            | Self::UpstreamCursor
            | Self::AmbiguousResourceServer
            | Self::InvalidServersHeader(_)
            | Self::Json(_)
            | Self::Query(_)
            | Self::Path(_)
            | Self::BodyTooLarge(_)
            | Self::Forbidden
            | Self::HostNotAllowed
            | Self::OriginNotAllowed
            | Self::ForeignSession
            | Self::UnknownSession
            | Self::SessionBusy
            | Self::InvalidPagination
            | Self::UnexpectedResult
            | Self::Cancelled => Report::new(self).to_string(),
        }
    }

    fn upstream(&self) -> Option<&ServiceError> {
        match self {
            Self::Upstream(error) => Some(error),
            Self::MissingField(_)
            | Self::UpstreamCursor
            | Self::AmbiguousResourceServer
            | Self::InvalidServersHeader(_)
            | Self::Json(_)
            | Self::Query(_)
            | Self::Path(_)
            | Self::BodyTooLarge(_)
            | Self::Forbidden
            | Self::HostNotAllowed
            | Self::OriginNotAllowed
            | Self::ForeignSession
            | Self::UnknownSession
            | Self::SessionBusy
            | Self::InvalidPagination
            | Self::UnexpectedResult
            | Self::AuthorizationUnavailable(_)
            | Self::Cancelled => None,
        }
    }

    pub(crate) fn is_upstream(&self) -> bool {
        self.upstream().is_some()
    }

    pub(crate) fn is_timeout(&self) -> bool {
        self.upstream().is_some_and(|error| match error {
            ServiceError::Timeout { .. } => true,
            ServiceError::McpError(_)
            | ServiceError::TransportSend(_)
            | ServiceError::TransportClosed
            | ServiceError::UnexpectedResponse
            | ServiceError::SubscriptionLagged { .. }
            | ServiceError::Cancelled { .. }
            | ServiceError::InputRequiredRoundsExceeded { .. } => false,
            _ => false,
        })
    }

    pub(crate) fn is_unreachable(&self) -> bool {
        self.upstream().is_some_and(|error| match error {
            ServiceError::TransportSend(_) | ServiceError::TransportClosed => true,
            ServiceError::McpError(_)
            | ServiceError::Timeout { .. }
            | ServiceError::UnexpectedResponse
            | ServiceError::SubscriptionLagged { .. }
            | ServiceError::Cancelled { .. }
            | ServiceError::InputRequiredRoundsExceeded { .. } => false,
            _ => false,
        })
    }

    pub(crate) fn mcp_error(&self) -> Option<&ErrorData> {
        self.upstream().and_then(|error| match error {
            ServiceError::McpError(error) => Some(error),
            ServiceError::TransportSend(_)
            | ServiceError::TransportClosed
            | ServiceError::Timeout { .. }
            | ServiceError::UnexpectedResponse
            | ServiceError::SubscriptionLagged { .. }
            | ServiceError::Cancelled { .. }
            | ServiceError::InputRequiredRoundsExceeded { .. } => None,
            _ => None,
        })
    }

    pub fn into_mcp(self) -> ErrorData {
        if let Some(error) = self.mcp_error() {
            return error.clone();
        }
        let message = self.detail();
        if self.status() == StatusCode::FORBIDDEN {
            ErrorData::new(ErrorCode(-32003), message, None)
        } else if self.is_request() {
            ErrorData::invalid_params(message, None)
        } else {
            ErrorData::internal_error(message, None)
        }
    }
}

impl IntoResponse for Error {
    fn into_response(self) -> Response {
        (self.status(), Json(json!({"detail": self.detail()}))).into_response()
    }
}

#[derive(thiserror::Error, veil::Redact)]
pub enum ConnectError {
    #[error("MCP server {server} is misconfigured")]
    Configuration {
        server: String,
        #[source]
        fault: ConfigFault,
    },
    #[error("could not resolve MCP credentials")]
    Secret(#[from] litellm_secrets::Error),
    #[error("could not start MCP child process")]
    Process(#[from] std::io::Error),
    #[error("could not initialize MCP upstream {server}")]
    Initialize {
        server: String,
        #[source]
        #[redact]
        source: Box<ClientInitializeError>,
    },
    #[error("MCP upstream {0} initialization timed out")]
    Timeout(String),
    #[error("MCP upstream startup cancelled")]
    Cancelled,
    #[error("MCP server identifiers must not be empty")]
    EmptyIdentifier,
    #[error("MCP server identifiers and prefixes must be unique")]
    DuplicateIdentifier,
}

#[derive(Debug, thiserror::Error)]
pub enum ConfigFault {
    #[error("timeout must be finite and positive")]
    InvalidTimeout(#[source] TryFromFloatSecsError),
    #[error("timeout must be positive")]
    ZeroTimeout,
    #[error("server identifiers must not be blank")]
    BlankIdentifier,
    #[error("setting '{0}' is not supported by the Rust MCP host")]
    UnsupportedSetting(String),
    #[error("max_concurrent_requests is out of range")]
    ConcurrencyOutOfRange,
    #[error("mcp_info protocol and cost settings are not supported")]
    UnsupportedMcpInfo,
    #[error("configured upstream auth mode is not supported by the Rust MCP host")]
    UnsupportedAuth,
    #[error("legacy SSE upstreams are not supported by the pinned SDK; use transport: http")]
    SseTransport,
    #[error("HTTP transport requires url and does not accept command, args or env")]
    HttpShape,
    #[error("stdio transport requires command and does not accept url")]
    StdioShape,
    #[error("stdio credentials must be supplied through env")]
    StdioCredentials,
    #[error("authentication_token and upstream_token_header require an auth_type")]
    TokenWithoutAuth,
    #[error("configured auth_type requires authentication_token")]
    AuthWithoutToken,
    #[error("referenced secret is unavailable")]
    MissingSecret,
    #[error("upstream credential is empty")]
    EmptyCredential,
    #[error("invalid upstream header name")]
    HeaderName(#[source] InvalidHeaderName),
    #[error("invalid upstream header value")]
    HeaderValue(#[source] InvalidHeaderValue),
    #[error("upstream header conflicts with HTTP framing")]
    FramingHeader,
    #[error("invalid HTTP URL")]
    InvalidUrl(#[source] url::ParseError),
    #[error("HTTP URL must use http or https without user credentials or fragments")]
    UrlShape,
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    const SECRET_URL: &str = "https://example.test/mcp?token=private-secret";

    #[rstest]
    fn startup_diagnostics_do_not_expose_upstream_details() {
        let error = ConnectError::Initialize {
            server: "docs".into(),
            source: Box::new(ClientInitializeError::ConnectionClosed(SECRET_URL.into())),
        };
        let debug = format!("{error:?}");
        assert!(debug.contains("docs"));
        assert!(!debug.contains("private-secret"));
        assert!(std::error::Error::source(&error).is_some());
    }

    #[rstest]
    fn configuration_faults_reach_the_startup_diagnostic() {
        let error = ConnectError::Configuration {
            server: "docs".into(),
            fault: ConfigFault::ZeroTimeout,
        };
        assert!(format!("{error:?}").contains("ZeroTimeout"));
        let report = Report::new(&error).to_string();
        assert!(report.contains("docs"), "{report}");
        assert_eq!(
            report.matches("timeout must be positive").count(),
            1,
            "{report}"
        );
    }

    #[rstest]
    #[case::missing_field(Error::MissingField("server_id"), StatusCode::BAD_REQUEST)]
    #[case::upstream_cursor(Error::UpstreamCursor, StatusCode::BAD_REQUEST)]
    #[case::forbidden(Error::Forbidden, StatusCode::FORBIDDEN)]
    #[case::foreign_session(Error::ForeignSession, StatusCode::FORBIDDEN)]
    #[case::unknown_session(Error::UnknownSession, StatusCode::NOT_FOUND)]
    #[case::busy_session(Error::SessionBusy, StatusCode::TOO_MANY_REQUESTS)]
    #[case::invalid_pagination(Error::InvalidPagination, StatusCode::BAD_GATEWAY)]
    #[case::upstream(
        Error::Upstream(ServiceError::TransportClosed),
        StatusCode::BAD_GATEWAY
    )]
    #[case::cancelled(Error::Cancelled, StatusCode::REQUEST_TIMEOUT)]
    fn status_follows_who_is_at_fault(#[case] error: Error, #[case] status: StatusCode) {
        assert_eq!(error.status(), status);
    }

    fn mcp_failure() -> Error {
        Error::Upstream(ServiceError::McpError(ErrorData::invalid_params(
            "bad", None,
        )))
    }

    #[rstest]
    #[case::timeout(
        Error::Upstream(ServiceError::Timeout { timeout: std::time::Duration::from_secs(1) }),
        true,
        false,
        false
    )]
    #[case::closed(Error::Upstream(ServiceError::TransportClosed), false, true, false)]
    #[case::mcp_error(mcp_failure(), false, false, true)]
    #[case::unexpected_response(
        Error::Upstream(ServiceError::UnexpectedResponse),
        false,
        false,
        false
    )]
    #[case::gateway_cancelled(Error::Cancelled, false, false, false)]
    fn upstream_failures_answer_what_went_wrong(
        #[case] error: Error,
        #[case] timeout: bool,
        #[case] unreachable: bool,
        #[case] mcp_error: bool,
    ) {
        assert_eq!(error.is_timeout(), timeout);
        assert_eq!(error.is_unreachable(), unreachable);
        assert_eq!(error.mcp_error().is_some(), mcp_error);
    }

    #[rstest]
    fn an_upstream_mcp_error_passes_through_unchanged() {
        let mapped = mcp_failure().into_mcp();
        assert_eq!(mapped.code, ErrorCode::INVALID_PARAMS);
        assert_eq!(mapped.message, "bad");
    }

    #[rstest]
    #[case::request(Error::MissingField("name"), ErrorCode::INVALID_PARAMS)]
    #[case::forbidden(Error::Forbidden, ErrorCode(-32003))]
    #[case::upstream_fault(Error::InvalidPagination, ErrorCode::INTERNAL_ERROR)]
    fn mcp_error_code_follows_the_fault(#[case] error: Error, #[case] code: ErrorCode) {
        assert_eq!(error.into_mcp().code, code);
    }

    #[rstest]
    #[case::upstream(Error::Upstream(ServiceError::Cancelled {
        reason: Some(SECRET_URL.into())
    }))]
    #[case::authorization(Error::AuthorizationUnavailable(Source::new(std::io::Error::other(
        SECRET_URL
    ))))]
    fn private_causes_stay_out_of_client_messages(#[case] error: Error) {
        assert!(Report::new(&error).to_string().contains("private-secret"));
        assert!(!error.detail().contains("private-secret"));
        assert!(!error.into_mcp().message.contains("private-secret"));
    }
}
