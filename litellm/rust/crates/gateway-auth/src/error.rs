use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
};
use litellm_error::Source;

#[derive(Debug, thiserror::Error, strum::IntoStaticStr)]
pub enum Error {
    #[error("missing or invalid bearer token")]
    #[strum(serialize = "invalid_token")]
    InvalidToken,
    #[error("missing or invalid API key")]
    #[strum(serialize = "invalid_api_key")]
    InvalidApiKey,
    #[error("credential has expired")]
    #[strum(serialize = "expired")]
    Expired,
    #[error("operation is not permitted")]
    #[strum(serialize = "forbidden")]
    Forbidden,
    #[error("authentication service unavailable")]
    #[strum(serialize = "unavailable")]
    Unavailable(#[source] Source),
    #[error("gateway auth not configured")]
    #[strum(serialize = "unconfigured")]
    Unconfigured,
    #[error("authenticated request context is missing")]
    #[strum(serialize = "missing_identity")]
    MissingIdentity,
}

impl Error {
    pub fn status(&self) -> StatusCode {
        match self {
            Self::InvalidToken | Self::InvalidApiKey | Self::Expired => StatusCode::UNAUTHORIZED,
            Self::Forbidden => StatusCode::FORBIDDEN,
            Self::Unavailable(_) => StatusCode::SERVICE_UNAVAILABLE,
            Self::Unconfigured | Self::MissingIdentity => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    pub fn is_denied(&self) -> bool {
        match self {
            Self::InvalidToken | Self::InvalidApiKey | Self::Expired | Self::Forbidden => true,
            Self::Unavailable(_) | Self::Unconfigured | Self::MissingIdentity => false,
        }
    }
}

impl IntoResponse for Error {
    fn into_response(self) -> Response {
        (self.status(), self.to_string()).into_response()
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[tokio::test]
    async fn rejection_body_hides_backend_causes() {
        let error =
            Error::Unavailable(Source::new(std::io::Error::other("private-backend-detail")));
        assert!(
            litellm_error::Report::new(&error)
                .to_string()
                .contains("private-backend-detail")
        );
        let response = error.into_response();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        let body = axum::body::to_bytes(response.into_body(), 4096)
            .await
            .unwrap();
        assert!(
            !std::str::from_utf8(&body)
                .unwrap()
                .contains("private-backend-detail")
        );
    }
}
