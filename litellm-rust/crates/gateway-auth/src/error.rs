use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
};
use litellm_error::Source;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("missing or invalid API key")]
    InvalidToken,
    #[error("invalid UI credentials or session")]
    InvalidSession,
    #[error("credential has expired")]
    Expired,
    #[error("operation is not permitted")]
    Forbidden,
    #[error("authentication service unavailable")]
    Unavailable(#[source] Source),
    #[error("gateway authentication unavailable")]
    Secret(#[from] litellm_secrets::Error),
    #[error("UI session unavailable")]
    Session(#[from] tower_sessions::session::Error),
    #[error("gateway auth not configured")]
    Unconfigured,
    #[error("UI_USERNAME and UI_PASSWORD must be nonempty")]
    UiUnconfigured,
    #[error("authenticated request context is missing")]
    MissingIdentity,
    #[error("UI session layer is not installed")]
    MissingSessionLayer,
}

impl Error {
    pub fn status(&self) -> StatusCode {
        match self {
            Self::InvalidToken | Self::InvalidSession | Self::Expired => StatusCode::UNAUTHORIZED,
            Self::Forbidden => StatusCode::FORBIDDEN,
            Self::Unavailable(_) => StatusCode::SERVICE_UNAVAILABLE,
            Self::Secret(_)
            | Self::Session(_)
            | Self::Unconfigured
            | Self::UiUnconfigured
            | Self::MissingIdentity
            | Self::MissingSessionLayer => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    pub fn is_denied(&self) -> bool {
        match self {
            Self::InvalidToken | Self::InvalidSession | Self::Expired | Self::Forbidden => true,
            Self::Unavailable(_)
            | Self::Secret(_)
            | Self::Session(_)
            | Self::Unconfigured
            | Self::UiUnconfigured
            | Self::MissingIdentity
            | Self::MissingSessionLayer => false,
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
