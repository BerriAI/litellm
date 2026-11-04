use axum::{
    Json,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use litellm_error::Report;
use serde_json::json;

/// Why a fetched map was not installed. The current one stays
#[derive(Debug, thiserror::Error)]
pub enum Unavailable {
    /// `model-catalog-refresh` renders the fetch failure, retries included, into its reason
    #[error("{reason}")]
    Fetch { reason: String },
    #[error("could not fetch {url}")]
    Source {
        url: String,
        #[source]
        cause: std::sync::Arc<litellm_model_catalog_refresh::Error>,
    },
    #[error("{url} was not modified")]
    NotModified { url: String },
    #[error("invalid model cost map from {url}")]
    Invalid {
        url: String,
        #[source]
        cause: litellm_model_catalog::Error,
    },
    #[error("model cost map from {url} failed integrity validation")]
    Integrity {
        url: String,
        #[source]
        cause: litellm_model_catalog::Error,
    },
    #[error("the gateway is shutting down")]
    ShuttingDown,
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("could not reload the model cost map")]
    Reload(#[source] Unavailable),
}

impl Error {
    pub fn status(&self) -> StatusCode {
        match self {
            Self::Reload(_) => StatusCode::BAD_GATEWAY,
        }
    }

    /// The `detail` the Python proxy's `/reload/model_cost_map` sends
    fn detail(&self) -> String {
        match self {
            Self::Reload(cause) => format!(
                "Failed to reload model cost map: {}. Current pricing data was kept.",
                Report::new(cause)
            ),
        }
    }
}

impl IntoResponse for Error {
    fn into_response(self) -> Response {
        (self.status(), Json(json!({"detail": self.detail()}))).into_response()
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    fn a_failed_reload_reports_its_whole_cause_in_python_s_detail() {
        let cause = litellm_model_catalog::Catalog::parse(b"[]").unwrap_err();
        let error = Error::Reload(Unavailable::Invalid {
            url: "file:///map.json".to_owned(),
            cause,
        });
        assert_eq!(error.status(), StatusCode::BAD_GATEWAY);
        let detail = error.detail();
        assert!(
            detail.starts_with(
                "Failed to reload model cost map: invalid model cost map from file:///map.json: "
            ),
            "{detail}"
        );
        assert!(
            detail.ends_with(". Current pricing data was kept."),
            "{detail}"
        );
    }
}
