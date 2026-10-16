use axum::{
    Json,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde_json::{Value, json};

use crate::keys::{ModelDenial, Refusal};

#[derive(Debug, thiserror::Error)]
pub enum KeyError {
    #[error("missing or invalid virtual key")]
    Invalid,
    #[error("virtual key has expired")]
    Expired,
    #[error("virtual key is blocked")]
    Blocked,
    #[error("virtual key refused: {0}")]
    Refused(Refusal),
    #[error("virtual key lookup unavailable")]
    Lookup(#[source] Box<dyn std::error::Error + Send + Sync>),
}

/// Messages follow the proxy's `ProxyException`s so clients see the same text from either.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("operation is not permitted")]
    Forbidden,
    #[error(
        "The requested model '{model}' is not available for this API key, or the model name is invalid. Check the models available to you and try again."
    )]
    ModelAccessDenied { model: String, by: ModelDenial },
    #[error(
        "Virtual key is not allowed to call this route. Only allowed to call routes: {allowed}. Tried to call route: {route}"
    )]
    RouteNotAllowed { allowed: String, route: String },
    #[error("Authentication Error - Expired Key.")]
    Expired,
    #[error("Authentication Error, Key is blocked. Update via `/key/unblock` if you're an admin.")]
    Blocked,
    #[error("{0}")]
    Refused(Refusal),
    /// `is_request_body_safe`'s `ValueError`, which `_as_proxy_exception` prefixes and answers
    /// with a 401.
    #[error(
        "Authentication Error, Rejected Request: {0} is not allowed in request body. Clientside passthrough requires explicit admin opt-in via either `general_settings.allow_client_side_credentials = true` (proxy-wide) or `configurable_clientside_auth_params` on the deployment in your proxy config.yaml. Relevant Issue: https://huntr.com/bounties/4001e1a2-7b7a-4776-a3ae-e6692ec3d997"
    )]
    BannedParam(&'static str),
    #[error(
        "Authentication Error, Rejected Request: model_list is not allowed in the request body."
    )]
    ModelListInBody,
    #[error("authenticated request context is missing")]
    MissingIdentity,
    #[error("authentication service unavailable")]
    Unavailable,
    #[error("gateway auth not configured")]
    Unconfigured,
    #[error("Authentication Error, No api key passed in.")]
    MissingToken,
    #[error("Authentication Error, Invalid API key provided.")]
    InvalidToken,
    #[error("gateway authentication unavailable")]
    Secret(#[from] litellm_secrets::Error),
}

impl From<KeyError> for Error {
    fn from(error: KeyError) -> Self {
        match error {
            KeyError::Invalid => Self::InvalidToken,
            KeyError::Expired => Self::Expired,
            KeyError::Blocked => Self::Blocked,
            KeyError::Refused(refusal) => Self::Refused(refusal),
            KeyError::Lookup(_) => Self::Unavailable,
        }
    }
}

impl Error {
    pub fn status(&self) -> StatusCode {
        match self {
            Self::MissingToken
            | Self::InvalidToken
            | Self::Expired
            | Self::Blocked
            | Self::BannedParam(_)
            | Self::ModelListInBody => StatusCode::UNAUTHORIZED,
            Self::Refused(refusal) => match refusal {
                Refusal::TeamMissing { .. } => StatusCode::NOT_FOUND,
                Refusal::TeamBlocked { .. } | Refusal::Unsupported { .. } => {
                    StatusCode::UNAUTHORIZED
                }
                // `litellm.budget_exceeded_status_code`.
                Refusal::BudgetExceeded { .. } => StatusCode::UNPROCESSABLE_ENTITY,
            },
            Self::Forbidden | Self::ModelAccessDenied { .. } | Self::RouteNotAllowed { .. } => {
                StatusCode::FORBIDDEN
            }
            Self::Unavailable => StatusCode::SERVICE_UNAVAILABLE,
            Self::MissingIdentity | Self::Unconfigured | Self::Secret(_) => {
                StatusCode::INTERNAL_SERVER_ERROR
            }
        }
    }

    /// The proxy's `ProxyErrorTypes` value and `param` for this failure.
    fn proxy_type(&self) -> (&'static str, Value) {
        match self {
            Self::InvalidToken => ("token_not_found_in_db", "key".into()),
            Self::Expired => ("expired_key", "None".into()),
            Self::ModelAccessDenied {
                by: ModelDenial::Key,
                ..
            } => ("key_model_access_denied", "model".into()),
            Self::ModelAccessDenied {
                by: ModelDenial::Team,
                ..
            } => ("team_model_access_denied", "model".into()),
            Self::Refused(Refusal::BudgetExceeded { .. }) => ("budget_exceeded", Value::Null),
            _ => ("auth_error", "None".into()),
        }
    }

    /// The JSON body the proxy returns for a `ProxyException`: `code` is the status as a string.
    pub fn body(&self) -> Value {
        let (kind, param) = self.proxy_type();
        json!({"error": {
            "message": self.to_string(),
            "type": kind,
            "param": param,
            "code": self.status().as_u16().to_string(),
        }})
    }
}

impl IntoResponse for Error {
    fn into_response(self) -> Response {
        (self.status(), Json(self.body())).into_response()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum UiAuthError {
    #[error("UI_USERNAME and UI_PASSWORD must be nonempty")]
    Unconfigured,
    #[error("invalid UI credentials or session")]
    Unauthorized,
    #[error("UI authentication unavailable")]
    Unavailable,
    #[error("UI session unavailable")]
    Session(#[from] tower_sessions::session::Error),
}

impl IntoResponse for UiAuthError {
    fn into_response(self) -> Response {
        let status = match self {
            Self::Unauthorized => StatusCode::UNAUTHORIZED,
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        };
        (status, self.to_string()).into_response()
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;
    use crate::keys::{Owner, Restriction};

    #[rstest]
    #[case::missing(Error::MissingToken, 401, "auth_error", json!("None"))]
    #[case::invalid(Error::InvalidToken, 401, "token_not_found_in_db", json!("key"))]
    #[case::expired(Error::Expired, 401, "expired_key", json!("None"))]
    #[case::blocked(Error::Blocked, 401, "auth_error", json!("None"))]
    #[case::key_model(
        Error::ModelAccessDenied { model: "gpt-b".into(), by: ModelDenial::Key },
        403,
        "key_model_access_denied",
        json!("model")
    )]
    #[case::team_model(
        Error::ModelAccessDenied { model: "gpt-b".into(), by: ModelDenial::Team },
        403,
        "team_model_access_denied",
        json!("model")
    )]
    #[case::route(
        Error::RouteNotAllowed { allowed: "['/v1/messages']".into(), route: "/chat/completions".into() },
        403,
        "auth_error",
        json!("None")
    )]
    #[case::team_missing(
        Error::Refused(Refusal::TeamMissing { team_id: "t".into() }),
        404,
        "auth_error",
        json!("None")
    )]
    #[case::team_blocked(
        Error::Refused(Refusal::TeamBlocked { team_id: "t".into() }),
        401,
        "auth_error",
        json!("None")
    )]
    #[case::budget_exceeded(
        Error::Refused(Refusal::BudgetExceeded {
            owner: "Key=key".into(),
            spend: "1.0".into(),
            max_budget: "1.0".into(),
        }),
        422,
        "budget_exceeded",
        Value::Null
    )]
    #[case::unsupported(
        Error::Refused(Refusal::Unsupported {
            restriction: Restriction::RateLimits,
            owner: Owner::Team,
        }),
        401,
        "auth_error",
        json!("None")
    )]
    #[case::unavailable(Error::Unavailable, 503, "auth_error", json!("None"))]
    #[case::banned_param(Error::BannedParam("api_base"), 401, "auth_error", json!("None"))]
    #[case::model_list(Error::ModelListInBody, 401, "auth_error", json!("None"))]
    fn body_matches_the_proxy_exception_shape(
        #[case] error: Error,
        #[case] status: u16,
        #[case] kind: &str,
        #[case] param: Value,
    ) {
        let body = error.body();
        assert_eq!(error.status().as_u16(), status);
        assert_eq!(body["error"]["type"], kind);
        assert_eq!(body["error"]["param"], param);
        assert_eq!(body["error"]["code"], status.to_string());
        assert_eq!(body["error"]["message"], error.to_string());
    }

    #[rstest]
    #[case::model(
        Error::ModelAccessDenied { model: "gpt-b".into(), by: ModelDenial::Team },
        "The requested model 'gpt-b' is not available for this API key, or the model name is invalid. Check the models available to you and try again."
    )]
    #[case::route(
        Error::RouteNotAllowed { allowed: "['llm_api_routes']".into(), route: "/mcp/sse".into() },
        "Virtual key is not allowed to call this route. Only allowed to call routes: ['llm_api_routes']. Tried to call route: /mcp/sse"
    )]
    #[case::team_missing(
        Error::Refused(Refusal::TeamMissing { team_id: "team-1".into() }),
        "{'error': \"Team doesn't exist in db. Team=team-1. Create team via `/team/new` call.\"}"
    )]
    #[case::team_blocked(
        Error::Refused(Refusal::TeamBlocked { team_id: "team-1".into() }),
        "Authentication Error, Team=team-1 is blocked. Update via `/team/unblock` if you're an admin."
    )]
    #[case::budget_exceeded(
        Error::Refused(Refusal::BudgetExceeded {
            owner: "Team=team-1".into(),
            spend: "10.5".into(),
            max_budget: "10.0".into(),
        }),
        "Budget has been exceeded! Team=team-1 Current cost: 10.5, Max budget: 10.0"
    )]
    #[case::unsupported(
        Error::Refused(Refusal::Unsupported {
            restriction: Restriction::RateLimits,
            owner: Owner::TeamMembership,
        }),
        "Authentication Error, keys with rate limits set on its team membership are not supported by this gateway yet."
    )]
    #[case::banned_param(
        Error::BannedParam("aws_bedrock_runtime_endpoint"),
        "Authentication Error, Rejected Request: aws_bedrock_runtime_endpoint is not allowed in request body. Clientside passthrough requires explicit admin opt-in via either `general_settings.allow_client_side_credentials = true` (proxy-wide) or `configurable_clientside_auth_params` on the deployment in your proxy config.yaml. Relevant Issue: https://huntr.com/bounties/4001e1a2-7b7a-4776-a3ae-e6692ec3d997"
    )]
    #[case::model_list(
        Error::ModelListInBody,
        "Authentication Error, Rejected Request: model_list is not allowed in the request body."
    )]
    fn messages_match_the_proxy(#[case] error: Error, #[case] message: &str) {
        assert_eq!(error.to_string(), message);
    }
}
