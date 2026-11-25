use axum::{
    Json,
    http::{HeaderName, HeaderValue, StatusCode},
    response::{AppendHeaders, IntoResponse, Response},
};
use serde_json::{Value, json};

use crate::hooks::Refusal;

const MESSAGES: &str = "/v1/messages";
/// The `type` and the `code` of Python's `ProxyRateLimitError`.
const THROTTLING: &str = "throttling_error";
const CODE: &str = "429";

/// Python's `ProxyRateLimitError`: a limit on the caller refused the call before it ran.
#[derive(Debug, PartialEq)]
pub struct RateLimited {
    pub message: String,
    /// In the order Python sends them.
    pub headers: Vec<(String, String)>,
}

pub(crate) fn header_pairs(headers: Vec<(String, String)>) -> Vec<(HeaderName, HeaderValue)> {
    headers
        .into_iter()
        .filter_map(|(name, value)| {
            Some((
                HeaderName::try_from(name).ok()?,
                HeaderValue::try_from(value).ok()?,
            ))
        })
        .collect()
}

impl RateLimited {
    /// The 429 Python's authentication answers before the handler of the route matched at `path`
    /// runs: the Anthropic envelope on `/v1/messages`, the OpenAI one everywhere else, each with
    /// the limit's headers and the error type the proxy adds. The refusal of a route's own
    /// handler is a [`Refusal`], which the route answers as it answers any other error.
    pub fn response(self, path: &str) -> Response {
        (
            StatusCode::TOO_MANY_REQUESTS,
            AppendHeaders(header_pairs(self.headers)),
            Json(body(path, self.message)),
        )
            .into_response()
    }
}

impl From<RateLimited> for Refusal {
    fn from(limited: RateLimited) -> Self {
        Self {
            status: StatusCode::TOO_MANY_REQUESTS,
            message: limited.message,
            kind: Some(THROTTLING.to_owned()),
            code: Some(CODE.to_owned()),
            headers: limited.headers,
            at_authentication: false,
        }
    }
}

fn body(path: &str, message: String) -> Value {
    if path == MESSAGES {
        return json!({"type": "error", "error": {
            "type": "rate_limit_error",
            "message": message,
            "litellm_error_type": THROTTLING,
        }});
    }
    json!({"error": {
        "message": message,
        "type": THROTTLING,
        "param": null,
        "code": CODE,
    }})
}

#[cfg(test)]
mod tests {
    use axum::body::to_bytes;
    use rstest::rstest;

    use super::*;
    use crate::Error;

    fn limited() -> RateLimited {
        RateLimited {
            message: "over".into(),
            headers: vec![
                ("retry-after".into(), "60".into()),
                ("rate_limited_by".into(), "api_key".into()),
            ],
        }
    }

    #[rstest]
    #[case::messages(
        "/v1/messages",
        json!({"type": "error", "error": {
            "type": "rate_limit_error",
            "message": "over",
            "litellm_error_type": "throttling_error",
        }})
    )]
    #[case::ocr(
        "/v1/ocr",
        json!({"error": {"message": "over", "type": "throttling_error", "param": null, "code": "429"}})
    )]
    #[tokio::test]
    async fn authentications_refusal_is_a_429_in_the_routes_own_envelope_with_its_headers(
        #[case] path: &str,
        #[case] expected: Value,
    ) {
        let response = limited().response(path);

        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(response.headers()["retry-after"], "60");
        assert_eq!(response.headers()["rate_limited_by"], "api_key");
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        assert_eq!(serde_json::from_slice::<Value>(&body).unwrap(), expected);
    }

    #[tokio::test]
    async fn an_openai_handlers_refusal_is_the_same_answer_authentication_gives() {
        let response = Error::Refused(Refusal::from(limited())).openai_response();

        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(response.headers()["retry-after"], "60");
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        assert_eq!(
            serde_json::from_slice::<Value>(&body).unwrap(),
            json!({"error": {"message": "over", "type": "throttling_error", "param": null, "code": "429"}})
        );
    }

    #[rstest]
    fn the_messages_handlers_own_refusal_carries_neither_the_limits_headers_nor_its_type() {
        let error = Error::Refused(Refusal::from(limited()));

        assert_eq!(
            (error.status(), error.body(None)),
            (
                StatusCode::TOO_MANY_REQUESTS,
                json!({"type": "error", "error": {"type": "rate_limit_error", "message": "over"}})
            )
        );
    }
}
