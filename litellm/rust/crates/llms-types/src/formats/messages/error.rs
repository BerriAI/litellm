/// Every error type on https://platform.claude.com/docs/en/api/errors as of 2026-11-18, the
/// `type` of the `error` in an error response and in a stream's `error` event. A type a host
/// adds rides in `Other` untouched.
#[derive(
    Clone,
    Debug,
    PartialEq,
    Eq,
    strum::EnumIter,
    strum::EnumString,
    strum::Display,
    serde_with::DeserializeFromStr,
    serde_with::SerializeDisplay,
)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[cfg_attr(feature = "schema", schemars(from = "String", into = "String"))]
#[strum(serialize_all = "snake_case")]
pub enum MessagesErrorType {
    InvalidRequestError,
    AuthenticationError,
    BillingError,
    PermissionError,
    NotFoundError,
    ConflictError,
    RequestTooLarge,
    RateLimitError,
    ApiError,
    TimeoutError,
    OverloadedError,
    #[strum(default, transparent)]
    Other(String),
}

impl MessagesErrorType {
    /// The status Anthropic answers a request with when it fails with this error, which an
    /// `error` event of this type in a stream stands for. `timeout_error` is the 504 of
    /// `ANTHROPIC_ERROR_STATUS_CODE_MAP` in `litellm/llms/anthropic/common_utils.py`, and a type
    /// the page gives no status of its own is a 500.
    pub fn status(&self) -> u16 {
        match self {
            Self::InvalidRequestError => 400,
            Self::AuthenticationError => 401,
            Self::PermissionError => 403,
            Self::NotFoundError => 404,
            Self::RequestTooLarge => 413,
            Self::RateLimitError => 429,
            Self::TimeoutError => 504,
            Self::OverloadedError => 529,
            Self::BillingError | Self::ConflictError | Self::ApiError | Self::Other(_) => 500,
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::{Value, json};

    use super::*;

    #[rstest]
    #[case::suffixed(json!("overloaded_error"), MessagesErrorType::OverloadedError)]
    #[case::unsuffixed(json!("request_too_large"), MessagesErrorType::RequestTooLarge)]
    #[case::unknown(json!("future_error"), MessagesErrorType::Other("future_error".into()))]
    #[case::case_sensitive(json!("Api_Error"), MessagesErrorType::Other("Api_Error".into()))]
    fn an_error_type_round_trips_as_its_wire_spelling(
        #[case] wire: Value,
        #[case] expected: MessagesErrorType,
    ) {
        let read: MessagesErrorType = serde_json::from_value(wire.clone()).unwrap();

        assert_eq!(
            (&read, serde_json::to_value(&read).unwrap()),
            (&expected, wire)
        );
    }

    /// The statuses are the ones on https://platform.claude.com/docs/en/api/errors as of
    /// 2026-11-18. An overloaded provider has to stay apart from a plain server error, since the
    /// router retries and cools down by status.
    #[rstest]
    #[case::overloaded(MessagesErrorType::OverloadedError, 529)]
    #[case::a_type_a_host_adds(MessagesErrorType::Other("future_error".into()), 500)]
    fn an_error_type_stands_for_the_status_its_request_is_answered_with(
        #[case] error_type: MessagesErrorType,
        #[case] status: u16,
    ) {
        assert_eq!(error_type.status(), status);
    }

    #[rstest]
    fn a_type_that_is_not_text_is_rejected(
        #[values(json!(529), json!(null), json!({}))] wire: Value,
    ) {
        assert!(serde_json::from_value::<MessagesErrorType>(wire).is_err());
    }
}
