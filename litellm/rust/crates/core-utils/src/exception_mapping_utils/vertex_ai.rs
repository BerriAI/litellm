use super::public::PublicError;
use super::rules::{Rule, body_error_code, contains_any, is_context_window_exceeded};

const QUOTA_MARKERS: &[&str] = &[
    "429 Quota exceeded",
    "Quota exceeded for",
    "Resource exhausted",
    "429 Unable to submit request because the service is temporarily out of capacity.",
];

/// The text branches of `_map_vertex_exception`, in its order.
pub(super) const RULES: &[Rule] = &[
    Rule::new(
        |mapping| {
            contains_any(
                &mapping.error_str,
                &[
                    "Vertex AI API has not been used in project",
                    "Unable to find your project",
                ],
            )
        },
        PublicError::BadRequest,
    ),
    Rule::new(
        |mapping| {
            mapping
                .error_str
                .contains("400 Request payload size exceeds")
                || is_context_window_exceeded(&mapping.error_str)
        },
        PublicError::ContextWindowExceeded,
    ),
    Rule::new(
        |mapping| {
            contains_any(
                &mapping.error_str,
                &["None Unknown Error.", "Content has no parts."],
            )
        },
        PublicError::InternalServer,
    ),
    Rule::new(
        |mapping| mapping.error_str.contains("API key not valid."),
        PublicError::Authentication,
    ),
    Rule::new(
        |mapping| {
            contains_any(
                &mapping.error_str,
                &[
                    "The response was blocked.",
                    "Output blocked by content filtering policy",
                ],
            )
        },
        PublicError::ContentPolicyViolation,
    ),
    Rule::new(
        |mapping| {
            contains_any(&mapping.error_str, QUOTA_MARKERS)
                || (mapping
                    .status
                    .is_some_and(|status| (500..600).contains(&status))
                    && body_error_code(&mapping.error_str) == Some(429))
        },
        PublicError::RateLimit,
    ),
    Rule::new(
        |mapping| {
            contains_any(
                &mapping.error_str,
                &["500 Internal Server Error", "The model is overloaded."],
            )
        },
        PublicError::InternalServer,
    ),
];

#[cfg(test)]
mod tests {
    use super::super::rules::first_match;
    use super::super::testing::mapping;
    use super::*;

    fn classified(status: Option<u16>, text: &str) -> Option<PublicError> {
        first_match(RULES, &mapping(status, text)).map(|rule| rule.error)
    }

    #[rstest::rstest]
    #[case::api_not_enabled(
        "Vertex AI API has not been used in project x",
        PublicError::BadRequest
    )]
    #[case::project_not_found("Unable to find your project", PublicError::BadRequest)]
    #[case::payload_too_large(
        "400 Request payload size exceeds the limit",
        PublicError::ContextWindowExceeded
    )]
    #[case::context_window(
        "This model's maximum context length is 10",
        PublicError::ContextWindowExceeded
    )]
    #[case::unknown_error("None Unknown Error.", PublicError::InternalServer)]
    #[case::no_parts("Content has no parts.", PublicError::InternalServer)]
    #[case::api_key_not_valid("API key not valid.", PublicError::Authentication)]
    #[case::response_blocked("The response was blocked.", PublicError::ContentPolicyViolation)]
    #[case::output_blocked(
        "Output blocked by content filtering policy",
        PublicError::ContentPolicyViolation
    )]
    #[case::quota_exceeded_429("429 Quota exceeded", PublicError::RateLimit)]
    #[case::quota_exceeded_for("Quota exceeded for aiplatform", PublicError::RateLimit)]
    #[case::resource_exhausted("Resource exhausted", PublicError::RateLimit)]
    #[case::out_of_capacity(
        "429 Unable to submit request because the service is temporarily out of capacity.",
        PublicError::RateLimit
    )]
    #[case::internal_server_text("500 Internal Server Error", PublicError::InternalServer)]
    #[case::overloaded("The model is overloaded.", PublicError::InternalServer)]
    fn each_text_rule_claims_its_marker(#[case] text: &str, #[case] expected: PublicError) {
        assert_eq!(classified(Some(400), text), Some(expected));
    }

    #[rstest::rstest]
    #[case::server_error_wrapping_a_429(Some(503), Some(PublicError::RateLimit))]
    #[case::lowest_server_error(Some(500), Some(PublicError::RateLimit))]
    #[case::highest_server_error(Some(599), Some(PublicError::RateLimit))]
    #[case::client_error(Some(400), None)]
    #[case::no_status(None, None)]
    fn a_wrapped_429_is_a_rate_limit_only_behind_a_server_error(
        #[case] status: Option<u16>,
        #[case] expected: Option<PublicError>,
    ) {
        assert_eq!(
            classified(status, r#"{"error": {"code": "429"}}"#),
            expected
        );
    }

    #[rstest::rstest]
    #[case::project_before_payload_size(
        "Unable to find your project 400 Request payload size exceeds",
        PublicError::BadRequest
    )]
    #[case::context_window_before_unknown_error(
        "This model's maximum context length is 10 None Unknown Error.",
        PublicError::ContextWindowExceeded
    )]
    #[case::unknown_error_before_api_key(
        "Content has no parts. API key not valid.",
        PublicError::InternalServer
    )]
    #[case::api_key_before_blocked(
        "API key not valid. The response was blocked.",
        PublicError::Authentication
    )]
    #[case::blocked_before_quota(
        "The response was blocked. Resource exhausted",
        PublicError::ContentPolicyViolation
    )]
    #[case::quota_before_overloaded(
        "Resource exhausted The model is overloaded.",
        PublicError::RateLimit
    )]
    fn the_earlier_rule_wins_when_two_apply(#[case] text: &str, #[case] expected: PublicError) {
        assert_eq!(classified(Some(400), text), Some(expected));
    }

    #[rstest::rstest]
    #[case::a_403_in_the_text("got a 403 from 4031 tokens")]
    #[case::python_client_crash("IndexError: list index out of range")]
    #[case::unmarked("rejected")]
    fn text_without_a_marker_is_left_to_the_status_table(#[case] text: &str) {
        assert_eq!(classified(Some(400), text), None);
    }
}
