use super::public::PublicError;
use super::rules::{Rule, contains_any, is_context_window_exceeded};

/// The branches of `_map_anthropic_exception` that the shared status table does not already
/// give: its text rules, in its order, and an overloaded provider's 529, which it raises as an
/// `InternalServerError` where the status table has no class for it.
pub(super) const RULES: &[Rule] = &[
    Rule::new(
        |mapping| {
            contains_any(
                &mapping.error_str,
                &["prompt is too long", "prompt: length"],
            ) || is_context_window_exceeded(&mapping.error_str)
        },
        PublicError::ContextWindowExceeded,
    ),
    Rule::new(
        |mapping| contains_any(&mapping.error_str, &["overloaded_error", "Overloaded"]),
        PublicError::InternalServer,
    ),
    Rule::new(
        |mapping| mapping.error_str.contains("Invalid API Key"),
        PublicError::Authentication,
    ),
    Rule::new(
        |mapping| mapping.error_str.contains("content filtering policy"),
        PublicError::ContentPolicyViolation,
    ),
    Rule::new(
        |mapping| mapping.error_str.contains("Client error '400 Bad Request'"),
        PublicError::BadRequest,
    ),
    Rule::new(
        |mapping| mapping.status == Some(529),
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
    #[case::prompt_too_long(
        Some(400),
        "prompt is too long: 210000 tokens",
        Some(PublicError::ContextWindowExceeded)
    )]
    #[case::prompt_length(
        Some(400),
        "prompt: length must be",
        Some(PublicError::ContextWindowExceeded)
    )]
    #[case::shared_context_marker(
        Some(400),
        "exceeds the maximum number of tokens allowed",
        Some(PublicError::ContextWindowExceeded)
    )]
    #[case::overloaded_body(
        Some(529),
        r#"{"type":"error","error":{"type":"overloaded_error"}}"#,
        Some(PublicError::InternalServer)
    )]
    #[case::invalid_key(Some(400), "Invalid API Key", Some(PublicError::Authentication))]
    #[case::content_filter(
        Some(400),
        "blocked by our content filtering policy",
        Some(PublicError::ContentPolicyViolation)
    )]
    #[case::httpx_400(
        None,
        "Client error '400 Bad Request' for url",
        Some(PublicError::BadRequest)
    )]
    #[case::bare_529(Some(529), "busy", Some(PublicError::InternalServer))]
    #[case::left_to_the_status_table(Some(429), "slow down", None)]
    fn each_branch_claims_its_failure(
        #[case] status: Option<u16>,
        #[case] text: &str,
        #[case] expected: Option<PublicError>,
    ) {
        assert_eq!(classified(status, text), expected);
    }

    #[rstest::rstest]
    fn the_context_window_beats_the_overload() {
        assert_eq!(
            classified(Some(529), "Overloaded: prompt is too long"),
            Some(PublicError::ContextWindowExceeded)
        );
    }
}
