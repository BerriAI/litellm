use super::public::PublicError;
use super::rules::{Rule, contains_any};

/// The text branches of `_map_cohere_exception`, in its order.
pub(super) const RULES: &[Rule] = &[
    Rule::new(
        |mapping| {
            contains_any(
                &mapping.error_str,
                &["invalid api token", "No API key provided."],
            )
        },
        PublicError::Authentication,
    ),
    Rule::new(
        |mapping| mapping.error_str.contains("invalid type: parameter"),
        PublicError::BadRequest,
    ),
    Rule::new(
        |mapping| mapping.error_str.contains("too many tokens"),
        PublicError::ContextWindowExceeded,
    ),
    Rule::new(
        |mapping| {
            mapping
                .error_str
                .to_lowercase()
                .contains("internal server error")
        },
        PublicError::InternalServer,
    ),
    Rule::new(
        |mapping| mapping.status.is_none() && mapping.error_str.contains("invalid type:"),
        PublicError::BadRequest,
    ),
    Rule::new(
        |mapping| mapping.status.is_none() && mapping.error_str.contains("Unexpected server error"),
        PublicError::InternalServer,
    ),
];

#[cfg(test)]
mod tests {
    use super::super::rules::first_match;
    use super::super::testing::mapping;
    use super::*;

    fn classified(text: &str) -> Option<PublicError> {
        classified_with(Some(400), text)
    }

    fn classified_with(status: Option<u16>, text: &str) -> Option<PublicError> {
        first_match(RULES, &mapping(status, text)).map(|rule| rule.error)
    }

    #[rstest::rstest]
    #[case::invalid_token("invalid api token", PublicError::Authentication)]
    #[case::no_api_key("No API key provided.", PublicError::Authentication)]
    #[case::invalid_parameter("invalid type: parameter x", PublicError::BadRequest)]
    #[case::too_many_tokens("too many tokens", PublicError::ContextWindowExceeded)]
    #[case::internal_server_text("Internal Server Error", PublicError::InternalServer)]
    #[case::internal_server_any_case("INTERNAL server ERROR", PublicError::InternalServer)]
    fn each_text_rule_claims_its_marker(#[case] text: &str, #[case] expected: PublicError) {
        assert_eq!(classified(text), Some(expected));
    }

    #[rstest::rstest]
    #[case::token_before_parameter(
        "invalid api token invalid type: parameter",
        PublicError::Authentication
    )]
    #[case::parameter_before_tokens(
        "invalid type: parameter too many tokens",
        PublicError::BadRequest
    )]
    #[case::tokens_before_internal(
        "too many tokens Internal Server Error",
        PublicError::ContextWindowExceeded
    )]
    fn the_earlier_rule_wins_when_two_apply(#[case] text: &str, #[case] expected: PublicError) {
        assert_eq!(classified(text), Some(expected));
    }

    #[rstest::rstest]
    #[case::invalid_type(None, "invalid type: x", Some(PublicError::BadRequest))]
    #[case::unexpected_server_error(
        None,
        "Unexpected server error",
        Some(PublicError::InternalServer)
    )]
    #[case::invalid_type_before_unexpected(
        None,
        "invalid type: x Unexpected server error",
        Some(PublicError::BadRequest)
    )]
    #[case::internal_before_invalid_type(
        None,
        "internal server error invalid type: x",
        Some(PublicError::InternalServer)
    )]
    #[case::invalid_type_with_a_status(Some(500), "invalid type: x", None)]
    #[case::unexpected_with_a_status(Some(400), "Unexpected server error", None)]
    fn the_trailing_rules_only_claim_failures_without_a_status(
        #[case] status: Option<u16>,
        #[case] text: &str,
        #[case] expected: Option<PublicError>,
    ) {
        assert_eq!(classified_with(status, text), expected);
    }

    #[test]
    fn text_without_a_marker_is_left_to_the_status_table() {
        assert_eq!(classified("rejected"), None);
    }
}
