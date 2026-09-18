use super::public::PublicError;
use super::rules::{Rule, contains_any, is_context_window_exceeded, is_rate_limit};

const ENCRYPTED_CONTENT_HELP: &str = "\n\n This error occurs when load balancing Responses API across deployments with different API keys.\n   Encrypted content is tied to the organization that created it and cannot be decrypted by other organizations.\n\n   Solution: Enable 'encrypted_content_affinity' to route follow-up requests to the correct deployment:\n\n   router_settings:\n     enable_pre_call_checks: true\n     optional_pre_call_checks:\n       - encrypted_content_affinity\n\n   Learn more: https://docs.litellm.ai/docs/response_api#encrypted-content-affinity-multi-region-load-balancing";

/// The text branches of `_map_openai_exception`, in its order.
pub(super) const RULES: &[Rule] = &[
    Rule::new(
        |mapping| is_rate_limit(&mapping.error_str, mapping.status),
        PublicError::RateLimit,
    ),
    Rule::new(
        |mapping| is_context_window_exceeded(&mapping.error_str),
        PublicError::ContextWindowExceeded,
    ),
    Rule::new(
        |mapping| {
            mapping.error_str.contains("invalid_request_error")
                && mapping.error_str.contains("model_not_found")
        },
        PublicError::NotFound,
    ),
    Rule::new(
        |mapping| mapping.error_str.contains("A timeout occurred"),
        PublicError::Timeout { status: 408 },
    ),
    Rule::new(
        |mapping| {
            let error_str = &mapping.error_str;
            (error_str.contains("invalid_request_error")
                && error_str.contains("content_policy_violation"))
                || (error_str.contains("Invalid prompt")
                    && error_str.contains("violating our usage policy"))
                || error_str
                    .to_lowercase()
                    .contains("request was rejected as a result of the safety system")
        },
        PublicError::ContentPolicyViolation,
    ),
    Rule {
        hint: ENCRYPTED_CONTENT_HELP,
        ..Rule::new(
            |mapping| {
                contains_any(
                    &mapping.error_str,
                    &["invalid_encrypted_content", "could not be verified"],
                )
            },
            PublicError::BadRequest,
        )
    },
    Rule::new(
        |mapping| {
            mapping.error_str.contains("invalid_request_error")
                && !mapping.error_str.contains("Incorrect API key provided")
        },
        PublicError::BadRequest,
    ),
    Rule::new(
        |mapping| {
            contains_any(
                &mapping.error_str,
                &[
                    "Web server is returning an unknown error",
                    "The server had an error processing your request.",
                ],
            )
        },
        PublicError::InternalServer,
    ),
    Rule::new(
        |mapping| mapping.error_str.contains("Request too large"),
        PublicError::RateLimit,
    ),
    Rule::new(
        |mapping| {
            mapping
                .error_str
                .contains("Mistral API raised a streaming error")
        },
        PublicError::Api { status: 500 },
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
    #[case::rate_limit_phrase("rate limit reached", PublicError::RateLimit)]
    #[case::context_window(
        "This model's maximum context length is 10",
        PublicError::ContextWindowExceeded
    )]
    #[case::model_not_found("invalid_request_error model_not_found", PublicError::NotFound)]
    #[case::timeout_occurred("A timeout occurred", PublicError::Timeout { status: 408 })]
    #[case::content_policy_error_code(
        "invalid_request_error content_policy_violation",
        PublicError::ContentPolicyViolation
    )]
    #[case::content_policy_usage_policy(
        "Invalid prompt violating our usage policy",
        PublicError::ContentPolicyViolation
    )]
    #[case::content_policy_safety_system(
        "Request was rejected as a result of the safety system",
        PublicError::ContentPolicyViolation
    )]
    #[case::encrypted_content("invalid_encrypted_content", PublicError::BadRequest)]
    #[case::unverifiable_content("could not be verified", PublicError::BadRequest)]
    #[case::invalid_request("invalid_request_error bad field", PublicError::BadRequest)]
    #[case::unknown_server_error(
        "Web server is returning an unknown error",
        PublicError::InternalServer
    )]
    #[case::server_had_an_error(
        "The server had an error processing your request.",
        PublicError::InternalServer
    )]
    #[case::request_too_large("Request too large", PublicError::RateLimit)]
    #[case::mistral_streaming_error(
        "Mistral API raised a streaming error",
        PublicError::Api { status: 500 }
    )]
    fn each_text_rule_claims_its_marker(#[case] text: &str, #[case] expected: PublicError) {
        assert_eq!(classified(Some(400), text), Some(expected));
    }

    #[rstest::rstest]
    #[case::rate_limit_before_context_window(
        "rate limit and This model's maximum context length is 10",
        PublicError::RateLimit
    )]
    #[case::context_window_before_content_policy(
        "This model's maximum context length is 10 invalid_request_error content_policy_violation",
        PublicError::ContextWindowExceeded
    )]
    #[case::model_not_found_before_invalid_request(
        "invalid_request_error model_not_found",
        PublicError::NotFound
    )]
    #[case::timeout_before_invalid_request(
        "A timeout occurred invalid_request_error",
        PublicError::Timeout { status: 408 }
    )]
    #[case::content_policy_before_invalid_request(
        "invalid_request_error content_policy_violation",
        PublicError::ContentPolicyViolation
    )]
    #[case::encrypted_content_before_invalid_request(
        "invalid_request_error invalid_encrypted_content",
        PublicError::BadRequest
    )]
    fn the_earlier_rule_wins_when_two_apply(#[case] text: &str, #[case] expected: PublicError) {
        assert_eq!(classified(Some(400), text), Some(expected));
    }

    #[rstest::rstest]
    #[case::encrypted_content("invalid_encrypted_content", ENCRYPTED_CONTENT_HELP)]
    #[case::plain_invalid_request("invalid_request_error bad field", "")]
    fn only_encrypted_content_failures_carry_the_affinity_help(
        #[case] text: &str,
        #[case] hint: &str,
    ) {
        assert_eq!(
            first_match(RULES, &mapping(Some(400), text)).map(|rule| rule.hint),
            Some(hint)
        );
    }

    #[rstest::rstest]
    #[case::bad_key_is_left_to_the_status("invalid_request_error Incorrect API key provided")]
    #[case::echoed_429_is_not_a_rate_limit("token 429 in the prompt")]
    #[case::unmarked("rejected")]
    fn text_without_a_marker_is_left_to_the_status_table(#[case] text: &str) {
        assert_eq!(classified(Some(400), text), None);
    }

    #[test]
    fn a_standalone_429_counts_only_with_a_429_status() {
        assert_eq!(
            classified(Some(429), "got 429 back"),
            Some(PublicError::RateLimit)
        );
    }
}
