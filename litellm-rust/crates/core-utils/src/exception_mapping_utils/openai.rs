use super::public::{PublicFailure, StatusClass};
use super::rules::{
    ApiStatus, Kind, ResponseChoice, Rule, apply, contains_any, is_context_window_exceeded,
    is_rate_limit,
};
use super::{DOCS_URL, Mapping};

const OPENAI_URL: &str = "https://api.openai.com/v1";

const ENCRYPTED_CONTENT_HELP: &str = "\n\n This error occurs when load balancing Responses API across deployments with different API keys.\n   Encrypted content is tied to the organization that created it and cannot be decrypted by other organizations.\n\n   Solution: Enable 'encrypted_content_affinity' to route follow-up requests to the correct deployment:\n\n   router_settings:\n     enable_pre_call_checks: true\n     optional_pre_call_checks:\n       - encrypted_content_affinity\n\n   Learn more: https://docs.litellm.ai/docs/response_api#encrypted-content-affinity-multi-region-load-balancing";

const fn with_response(class: StatusClass) -> Kind {
    Kind::Status {
        class,
        response: ResponseChoice::Provider,
    }
}

fn exception_provider(mapping: &Mapping<'_>) -> String {
    if mapping.provider == "openai" {
        "OpenAIException".to_string()
    } else {
        super::exception_provider(mapping.provider)
    }
}

/// The raw message with OpenAI's own names swapped for the provider's.
fn message(mapping: &Mapping<'_>) -> String {
    let provider = mapping.provider;
    mapping
        .original
        .message
        .replace("OPENAI", &provider.to_uppercase())
        .replace("openai.OpenAIError", &format!("{provider}.{provider}Error"))
}

fn prefixed(mapping: &Mapping<'_>, label: &str) -> String {
    format!(
        "{label}{} - {}",
        exception_provider(mapping),
        message(mapping)
    )
}

fn status_is(mapping: &Mapping<'_>, statuses: &[u16]) -> bool {
    mapping
        .original
        .status
        .is_some_and(|status| statuses.contains(&status))
}

/// `_map_openai_exception`, in its branch order.
const RULES: &[Rule] = &[
    Rule {
        when: |mapping| is_rate_limit(&mapping.error_str, mapping.original.status),
        kind: with_response(StatusClass::RateLimit),
        message: |mapping| prefixed(mapping, "RateLimitError: "),
        debug: false,
    },
    Rule {
        when: |mapping| is_context_window_exceeded(&mapping.error_str),
        kind: with_response(StatusClass::ContextWindowExceeded),
        message: |mapping| prefixed(mapping, "ContextWindowExceededError: "),
        debug: true,
    },
    Rule {
        when: |mapping| {
            mapping.error_str.contains("invalid_request_error")
                && mapping.error_str.contains("model_not_found")
        },
        kind: with_response(StatusClass::NotFound),
        message: |mapping| prefixed(mapping, ""),
        debug: true,
    },
    Rule {
        when: |mapping| mapping.error_str.contains("A timeout occurred"),
        kind: Kind::Timeout(None),
        message: |mapping| prefixed(mapping, ""),
        debug: true,
    },
    Rule {
        when: |mapping| {
            let error_str = &mapping.error_str;
            (error_str.contains("invalid_request_error")
                && error_str.contains("content_policy_violation"))
                || (error_str.contains("Invalid prompt")
                    && error_str.contains("violating our usage policy"))
                || error_str
                    .to_lowercase()
                    .contains("request was rejected as a result of the safety system")
        },
        kind: with_response(StatusClass::ContentPolicyViolation),
        message: |mapping| prefixed(mapping, "ContentPolicyViolationError: "),
        debug: true,
    },
    Rule {
        when: |mapping| {
            contains_any(
                &mapping.error_str,
                &["invalid_encrypted_content", "could not be verified"],
            )
        },
        kind: with_response(StatusClass::BadRequest),
        message: |mapping| {
            format!(
                "{} - {}{ENCRYPTED_CONTENT_HELP}",
                exception_provider(mapping),
                message(mapping)
            )
        },
        debug: true,
    },
    Rule {
        when: |mapping| {
            mapping.error_str.contains("invalid_request_error")
                && !mapping.error_str.contains("Incorrect API key provided")
        },
        kind: with_response(StatusClass::BadRequest),
        message: |mapping| prefixed(mapping, ""),
        debug: true,
    },
    Rule {
        when: |mapping| {
            contains_any(
                &mapping.error_str,
                &[
                    "Web server is returning an unknown error",
                    "The server had an error processing your request.",
                ],
            )
        },
        kind: Kind::Status {
            class: StatusClass::InternalServer,
            response: ResponseChoice::Omitted,
        },
        message: |mapping| prefixed(mapping, ""),
        debug: false,
    },
    Rule {
        when: |mapping| mapping.error_str.contains("Request too large"),
        kind: with_response(StatusClass::RateLimit),
        message: |mapping| prefixed(mapping, "RateLimitError: "),
        debug: true,
    },
    Rule {
        when: |mapping| {
            mapping.error_str.contains("The api_key client option must be set either by passing api_key to the client or by setting the OPENAI_API_KEY environment variable")
        },
        kind: with_response(StatusClass::Authentication),
        message: |mapping| prefixed(mapping, "AuthenticationError: "),
        debug: true,
    },
    Rule {
        when: |mapping| {
            mapping
                .error_str
                .contains("Mistral API raised a streaming error")
        },
        kind: Kind::Api {
            status: ApiStatus::Fixed(500),
            request_url: OPENAI_URL,
        },
        message: |mapping| prefixed(mapping, ""),
        debug: true,
    },
    Rule {
        when: |mapping| mapping.original.status.is_none(),
        kind: Kind::ApiConnection,
        message: |mapping| prefixed(mapping, "APIConnectionError: "),
        debug: true,
    },
    Rule {
        when: |mapping| status_is(mapping, &[400, 422]),
        kind: with_response(StatusClass::BadRequest),
        message: |mapping| prefixed(mapping, ""),
        debug: true,
    },
    Rule {
        when: |mapping| status_is(mapping, &[401]),
        kind: with_response(StatusClass::Authentication),
        message: |mapping| prefixed(mapping, "AuthenticationError: "),
        debug: true,
    },
    Rule {
        when: |mapping| status_is(mapping, &[404]),
        kind: with_response(StatusClass::NotFound),
        message: |mapping| prefixed(mapping, "NotFoundError: "),
        debug: true,
    },
    Rule {
        when: |mapping| status_is(mapping, &[408]),
        kind: Kind::Timeout(None),
        message: |mapping| prefixed(mapping, "Timeout Error: "),
        debug: true,
    },
    Rule {
        when: |mapping| status_is(mapping, &[429]),
        kind: with_response(StatusClass::RateLimit),
        message: |mapping| prefixed(mapping, "RateLimitError: "),
        debug: true,
    },
    Rule {
        when: |mapping| status_is(mapping, &[500]),
        kind: with_response(StatusClass::InternalServer),
        message: |mapping| prefixed(mapping, "InternalServerError: "),
        debug: true,
    },
    Rule {
        when: |mapping| status_is(mapping, &[502]),
        kind: with_response(StatusClass::BadGateway),
        message: |mapping| prefixed(mapping, "BadGatewayError: "),
        debug: true,
    },
    Rule {
        when: |mapping| status_is(mapping, &[503]),
        kind: with_response(StatusClass::ServiceUnavailable),
        message: |mapping| prefixed(mapping, "ServiceUnavailableError: "),
        debug: true,
    },
    Rule {
        when: |mapping| status_is(mapping, &[504]),
        kind: Kind::Timeout(Some(504)),
        message: |mapping| prefixed(mapping, "Timeout Error: "),
        debug: true,
    },
    Rule {
        when: |_| true,
        kind: Kind::Api {
            status: ApiStatus::Original,
            request_url: DOCS_URL,
        },
        message: |mapping| prefixed(mapping, "APIError: "),
        debug: true,
    },
];

pub(super) fn map(mapping: &Mapping<'_>) -> Option<PublicFailure> {
    apply(RULES, mapping)
}

#[cfg(test)]
mod tests {
    use super::super::testing::{context, failure, http, status, upstream, with_debug};
    use super::super::{ExceptionFamily, OriginalException, PublicKind};
    use super::*;

    fn mapped(provider: &str, original: &OriginalException) -> PublicFailure {
        let context = context(provider, ExceptionFamily::OpenAiCompatible);
        map(&Mapping::new(&context, original)).expect("the OpenAI table ends in a catch-all")
    }

    fn kind(class: StatusClass, status_code: u16, body: &str) -> PublicKind {
        status(class, upstream(status_code, body))
    }

    #[rstest::rstest]
    #[case::rate_limit_phrase(
        400,
        "rate limit reached",
        failure(
            kind(StatusClass::RateLimit, 400, "rate limit reached"),
            "RateLimitError: MistralException - rate limit reached",
            "mistral",
        )
    )]
    #[case::context_window(
        500,
        "This model's maximum context length is 10",
        with_debug(failure(
            kind(
                StatusClass::ContextWindowExceeded,
                500,
                "This model's maximum context length is 10"
            ),
            "ContextWindowExceededError: MistralException - This model's maximum context length is 10",
            "mistral",
        ))
    )]
    #[case::model_not_found(
        400,
        "invalid_request_error model_not_found",
        with_debug(failure(
            kind(StatusClass::NotFound, 400, "invalid_request_error model_not_found"),
            "MistralException - invalid_request_error model_not_found",
            "mistral",
        ))
    )]
    #[case::timeout_occurred(400, "A timeout occurred", with_debug(failure(
        PublicKind::Timeout { status: None },
        "MistralException - A timeout occurred",
        "mistral",
    )))]
    #[case::content_policy_error_code(
        400,
        "invalid_request_error content_policy_violation",
        with_debug(failure(
            kind(
                StatusClass::ContentPolicyViolation,
                400,
                "invalid_request_error content_policy_violation"
            ),
            "ContentPolicyViolationError: MistralException - invalid_request_error content_policy_violation",
            "mistral",
        ))
    )]
    #[case::content_policy_usage_policy(
        400,
        "Invalid prompt violating our usage policy",
        with_debug(failure(
            kind(
                StatusClass::ContentPolicyViolation,
                400,
                "Invalid prompt violating our usage policy"
            ),
            "ContentPolicyViolationError: MistralException - Invalid prompt violating our usage policy",
            "mistral",
        ))
    )]
    #[case::content_policy_safety_system(
        400,
        "Request was rejected as a result of the safety system",
        with_debug(failure(
            kind(
                StatusClass::ContentPolicyViolation,
                400,
                "Request was rejected as a result of the safety system"
            ),
            "ContentPolicyViolationError: MistralException - Request was rejected as a result of the safety system",
            "mistral",
        ))
    )]
    #[case::encrypted_content(400, "invalid_encrypted_content", with_debug(failure(
        kind(StatusClass::BadRequest, 400, "invalid_encrypted_content"),
        &format!("MistralException - invalid_encrypted_content{ENCRYPTED_CONTENT_HELP}"),
        "mistral",
    )))]
    #[case::unverifiable_content(400, "could not be verified", with_debug(failure(
        kind(StatusClass::BadRequest, 400, "could not be verified"),
        &format!("MistralException - could not be verified{ENCRYPTED_CONTENT_HELP}"),
        "mistral",
    )))]
    #[case::invalid_request(
        429,
        "invalid_request_error bad field",
        with_debug(failure(
            kind(StatusClass::BadRequest, 429, "invalid_request_error bad field"),
            "MistralException - invalid_request_error bad field",
            "mistral",
        ))
    )]
    #[case::unknown_server_error(
        400,
        "Web server is returning an unknown error",
        failure(
            status(StatusClass::InternalServer, None),
            "MistralException - Web server is returning an unknown error",
            "mistral",
        )
    )]
    #[case::server_had_an_error(
        400,
        "The server had an error processing your request.",
        failure(
            status(StatusClass::InternalServer, None),
            "MistralException - The server had an error processing your request.",
            "mistral",
        )
    )]
    #[case::request_too_large(
        400,
        "Request too large",
        with_debug(failure(
            kind(StatusClass::RateLimit, 400, "Request too large"),
            "RateLimitError: MistralException - Request too large",
            "mistral",
        ))
    )]
    #[case::missing_client_api_key(
        400,
        "The api_key client option must be set either by passing api_key to the client or by setting the OPENAI_API_KEY environment variable",
        with_debug(failure(
            kind(
                StatusClass::Authentication,
                400,
                "The api_key client option must be set either by passing api_key to the client or by setting the OPENAI_API_KEY environment variable",
            ),
            "AuthenticationError: MistralException - The api_key client option must be set either by passing api_key to the client or by setting the MISTRAL_API_KEY environment variable",
            "mistral",
        ))
    )]
    #[case::mistral_streaming_error(400, "Mistral API raised a streaming error", with_debug(failure(
        PublicKind::Api { status: 500, request_url: OPENAI_URL },
        "MistralException - Mistral API raised a streaming error",
        "mistral",
    )))]
    fn each_text_rule_maps_by_the_body(
        #[case] status_code: u16,
        #[case] body: &str,
        #[case] expected: PublicFailure,
    ) {
        assert_eq!(mapped("mistral", &http(status_code, body)), expected);
    }

    #[rstest::rstest]
    #[case::bad_request(
        400,
        kind(StatusClass::BadRequest, 400, "rejected"),
        "MistralException - rejected"
    )]
    #[case::unprocessable(
        422,
        kind(StatusClass::BadRequest, 422, "rejected"),
        "MistralException - rejected"
    )]
    #[case::authentication(
        401,
        kind(StatusClass::Authentication, 401, "rejected"),
        "AuthenticationError: MistralException - rejected"
    )]
    #[case::not_found(
        404,
        kind(StatusClass::NotFound, 404, "rejected"),
        "NotFoundError: MistralException - rejected"
    )]
    #[case::request_timeout(408, PublicKind::Timeout { status: None }, "Timeout Error: MistralException - rejected")]
    #[case::rate_limited(
        429,
        kind(StatusClass::RateLimit, 429, "rejected"),
        "RateLimitError: MistralException - rejected"
    )]
    #[case::internal_server(
        500,
        kind(StatusClass::InternalServer, 500, "rejected"),
        "InternalServerError: MistralException - rejected"
    )]
    #[case::bad_gateway(
        502,
        kind(StatusClass::BadGateway, 502, "rejected"),
        "BadGatewayError: MistralException - rejected"
    )]
    #[case::service_unavailable(
        503,
        kind(StatusClass::ServiceUnavailable, 503, "rejected"),
        "ServiceUnavailableError: MistralException - rejected"
    )]
    #[case::gateway_timeout(504, PublicKind::Timeout { status: Some(504) }, "Timeout Error: MistralException - rejected")]
    #[case::any_other_status(409, PublicKind::Api { status: 409, request_url: DOCS_URL }, "APIError: MistralException - rejected")]
    fn each_status_rule_maps_by_the_status(
        #[case] status_code: u16,
        #[case] kind: PublicKind,
        #[case] message: &str,
    ) {
        assert_eq!(
            mapped("mistral", &http(status_code, "rejected")),
            with_debug(failure(kind, message, "mistral"))
        );
    }

    #[test]
    fn a_failure_without_a_status_is_a_connection_error() {
        let original = OriginalException::Response {
            message: "invalid OCR response field: pages".into(),
        };
        assert_eq!(
            mapped("mistral", &original),
            with_debug(failure(
                PublicKind::ApiConnection,
                "APIConnectionError: MistralException - invalid OCR response field: pages",
                "mistral"
            ))
        );
    }

    #[rstest::rstest]
    #[case::rate_limit_before_context_window(
        400,
        "rate limit and This model's maximum context length is 10",
        kind(
            StatusClass::RateLimit,
            400,
            "rate limit and This model's maximum context length is 10"
        ),
        "RateLimitError: MistralException - rate limit and This model's maximum context length is 10",
        false
    )]
    #[case::context_window_before_content_policy(
        400,
        "This model's maximum context length is 10 invalid_request_error content_policy_violation",
        kind(
            StatusClass::ContextWindowExceeded,
            400,
            "This model's maximum context length is 10 invalid_request_error content_policy_violation"
        ),
        "ContextWindowExceededError: MistralException - This model's maximum context length is 10 invalid_request_error content_policy_violation",
        true
    )]
    #[case::model_not_found_before_invalid_request(
        400,
        "invalid_request_error model_not_found",
        kind(StatusClass::NotFound, 400, "invalid_request_error model_not_found"),
        "MistralException - invalid_request_error model_not_found",
        true
    )]
    #[case::timeout_before_invalid_request(
        400,
        "A timeout occurred invalid_request_error",
        PublicKind::Timeout { status: None },
        "MistralException - A timeout occurred invalid_request_error",
        true
    )]
    #[case::content_policy_before_invalid_request(
        400,
        "invalid_request_error content_policy_violation",
        kind(
            StatusClass::ContentPolicyViolation,
            400,
            "invalid_request_error content_policy_violation"
        ),
        "ContentPolicyViolationError: MistralException - invalid_request_error content_policy_violation",
        true
    )]
    #[case::invalid_request_with_a_bad_key_falls_to_the_status(
        401,
        "invalid_request_error Incorrect API key provided",
        kind(
            StatusClass::Authentication,
            401,
            "invalid_request_error Incorrect API key provided"
        ),
        "AuthenticationError: MistralException - invalid_request_error Incorrect API key provided",
        true
    )]
    #[case::text_rules_before_status(
        429,
        "invalid_request_error bad field",
        kind(StatusClass::BadRequest, 429, "invalid_request_error bad field"),
        "MistralException - invalid_request_error bad field",
        true
    )]
    #[case::echoed_429_is_not_a_rate_limit(
        400,
        "token 429 in the prompt",
        kind(StatusClass::BadRequest, 400, "token 429 in the prompt"),
        "MistralException - token 429 in the prompt",
        true
    )]
    fn the_earlier_rule_wins_when_two_apply(
        #[case] status_code: u16,
        #[case] body: &str,
        #[case] kind: PublicKind,
        #[case] message: &str,
        #[case] debug: bool,
    ) {
        let expected = failure(kind, message, "mistral");
        assert_eq!(
            mapped("mistral", &http(status_code, body)),
            if debug {
                with_debug(expected)
            } else {
                expected
            }
        );
    }

    #[rstest::rstest]
    #[case::provider_names_replace_openai(
        "azure_ai",
        "OPENAI said openai.OpenAIError",
        "Azure_aiException - AZURE_AI said azure_ai.azure_aiError"
    )]
    #[case::openai_keeps_its_own_name("openai", "rejected", "OpenAIException - rejected")]
    fn the_message_names_the_provider(
        #[case] provider: &str,
        #[case] body: &str,
        #[case] message: &str,
    ) {
        assert_eq!(
            mapped(provider, &http(400, body)),
            with_debug(failure(
                kind(StatusClass::BadRequest, 400, body),
                message,
                provider
            ))
        );
    }
}
