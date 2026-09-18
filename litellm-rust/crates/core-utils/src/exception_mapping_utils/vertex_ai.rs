use super::public::{PublicFailure, PublicKind, ResponseArg, StatusClass, UpstreamResponse};
use super::rules::{
    Kind, ResponseChoice, Rule, apply, body_error_code, contains_any, is_context_window_exceeded,
};
use super::{Mapping, python_capitalize};

const VERTEX_URL: &str = "https://cloud.google.com/vertex-ai/";
const VERTEX_URL_WITH_SPACE: &str = " https://cloud.google.com/vertex-ai/";

const QUOTA_MARKERS: &[&str] = &[
    "429 Quota exceeded",
    "Quota exceeded for",
    "Resource exhausted",
    "IndexError: list index out of range",
    "429 Unable to submit request because the service is temporarily out of capacity.",
];

const fn stubbed(class: StatusClass, status: u16, url: &'static str) -> Kind {
    Kind::Status {
        class,
        response: ResponseChoice::Stub { status, url },
    }
}

const fn bare(class: StatusClass) -> Kind {
    Kind::Status {
        class,
        response: ResponseChoice::Omitted,
    }
}

/// `{Provider}Exception{label} - {error_str}` with Python's `str.capitalize()`.
fn capitalized(mapping: &Mapping<'_>, label: &str) -> String {
    format!(
        "{}Exception{label} - {}",
        python_capitalize(mapping.provider),
        mapping.error_str
    )
}

/// `litellm.{Class}: {provider}Exception - {error_str}` with the provider as given.
fn litellm_prefixed(mapping: &Mapping<'_>, class: &str) -> String {
    format!(
        "litellm.{class}: {}Exception - {}",
        mapping.provider, mapping.error_str
    )
}

fn status_is(mapping: &Mapping<'_>, status: u16) -> bool {
    mapping.original.status == Some(status)
}

/// `_map_vertex_exception`, in its branch order. A failure no rule claims falls through
/// to the status table.
const RULES: &[Rule] = &[
    Rule {
        when: |mapping| {
            contains_any(
                &mapping.error_str,
                &[
                    "Vertex AI API has not been used in project",
                    "Unable to find your project",
                ],
            )
        },
        kind: stubbed(StatusClass::BadRequest, 400, VERTEX_URL_WITH_SPACE),
        message: |mapping| litellm_prefixed(mapping, "BadRequestError"),
        debug: true,
    },
    Rule {
        when: |mapping| {
            mapping
                .error_str
                .contains("400 Request payload size exceeds")
        },
        kind: bare(StatusClass::ContextWindowExceeded),
        message: |mapping| capitalized(mapping, ""),
        debug: false,
    },
    Rule {
        when: |mapping| is_context_window_exceeded(&mapping.error_str),
        kind: bare(StatusClass::ContextWindowExceeded),
        message: |mapping| format!("ContextWindowExceededError: {}", capitalized(mapping, "")),
        debug: true,
    },
    Rule {
        when: |mapping| {
            contains_any(
                &mapping.error_str,
                &["None Unknown Error.", "Content has no parts."],
            )
        },
        kind: Kind::Status {
            class: StatusClass::InternalServer,
            response: ResponseChoice::InternalServerStub,
        },
        message: |mapping| litellm_prefixed(mapping, "InternalServerError"),
        debug: true,
    },
    Rule {
        when: |mapping| mapping.error_str.contains("API key not valid."),
        kind: bare(StatusClass::Authentication),
        message: |mapping| capitalized(mapping, ""),
        debug: true,
    },
    Rule {
        when: |mapping| mapping.error_str.contains("403"),
        kind: stubbed(StatusClass::BadRequest, 403, VERTEX_URL_WITH_SPACE),
        message: |mapping| capitalized(mapping, " BadRequestError"),
        debug: true,
    },
    Rule {
        when: |mapping| {
            contains_any(
                &mapping.error_str,
                &[
                    "The response was blocked.",
                    "Output blocked by content filtering policy",
                ],
            )
        },
        kind: stubbed(
            StatusClass::ContentPolicyViolation,
            400,
            VERTEX_URL_WITH_SPACE,
        ),
        message: |mapping| capitalized(mapping, " ContentPolicyViolationError"),
        debug: true,
    },
    Rule {
        when: |mapping| {
            contains_any(&mapping.error_str, QUOTA_MARKERS)
                || (mapping
                    .original
                    .status
                    .is_some_and(|status| (500..600).contains(&status))
                    && body_error_code(&mapping.error_str) == Some(429))
        },
        kind: stubbed(StatusClass::RateLimit, 429, VERTEX_URL_WITH_SPACE),
        message: |mapping| litellm_prefixed(mapping, "RateLimitError"),
        debug: true,
    },
    Rule {
        when: |mapping| {
            contains_any(
                &mapping.error_str,
                &["500 Internal Server Error", "The model is overloaded."],
            )
        },
        kind: bare(StatusClass::InternalServer),
        message: |mapping| litellm_prefixed(mapping, "InternalServerError"),
        debug: true,
    },
    Rule {
        when: |mapping| status_is(mapping, 400),
        kind: stubbed(StatusClass::BadRequest, 400, VERTEX_URL),
        message: |mapping| capitalized(mapping, " BadRequestError"),
        debug: true,
    },
    Rule {
        when: |mapping| status_is(mapping, 401),
        kind: bare(StatusClass::Authentication),
        message: |mapping| capitalized(mapping, ""),
        debug: false,
    },
    Rule {
        when: |mapping| status_is(mapping, 403),
        kind: stubbed(StatusClass::PermissionDenied, 403, VERTEX_URL),
        message: |mapping| capitalized(mapping, ""),
        debug: false,
    },
    Rule {
        when: |mapping| status_is(mapping, 404),
        kind: bare(StatusClass::NotFound),
        message: |mapping| capitalized(mapping, ""),
        debug: false,
    },
    Rule {
        when: |mapping| status_is(mapping, 408),
        kind: Kind::Timeout(None),
        message: |mapping| capitalized(mapping, ""),
        debug: false,
    },
    Rule {
        when: |mapping| status_is(mapping, 429),
        kind: stubbed(StatusClass::RateLimit, 429, VERTEX_URL_WITH_SPACE),
        message: |mapping| format!("litellm.RateLimitError: {}", capitalized(mapping, "")),
        debug: true,
    },
    Rule {
        when: |mapping| status_is(mapping, 500),
        kind: Kind::Status {
            class: StatusClass::InternalServer,
            response: ResponseChoice::InternalServerStub,
        },
        message: |mapping| capitalized(mapping, " InternalServerError"),
        debug: true,
    },
    Rule {
        when: |mapping| status_is(mapping, 502),
        kind: Kind::ApiConnection,
        message: |mapping| capitalized(mapping, ""),
        debug: false,
    },
    Rule {
        when: |mapping| status_is(mapping, 503),
        kind: bare(StatusClass::ServiceUnavailable),
        message: |mapping| capitalized(mapping, ""),
        debug: false,
    },
];

pub(super) fn map(mapping: &Mapping<'_>) -> Option<PublicFailure> {
    apply(RULES, mapping).map(|failure| keep_upstream_response(mapping, failure))
}

/// Deliberate divergence from `_map_vertex_exception`, which replaces the provider response
/// with a stub and so drops the upstream body and `retry-after`. The response keeps the
/// status the public class carries.
fn keep_upstream_response(mapping: &Mapping<'_>, failure: PublicFailure) -> PublicFailure {
    let (PublicKind::Status { status_class, .. }, Some(upstream), false) = (
        &failure.kind,
        &mapping.original.response,
        mapping.original.status_is_synthesized,
    ) else {
        return failure;
    };
    PublicFailure {
        kind: PublicKind::Status {
            status_class: *status_class,
            response: Some(ResponseArg::Upstream(UpstreamResponse {
                status: status_class.status_code(),
                ..upstream.clone()
            })),
        },
        ..failure
    }
}

#[cfg(test)]
mod tests {
    use super::super::testing::{context, failure, http, status, upstream, with_debug};
    use super::super::{ExceptionFamily, HttpStub, OriginalException};
    use super::*;

    fn mapped(original: &OriginalException) -> Option<PublicFailure> {
        let context = context("vertex_ai", ExceptionFamily::VertexAi);
        map(&Mapping::new(&context, original))
    }

    fn kept(class: StatusClass, body: &str) -> PublicKind {
        status(class, upstream(class.status_code(), body))
    }

    #[rstest::rstest]
    #[case::api_not_enabled(
        400,
        "Vertex AI API has not been used in project x",
        with_debug(failure(
            kept(
                StatusClass::BadRequest,
                "Vertex AI API has not been used in project x"
            ),
            "litellm.BadRequestError: vertex_aiException - Vertex AI API has not been used in project x",
            "vertex_ai",
        ))
    )]
    #[case::project_not_found(
        400,
        "Unable to find your project",
        with_debug(failure(
            kept(StatusClass::BadRequest, "Unable to find your project"),
            "litellm.BadRequestError: vertex_aiException - Unable to find your project",
            "vertex_ai",
        ))
    )]
    #[case::payload_too_large(
        400,
        "400 Request payload size exceeds the limit",
        failure(
            kept(
                StatusClass::ContextWindowExceeded,
                "400 Request payload size exceeds the limit"
            ),
            "Vertex_aiException - 400 Request payload size exceeds the limit",
            "vertex_ai",
        )
    )]
    #[case::context_window(
        500,
        "This model's maximum context length is 10",
        with_debug(failure(
            kept(
                StatusClass::ContextWindowExceeded,
                "This model's maximum context length is 10"
            ),
            "ContextWindowExceededError: Vertex_aiException - This model's maximum context length is 10",
            "vertex_ai",
        ))
    )]
    #[case::unknown_error(
        400,
        "None Unknown Error.",
        with_debug(failure(
            kept(StatusClass::InternalServer, "None Unknown Error."),
            "litellm.InternalServerError: vertex_aiException - None Unknown Error.",
            "vertex_ai",
        ))
    )]
    #[case::no_parts(
        400,
        "Content has no parts.",
        with_debug(failure(
            kept(StatusClass::InternalServer, "Content has no parts."),
            "litellm.InternalServerError: vertex_aiException - Content has no parts.",
            "vertex_ai",
        ))
    )]
    #[case::api_key_not_valid(
        400,
        "API key not valid.",
        with_debug(failure(
            kept(StatusClass::Authentication, "API key not valid."),
            "Vertex_aiException - API key not valid.",
            "vertex_ai",
        ))
    )]
    #[case::forbidden_text(
        400,
        "got a 403",
        with_debug(failure(
            kept(StatusClass::BadRequest, "got a 403"),
            "Vertex_aiException BadRequestError - got a 403",
            "vertex_ai",
        ))
    )]
    #[case::response_blocked(
        400,
        "The response was blocked.",
        with_debug(failure(
            kept(StatusClass::ContentPolicyViolation, "The response was blocked."),
            "Vertex_aiException ContentPolicyViolationError - The response was blocked.",
            "vertex_ai",
        ))
    )]
    #[case::output_blocked(
        400,
        "Output blocked by content filtering policy",
        with_debug(failure(
            kept(
                StatusClass::ContentPolicyViolation,
                "Output blocked by content filtering policy"
            ),
            "Vertex_aiException ContentPolicyViolationError - Output blocked by content filtering policy",
            "vertex_ai",
        ))
    )]
    #[case::quota_marker(
        400,
        "Quota exceeded for aiplatform",
        with_debug(failure(
            kept(StatusClass::RateLimit, "Quota exceeded for aiplatform"),
            "litellm.RateLimitError: vertex_aiException - Quota exceeded for aiplatform",
            "vertex_ai",
        ))
    )]
    #[case::wrapped_429(
        503,
        r#"{"error": {"code": "429"}}"#,
        with_debug(failure(
            kept(StatusClass::RateLimit, r#"{"error": {"code": "429"}}"#),
            r#"litellm.RateLimitError: vertex_aiException - {"error": {"code": "429"}}"#,
            "vertex_ai",
        ))
    )]
    #[case::overloaded(
        400,
        "The model is overloaded.",
        with_debug(failure(
            kept(StatusClass::InternalServer, "The model is overloaded."),
            "litellm.InternalServerError: vertex_aiException - The model is overloaded.",
            "vertex_ai",
        ))
    )]
    #[case::internal_server_text(
        400,
        "500 Internal Server Error",
        with_debug(failure(
            kept(StatusClass::InternalServer, "500 Internal Server Error"),
            "litellm.InternalServerError: vertex_aiException - 500 Internal Server Error",
            "vertex_ai",
        ))
    )]
    fn each_text_rule_maps_by_the_body(
        #[case] status_code: u16,
        #[case] body: &str,
        #[case] expected: PublicFailure,
    ) {
        assert_eq!(mapped(&http(status_code, body)), Some(expected));
    }

    #[rstest::rstest]
    #[case::bad_request(
        400,
        with_debug(failure(
            kept(StatusClass::BadRequest, "rejected"),
            "Vertex_aiException BadRequestError - rejected",
            "vertex_ai"
        ))
    )]
    #[case::authentication(
        401,
        failure(
            kept(StatusClass::Authentication, "rejected"),
            "Vertex_aiException - rejected",
            "vertex_ai"
        )
    )]
    #[case::permission_denied(
        403,
        failure(
            kept(StatusClass::PermissionDenied, "rejected"),
            "Vertex_aiException - rejected",
            "vertex_ai"
        )
    )]
    #[case::not_found(
        404,
        failure(
            kept(StatusClass::NotFound, "rejected"),
            "Vertex_aiException - rejected",
            "vertex_ai"
        )
    )]
    #[case::request_timeout(408, failure(PublicKind::Timeout { status: None }, "Vertex_aiException - rejected", "vertex_ai"))]
    #[case::rate_limited(
        429,
        with_debug(failure(
            kept(StatusClass::RateLimit, "rejected"),
            "litellm.RateLimitError: Vertex_aiException - rejected",
            "vertex_ai"
        ))
    )]
    #[case::internal_server(
        500,
        with_debug(failure(
            kept(StatusClass::InternalServer, "rejected"),
            "Vertex_aiException InternalServerError - rejected",
            "vertex_ai"
        ))
    )]
    #[case::bad_gateway(
        502,
        failure(
            PublicKind::ApiConnection,
            "Vertex_aiException - rejected",
            "vertex_ai"
        )
    )]
    #[case::service_unavailable(
        503,
        failure(
            kept(StatusClass::ServiceUnavailable, "rejected"),
            "Vertex_aiException - rejected",
            "vertex_ai"
        )
    )]
    fn each_status_rule_maps_by_the_status(
        #[case] status_code: u16,
        #[case] expected: PublicFailure,
    ) {
        assert_eq!(mapped(&http(status_code, "rejected")), Some(expected));
    }

    #[rstest::rstest]
    #[case::unmapped_status(409)]
    #[case::gateway_timeout(504)]
    fn statuses_without_a_rule_fall_through(#[case] status_code: u16) {
        assert_eq!(mapped(&http(status_code, "rejected")), None);
    }

    #[rstest::rstest]
    #[case::stub_without_an_upstream_response(
        OriginalException::Response { message: "got a 403".into() },
        status(StatusClass::BadRequest, Some(ResponseArg::Stub(HttpStub { status: 403, method: "POST", url: VERTEX_URL_WITH_SPACE, content: None })))
    )]
    #[case::stub_for_a_synthesized_status(
        OriginalException::Connection { message: "got a 403".into() },
        status(StatusClass::BadRequest, Some(ResponseArg::Stub(HttpStub { status: 403, method: "POST", url: VERTEX_URL_WITH_SPACE, content: None })))
    )]
    fn the_rule_response_stays_when_there_is_no_real_upstream_response(
        #[case] original: OriginalException,
        #[case] kind: PublicKind,
    ) {
        assert_eq!(mapped(&original).map(|failure| failure.kind), Some(kind));
    }

    #[test]
    fn a_synthesized_500_keeps_the_internal_server_stub() {
        let original = OriginalException::Connection {
            message: "refused".into(),
        };
        assert_eq!(
            mapped(&original),
            Some(with_debug(failure(
                status(
                    StatusClass::InternalServer,
                    Some(ResponseArg::Stub(HttpStub {
                        status: 500,
                        method: "completion",
                        url: "https://github.com/BerriAI/litellm",
                        content: Some("refused".into()),
                    }))
                ),
                "Vertex_aiException InternalServerError - refused",
                "vertex_ai"
            )))
        );
    }

    #[rstest::rstest]
    #[case::project_before_payload_size(
        "Unable to find your project 400 Request payload size exceeds",
        StatusClass::BadRequest,
        "litellm.BadRequestError: vertex_aiException - Unable to find your project 400 Request payload size exceeds",
        true
    )]
    #[case::payload_size_before_context_window(
        "400 Request payload size exceeds; This model's maximum context length is 10",
        StatusClass::ContextWindowExceeded,
        "Vertex_aiException - 400 Request payload size exceeds; This model's maximum context length is 10",
        false
    )]
    #[case::api_key_before_forbidden(
        "API key not valid. 403",
        StatusClass::Authentication,
        "Vertex_aiException - API key not valid. 403",
        true
    )]
    #[case::forbidden_before_blocked(
        "403 The response was blocked.",
        StatusClass::BadRequest,
        "Vertex_aiException BadRequestError - 403 The response was blocked.",
        true
    )]
    #[case::blocked_before_quota(
        "The response was blocked. Resource exhausted",
        StatusClass::ContentPolicyViolation,
        "Vertex_aiException ContentPolicyViolationError - The response was blocked. Resource exhausted",
        true
    )]
    #[case::quota_before_overloaded(
        "Resource exhausted The model is overloaded.",
        StatusClass::RateLimit,
        "litellm.RateLimitError: vertex_aiException - Resource exhausted The model is overloaded.",
        true
    )]
    fn the_earlier_rule_wins_when_two_apply(
        #[case] body: &str,
        #[case] class: StatusClass,
        #[case] message: &str,
        #[case] debug: bool,
    ) {
        let expected = failure(kept(class, body), message, "vertex_ai");
        assert_eq!(
            mapped(&http(401, body)),
            Some(if debug {
                with_debug(expected)
            } else {
                expected
            })
        );
    }
}
