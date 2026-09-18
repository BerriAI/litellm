use super::Mapping;
use super::public::{PublicFailure, StatusClass};
use super::rules::{Kind, ResponseChoice, Rule, apply, contains_any};

const fn with_response(class: StatusClass) -> Kind {
    Kind::Status {
        class,
        response: ResponseChoice::Provider,
    }
}

fn original(mapping: &Mapping<'_>) -> String {
    format!("CohereException - {}", mapping.original.message)
}

fn status_is(mapping: &Mapping<'_>, statuses: &[u16]) -> bool {
    mapping
        .original
        .status
        .is_some_and(|status| statuses.contains(&status))
}

/// `_map_cohere_exception`, in its branch order. A failure no rule claims falls through to
/// the status table.
const RULES: &[Rule] = &[
    Rule {
        when: |mapping| {
            contains_any(
                &mapping.error_str,
                &["invalid api token", "No API key provided."],
            )
        },
        kind: with_response(StatusClass::Authentication),
        message: original,
        debug: false,
    },
    Rule {
        when: |mapping| mapping.error_str.contains("invalid type: parameter"),
        kind: with_response(StatusClass::BadRequest),
        message: original,
        debug: false,
    },
    Rule {
        when: |mapping| mapping.error_str.contains("too many tokens"),
        kind: with_response(StatusClass::ContextWindowExceeded),
        message: original,
        debug: false,
    },
    Rule {
        when: |mapping| {
            mapping
                .error_str
                .to_lowercase()
                .contains("internal server error")
        },
        kind: with_response(StatusClass::InternalServer),
        message: |mapping| format!("CohereException - {}", mapping.error_str),
        debug: false,
    },
    Rule {
        when: |mapping| status_is(mapping, &[400, 498]),
        kind: with_response(StatusClass::BadRequest),
        message: original,
        debug: false,
    },
    Rule {
        when: |mapping| status_is(mapping, &[408]),
        kind: Kind::Timeout(None),
        message: original,
        debug: false,
    },
    Rule {
        when: |mapping| status_is(mapping, &[500]),
        kind: with_response(StatusClass::InternalServer),
        message: original,
        debug: false,
    },
];

pub(super) fn map(mapping: &Mapping<'_>) -> Option<PublicFailure> {
    apply(RULES, mapping).map(|failure| PublicFailure {
        llm_provider: Some("cohere".to_string()),
        ..failure
    })
}

#[cfg(test)]
mod tests {
    use super::super::testing::{context, failure, http, status, upstream};
    use super::super::{ExceptionFamily, OriginalException, PublicKind};
    use super::*;

    fn mapped(provider: &str, original: &OriginalException) -> Option<PublicFailure> {
        let context = context(provider, ExceptionFamily::Cohere);
        map(&Mapping::new(&context, original))
    }

    fn cohere(class: StatusClass, status_code: u16, body: &str, message: &str) -> PublicFailure {
        failure(
            status(class, upstream(status_code, body)),
            message,
            "cohere",
        )
    }

    #[rstest::rstest]
    #[case::invalid_token(
        500,
        "invalid api token",
        cohere(
            StatusClass::Authentication,
            500,
            "invalid api token",
            "CohereException - invalid api token"
        )
    )]
    #[case::no_api_key(
        500,
        "No API key provided.",
        cohere(
            StatusClass::Authentication,
            500,
            "No API key provided.",
            "CohereException - No API key provided."
        )
    )]
    #[case::invalid_parameter(
        500,
        "invalid type: parameter x",
        cohere(
            StatusClass::BadRequest,
            500,
            "invalid type: parameter x",
            "CohereException - invalid type: parameter x"
        )
    )]
    #[case::too_many_tokens(
        500,
        "too many tokens",
        cohere(
            StatusClass::ContextWindowExceeded,
            500,
            "too many tokens",
            "CohereException - too many tokens"
        )
    )]
    #[case::internal_server_text(
        400,
        "Internal Server Error",
        cohere(
            StatusClass::InternalServer,
            400,
            "Internal Server Error",
            "CohereException - Internal Server Error"
        )
    )]
    #[case::bad_request(
        400,
        "rejected",
        cohere(StatusClass::BadRequest, 400, "rejected", "CohereException - rejected")
    )]
    #[case::invalid_token_status(
        498,
        "rejected",
        cohere(StatusClass::BadRequest, 498, "rejected", "CohereException - rejected")
    )]
    #[case::request_timeout(408, "rejected", failure(PublicKind::Timeout { status: None }, "CohereException - rejected", "cohere"))]
    #[case::internal_server(
        500,
        "rejected",
        cohere(
            StatusClass::InternalServer,
            500,
            "rejected",
            "CohereException - rejected"
        )
    )]
    fn each_rule_maps_and_reports_cohere(
        #[case] status_code: u16,
        #[case] body: &str,
        #[case] expected: PublicFailure,
    ) {
        assert_eq!(mapped("azure_ai", &http(status_code, body)), Some(expected));
    }

    #[rstest::rstest]
    #[case::unmapped_status(409)]
    #[case::unauthorized(401)]
    fn statuses_without_a_rule_fall_through(#[case] status_code: u16) {
        assert_eq!(mapped("cohere", &http(status_code, "rejected")), None);
    }

    #[test]
    fn the_internal_server_rule_uses_the_redacted_text() {
        let body = "internal server error Bearer abcdefghijklmnop";
        assert_eq!(
            mapped("cohere", &http(400, body)),
            Some(cohere(
                StatusClass::InternalServer,
                400,
                body,
                "CohereException - internal server error REDACTED"
            ))
        );
    }

    #[rstest::rstest]
    #[case::token_before_parameter(
        "invalid api token invalid type: parameter",
        StatusClass::Authentication
    )]
    #[case::parameter_before_tokens(
        "invalid type: parameter too many tokens",
        StatusClass::BadRequest
    )]
    #[case::tokens_before_internal(
        "too many tokens Internal Server Error",
        StatusClass::ContextWindowExceeded
    )]
    #[case::internal_before_status("Internal Server Error", StatusClass::InternalServer)]
    fn the_earlier_rule_wins_when_two_apply(#[case] body: &str, #[case] class: StatusClass) {
        assert_eq!(
            mapped("cohere", &http(400, body)),
            Some(cohere(
                class,
                400,
                body,
                &format!("CohereException - {body}")
            ))
        );
    }
}
