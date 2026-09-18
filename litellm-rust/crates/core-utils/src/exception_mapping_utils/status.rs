use super::public::{PublicFailure, StatusClass};
use super::rules::{ApiStatus, Kind, ResponseChoice, Rule, apply};
use super::{DOCS_URL, Mapping};

const fn with_response(class: StatusClass) -> Kind {
    Kind::Status {
        class,
        response: ResponseChoice::Provider,
    }
}

fn message(mapping: &Mapping<'_>) -> String {
    format!("{} - {}", mapping.exception_provider, mapping.error_str)
}

fn status(mapping: &Mapping<'_>) -> u16 {
    mapping.original.status.unwrap_or_default()
}

/// `_map_exception_by_status`, the fallback for a provider error no provider mapper claimed.
const RULES: &[Rule] = &[
    Rule {
        when: |mapping| status(mapping) == 401,
        kind: with_response(StatusClass::Authentication),
        message,
        debug: true,
    },
    Rule {
        when: |mapping| status(mapping) == 403,
        kind: with_response(StatusClass::PermissionDenied),
        message,
        debug: true,
    },
    Rule {
        when: |mapping| status(mapping) == 404,
        kind: with_response(StatusClass::NotFound),
        message,
        debug: true,
    },
    Rule {
        when: |mapping| status(mapping) == 408,
        kind: Kind::Timeout(None),
        message,
        debug: true,
    },
    Rule {
        when: |mapping| status(mapping) == 429,
        kind: with_response(StatusClass::RateLimit),
        message,
        debug: true,
    },
    Rule {
        when: |mapping| status(mapping) == 500,
        kind: with_response(StatusClass::InternalServer),
        message,
        debug: true,
    },
    Rule {
        when: |mapping| status(mapping) == 502,
        kind: with_response(StatusClass::BadGateway),
        message,
        debug: true,
    },
    Rule {
        when: |mapping| status(mapping) == 503,
        kind: with_response(StatusClass::ServiceUnavailable),
        message,
        debug: true,
    },
    Rule {
        when: |mapping| status(mapping) == 504,
        kind: Kind::Timeout(Some(504)),
        message,
        debug: true,
    },
    Rule {
        when: |mapping| status(mapping) < 500,
        kind: with_response(StatusClass::BadRequest),
        message,
        debug: true,
    },
    Rule {
        when: |_| true,
        kind: Kind::Api {
            status: ApiStatus::Original,
            request_url: DOCS_URL,
        },
        message,
        debug: true,
    },
];

/// Only a real provider status of 400 or more reaches the table; a status the HTTP handler
/// synthesized for a failure without a response does not.
pub(super) fn map(mapping: &Mapping<'_>) -> Option<PublicFailure> {
    let status = mapping.original.status?;
    if status < 400 || mapping.original.status_is_synthesized {
        return None;
    }
    apply(RULES, mapping)
}

#[cfg(test)]
mod tests {
    use super::super::testing::{context, failure, http, upstream, with_debug};
    use super::super::{ExceptionFamily, OriginalException, PublicKind};
    use super::*;

    fn mapped(original: &OriginalException) -> Option<PublicFailure> {
        let context = context("reducto", ExceptionFamily::Other);
        map(&Mapping::new(&context, original))
    }

    fn classified(class: StatusClass, status_code: u16) -> PublicKind {
        PublicKind::Status {
            status_class: class,
            response: upstream(status_code, "rejected"),
        }
    }

    #[rstest::rstest]
    #[case::authentication(401, classified(StatusClass::Authentication, 401))]
    #[case::permission_denied(403, classified(StatusClass::PermissionDenied, 403))]
    #[case::not_found(404, classified(StatusClass::NotFound, 404))]
    #[case::request_timeout(408, PublicKind::Timeout { status: None })]
    #[case::rate_limited(429, classified(StatusClass::RateLimit, 429))]
    #[case::internal_server(500, classified(StatusClass::InternalServer, 500))]
    #[case::bad_gateway(502, classified(StatusClass::BadGateway, 502))]
    #[case::service_unavailable(503, classified(StatusClass::ServiceUnavailable, 503))]
    #[case::gateway_timeout(504, PublicKind::Timeout { status: Some(504) })]
    #[case::lowest_client_error(400, classified(StatusClass::BadRequest, 400))]
    #[case::other_client_error(409, classified(StatusClass::BadRequest, 409))]
    #[case::highest_client_error(499, classified(StatusClass::BadRequest, 499))]
    #[case::other_server_error(501, PublicKind::Api { status: 501, request_url: DOCS_URL })]
    fn every_mapped_status_and_the_fallback(#[case] status_code: u16, #[case] kind: PublicKind) {
        assert_eq!(
            mapped(&http(status_code, "rejected")),
            Some(with_debug(failure(
                kind,
                "ReductoException - rejected",
                "reducto"
            )))
        );
    }

    #[rstest::rstest]
    #[case::below_client_errors(http(399, "rejected"))]
    #[case::synthesized(OriginalException::Connection { message: "refused".into() })]
    #[case::no_status(OriginalException::Response { message: "bad body".into() })]
    fn failures_the_table_does_not_claim(#[case] original: OriginalException) {
        assert_eq!(mapped(&original), None);
    }
}
